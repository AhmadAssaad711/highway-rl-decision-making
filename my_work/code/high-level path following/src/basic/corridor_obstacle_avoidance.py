"""
Corridor Selection on a 3-Lane Highway with Varying-Width Static Obstacles
Using highway-env (Gymnasium) + Tabular Q-Learning

Key differences from static_obstacle_avoidance.py:
  - Obstacles can span 1 or 2 adjacent lanes (width ∈ {1, 2})
  - All 3 lanes are never blocked simultaneously
  - Obstacle spacing is randomised within [d_min, d_max]
  - A minimum reaction horizon (≥ 5 decision steps) is enforced
  - Centre-lane preference reward term
  - Rule-based baseline + decision-accuracy metric

Extensible later for:
  - Moving vehicles
  - Variable speeds
  - Larger occupancy grids
  - Curriculum learning
"""

import os
import pickle
import time

import gymnasium as gym
import highway_env  # noqa: F401 – registers highway-v0 etc.
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path where the trained Q-table is saved / loaded
ARTIFACTS_DIR = os.path.join("artifacts", "basic")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
Q_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "corridor_q_table.pkl")


# ======================================================================
# 1. Environment configuration
# ======================================================================

LANE_WIDTH_HWY = 4.0   # highway-env default lane width (metres)
NUM_LANES = 3

# Vehicle speed ≈ 25 m/s, policy_frequency = 5 → Δx ≈ 5 m/step
# d_min ≥ 5 * Δx = 25 m  (5 decision steps of reaction time)
OBSTACLE_D_MIN = 30.0    # metres – minimum spacing between obstacles
OBSTACLE_D_MAX = 60.0    # metres – maximum spacing (stays inside grid horizon)
OBSTACLE_LOOKAHEAD = 150 # metres ahead to keep populated

ENV_CONFIG = {
    # Road
    "lanes_count": NUM_LANES,
    "vehicles_count": 0,            # we spawn our own obstacles in the wrapper
    "vehicles_density": 0,
    "initial_lane_id": 1,           # ego starts in centre lane
    "ego_spacing": 3,

    # Episode length: 300 policy steps (5 Hz × 60 s)
    "duration": 60,
    "policy_frequency": 5,
    "simulation_frequency": 15,

    # Reward shaping  (base components handled by highway-env)
    "collision_reward": -10,
    "right_lane_reward": 0,         # we add our own centre-lane term
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.05,    # jerk / oscillation penalty
    "normalize_reward": False,

    # Only lane-change actions (LANE_LEFT=0, IDLE=1, LANE_RIGHT=2)
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False,
        "lateral": True,
    },

    # Kinematics observation (ego + nearest vehicles)
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 8,         # more slots → see multi-lane obstacles
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": False,
        "order": "sorted",
    },

    "offroad_terminal": True,
}

# Additional reward constants applied by the wrapper
REWARD_SURVIVAL = 0.1
REWARD_LANE_PREFERENCE_LAMBDA = 0.01   # r_lane = -λ|l - 1|


# ======================================================================
# 2. Corridor-obstacle wrapper
# ======================================================================

# All valid (width, lanes) patterns that do NOT block all 3 lanes
_OBSTACLE_PATTERNS: list[list[int]] = [
    # width = 1
    [0],
    [1],
    [2],
    # width = 2 (adjacent pairs)
    [0, 1],
    [1, 2],
]


class CorridorObstacleWrapper(gym.Wrapper):
    """
    Replace default traffic with continuously-spawned static obstacles
    of random width (1 or 2 lanes).

    At each spawn position a random pattern from _OBSTACLE_PATTERNS is
    chosen, placing one Vehicle per blocked lane.  Spacing between
    successive obstacles is drawn uniformly from [d_min, d_max].
    """

    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()
        self._next_x = 0.0
        self._obstacles: list = []          # Vehicle instances we own
        self._obstacle_lane_map: list[tuple[float, list[int]]] = []  # (x, lanes)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        seed = kwargs.get("seed")
        self._rng = np.random.default_rng(seed)
        self._obstacles = []
        self._obstacle_lane_map = []

        # Remove any default vehicles (except ego)
        ego = self.env.unwrapped.vehicle
        self.env.unwrapped.road.vehicles[:] = [ego]

        # First obstacle respects reaction-time constraint
        self._next_x = ego.position[0] + OBSTACLE_D_MIN
        self._populate_ahead()

        obs = self.env.unwrapped.observation_type.observe()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._populate_ahead()
        self._prune_behind()

        # --- extra reward terms applied on top of highway-env reward ---
        ego_lane = self.env.unwrapped.vehicle.lane_index[2]

        # Survival bonus
        if not info.get("crashed", False):
            reward += REWARD_SURVIVAL

        # Centre-lane preference: -λ|l - 1|
        reward -= REWARD_LANE_PREFERENCE_LAMBDA * abs(ego_lane - 1)

        return obs, reward, terminated, truncated, info

    # ---- internal helpers ----

    def _populate_ahead(self):
        """Spawn obstacles up to OBSTACLE_LOOKAHEAD metres ahead of ego."""
        from highway_env.vehicle.kinematics import Vehicle

        ego_x = self.env.unwrapped.vehicle.position[0]
        road = self.env.unwrapped.road

        while self._next_x <= ego_x + OBSTACLE_LOOKAHEAD:
            pattern = _OBSTACLE_PATTERNS[
                int(self._rng.integers(0, len(_OBSTACLE_PATTERNS)))
            ]
            for lane_idx in pattern:
                y = lane_idx * LANE_WIDTH_HWY
                obj = Vehicle(road, position=[self._next_x, y], speed=0)
                road.vehicles.append(obj)
                self._obstacles.append(obj)

            self._obstacle_lane_map.append((self._next_x, list(pattern)))

            # Random spacing for the next obstacle
            spacing = self._rng.uniform(OBSTACLE_D_MIN, OBSTACLE_D_MAX)
            self._next_x += spacing

    def _prune_behind(self):
        """Remove obstacles far behind the ego to save memory."""
        ego_x = self.env.unwrapped.vehicle.position[0]
        road = self.env.unwrapped.road
        keep = []
        for o in self._obstacles:
            if o.position[0] < ego_x - 50:
                if o in road.vehicles:
                    road.vehicles.remove(o)
            else:
                keep.append(o)
        self._obstacles = keep
        self._obstacle_lane_map = [
            (x, lanes) for x, lanes in self._obstacle_lane_map
            if x >= ego_x - 50
        ]

    def get_obstacle_lanes_ahead(self, horizon: float = 80.0) -> list[int]:
        """
        Return the set of blocked lanes for the nearest obstacle group
        within *horizon* metres ahead.  Used by the rule-based baseline.
        """
        ego_x = self.env.unwrapped.vehicle.position[0]
        for x, lanes in self._obstacle_lane_map:
            if x > ego_x and (x - ego_x) <= horizon:
                return lanes
        return []


def make_env(render_mode=None):
    """Create a configured corridor-obstacle highway environment."""
    env = gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return CorridorObstacleWrapper(env)


# ======================================================================
# 3. Observation discretiser (for tabular Q-learning)
# ======================================================================

def discretize_obs(obs, env) -> tuple:
    """
    Map continuous Kinematics observation → compact hashable state.

    State = (ego_lane, danger_left, danger_same, danger_right)
      danger ∈ {0=clear, 1=far, 2=medium, 3=close, 4=imminent}

    State-space size = 3 × 5³ = 375
    """
    ego_lane = env.unwrapped.vehicle.lane_index[2]
    danger = [0, 0, 0]   # [left-of-ego, same-lane, right-of-ego]

    for i in range(1, len(obs)):
        if obs[i, 0] < 0.5:
            continue
        x, y = obs[i, 1], obs[i, 2]
        if x < 0:
            continue

        rel = int(round(y / LANE_WIDTH_HWY)) + 1
        if not (0 <= rel <= 2):
            continue

        if x < 20:
            d = 4
        elif x < 40:
            d = 3
        elif x < 60:
            d = 2
        elif x < 80:
            d = 1
        else:
            continue

        danger[rel] = max(danger[rel], d)

    return (ego_lane, danger[0], danger[1], danger[2])


# ======================================================================
# 4. Rule-based baseline
# ======================================================================

def rule_based_action(env) -> int:
    """
    Optimal rule-based controller for corridor selection.

    Logic:
      1. If current lane is clear ahead → IDLE (1)
      2. Else → move to nearest clear lane (prefer centre)
    """
    wrapper = env  # should be CorridorObstacleWrapper
    blocked = wrapper.get_obstacle_lanes_ahead(horizon=80.0)
    ego_lane = env.unwrapped.vehicle.lane_index[2]

    if ego_lane not in blocked:
        return 1  # IDLE – current lane is safe

    # Find clear lanes, prefer one closer to centre (lane 1)
    clear = [l for l in range(NUM_LANES) if l not in blocked]
    if not clear:
        return 1  # nothing we can do

    # Sort by distance to ego lane, break ties by closeness to centre
    clear.sort(key=lambda l: (abs(l - ego_lane), abs(l - 1)))
    target = clear[0]

    if target < ego_lane:
        return 0  # LANE_LEFT
    elif target > ego_lane:
        return 2  # LANE_RIGHT
    else:
        return 1  # already there


def compute_decision_accuracy(
    q_table: dict,
    episodes: int = 50,
    seed: int = 77,
) -> float:
    """
    Compute the fraction of steps where the RL greedy action matches
    the rule-based optimal action.

    Returns accuracy in [0, 1].
    """
    env = make_env()
    rng = np.random.default_rng(seed)

    matches = 0
    total = 0

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_obs(obs, env)

        while True:
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            rl_action = int(np.argmax(q_vals))
            opt_action = rule_based_action(env)

            if rl_action == opt_action:
                matches += 1
            total += 1

            obs, _, terminated, truncated, _ = env.step(rl_action)
            state = discretize_obs(obs, env)

            if terminated or truncated:
                break

    env.close()
    accuracy = matches / total if total > 0 else 0.0
    return accuracy


# ======================================================================
# 5. Q-Learning
# ======================================================================

def train_q_learning(
    total_episodes: int = 5_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    log_interval: int = 500,
    seed: int = 42,
):
    """
    Train a tabular Q-learning agent on the corridor-obstacle highway.

    Returns
    -------
    q_table        : dict   state → np.ndarray(n_actions)
    rewards_log    : list[float]
    lengths_log    : list[int]
    collisions_log : list[bool]
    """
    env = make_env()
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    q_table: dict[tuple, np.ndarray] = {}
    rewards_log: list[float] = []
    lengths_log: list[int] = []
    collisions_log: list[bool] = []

    epsilon = epsilon_start

    def Q(s):
        if s not in q_table:
            q_table[s] = np.zeros(n_actions, dtype=np.float64)
        return q_table[s]

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_obs(obs, env)
        ep_reward = 0.0
        steps = 0
        collided = False

        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q(state)))

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_obs(obs, env)
            steps += 1

            best_next = np.max(Q(next_state))
            td_target = reward + gamma * best_next * (not terminated)
            Q(state)[action] += alpha * (td_target - Q(state)[action])

            state = next_state
            ep_reward += reward

            if terminated:
                collided = info.get("crashed", False)
                break
            if truncated:
                break

        rewards_log.append(ep_reward)
        lengths_log.append(steps)
        collisions_log.append(collided)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % log_interval == 0:
            avg_r = np.mean(rewards_log[-log_interval:])
            avg_l = np.mean(lengths_log[-log_interval:])
            col = np.mean(collisions_log[-log_interval:]) * 100
            print(
                f"Ep {ep:>5d} | reward {avg_r:+7.2f} | "
                f"len {avg_l:5.0f} | col {col:4.1f}% | "
                f"ε {epsilon:.4f} | Q-states {len(q_table)}"
            )

    env.close()

    # Save Q-table
    os.makedirs("results", exist_ok=True)
    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved → {Q_TABLE_PATH}")

    return q_table, rewards_log, lengths_log, collisions_log


def load_q_table(path: str = Q_TABLE_PATH) -> dict:
    """Load a previously saved Q-table from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
# 6. Training plots
# ======================================================================

def plot_training(rewards, lengths, collisions, window=200):
    """Plot reward, episode length, and collision rate over training."""
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    episodes = np.arange(1, len(rewards) + 1)

    # --- 1. Episode reward ---
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.12, color="steelblue", label="raw")
    smooth_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(episodes[window - 1:], smooth_r, color="steelblue",
            label=f"MA-{window}")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.set_title("Q-Learning — Corridor Obstacle Avoidance (highway-env)")

    # --- 2. Episode length ---
    ax = axes[1]
    ax.plot(episodes, lengths, alpha=0.12, color="darkorange")
    smooth_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
    ax.plot(episodes[window - 1:], smooth_l, color="darkorange")
    ax.set_ylabel("Episode Length (steps)")
    ax.axhline(300, ls="--", color="green", alpha=0.6, label="success = 300")
    ax.legend()

    # --- 3. Collision rate ---
    ax = axes[2]
    col = np.array(collisions, dtype=float)
    smooth_c = np.convolve(col, np.ones(window) / window, mode="valid") * 100
    ax.plot(episodes[window - 1:], smooth_c, color="crimson")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    save_path = os.path.join(ARTIFACTS_DIR, "corridor_q_learning_training.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot → {save_path}")
    plt.show()


# ======================================================================
# 7. Visual evaluation
# ======================================================================

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}


def evaluate(q_table: dict, episodes: int = 10):
    """
    Run the greedy policy with highway-env rendered on screen.
    The agent "graduates" if it survives all 300 steps.
    """
    env = make_env(render_mode="human")
    rng = np.random.default_rng(99)

    total_rewards = []
    collisions = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_obs(obs, env)
        ep_reward = 0.0
        steps = 0

        while True:
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            action = int(np.argmax(q_vals))

            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_obs(obs, env)
            ep_reward += reward
            steps += 1
            time.sleep(0.05)

            if terminated or truncated:
                break

        result = "COLLISION" if info.get("crashed") else "SURVIVED 300 steps"
        if info.get("crashed"):
            collisions += 1
        total_rewards.append(ep_reward)
        print(f"Episode {ep}/{episodes}  |  steps={steps:>3d}  |  "
              f"reward={ep_reward:+.2f}  |  {result}")
        time.sleep(0.5)

    env.close()
    avg_r = np.mean(total_rewards)
    col_rate = collisions / episodes * 100
    print(f"\nEvaluation summary ({episodes} episodes):")
    print(f"  Average reward : {avg_r:+.2f}")
    print(f"  Collision rate : {col_rate:.1f}%")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Q-Learning — Corridor Obstacle Avoidance — highway-env")
    print("=" * 60)

    # ---- Train ----
    q_table, rewards, lengths, collisions = train_q_learning(
        total_episodes=5_000,
    )

    # ---- Plot learning curves ----
    plot_training(rewards, lengths, collisions)

    # ---- Decision accuracy vs rule-based baseline ----
    accuracy = compute_decision_accuracy(q_table, episodes=50)
    print(f"\nDecision accuracy (vs rule-based): {accuracy:.1%}")

    # NOTE: To visually evaluate with the highway-env renderer,
    #       run the separate test script:
    #
    #         python src/test_corridor.py
    #
    #       It loads the saved Q-table from artifacts/basic/corridor_q_table.pkl
    #       and renders 10 episodes so you can watch the learned policy.
