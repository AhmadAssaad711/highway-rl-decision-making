"""
Static Obstacle Avoidance on a 3-Lane Highway
Using highway-env (Gymnasium) + Tabular Q-Learning

Configured for:
  - 3 lanes, static obstacles (other vehicles frozen at speed 0)
  - Discrete lane-change-only actions (LANE_LEFT / IDLE / LANE_RIGHT)
  - Kinematics observation (discretised for Q-learning)
  - 300-step episode limit (agent "graduates" if it survives all 300)

Extensible later for:
  - Moving vehicles (remove the StaticObstacleWrapper)
  - Variable speeds (enable longitudinal actions)
  - Larger observation windows
  - Curriculum learning
"""

import os
import pickle
import time

import gymnasium as gym
import highway_env  # noqa: F401  – registers highway-v0 etc.
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path where the trained Q-table is saved / loaded
ARTIFACTS_DIR = os.path.join("artifacts", "basic")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
Q_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "q_table.pkl")


# ======================================================================
# 1. Environment configuration
# ======================================================================

ENV_CONFIG = {
    # Road
    "lanes_count": 3,
    "vehicles_count": 0,            # we spawn our own obstacles in the wrapper
    "vehicles_density": 0,
    "initial_lane_id": 1,           # ego starts in centre lane
    "ego_spacing": 3,               # initial gap around ego at spawn

    # Episode length: 300 policy steps  (5 Hz × 60 s = 300 steps)
    "duration": 60,
    "policy_frequency": 5,
    "simulation_frequency": 15,

    # Reward shaping
    "collision_reward": -10,
    "right_lane_reward": 0,         # no lane preference
    "high_speed_reward": 0.4,       # small reward for maintaining speed
    "lane_change_reward": -0.05,    # jerk / oscillation penalty
    "normalize_reward": False,

    # Only lane-change actions (LANE_LEFT=0, IDLE=1, LANE_RIGHT=2)
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False,
        "lateral": True,
    },

    # Kinematics observation (ego row + 4 nearest vehicles)
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": False,
        "order": "sorted",
    },

    "offroad_terminal": True,
}

# Obstacle-spawning parameters
OBSTACLE_SPACING = 15.0   # metres between consecutive obstacles
OBSTACLE_LOOKAHEAD = 120  # metres ahead to keep obstacles populated
LANE_WIDTH_HWY = 4.0      # highway-env default lane width


# ======================================================================
# 2. Static-obstacle wrapper
# ======================================================================

class StaticObstacleWrapper(gym.Wrapper):
    """
    Replace default traffic with continuously-spawned static obstacles.

    Obstacles are placed every OBSTACLE_SPACING metres in a random lane,
    always keeping the region up to OBSTACLE_LOOKAHEAD metres ahead of
    the ego populated.  Obstacles that fall behind are pruned each step.
    """

    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()
        self._next_x = 0.0          # x-position of the next obstacle to spawn
        self._obstacles = []         # list of Obstacle instances we own

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        seed = kwargs.get("seed")
        self._rng = np.random.default_rng(seed)
        self._obstacles = []

        # Remove any default vehicles (except ego)
        ego = self.env.unwrapped.vehicle
        self.env.unwrapped.road.vehicles[:] = [ego]

        # Start spawning a safe distance ahead of the ego
        self._next_x = ego.position[0] + OBSTACLE_SPACING
        self._populate_ahead()

        obs = self.env.unwrapped.observation_type.observe()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._populate_ahead()
        self._prune_behind()
        return obs, reward, terminated, truncated, info

    # ---- internal helpers ----

    def _populate_ahead(self):
        """Spawn obstacles up to OBSTACLE_LOOKAHEAD metres ahead of ego."""
        from highway_env.vehicle.kinematics import Vehicle

        ego_x = self.env.unwrapped.vehicle.position[0]
        road = self.env.unwrapped.road
        n_lanes = ENV_CONFIG["lanes_count"]

        while self._next_x <= ego_x + OBSTACLE_LOOKAHEAD:
            lane_idx = int(self._rng.integers(0, n_lanes))
            y = lane_idx * LANE_WIDTH_HWY   # lanes at y = 0, 4, 8 …
            obj = Vehicle(road, position=[self._next_x, y], speed=0)
            road.vehicles.append(obj)
            self._obstacles.append(obj)
            self._next_x += OBSTACLE_SPACING

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


def make_env(render_mode=None):
    """Create a configured static-obstacle highway environment."""
    env = gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return StaticObstacleWrapper(env)


# ======================================================================
# 3. Observation discretiser (for tabular Q-learning)
# ======================================================================

LANE_WIDTH = 4.0   # highway-env default lane width (metres)


def discretize_obs(obs, env) -> tuple:
    """
    Map continuous Kinematics observation → compact hashable state.

    State = (ego_lane, danger_left, danger_same, danger_right)
      danger ∈ {0=clear, 1=far >60m, 2=medium 40-60m, 3=close <40m, 4=imminent <20m}

    State-space size = 3 × 5³ = 375
    """
    ego_lane = env.unwrapped.vehicle.lane_index[2]
    danger = [0, 0, 0]   # [left-of-ego, same-lane, right-of-ego]

    for i in range(1, len(obs)):          # skip row 0 (ego)
        if obs[i, 0] < 0.5:              # not present
            continue
        x, y = obs[i, 1], obs[i, 2]      # relative longitudinal / lateral
        if x < 0:                         # behind us → ignore
            continue

        # Which relative lane?  y ≈ −4 → left, 0 → same, +4 → right
        rel = int(round(y / LANE_WIDTH)) + 1   # map -1→0, 0→1, +1→2
        if not (0 <= rel <= 2):
            continue

        # Distance-ahead bin
        if x < 20:
            d = 4   # imminent
        elif x < 40:
            d = 3   # close
        elif x < 60:
            d = 2   # medium
        elif x < 80:
            d = 1   # far
        else:
            continue  # too far to matter

        danger[rel] = max(danger[rel], d)

    return (ego_lane, danger[0], danger[1], danger[2])


# ======================================================================
# 4. Q-Learning
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
    Train a tabular Q-learning agent on the static-obstacle highway.

    Returns
    -------
    q_table       : dict   state → np.ndarray(n_actions)
    rewards_log   : list[float]
    lengths_log   : list[int]
    collisions_log: list[bool]
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
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q(state)))

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_obs(obs, env)
            steps += 1

            # Q-learning update
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

    # Save Q-table so the test script can load it
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
# 5. Training plots
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
    ax.set_title("Q-Learning Training — Static Obstacle Avoidance (highway-env)")

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
    save_path = os.path.join(ARTIFACTS_DIR, "q_learning_training.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot → {save_path}")
    plt.show()


# ======================================================================
# 6. Visual evaluation
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
            time.sleep(0.05)   # slow down so the human can watch

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
    print("  Q-Learning — Static Obstacles — highway-env")
    print("=" * 60)

    # ---- Train ----
    q_table, rewards, lengths, collisions = train_q_learning(
        total_episodes=5_000,
    )

    # ---- Plot learning curves ----
    plot_training(rewards, lengths, collisions)

    # NOTE: To visually evaluate with the highway-env renderer,
    #       run the separate test script:
    #
    #         python src/test_env.py
    #
    #       It loads the saved Q-table from artifacts/basic/q_table.pkl
    #       and renders 10 episodes so you can watch the learned policy.
