"""
Dynamic Obstacle Avoidance on a 3-Lane Highway — Constant Ego Velocity
Using highway-env (Gymnasium) + Tabular Q-Learning

Key features:
  - 3 lanes, dynamic vehicles with random speeds (v ∈ [15, 30] m/s)
  - Ego vehicle at constant speed (≈25 m/s, controlled by highway-env)
  - Discrete lane-change-only actions (LANE_LEFT / IDLE / LANE_RIGHT)
  - Observation discretised as (ego_lane, danger_left, danger_center,
    danger_right, ttc_left, ttc_center, ttc_right)
  - TTC-based reward shaping (penalty when TTC < T_safe)
  - Survival bonus + collision penalty + lane-change penalty
  - 300-step episode limit

Extensible later for:
  - Variable ego speed
  - Curriculum learning
  - Function approximation (DQN / PPO via Stable-Baselines3)
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
Q_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "dynamic_q_table.pkl")


# ======================================================================
# 1. Environment configuration
# ======================================================================

LANE_WIDTH_HWY = 4.0   # highway-env default lane width (metres)
NUM_LANES = 3

# Traffic vehicle speed range
V_OTHER_MIN = 15.0   # m/s
V_OTHER_MAX = 30.0   # m/s

# Spacing between spawns
VEHICLE_SPACING_MIN = 25.0   # metres
VEHICLE_SPACING_MAX = 60.0   # metres
VEHICLE_LOOKAHEAD = 150      # metres ahead to keep populated

# TTC / danger parameters
TTC_MAX = 10.0    # seconds – cap value (no threat)
T_SAFE = 4.0      # seconds – safety threshold for penalty
D_SCALE = 30.0    # metres – distance scaling for danger feature

# Extra reward terms applied by the wrapper
REWARD_SURVIVAL = 0.1
ALPHA_TTC = 0.5           # TTC penalty coefficient
BETA_LANE_CHANGE = 0.05   # lane-change penalty

ENV_CONFIG = {
    # Road
    "lanes_count": NUM_LANES,
    "vehicles_count": 0,            # we spawn our own vehicles in the wrapper
    "vehicles_density": 0,
    "initial_lane_id": 1,           # ego starts in centre lane
    "ego_spacing": 3,

    # Episode length: 300 policy steps (5 Hz × 60 s)
    "duration": 60,
    "policy_frequency": 5,
    "simulation_frequency": 15,

    # Reward shaping (base components from highway-env)
    "collision_reward": -10,
    "right_lane_reward": 0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,        # we apply our own via the wrapper
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
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": False,
        "order": "sorted",
    },

    "offroad_terminal": True,
}


# ======================================================================
# 2. Dynamic-traffic wrapper
# ======================================================================

class DynamicTrafficWrapper(gym.Wrapper):
    """
    Replace default traffic with continuously-spawned dynamic vehicles.

    Vehicles are placed at random lanes with random speeds drawn from
    U(V_OTHER_MIN, V_OTHER_MAX).  Spacing is randomised within
    [VEHICLE_SPACING_MIN, VEHICLE_SPACING_MAX].

    The wrapper also computes:
      - per-lane danger features  D_i = exp(-d_i / D_SCALE)
      - per-lane TTC values
    and applies extra reward shaping (survival, TTC penalty, lane-change).
    """

    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()
        self._next_x = 0.0
        self._traffic: list = []          # Vehicle instances we own
        self._prev_lane: int = 1

        # Cached per-lane features (updated each step)
        self.danger = np.zeros(NUM_LANES, dtype=np.float64)
        self.ttc = np.full(NUM_LANES, TTC_MAX, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        seed = kwargs.get("seed")
        self._rng = np.random.default_rng(seed)
        self._traffic = []

        # Remove any default vehicles (except ego)
        ego = self.env.unwrapped.vehicle
        self.env.unwrapped.road.vehicles[:] = [ego]
        self._prev_lane = ego.lane_index[2]

        # Start spawning ahead
        self._next_x = ego.position[0] + VEHICLE_SPACING_MIN
        self._populate_ahead()

        obs = self.env.unwrapped.observation_type.observe()
        self._update_features(obs)
        info["danger"] = self.danger.tolist()
        info["ttc"] = self.ttc.tolist()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._populate_ahead()
        self._prune_behind()

        # Update lane-level features from fresh observation
        self._update_features(obs)

        ego_lane = self.env.unwrapped.vehicle.lane_index[2]

        # --- extra reward terms ---
        if not info.get("crashed", False):
            reward += REWARD_SURVIVAL

        # TTC penalty for current lane
        ttc_cur = float(self.ttc[ego_lane])
        if ttc_cur < T_SAFE:
            reward -= ALPHA_TTC * (T_SAFE - ttc_cur)

        # Lane-change penalty
        if ego_lane != self._prev_lane:
            reward -= BETA_LANE_CHANGE
        self._prev_lane = ego_lane

        info["danger"] = self.danger.tolist()
        info["ttc"] = self.ttc.tolist()
        return obs, reward, terminated, truncated, info

    # ---- feature computation ----

    def _update_features(self, obs):
        """Compute per-lane danger and TTC from the kinematics observation."""
        ego = self.env.unwrapped.vehicle
        ego_lane = ego.lane_index[2]

        closest_dist = np.full(NUM_LANES, np.inf)
        closest_vrel = np.zeros(NUM_LANES)

        for i in range(1, len(obs)):
            if obs[i, 0] < 0.5:       # not present
                continue
            dx = obs[i, 1]             # relative longitudinal distance
            dy = obs[i, 2]             # relative lateral distance
            vx_rel = obs[i, 3]         # relative vx (other − ego)

            if dx < 0:                 # behind us
                continue

            # Map lateral offset to absolute lane index
            rel_lane = int(round(dy / LANE_WIDTH_HWY))
            abs_lane = ego_lane + rel_lane
            if not (0 <= abs_lane < NUM_LANES):
                continue

            if dx < closest_dist[abs_lane]:
                closest_dist[abs_lane] = dx
                # Closing speed: positive means ego is catching up
                closest_vrel[abs_lane] = -vx_rel

        # Danger: D_i = exp(-d_i / D_SCALE)
        for lane in range(NUM_LANES):
            if closest_dist[lane] < np.inf:
                self.danger[lane] = np.exp(-closest_dist[lane] / D_SCALE)
            else:
                self.danger[lane] = 0.0

        # TTC
        self.ttc[:] = TTC_MAX
        for lane in range(NUM_LANES):
            if closest_dist[lane] < np.inf and closest_vrel[lane] > 0:
                self.ttc[lane] = min(
                    closest_dist[lane] / closest_vrel[lane], TTC_MAX
                )

    # ---- vehicle management ----

    def _populate_ahead(self):
        """Spawn dynamic vehicles up to VEHICLE_LOOKAHEAD metres ahead."""
        from highway_env.vehicle.kinematics import Vehicle

        ego_x = self.env.unwrapped.vehicle.position[0]
        road = self.env.unwrapped.road

        while self._next_x <= ego_x + VEHICLE_LOOKAHEAD:
            lane_idx = int(self._rng.integers(0, NUM_LANES))
            y = lane_idx * LANE_WIDTH_HWY
            speed = float(self._rng.uniform(V_OTHER_MIN, V_OTHER_MAX))

            veh = Vehicle(road, position=[self._next_x, y], speed=speed)
            road.vehicles.append(veh)
            self._traffic.append(veh)

            spacing = self._rng.uniform(VEHICLE_SPACING_MIN, VEHICLE_SPACING_MAX)
            self._next_x += spacing

    def _prune_behind(self):
        """Remove vehicles far behind the ego to save memory."""
        ego_x = self.env.unwrapped.vehicle.position[0]
        road = self.env.unwrapped.road
        keep = []
        for v in self._traffic:
            if v.position[0] < ego_x - 50:
                if v in road.vehicles:
                    road.vehicles.remove(v)
            else:
                keep.append(v)
        self._traffic = keep


def make_env(render_mode=None):
    """Create a configured dynamic-traffic highway environment."""
    env = gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return DynamicTrafficWrapper(env)


# ======================================================================
# 3. Observation discretiser (for tabular Q-learning)
# ======================================================================

def discretize_obs(obs, env) -> tuple:
    """
    Map continuous observation → compact hashable state.

    State = (ego_lane, danger_left, danger_center, danger_right,
             ttc_left, ttc_center, ttc_right)

    Danger bins  : {0=clear, 1=far, 2=medium, 3=close, 4=imminent}
    TTC bins     : {0=critical, 1=unsafe, 2=marginal, 3=safe, 4=clear}

    State-space size = 3 × 5³ × 5³ = 46 875
    """
    wrapper = env   # DynamicTrafficWrapper
    ego_lane = env.unwrapped.vehicle.lane_index[2]

    # --- Danger discretisation ---
    danger_bins = [0, 0, 0]
    for i in range(NUM_LANES):
        d = wrapper.danger[i]
        if d > 0.7:
            danger_bins[i] = 4   # imminent
        elif d > 0.4:
            danger_bins[i] = 3   # close
        elif d > 0.2:
            danger_bins[i] = 2   # medium
        elif d > 0.05:
            danger_bins[i] = 1   # far
        else:
            danger_bins[i] = 0   # clear

    # --- TTC discretisation ---
    ttc_bins = [0, 0, 0]
    for i in range(NUM_LANES):
        t = wrapper.ttc[i]
        if t < 2.0:
            ttc_bins[i] = 0   # critical
        elif t < 4.0:
            ttc_bins[i] = 1   # unsafe
        elif t < 6.0:
            ttc_bins[i] = 2   # marginal
        elif t < 8.0:
            ttc_bins[i] = 3   # safe
        else:
            ttc_bins[i] = 4   # clear

    return (ego_lane,
            danger_bins[0], danger_bins[1], danger_bins[2],
            ttc_bins[0], ttc_bins[1], ttc_bins[2])


# ======================================================================
# 4. Rule-based baseline
# ======================================================================

def rule_based_action(env) -> int:
    """
    Heuristic controller: pick the lane with the highest TTC.
    Prefer staying if current lane TTC is within T_SAFE.
    """
    wrapper = env
    ego_lane = env.unwrapped.vehicle.lane_index[2]
    ttc = wrapper.ttc

    # If current lane is safe, stay
    if ttc[ego_lane] >= T_SAFE:
        return 1  # IDLE

    # Otherwise move toward lane with the best TTC
    best_lane = int(np.argmax(ttc))
    if best_lane < ego_lane:
        return 0  # LANE_LEFT
    elif best_lane > ego_lane:
        return 2  # LANE_RIGHT
    else:
        return 1  # already best


def compute_decision_accuracy(
    q_table: dict,
    episodes: int = 50,
    seed: int = 77,
) -> float:
    """
    Fraction of steps where the RL greedy action matches the
    rule-based heuristic.
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
    return matches / total if total > 0 else 0.0


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
    Train a tabular Q-learning agent on the dynamic-traffic highway.

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
    ax.set_title("Q-Learning — Dynamic Traffic Avoidance (highway-env)")

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
    save_path = os.path.join(ARTIFACTS_DIR, "dynamic_q_learning_training.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot → {save_path}")
    plt.show()


# ======================================================================
# 7. Visual evaluation
# ======================================================================

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}


def evaluate(q_table: dict, episodes: int = 10):
    """
    Run the greedy policy with highway-env rendered on screen (Pygame).
    The agent "graduates" if it survives all 300 steps.
    """
    env = make_env(render_mode="human")
    rng = np.random.default_rng(99)

    total_rewards = []
    collisions = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        env.render()
        state = discretize_obs(obs, env)
        ep_reward = 0.0
        steps = 0

        while True:
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            action = int(np.argmax(q_vals))

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            state = discretize_obs(obs, env)
            ep_reward += reward
            steps += 1
            time.sleep(0.05)   # slow down for human viewing

            if terminated or truncated:
                break

        result = "COLLISION" if info.get("crashed") else "SURVIVED 300 steps"
        if info.get("crashed"):
            collisions += 1
        total_rewards.append(ep_reward)
        print(f"Episode {ep}/{episodes}  |  steps={steps:>3d}  |  "
              f"reward={ep_reward:+.2f}  |  "
              f"D={[f'{d:.2f}' for d in info['danger']]}  "
              f"TTC={[f'{t:.1f}' for t in info['ttc']]}  |  {result}")
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
    print("  Q-Learning — Dynamic Traffic Avoidance — highway-env")
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

    # NOTE: To visually evaluate with the highway-env Pygame renderer,
    #       run the separate test script:
    #
    #         python src/test_dynamic.py
    #
    #       It loads the saved Q-table from artifacts/basic/dynamic_q_table.pkl
    #       and renders 10 episodes so you can watch the learned policy.
