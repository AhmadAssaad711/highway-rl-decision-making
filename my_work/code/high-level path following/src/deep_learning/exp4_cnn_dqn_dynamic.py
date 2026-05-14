"""
CNN-Based DQN for Lane Selection in Dynamic Highway Traffic
Using highway-env (Gymnasium) + PyTorch Deep Q-Network

Key features:
  - 3 lanes, dynamic vehicles with random speeds (v ∈ [15, 30] m/s)
  - Ego vehicle at constant speed (≈25 m/s, controlled by highway-env)
  - Discrete lane-change-only actions (LANE_LEFT=0, IDLE=1, LANE_RIGHT=2)
  - Occupancy-grid observation  (H × 3 × 2):
       channel 0 → occupancy (1 if cell occupied, 0 otherwise)
       channel 1 → normalised relative velocity
  - CNN-based Deep Q-Network with experience replay & target network
  - Reward: survival bonus + collision penalty + TTC penalty + lane-change penalty
  - 300-step episode limit (5 Hz × 60 s)

Extensible later for:
  - Variable ego speed
  - Curriculum learning
  - Larger grids / more channels
"""

import os
import time
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 – registers highway-v0 etc.
import numpy as np
import matplotlib
matplotlib.use(os.environ.get("EXP4_CNN_DQN_MPL_BACKEND", "TkAgg"))
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================================
# Paths
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("EXP4_CNN_DQN_OUTPUT_DIR", str(PROJECT_ROOT / "artifacts" / "dqn" / "exp4_cnn_dqn_dynamic")))
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
MODEL_PATH = str(MODEL_DIR / "exp4_dqn_cnn.pt")
PLOT_PATH = RESULTS_DIR / "exp4_cnn_dqn_training.png"


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

# Grid parameters
GRID_HEIGHT = 15      # number of forward-distance cells
CELL_LENGTH = 10.0    # metres per cell (grid covers 0…150 m ahead)
V_REL_CLIP = 20.0     # m/s – clip value for velocity normalisation

# TTC / reward parameters
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
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": False,
        "order": "sorted",
    },

    "offroad_terminal": True,
}

N_ACTIONS = 3   # LANE_LEFT, IDLE, LANE_RIGHT


# ======================================================================
# 2. Dynamic-traffic wrapper with occupancy-grid observation
# ======================================================================

class DynamicTrafficCNNWrapper(gym.Wrapper):
    """
    Wraps highway-v0 to:
      1. Spawn dynamic vehicles continuously ahead of ego.
      2. Convert kinematics observation → occupancy grid  (H × 3 × 2).
      3. Apply extra reward shaping (survival, TTC, lane-change).

    Observation tensor layout (numpy, dtype float32):
      grid[row, lane, 0]  = occupancy  (1.0 if occupied, 0.0 otherwise)
      grid[row, lane, 1]  = normalised relative velocity  (clipped to [-1, 1])
      row 0 = nearest cells, row H-1 = farthest cells.
    """

    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng()
        self._next_x = 0.0
        self._traffic: list = []
        self._prev_lane: int = 1

        # Override observation space to match the grid
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(GRID_HEIGHT, NUM_LANES, 2),
            dtype=np.float32,
        )

        # Cached per-lane TTC (for reward shaping)
        self.ttc = np.full(NUM_LANES, TTC_MAX, dtype=np.float64)

    # ----------------------------------------------------------------
    # Reset / Step
    # ----------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        seed = kwargs.get("seed")
        self._rng = np.random.default_rng(seed)
        self._traffic = []

        # Remove default vehicles (except ego)
        ego = self.env.unwrapped.vehicle
        self.env.unwrapped.road.vehicles[:] = [ego]
        self._prev_lane = ego.lane_index[2]

        # Start spawning ahead
        self._next_x = ego.position[0] + VEHICLE_SPACING_MIN
        self._populate_ahead()

        obs = self.env.unwrapped.observation_type.observe()
        grid = self._build_grid(obs)
        self._update_ttc(obs)

        info["ttc"] = self.ttc.tolist()
        return grid, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._populate_ahead()
        self._prune_behind()

        grid = self._build_grid(obs)
        self._update_ttc(obs)

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

        info["ttc"] = self.ttc.tolist()
        return grid, reward, terminated, truncated, info

    # ----------------------------------------------------------------
    # Occupancy-grid builder
    # ----------------------------------------------------------------

    def _build_grid(self, obs) -> np.ndarray:
        """Convert kinematics observation → (H, 3, 2) occupancy grid."""
        grid = np.zeros((GRID_HEIGHT, NUM_LANES, 2), dtype=np.float32)
        ego = self.env.unwrapped.vehicle
        ego_lane = ego.lane_index[2]

        for i in range(1, len(obs)):
            if obs[i, 0] < 0.5:       # not present
                continue
            dx = obs[i, 1]             # relative longitudinal distance
            dy = obs[i, 2]             # relative lateral distance
            vx_rel = obs[i, 3]         # relative vx (other − ego)

            if dx < 0:                 # behind us
                continue

            # Map to grid row
            row = int(dx / CELL_LENGTH)
            if not (0 <= row < GRID_HEIGHT):
                continue

            # Map lateral offset to absolute lane index
            rel_lane = int(round(dy / LANE_WIDTH_HWY))
            abs_lane = ego_lane + rel_lane
            if not (0 <= abs_lane < NUM_LANES):
                continue

            # Fill cell (keep closest / most dangerous)
            if grid[row, abs_lane, 0] < 0.5:
                grid[row, abs_lane, 0] = 1.0
                grid[row, abs_lane, 1] = np.clip(
                    -vx_rel / V_REL_CLIP, -1.0, 1.0
                )

        return grid

    # ----------------------------------------------------------------
    # TTC computation (for reward shaping)
    # ----------------------------------------------------------------

    def _update_ttc(self, obs):
        """Compute per-lane TTC from the kinematics observation."""
        ego = self.env.unwrapped.vehicle
        ego_lane = ego.lane_index[2]

        closest_dist = np.full(NUM_LANES, np.inf)
        closest_vrel = np.zeros(NUM_LANES)

        for i in range(1, len(obs)):
            if obs[i, 0] < 0.5:
                continue
            dx = obs[i, 1]
            dy = obs[i, 2]
            vx_rel = obs[i, 3]

            if dx < 0:
                continue

            rel_lane = int(round(dy / LANE_WIDTH_HWY))
            abs_lane = ego_lane + rel_lane
            if not (0 <= abs_lane < NUM_LANES):
                continue

            if dx < closest_dist[abs_lane]:
                closest_dist[abs_lane] = dx
                closest_vrel[abs_lane] = -vx_rel  # positive = closing

        self.ttc[:] = TTC_MAX
        for lane in range(NUM_LANES):
            if closest_dist[lane] < np.inf and closest_vrel[lane] > 0:
                self.ttc[lane] = min(
                    closest_dist[lane] / closest_vrel[lane], TTC_MAX
                )

    # ----------------------------------------------------------------
    # Vehicle management
    # ----------------------------------------------------------------

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
    """Create a configured dynamic-traffic highway environment with grid obs."""
    env = gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return DynamicTrafficCNNWrapper(env)


# ======================================================================
# 3. CNN-based DQN model
# ======================================================================

class CNNDQN(nn.Module):
    """
    CNN-based Deep Q-Network for occupancy-grid input.

    Input : (batch, 2, H, 3)       — channels-first from (H, 3, 2)
    Output: (batch, n_actions)      — Q-values per action
    """

    def __init__(
        self,
        grid_height: int = GRID_HEIGHT,
        n_lanes: int = NUM_LANES,
        n_channels: int = 2,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        conv_out_size = 64 * grid_height * n_lanes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        """x: (batch, channels, H, lanes)"""
        h = self.conv(x)
        return self.fc(h)


def grid_to_tensor(grid: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, lanes, channels) → (1, channels, H, lanes) float32 tensor."""
    t = torch.from_numpy(grid).float()          # (H, 3, 2)
    t = t.permute(2, 0, 1).unsqueeze(0)         # (1, 2, H, 3)
    return t.to(device)


# ======================================================================
# 4. Experience Replay Buffer
# ======================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


# ======================================================================
# 5. DQN Training
# ======================================================================

def train_dqn(
    total_episodes: int = 10000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    batch_size: int = 64,
    replay_capacity: int = 50_000,
    target_update_freq: int = 10,
    log_interval: int = 100,
    seed: int = 42,
):
    """
    Train a CNN-based DQN agent on the dynamic-traffic highway.

    Returns
    -------
    policy_net     : CNNDQN
    rewards_log    : list[float]
    lengths_log    : list[int]
    collisions_log : list[bool]
    losses_log     : list[float]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    env = make_env()
    rng = np.random.default_rng(seed)

    policy_net = CNNDQN().to(device)
    target_net = CNNDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    replay = ReplayBuffer(capacity=replay_capacity)

    rewards_log: list[float] = []
    lengths_log: list[int] = []
    collisions_log: list[bool] = []
    losses_log: list[float] = []

    epsilon = epsilon_start

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        steps = 0
        collided = False
        ep_losses: list[float] = []

        while True:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                with torch.no_grad():
                    q_vals = policy_net(grid_to_tensor(obs, device))
                    action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            # ----- Learning step -----
            if len(replay) >= batch_size:
                s, a, r, s2, d = replay.sample(batch_size)

                # (batch, H, 3, 2) → (batch, 2, H, 3)
                s_t = torch.from_numpy(s).float().permute(0, 3, 1, 2).to(device)
                s2_t = torch.from_numpy(s2).float().permute(0, 3, 1, 2).to(device)
                a_t = torch.from_numpy(a).long().to(device)
                r_t = torch.from_numpy(r).float().to(device)
                d_t = torch.from_numpy(d).float().to(device)

                q_values = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(s2_t).max(dim=1).values
                    td_target = r_t + gamma * next_q * (1.0 - d_t)

                loss = loss_fn(q_values, td_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

                ep_losses.append(loss.item())

            if terminated:
                collided = info.get("crashed", False)
                break
            if truncated:
                break

        rewards_log.append(ep_reward)
        lengths_log.append(steps)
        collisions_log.append(collided)
        losses_log.append(np.mean(ep_losses) if ep_losses else 0.0)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Periodic target network update
        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % log_interval == 0:
            avg_r = np.mean(rewards_log[-log_interval:])
            avg_l = np.mean(lengths_log[-log_interval:])
            col = np.mean(collisions_log[-log_interval:]) * 100
            avg_loss = np.mean(losses_log[-log_interval:])
            print(
                f"Ep {ep:>5d} | reward {avg_r:+7.2f} | "
                f"len {avg_l:5.0f} | col {col:4.1f}% | "
                f"loss {avg_loss:.4f} | ε {epsilon:.4f}"
            )

    env.close()

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    return policy_net, rewards_log, lengths_log, collisions_log, losses_log


def load_model(path: str = MODEL_PATH, device: torch.device = None) -> CNNDQN:
    """Load a previously saved CNN-DQN model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNDQN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


# ======================================================================
# 6. Training plots
# ======================================================================

def plot_training(rewards, lengths, collisions, losses, window=100):
    """Plot reward, episode length, collision rate, and loss over training."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    episodes = np.arange(1, len(rewards) + 1)

    # --- 1. Episode reward ---
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.12, color="steelblue", label="raw")
    smooth_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(episodes[window - 1:], smooth_r, color="steelblue",
            label=f"MA-{window}")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.set_title("CNN-DQN Training — Dynamic Traffic Lane Selection (highway-env)")

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
    ax.set_ylim(-5, 105)

    # --- 4. Training loss ---
    ax = axes[3]
    ax.plot(episodes, losses, alpha=0.12, color="purple")
    smooth_loss = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(episodes[window - 1:], smooth_loss, color="purple")
    ax.set_ylabel("Training Loss")
    ax.set_xlabel("Episode")

    plt.tight_layout()
    save_path = str(PLOT_PATH)
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot → {save_path}")
    plt.show()


# ======================================================================
# 7. Visual evaluation
# ======================================================================

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}


def evaluate(model: CNNDQN = None, episodes: int = 10, model_path: str = MODEL_PATH):
    """
    Run the greedy policy with highway-env rendered on screen (Pygame).
    The agent "graduates" if it survives all 300 steps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_model(model_path, device)

    env = make_env(render_mode="human")
    rng = np.random.default_rng(99)

    total_rewards = []
    collisions = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        env.render()
        ep_reward = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                q_vals = model(grid_to_tensor(obs, device))
                action = int(q_vals.argmax(dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            ep_reward += reward
            steps += 1
            time.sleep(0.05)   # slow down for human viewing

            if terminated or truncated:
                break

        result = "COLLISION" if info.get("crashed") else "SURVIVED 300 steps"
        if info.get("crashed"):
            collisions += 1
        total_rewards.append(ep_reward)
        print(
            f"Episode {ep}/{episodes}  |  steps={steps:>3d}  |  "
            f"reward={ep_reward:+.2f}  |  "
            f"TTC={[f'{t:.1f}' for t in info['ttc']]}  |  {result}"
        )
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
    print("  CNN-DQN — Dynamic Traffic Lane Selection — highway-env")
    print("=" * 60)

    # ---- Train ----
    model, rewards, lengths, collisions, losses = train_dqn(
        total_episodes=3_000,
    )

    # ---- Plot learning curves ----
    plot_training(rewards, lengths, collisions, losses)

    # NOTE: To visually evaluate with the highway-env Pygame renderer,
    #       run the separate test script:
    #
    #         python src/test_exp4_visualization.py
    #
    #       It loads the saved model from artifacts/dqn/exp4_cnn_dqn_dynamic/models/exp4_dqn_cnn.pt
    #       and renders 10 episodes so you can watch the learned policy.
