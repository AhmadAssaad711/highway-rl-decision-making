"""
Hybrid PPO training script using highway-env.

This baseline keeps lane decisions discrete at execution time
(LEFT / IDLE / RIGHT) while learning a continuous throttle signal.
It does not modify the original DQN baseline or its artifacts.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(PROJECT_ROOT / "artifacts" / "ppo" / "hybrid_ppo_baseline" / "elurant_ppo")
DEFAULT_EPISODE_DURATION = 50
DEFAULT_VEHICLES_COUNT = 30
DEFAULT_VEHICLES_DENSITY = 1.6
DEFAULT_EGO_SPACING = 1.0
DEFAULT_INITIAL_LANE_ID = 1


class TimestepMonitorCallback(BaseCallback):
    """Print training progress every fixed number of environment timesteps."""

    def __init__(self, every_n_steps: int = 100):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._next_print = self.every_n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            print(f"[train] timesteps={self.num_timesteps}")
            while self._next_print <= self.num_timesteps:
                self._next_print += self.every_n_steps
        return True


class LaneThrottleHybridWrapper(gym.Wrapper):
    """
    Exposes a 2D continuous action for PPO:
      action[0] -> lane signal in [-1, 1], mapped to {LEFT, IDLE, RIGHT}
      action[1] -> throttle command in [-1, 1], applied as target speed delta

    The wrapped highway environment still receives discrete lane actions.
    """

    LANE_LEFT = 0
    LANE_IDLE = 1
    LANE_RIGHT = 2

    def __init__(
        self,
        env: gym.Env,
        min_speed: float = 10.0,
        max_speed: float = 35.0,
        max_speed_delta_per_step: float = 1.25,
        lane_change_threshold: float = 0.33,
        lane_action_mode: str = "threshold",
        lane_intent_decay: float = 0.7,
        lane_intent_threshold: float = 0.35,
        lane_change_cooldown_steps: int = 3,
        lane_safety_checks: bool = False,
        min_target_front_gap: float = 12.0,
        min_target_rear_gap: float = 10.0,
        max_target_rear_closing_speed: float = 8.0,
        throttle_safety_checks: bool = False,
        min_same_lane_gap: float = 14.0,
        min_same_lane_ttc: float = 2.0,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = env.observation_space

        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.max_speed_delta_per_step = float(max_speed_delta_per_step)
        self.lane_change_threshold = float(np.clip(lane_change_threshold, 0.0, 1.0))
        self.lane_action_mode = str(lane_action_mode)
        self.lane_intent_decay = float(np.clip(lane_intent_decay, 0.0, 1.0))
        self.lane_intent_threshold = float(max(0.0, lane_intent_threshold))
        self.lane_change_cooldown_steps = max(0, int(lane_change_cooldown_steps))
        self.lane_safety_checks = bool(lane_safety_checks)
        self.min_target_front_gap = float(min_target_front_gap)
        self.min_target_rear_gap = float(min_target_rear_gap)
        self.max_target_rear_closing_speed = float(max_target_rear_closing_speed)
        self.throttle_safety_checks = bool(throttle_safety_checks)
        self.min_same_lane_gap = float(min_same_lane_gap)
        self.min_same_lane_ttc = float(max(0.1, min_same_lane_ttc))

        self.last_lane_action = self.LANE_IDLE
        self.last_throttle = 0.0
        self.last_lane_signal = 0.0
        self.lane_intent = 0.0
        self.cooldown_remaining = 0
        self.last_safety_clamped = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lane_action = self.LANE_IDLE
        self.last_throttle = 0.0
        self.last_lane_signal = 0.0
        self.lane_intent = 0.0
        self.cooldown_remaining = 0
        self.last_safety_clamped = False
        return obs, info

    def step(self, action):
        lane_signal, throttle = self._parse_action(action)

        lane_action = self._select_lane_action(lane_signal)

        self._apply_continuous_throttle(throttle)
        obs, reward, terminated, truncated, info = self.env.step(lane_action)

        self.last_lane_action = lane_action
        self.last_throttle = throttle
        self.last_lane_signal = lane_signal

        info["lane_action"] = lane_action
        info["throttle"] = throttle
        info["lane_signal"] = lane_signal
        info["lane_intent"] = self.lane_intent
        info["lane_cooldown_remaining"] = self.cooldown_remaining
        info["safety_clamped"] = self.last_safety_clamped
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < 2:
            raise ValueError(f"Expected hybrid action of shape (2,), got {a.shape}")
        lane_signal = float(np.clip(a[0], -1.0, 1.0))
        throttle = float(np.clip(a[1], -1.0, 1.0))
        return lane_signal, throttle

    def _select_lane_action(self, lane_signal: float) -> int:
        ego_lane = int(self.env.unwrapped.vehicle.lane_index[2])
        if self.lane_action_mode == "threshold":
            return self._lane_signal_to_discrete(lane_signal, ego_lane)
        if self.lane_action_mode == "intent":
            return self._lane_signal_to_intent(lane_signal, ego_lane)
        raise ValueError(f"Unsupported lane_action_mode={self.lane_action_mode!r}")

    def _lane_signal_to_discrete(self, lane_signal: float, ego_lane: int) -> int:
        if lane_signal < -self.lane_change_threshold:
            lane_action = self.LANE_LEFT
        elif lane_signal > self.lane_change_threshold:
            lane_action = self.LANE_RIGHT
        else:
            lane_action = self.LANE_IDLE

        return self._validate_lane_action(lane_action, ego_lane)

    def _lane_signal_to_intent(self, lane_signal: float, ego_lane: int) -> int:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.lane_intent *= self.lane_intent_decay
            return self.LANE_IDLE

        self.lane_intent = self.lane_intent * self.lane_intent_decay + lane_signal

        if self.lane_intent <= -self.lane_intent_threshold:
            lane_action = self.LANE_LEFT
        elif self.lane_intent >= self.lane_intent_threshold:
            lane_action = self.LANE_RIGHT
        else:
            return self.LANE_IDLE

        lane_action = self._validate_lane_action(lane_action, ego_lane)
        if lane_action == self.LANE_IDLE:
            self.lane_intent *= 0.5
            return lane_action

        if self.lane_safety_checks and not self._lane_action_is_safe(lane_action):
            self.lane_intent *= 0.4
            return self.LANE_IDLE

        self.lane_intent = 0.0
        self.cooldown_remaining = self.lane_change_cooldown_steps
        return lane_action

    def _validate_lane_action(self, lane_action: int, ego_lane: int) -> int:
        lane_count = len(self.env.unwrapped.road.network.all_side_lanes(self.env.unwrapped.vehicle.lane_index))
        # Keep lane actions valid at road boundaries.
        if ego_lane == 0 and lane_action == self.LANE_LEFT:
            return self.LANE_IDLE
        if ego_lane == lane_count - 1 and lane_action == self.LANE_RIGHT:
            return self.LANE_IDLE
        return lane_action

    def _lane_action_is_safe(self, lane_action: int) -> bool:
        vehicle = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road
        lane_offset = -1 if lane_action == self.LANE_LEFT else 1
        target_lane_index = (
            vehicle.lane_index[0],
            vehicle.lane_index[1],
            int(vehicle.lane_index[2] + lane_offset),
        )
        lane = road.network.get_lane(target_lane_index)
        front, rear = road.neighbour_vehicles(vehicle, target_lane_index)

        if front is not None:
            front_gap = vehicle.lane_distance_to(front, lane)
            if front_gap < self.min_target_front_gap:
                return False

        if rear is not None:
            rear_gap = -vehicle.lane_distance_to(rear, lane)
            closing_speed = float(rear.speed - vehicle.speed)
            if rear_gap < self.min_target_rear_gap and closing_speed > 0.0:
                return False
            if closing_speed > self.max_target_rear_closing_speed:
                return False

        return True

    def _apply_continuous_throttle(self, throttle: float) -> None:
        vehicle = self.env.unwrapped.vehicle
        current_target = float(getattr(vehicle, "target_speed", vehicle.speed))
        delta = throttle * self.max_speed_delta_per_step
        new_target = float(np.clip(current_target + delta, self.min_speed, self.max_speed))
        self.last_safety_clamped = False

        if self.throttle_safety_checks:
            safe_target = self._apply_same_lane_safety_cap(new_target)
            self.last_safety_clamped = safe_target < new_target - 1e-6
            new_target = safe_target

        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = new_target
        else:
            vehicle.speed = new_target

    def _apply_same_lane_safety_cap(self, target_speed: float) -> float:
        vehicle = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road
        front, _ = road.neighbour_vehicles(vehicle, vehicle.lane_index)
        if front is None:
            return target_speed

        gap = float(max(vehicle.lane_distance_to(front), 0.0))
        closing_speed = float(max(vehicle.speed - front.speed, 0.0))
        capped_target = target_speed
        if gap < self.min_same_lane_gap:
            capped_target = min(capped_target, max(self.min_speed, float(front.speed)))
        if closing_speed > 0.0:
            ttc = gap / closing_speed if closing_speed > 1e-6 else np.inf
            if ttc < self.min_same_lane_ttc:
                capped_target = min(capped_target, max(self.min_speed, float(front.speed)))
        return float(max(self.min_speed, capped_target))


def make_config(duration: int = DEFAULT_EPISODE_DURATION) -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": False,
            "lateral": True,
        },
        "lanes_count": 3,
        "vehicles_count": DEFAULT_VEHICLES_COUNT,
        "vehicles_density": DEFAULT_VEHICLES_DENSITY,
        "ego_spacing": DEFAULT_EGO_SPACING,
        "initial_lane_id": DEFAULT_INITIAL_LANE_ID,
        "duration": int(duration),
    }


def make_env(render_mode: str, *, duration: int = DEFAULT_EPISODE_DURATION, **wrapper_kwargs):
    base_env = gym.make("highway-v0", render_mode=render_mode, config=make_config(duration=duration))
    return LaneThrottleHybridWrapper(base_env, **wrapper_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid elurant PPO agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total PPO training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of post-training evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="PPO rollout steps per update (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size (default: 64)",
    )
    parser.add_argument(
        "--monitor-every-steps",
        type=int,
        default=100,
        help="Print training status every N timesteps (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size > args.n_steps:
        raise ValueError("--batch-size must be <= --n-steps")

    env = make_env(render_mode="rgb_array")
    env.reset(seed=42)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    monitor_cb = TimestepMonitorCallback(every_n_steps=args.monitor_every_steps)
    model.learn(
        total_timesteps=args.timesteps,
        callback=monitor_cb,
        progress_bar=False,
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")

    eval_env = Monitor(make_env(render_mode="rgb_array"))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
    )
    print(
        f"Evaluation over {args.eval_episodes} episodes: "
        f"mean reward = {mean_reward:.2f}, std = {std_reward:.2f}"
    )

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
