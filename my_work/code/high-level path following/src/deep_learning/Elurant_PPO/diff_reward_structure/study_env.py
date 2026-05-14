from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from highway_env import utils


DEFAULT_REWARD_CONFIG: dict[str, Any] = {
    "collision_reward": -1.0,
    "offroad_penalty": 0.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}


DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "vehicles_count": 10,
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },
    "lanes_count": 4,
    "vehicles_count": 60,
    "vehicles_density": 2.0,
    "ego_spacing": 1.0,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "duration": 60,
    "policy_frequency": 5,
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base)
    if not updates:
        return merged

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_study_config(
    reward_config: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _deep_update(DEFAULT_ENV_CONFIG, DEFAULT_REWARD_CONFIG)
    config = _deep_update(config, reward_config)
    config = _deep_update(config, config_overrides)
    return config


class DiffRewardStudyEnv(gym.Wrapper):
    """
    Wrap highway-v0 for alternative reward-structure experiments.

    The wrapper recomputes reward in normalized coefficient space so new reward
    terms can be added without being erased by the native on-road multiplier.
    The current hypothesis treats off-road as a one-step terminal negative event.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        reward_config: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.study_config = build_study_config(
            reward_config=reward_config,
            config_overrides=config_overrides,
        )
        base_env = gym.make("highway-v0", render_mode=render_mode, config=self.study_config)
        super().__init__(base_env)
        self.reward_config = {
            key: deepcopy(self.study_config[key]) for key in DEFAULT_REWARD_CONFIG
        }
        self._reset_episode_tracking()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_tracking()
        return obs, info

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if isinstance(self.action_space, gym.spaces.Box):
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        obs, _native_reward, terminated, truncated, info = self.env.step(action_array)

        native_terms = self._native_reward_terms(action_array)
        offroad_now = bool(not self.unwrapped.vehicle.on_road)
        offroad_term = self._offroad_term(offroad_now)
        raw_total = self._raw_total(
            collision_term=native_terms["collision_term"],
            high_speed_term=native_terms["high_speed_term"],
            right_lane_term=native_terms["right_lane_term"],
            offroad_term=offroad_term,
        )
        normalized_reward = self._normalize_reward(raw_total)

        if offroad_now and bool(self.reward_config.get("offroad_terminal", False)):
            terminated = True

        self._update_episode_tracking(
            action=action_array,
            reward=normalized_reward,
            collision_term=native_terms["collision_term"],
            high_speed_term=native_terms["high_speed_term"],
            right_lane_term=native_terms["right_lane_term"],
            offroad_term=offroad_term,
            raw_total=raw_total,
            normalized_reward=normalized_reward,
        )

        info["forward_speed"] = self._current_forward_speed()
        info["right_lane_ratio"] = self._current_right_lane_ratio()
        info["collision"] = float(self.unwrapped.vehicle.crashed)
        info["offroad"] = float(offroad_now)
        info["collision_term"] = float(native_terms["collision_term"])
        info["high_speed_term"] = float(native_terms["high_speed_term"])
        info["right_lane_term"] = float(native_terms["right_lane_term"])
        info["offroad_term"] = float(offroad_term)
        info["raw_total"] = float(raw_total)
        info["normalized_reward"] = float(normalized_reward)

        if terminated or truncated:
            info["episode_metrics"] = self._build_episode_metrics()

        return obs, normalized_reward, terminated, truncated, info

    def _reset_episode_tracking(self) -> None:
        self._episode_reward = 0.0
        self._step_count = 0
        self._collision = False
        self._offroad = False
        self._forward_speeds: list[float] = []
        self._right_lane_ratios: list[float] = []
        self._throttle_commands: list[float] = []
        self._steering_commands: list[float] = []
        self._collision_terms: list[float] = []
        self._high_speed_terms: list[float] = []
        self._right_lane_terms: list[float] = []
        self._offroad_terms: list[float] = []
        self._raw_totals: list[float] = []
        self._normalized_rewards: list[float] = []

    def _update_episode_tracking(
        self,
        action: np.ndarray,
        reward: float,
        collision_term: float,
        high_speed_term: float,
        right_lane_term: float,
        offroad_term: float,
        raw_total: float,
        normalized_reward: float,
    ) -> None:
        self._episode_reward += float(reward)
        self._step_count += 1
        self._collision = self._collision or bool(self.unwrapped.vehicle.crashed)
        self._offroad = self._offroad or bool(not self.unwrapped.vehicle.on_road)
        self._forward_speeds.append(self._current_forward_speed())
        self._right_lane_ratios.append(self._current_right_lane_ratio())
        self._throttle_commands.append(float(action[0]) if action.size >= 1 else 0.0)
        self._steering_commands.append(float(action[1]) if action.size >= 2 else 0.0)
        self._collision_terms.append(float(collision_term))
        self._high_speed_terms.append(float(high_speed_term))
        self._right_lane_terms.append(float(right_lane_term))
        self._offroad_terms.append(float(offroad_term))
        self._raw_totals.append(float(raw_total))
        self._normalized_rewards.append(float(normalized_reward))

    def _current_forward_speed(self) -> float:
        vehicle = self.unwrapped.vehicle
        return float(vehicle.speed * np.cos(vehicle.heading))

    def _current_right_lane_ratio(self) -> float:
        vehicle = self.unwrapped.vehicle
        neighbours = self.unwrapped.road.network.all_side_lanes(vehicle.lane_index)
        lane_index = int(vehicle.lane_index[2])
        return float(lane_index / max(len(neighbours) - 1, 1))

    def _native_reward_terms(self, action: np.ndarray) -> dict[str, float]:
        rewards = dict(self.unwrapped._rewards(action))
        return {
            "collision_term": float(rewards.get("collision_reward", 0.0)),
            "high_speed_term": float(rewards.get("high_speed_reward", 0.0)),
            "right_lane_term": float(rewards.get("right_lane_reward", 0.0)),
        }

    def _offroad_term(self, offroad_now: bool) -> float:
        return float(offroad_now and not self._offroad)

    def _raw_total(
        self,
        collision_term: float,
        high_speed_term: float,
        right_lane_term: float,
        offroad_term: float,
    ) -> float:
        return (
            float(self.reward_config["collision_reward"]) * float(collision_term)
            + float(self.reward_config["high_speed_reward"]) * float(high_speed_term)
            + float(self.reward_config["right_lane_reward"]) * float(right_lane_term)
            + float(self.reward_config["offroad_penalty"]) * float(offroad_term)
        )

    def _normalize_reward(self, raw_total: float) -> float:
        if not bool(self.reward_config.get("normalize_reward", True)):
            return float(raw_total)

        min_reward = float(self.reward_config["collision_reward"]) + float(
            self.reward_config["offroad_penalty"]
        )
        max_reward = float(self.reward_config["high_speed_reward"]) + float(
            self.reward_config["right_lane_reward"]
        )
        if np.isclose(min_reward, max_reward):
            return 0.0
        normalized = utils.lmap(raw_total, [min_reward, max_reward], [0.0, 1.0])
        return float(np.clip(normalized, 0.0, 1.0))

    def _build_episode_metrics(self) -> dict[str, float]:
        throttle = np.asarray(self._throttle_commands, dtype=np.float32)
        steering = np.asarray(self._steering_commands, dtype=np.float32)
        forward_speed = np.asarray(self._forward_speeds, dtype=np.float32)
        right_lane = np.asarray(self._right_lane_ratios, dtype=np.float32)
        collision_term = np.asarray(self._collision_terms, dtype=np.float32)
        high_speed_term = np.asarray(self._high_speed_terms, dtype=np.float32)
        right_lane_term = np.asarray(self._right_lane_terms, dtype=np.float32)
        offroad_term = np.asarray(self._offroad_terms, dtype=np.float32)
        raw_total = np.asarray(self._raw_totals, dtype=np.float32)
        normalized_reward = np.asarray(self._normalized_rewards, dtype=np.float32)

        throttle_delta = np.diff(throttle) if throttle.size >= 2 else np.zeros(0, dtype=np.float32)
        steering_delta = np.diff(steering) if steering.size >= 2 else np.zeros(0, dtype=np.float32)

        return {
            "episode_reward": float(self._episode_reward),
            "episode_length": float(self._step_count),
            "collision": float(self._collision),
            "offroad": float(self._offroad),
            "mean_forward_speed": float(forward_speed.mean()) if forward_speed.size else 0.0,
            "right_lane_ratio": float(right_lane.mean()) if right_lane.size else 0.0,
            "mean_collision_term": float(collision_term.mean()) if collision_term.size else 0.0,
            "mean_high_speed_term": float(high_speed_term.mean()) if high_speed_term.size else 0.0,
            "mean_right_lane_term": float(right_lane_term.mean()) if right_lane_term.size else 0.0,
            "mean_offroad_term": float(offroad_term.mean()) if offroad_term.size else 0.0,
            "mean_raw_total": float(raw_total.mean()) if raw_total.size else 0.0,
            "mean_normalized_reward": (
                float(normalized_reward.mean()) if normalized_reward.size else 0.0
            ),
            "throttle_var": float(throttle.var()) if throttle.size else 0.0,
            "steering_var": float(steering.var()) if steering.size else 0.0,
            "mean_abs_delta_throttle": (
                float(np.abs(throttle_delta).mean()) if throttle_delta.size else 0.0
            ),
            "mean_abs_delta_steering": (
                float(np.abs(steering_delta).mean()) if steering_delta.size else 0.0
            ),
        }


def make_study_env(
    render_mode: str | None = None,
    reward_config: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> DiffRewardStudyEnv:
    return DiffRewardStudyEnv(
        render_mode=render_mode,
        reward_config=reward_config,
        config_overrides=config_overrides,
    )
