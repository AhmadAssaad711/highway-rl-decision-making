"""
TTC-based reward shaping for the Kourani DQN environments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import gymnasium as gym
import highway_env  # noqa: F401 - register highway-v0
import numpy as np
from highway_env import utils


DEFAULT_TTC_CONFIG: dict[str, Any] = {
    "ttc_safe_threshold": 4.0,
    "ttc_cap": 10.0,
    "ttc_penalty_weight": 0.5,
    "lane_scope": "target",
}

REAR_SPAWN_CONFIG_KEY = "rear_vehicle_spawning"
DEFAULT_REAR_SPAWN_CONFIG: dict[str, Any] = {
    "enabled": False,
    "spawn_probability": 0.25,
    "cooldown_policy_steps": 3,
    "min_ego_progress": 120.0,
    "min_spawn_distance": 30.0,
    "max_spawn_distance": 65.0,
    "min_lane_gap": 18.0,
    "speed_delta_range": [2.0, 7.0],
    "absolute_speed_range": [23.0, 34.0],
    "lane_scope": "ego_neighborhood",
    "max_extra_vehicles": 8,
}


def build_ttc_config(ttc_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_TTC_CONFIG)
    if ttc_config:
        merged.update(dict(ttc_config))
    return merged


def build_rear_spawn_config(
    rear_spawn_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_REAR_SPAWN_CONFIG)
    if rear_spawn_config:
        merged.update(dict(rear_spawn_config))
    return merged


def _forward_speed(vehicle) -> float:
    return float(vehicle.speed * np.cos(vehicle.heading))


class RearVehicleSpawningWrapper(gym.Wrapper):
    """
    Inject occasional rear traffic so the ego vehicle faces pressure from behind
    as well as from vehicles already ahead.
    """

    def __init__(
        self,
        env: gym.Env,
        rear_spawn_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.rear_spawn_config = build_rear_spawn_config(rear_spawn_config)
        self._policy_steps_since_spawn = int(self.rear_spawn_config["cooldown_policy_steps"])

    def reset(self, **kwargs):
        self._policy_steps_since_spawn = int(self.rear_spawn_config["cooldown_policy_steps"])
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info["rear_vehicle_spawned"] = False
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        spawned = False
        if not (terminated or truncated):
            self._policy_steps_since_spawn += 1
            spawned = self._maybe_spawn_rear_vehicle()
        info = dict(info)
        info["rear_vehicle_spawned"] = bool(spawned)
        return observation, reward, terminated, truncated, info

    def _maybe_spawn_rear_vehicle(self) -> bool:
        config = self.rear_spawn_config
        if not bool(config.get("enabled", False)):
            return False

        env = self.unwrapped
        road = getattr(env, "road", None)
        ego_vehicle = getattr(env, "vehicle", None)
        if road is None or ego_vehicle is None:
            return False

        if self._policy_steps_since_spawn < int(config["cooldown_policy_steps"]):
            return False

        if float(ego_vehicle.position[0]) < float(config["min_ego_progress"]):
            return False

        if road.np_random.uniform() > float(config["spawn_probability"]):
            return False

        base_traffic = int(env.config.get("vehicles_count", 0))
        controlled = len(getattr(env, "controlled_vehicles", []))
        current_other_vehicles = max(0, len(road.vehicles) - controlled)
        max_other_vehicles = base_traffic + int(config["max_extra_vehicles"])
        if current_other_vehicles >= max_other_vehicles:
            return False

        lane_indices = self._candidate_lane_indices(ego_vehicle)
        if not lane_indices:
            return False

        road.np_random.shuffle(lane_indices)
        for lane_index in lane_indices:
            spawn_s = self._sample_spawn_position(ego_vehicle, lane_index)
            if spawn_s is None:
                continue
            if not self._lane_has_clear_gap(lane_index, spawn_s):
                continue
            self._spawn_vehicle(lane_index, spawn_s, ego_vehicle)
            self._policy_steps_since_spawn = 0
            return True

        return False

    def _candidate_lane_indices(self, ego_vehicle) -> list[tuple]:
        lane_index = getattr(ego_vehicle, "lane_index", None)
        if lane_index is None:
            return []

        lane_scope = str(self.rear_spawn_config["lane_scope"]).lower()
        road = self.unwrapped.road
        lane_from, lane_to, lane_id = lane_index
        lane_count = len(road.network.graph[lane_from][lane_to])

        if lane_scope == "current":
            return [lane_index]
        if lane_scope == "all":
            return [(lane_from, lane_to, idx) for idx in range(lane_count)]

        candidates = [lane_index]
        for delta in (-1, 1):
            candidate_id = lane_id + delta
            if 0 <= candidate_id < lane_count:
                candidates.append((lane_from, lane_to, candidate_id))
        return candidates

    def _sample_spawn_position(self, ego_vehicle, lane_index) -> float | None:
        lane = self.unwrapped.road.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(ego_vehicle.position)
        spawn_distance = self.unwrapped.road.np_random.uniform(
            float(self.rear_spawn_config["min_spawn_distance"]),
            float(self.rear_spawn_config["max_spawn_distance"]),
        )
        spawn_s = float(ego_s - spawn_distance)
        if spawn_s <= 5.0:
            return None
        return spawn_s

    def _lane_has_clear_gap(self, lane_index, spawn_s: float) -> bool:
        min_lane_gap = float(self.rear_spawn_config["min_lane_gap"])
        road = self.unwrapped.road
        lane = road.network.get_lane(lane_index)
        for other in road.vehicles:
            other_lane_index = getattr(other, "lane_index", None)
            if other_lane_index != lane_index:
                continue
            other_s, _ = lane.local_coordinates(other.position)
            if abs(float(other_s) - float(spawn_s)) < min_lane_gap:
                return False
        return True

    def _spawn_vehicle(self, lane_index, spawn_s: float, ego_vehicle) -> None:
        road = self.unwrapped.road
        lane = road.network.get_lane(lane_index)
        vehicle_cls = utils.class_from_path(self.unwrapped.config["other_vehicles_type"])

        speed_delta_low, speed_delta_high = self.rear_spawn_config["speed_delta_range"]
        abs_speed_low, abs_speed_high = self.rear_spawn_config["absolute_speed_range"]
        sampled_speed = max(
            float(ego_vehicle.speed)
            + float(road.np_random.uniform(float(speed_delta_low), float(speed_delta_high))),
            float(abs_speed_low),
        )
        sampled_speed = float(np.clip(sampled_speed, float(abs_speed_low), float(abs_speed_high)))

        vehicle = vehicle_cls(
            road,
            lane.position(spawn_s, 0),
            lane.heading_at(spawn_s),
            sampled_speed,
        )
        if hasattr(vehicle, "randomize_behavior"):
            vehicle.randomize_behavior()
        road.vehicles.append(vehicle)


class TTCRewardWrapper(gym.Wrapper):
    """
    Add a time-to-collision penalty on top of the native highway-env reward.
    """

    def __init__(self, env: gym.Env, ttc_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env)
        self.ttc_config = build_ttc_config(ttc_config)
        self.ttc_safe_threshold = float(max(self.ttc_config["ttc_safe_threshold"], 1e-6))
        self.ttc_cap = float(max(self.ttc_config["ttc_cap"], self.ttc_safe_threshold))
        self.ttc_penalty_weight = float(max(self.ttc_config["ttc_penalty_weight"], 0.0))
        self.lane_scope = str(self.ttc_config["lane_scope"]).lower()
        if self.lane_scope not in {"target", "current"}:
            raise ValueError(
                f"Unsupported lane_scope={self.lane_scope!r}. Expected 'target' or 'current'."
            )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        ttc_current = self.compute_ttc()
        info = dict(info)
        info["ttc_current"] = float(ttc_current)
        info["ttc_penalty"] = float(self.compute_ttc_penalty(ttc_current))
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        ttc_current = self.compute_ttc()
        ttc_penalty = self.compute_ttc_penalty(ttc_current)
        shaped_reward = float(base_reward) - float(ttc_penalty)

        info = dict(info)
        info["ttc_current"] = float(ttc_current)
        info["ttc_penalty"] = float(ttc_penalty)
        info["base_reward"] = float(base_reward)
        info["shaped_reward"] = float(shaped_reward)
        return observation, shaped_reward, terminated, truncated, info

    def compute_ttc(self) -> float:
        vehicle = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        if vehicle is None or road is None:
            return self.ttc_cap

        lane_index = self._resolve_lane_index(vehicle)
        if lane_index is None:
            return self.ttc_cap

        front_vehicle, _ = road.neighbour_vehicles(vehicle, lane_index)
        if front_vehicle is None:
            return self.ttc_cap

        lane = road.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(vehicle.position)
        front_s, _ = lane.local_coordinates(front_vehicle.position)
        clearance = max(
            0.0,
            float(front_s - ego_s)
            - 0.5
            * float(getattr(vehicle, "LENGTH", 0.0) + getattr(front_vehicle, "LENGTH", 0.0)),
        )

        closing_speed = _forward_speed(vehicle) - _forward_speed(front_vehicle)
        if closing_speed <= 1e-6:
            return self.ttc_cap

        ttc = 0.0 if clearance <= 0.0 else clearance / closing_speed
        return float(np.clip(ttc, 0.0, self.ttc_cap))

    def compute_ttc_penalty(self, ttc_current: float) -> float:
        clipped_ttc = float(np.clip(ttc_current, 0.0, self.ttc_cap))
        normalized_shortfall = max(
            0.0,
            (self.ttc_safe_threshold - clipped_ttc) / self.ttc_safe_threshold,
        )
        return float(self.ttc_penalty_weight * normalized_shortfall)

    def _resolve_lane_index(self, vehicle):
        if self.lane_scope == "current":
            return getattr(vehicle, "lane_index", None)

        target_lane_index = getattr(vehicle, "target_lane_index", None)
        if target_lane_index is not None:
            return target_lane_index
        return getattr(vehicle, "lane_index", None)


def wrap_env_with_ttc(
    env: gym.Env,
    ttc_config: Mapping[str, Any] | None = None,
) -> TTCRewardWrapper:
    return TTCRewardWrapper(env, ttc_config=ttc_config)


def make_ttc_highway_env(
    render_mode: str = "rgb_array",
    config: Mapping[str, Any] | None = None,
    ttc_config: Mapping[str, Any] | None = None,
) -> TTCRewardWrapper:
    resolved_config = dict(config or {})
    rear_spawn_config = build_rear_spawn_config(resolved_config.get(REAR_SPAWN_CONFIG_KEY))
    base_env = gym.make("highway-v0", render_mode=render_mode, config=resolved_config)
    if rear_spawn_config.get("enabled", False):
        base_env = RearVehicleSpawningWrapper(base_env, rear_spawn_config=rear_spawn_config)
    return wrap_env_with_ttc(base_env, ttc_config=ttc_config)
