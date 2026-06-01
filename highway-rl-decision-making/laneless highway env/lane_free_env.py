from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.road.road import Road, RoadNetwork


LANE_FREE_ENV_ID = "lane-free-v0"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _install_lane_free_road_renderer() -> None:
    """Extend highway-env's RoadGraphics at runtime, without editing its source."""
    if not hasattr(RoadGraphics, "_lane_free_original_display"):
        RoadGraphics._lane_free_original_display = RoadGraphics.display

    def display_lane_free_aware(road: Road, surface: WorldSurface) -> None:
        if not getattr(road, "lane_free", False):
            RoadGraphics._lane_free_original_display(road, surface)
            return

        surface.fill(surface.GREY)
        length = float(getattr(road, "lane_free_length", 1000.0))
        width = float(getattr(road, "lane_free_width", 10.2))
        top_left = surface.pos2pix(0.0, 0.0)
        rect = pygame.Rect(
            top_left[0],
            top_left[1],
            max(1, surface.pix(length)),
            max(1, surface.pix(width)),
        )
        pygame.draw.rect(surface, (70, 70, 70), rect)
        pygame.draw.line(surface, (245, 245, 245), rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, (245, 245, 245), rect.bottomleft, rect.bottomright, 2)

    RoadGraphics.display = staticmethod(display_lane_free_aware)


def register_lane_free_env() -> None:
    """Register this extension module as a Gymnasium/highway-env environment."""
    existing_spec = registry.get(LANE_FREE_ENV_ID)
    if existing_spec is not None and existing_spec.entry_point != "lane_free_env:LaneFreeTrafficEnv":
        del registry[LANE_FREE_ENV_ID]
    if LANE_FREE_ENV_ID not in registry:
        register(id=LANE_FREE_ENV_ID, entry_point="lane_free_env:LaneFreeTrafficEnv")


@dataclass
class LaneFreeVehicleState:
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    desired_speed: float


class LaneFreeVehicle:
    """A lane-free vehicle compatible with highway-env's vehicle graphics."""

    HISTORY_SIZE = 30

    def __init__(self, road: Road, state: LaneFreeVehicleState, *, is_ego: bool = False) -> None:
        # Do not inherit from highway_env.vehicle.kinematics.Vehicle: its base
        # constructor immediately queries the closest lane.
        self.road = road
        self.position = np.array([state.x, state.y], dtype=np.float64)
        self.heading = 0.0
        self.speed = float(state.vx)
        self.vx = float(state.vx)
        self.vy = float(state.vy)
        self.length = float(state.length)
        self.width = float(state.width)
        self.LENGTH = self.length
        self.WIDTH = self.width
        self.diagonal = float(np.sqrt(self.length**2 + self.width**2))
        self.desired_speed = float(state.desired_speed)
        self.is_ego = bool(is_ego)

        self.lane_index = None
        self.lane = None
        self.target_lane_index = None
        self.route = None
        self.action = {"ax": 0.0, "ay": 0.0, "steering": 0.0, "acceleration": 0.0}
        self.ax = 0.0
        self.ay = 0.0

        self.collidable = True
        self.solid = True
        self.check_collisions = False
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(2, dtype=float)
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.color = (50, 200, 0) if self.is_ego else (100, 200, 255)
        self._sync_graphics_fields()

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def act(self, action: dict | None = None) -> None:
        if action:
            self.ax = float(action.get("ax", self.ax))
            self.ay = float(action.get("ay", self.ay))
            self.action.update({"ax": self.ax, "ay": self.ay})

    def step(self, dt: float) -> None:
        self.position[0] += self.vx * dt + 0.5 * self.ax * dt * dt
        self.position[1] += self.vy * dt + 0.5 * self.ay * dt * dt
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self._sync_graphics_fields()
        self.on_state_update()

    def on_state_update(self) -> None:
        if self.road and self.road.record_history:
            self.history.appendleft(self.create_from(self))

    def _sync_graphics_fields(self) -> None:
        self.speed = float(self.vx)
        self.heading = float(np.arctan2(self.vy, max(self.vx, 1e-6)))
        self.LENGTH = self.length
        self.WIDTH = self.width
        self.action["acceleration"] = self.ax
        self.action["steering"] = 0.0

    @classmethod
    def create_from(cls, vehicle: "LaneFreeVehicle") -> "LaneFreeVehicle":
        state = LaneFreeVehicleState(
            x=float(vehicle.position[0]),
            y=float(vehicle.position[1]),
            vx=float(vehicle.vx),
            vy=float(vehicle.vy),
            length=float(vehicle.length),
            width=float(vehicle.width),
            desired_speed=float(vehicle.desired_speed),
        )
        copy = cls(vehicle.road, state, is_ego=vehicle.is_ego)
        copy.crashed = vehicle.crashed
        copy.color = vehicle.color
        return copy


class LaneFreeTrafficEnv(AbstractEnv):
    """A highway-env extension for lane-free ring-road traffic.

    The environment uses highway-env's ``AbstractEnv`` and native ``EnvViewer``.
    All lane-free additions live in this module; highway-env source files stay
    untouched.
    """

    VEHICLE_DIMENSIONS = np.array(
        [
            [3.20, 1.60],
            [3.90, 1.70],
            [4.25, 1.80],
            [4.55, 1.82],
            [4.60, 1.77],
            [5.15, 1.84],
        ],
        dtype=float,
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        config = super().default_config()
        config.update(
            {
                "road_length": 1000.0,
                "road_width": 10.2,
                "dt": 1.0 / 15.0,
                "simulation_frequency": 15,
                "policy_frequency": 15,
                "vehicles_count": 30,
                "sensing_range": 80.0,
                "episode_steps": 2000,
                "duration": 2000,
                "gamma_nudge": 0.0,
                "ego_controlled": True,
                "neighbors_count": 8,
                "desired_speed_range": [25.0, 35.0],
                "initial_speed_fraction_range": [0.0, 0.2],
                "force": {
                    "k_target_x": 2.5,
                    "k_target_y": 2.0,
                    "k_rep": 8.0,
                    "k_nudge": 0.18,
                    "k_boundary": 0.18,
                    "sigma_x": 5.0,
                    "sigma_y": 1.0,
                    "T_x": 1.2,
                    "T_y": 0.8,
                },
                "bounds": {
                    "ax_min": -6.0,
                    "ax_max": 3.0,
                    "ay_min": -4.0,
                    "ay_max": 4.0,
                    "max_lateral_speed_ratio": 0.3,
                    "max_speed": 45.0,
                },
                "action": {"type": "LaneFreeContinuousAction"},
                "observation": {"type": "LaneFreeKinematics"},
                "screen_width": 900,
                "screen_height": 260,
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "real_time_rendering": True,
            }
        )
        return config

    def configure(self, config: dict | None) -> None:
        if config:
            _deep_update(self.config, config)

    def define_spaces(self) -> None:
        rows = 1 + int(self.config.get("neighbors_count", 8))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(rows, 7), dtype=np.float32)
        self.action_type = None
        self.observation_type = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._last_accelerations = np.zeros((int(self.config["vehicles_count"]), 2), dtype=float)
        self._last_boundary_violations = 0
        self._last_collision_count = 0
        self._last_active_collision_count = 0
        self._last_ego_collision = False
        self._cumulative_collision_count = 0
        self._active_collision_pairs: set[tuple[int, int]] = set()
        self._flow_count = 0
        self._reset()
        obs = self._observe()
        return obs, self._info(obs, self._last_action)

    def _reset(self) -> None:
        self.road = Road(
            network=RoadNetwork(),
            vehicles=[],
            np_random=self.np_random,
            record_history=bool(self.config.get("show_trajectories", False)),
        )
        self.road.lane_free = True
        self.road.lane_free_length = float(self.config["road_length"])
        self.road.lane_free_width = float(self.config["road_width"])
        self._create_vehicles()

    def _create_vehicles(self) -> None:
        count = int(self.config["vehicles_count"])
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        desired_low, desired_high = self.config["desired_speed_range"]
        speed_low, speed_high = self.config["initial_speed_fraction_range"]

        x_slots = np.linspace(0.0, road_length, count, endpoint=False)
        self.np_random.shuffle(x_slots)
        vehicles: list[LaneFreeVehicle] = []

        for index in range(count):
            vehicle_length, vehicle_width = self.VEHICLE_DIMENSIONS[
                self.np_random.integers(0, len(self.VEHICLE_DIMENSIONS))
            ]
            desired_speed = float(self.np_random.uniform(desired_low, desired_high))
            speed_fraction = float(self.np_random.uniform(speed_low, speed_high))
            x = float((x_slots[index] + self.np_random.uniform(-0.35, 0.35) * road_length / count) % road_length)
            y = float(self.np_random.uniform(vehicle_width / 2.0, road_width - vehicle_width / 2.0))
            state = LaneFreeVehicleState(
                x=x,
                y=y,
                vx=speed_fraction * desired_speed,
                vy=0.0,
                length=float(vehicle_length),
                width=float(vehicle_width),
                desired_speed=desired_speed,
            )
            vehicles.append(LaneFreeVehicle(self.road, state, is_ego=index == 0))

        self.road.vehicles = vehicles
        self.vehicle = vehicles[0]
        self.controlled_vehicles = [self.vehicle]

    def step(self, action: np.ndarray | list[float] | None) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.zeros(2, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
        action_array = np.clip(action_array, -1.0, 1.0)
        self._last_action = action_array

        frames = max(1, int(round(float(self.config["simulation_frequency"]) / float(self.config["policy_frequency"]))))
        dt = float(self.config.get("dt", 1.0 / float(self.config["simulation_frequency"])))
        for _ in range(frames):
            accelerations = self._compute_accelerations()
            if bool(self.config["ego_controlled"]):
                accelerations[0, 0] = self._map_action(action_array[0], "longitudinal")
                accelerations[0, 1] = self._map_action(action_array[1], "lateral") + self._boundary_force(self.vehicle)
            accelerations = self._clip_accelerations(accelerations)
            self._last_accelerations = accelerations.copy()
            self._integrate(accelerations, dt)
            self._detect_collisions()
            self.steps += 1
            self.time += dt

        obs = self._observe()
        reward = self._reward(action_array)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action_array)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _map_action(self, value: float, axis: str) -> float:
        bounds = self.config["bounds"]
        if axis == "longitudinal":
            low, high = float(bounds["ax_min"]), float(bounds["ax_max"])
        else:
            low, high = float(bounds["ay_min"]), float(bounds["ay_max"])
        return low + 0.5 * (float(value) + 1.0) * (high - low)

    def _compute_accelerations(self) -> np.ndarray:
        vehicles = self.road.vehicles
        force = self.config["force"]
        accelerations = np.zeros((len(vehicles), 2), dtype=float)

        for i, vehicle in enumerate(vehicles):
            accelerations[i, 0] += float(force["k_target_x"]) * np.tanh(
                (vehicle.desired_speed - vehicle.vx) / max(float(force["sigma_x"]), 1e-6)
            )
            accelerations[i, 1] += -float(force["k_target_y"]) * np.tanh(
                vehicle.vy / max(float(force["sigma_y"]), 1e-6)
            )
            accelerations[i, 1] += self._boundary_force(vehicle)

        sensing_range = float(self.config["sensing_range"])
        gamma_nudge = float(self.config["gamma_nudge"])
        k_rep = float(force["k_rep"])
        k_nudge = float(force["k_nudge"])
        t_x = float(force["T_x"])
        t_y = float(force["T_y"])

        for i, upstream in enumerate(vehicles):
            for j, downstream in enumerate(vehicles):
                if i == j:
                    continue
                dx = self._forward_distance(upstream.position[0], downstream.position[0])
                if not (0.0 < dx < sensing_range):
                    continue

                dy = float(downstream.position[1] - upstream.position[1])
                dx_scale = 0.5 * (upstream.length + downstream.length) + t_x * max(upstream.vx - downstream.vx, 0.0)
                dvy = downstream.vy - upstream.vy
                v_close_y = max(0.0, -(dy * dvy) / (abs(dy) + 1e-6))
                dy_scale = 0.5 * (upstream.width + downstream.width) + t_y * v_close_y
                dx_scale = max(dx_scale, 1e-6)
                dy_scale = max(dy_scale, 1e-6)

                potential = float(np.exp(-((dx / dx_scale) ** 2) - ((dy / dy_scale) ** 2)))
                norm = float(np.sqrt(dx * dx + dy * dy) + 1e-6)
                ux = dx / norm
                uy = dy / norm

                accelerations[i, 0] += -k_rep * potential * ux
                accelerations[i, 1] += -k_rep * potential * uy

                blocked = max(upstream.desired_speed - upstream.vx, 0.0)
                accelerations[j, 0] += gamma_nudge * k_nudge * blocked * potential * ux
                accelerations[j, 1] += gamma_nudge * k_nudge * blocked * potential * uy

        return accelerations

    def _boundary_force(self, vehicle: LaneFreeVehicle) -> float:
        road_width = float(self.config["road_width"])
        k_boundary = float(self.config["force"]["k_boundary"])
        eps = 1e-3
        left_clearance = max(float(vehicle.position[1] - vehicle.width / 2.0), eps)
        right_clearance = max(float(road_width - vehicle.width / 2.0 - vehicle.position[1]), eps)
        return k_boundary * (1.0 / (left_clearance + eps) ** 2 - 1.0 / (right_clearance + eps) ** 2)

    def _clip_accelerations(self, accelerations: np.ndarray) -> np.ndarray:
        bounds = self.config["bounds"]
        accelerations[:, 0] = np.clip(accelerations[:, 0], float(bounds["ax_min"]), float(bounds["ax_max"]))
        accelerations[:, 1] = np.clip(accelerations[:, 1], float(bounds["ay_min"]), float(bounds["ay_max"]))
        return accelerations

    def _integrate(self, accelerations: np.ndarray, dt: float) -> None:
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        max_speed = float(self.config["bounds"]["max_speed"])
        lateral_ratio = float(self.config["bounds"]["max_lateral_speed_ratio"])
        old_x = np.array([vehicle.position[0] for vehicle in self.road.vehicles], dtype=float)
        boundary_violations = 0

        for vehicle, (ax, ay) in zip(self.road.vehicles, accelerations):
            vehicle.ax = float(ax)
            vehicle.ay = float(ay)
            vehicle.step(dt)
            vehicle.position[0] = float(vehicle.position[0] % road_length)
            vehicle.vx = float(np.clip(vehicle.vx, 0.0, max_speed))
            max_lateral_speed = lateral_ratio * max(vehicle.vx, 1.0)
            vehicle.vy = float(np.clip(vehicle.vy, -max_lateral_speed, max_lateral_speed))

            y_min = vehicle.width / 2.0
            y_max = road_width - vehicle.width / 2.0
            if vehicle.position[1] < y_min or vehicle.position[1] > y_max:
                boundary_violations += 1
                vehicle.position[1] = float(np.clip(vehicle.position[1], y_min, y_max))
                vehicle.vy = 0.0
            vehicle._sync_graphics_fields()

        for previous_x, vehicle in zip(old_x, self.road.vehicles):
            if previous_x + max(vehicle.vx, 0.0) * dt >= road_length and vehicle.position[0] < previous_x:
                self._flow_count += 1
        self._last_boundary_violations = boundary_violations

    def _detect_collisions(self) -> None:
        active_pairs: set[tuple[int, int]] = set()
        ego_collision = False
        vehicles = self.road.vehicles
        for i, first in enumerate(vehicles):
            first.crashed = False
            for j, second in enumerate(vehicles[i + 1 :], start=i + 1):
                dx = abs(self._signed_distance(first.position[0], second.position[0]))
                dy = abs(float(second.position[1] - first.position[1]))
                if dx < 0.5 * (first.length + second.length) and dy < 0.5 * (first.width + second.width):
                    active_pairs.add((i, j))
                    first.crashed = True
                    second.crashed = True
                    ego_collision = ego_collision or first.is_ego or second.is_ego

        new_pairs = active_pairs - self._active_collision_pairs
        self._last_collision_count = len(new_pairs)
        self._last_active_collision_count = len(active_pairs)
        self._last_ego_collision = ego_collision
        self._cumulative_collision_count += len(new_pairs)
        self._active_collision_pairs = active_pairs

    def _observe(self) -> np.ndarray:
        ego = self.vehicle
        neighbors = sorted(
            [vehicle for vehicle in self.road.vehicles if vehicle is not ego],
            key=lambda vehicle: self._distance_to_ego(vehicle),
        )
        selected = [ego] + neighbors[: int(self.config["neighbors_count"])]
        rows = [self._observation_row(vehicle, ego) for vehicle in selected]
        while len(rows) < 1 + int(self.config["neighbors_count"]):
            rows.append(np.zeros(7, dtype=np.float32))
        return np.asarray(rows, dtype=np.float32)

    def _observation_row(self, vehicle: LaneFreeVehicle, ego: LaneFreeVehicle) -> np.ndarray:
        road_width = float(self.config["road_width"])
        sensing_range = float(self.config["sensing_range"])
        signed_dx = 0.0 if vehicle is ego else self._signed_distance(ego.position[0], vehicle.position[0])
        dy = 0.0 if vehicle is ego else float(vehicle.position[1] - ego.position[1])
        return np.array(
            [
                np.clip(signed_dx / sensing_range, -1.0, 1.0),
                np.clip(dy / road_width, -1.0, 1.0),
                vehicle.vx / 40.0,
                vehicle.vy / 12.0,
                vehicle.length / 5.15,
                vehicle.width / 1.84,
                vehicle.desired_speed / 35.0,
            ],
            dtype=np.float32,
        )

    def _reward(self, action: np.ndarray) -> float:
        ego = self.vehicle
        ay_ego = float(self._last_accelerations[0, 1]) if len(self._last_accelerations) else 0.0
        boundary_penalty = 2.0 if self._last_boundary_violations > 0 else 0.0
        return float(
            ego.vx / max(ego.desired_speed, 1e-6)
            - 5.0 * float(self._last_ego_collision)
            - 0.05 * abs(ay_ego)
            - 0.02 * abs(ego.vy)
            - boundary_penalty
        )

    def _is_terminated(self) -> bool:
        return bool(self._last_ego_collision)

    def _is_truncated(self) -> bool:
        return self.steps >= int(self.config.get("episode_steps", self.config.get("duration", 2000)))

    def _info(self, obs: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
        elapsed_hours = max(self.time / 3600.0, 1e-9)
        return {
            "speed": float(self.vehicle.vx),
            "mean_speed": self.mean_speed,
            "collisions": int(self._last_collision_count),
            "active_collisions": int(self._last_active_collision_count),
            "cumulative_collisions": int(self._cumulative_collision_count),
            "ego_collision": bool(self._last_ego_collision),
            "boundary_violations": int(self._last_boundary_violations),
            "flow_count": int(self._flow_count),
            "flow_per_hour": float(self._flow_count / elapsed_hours),
        }

    @property
    def mean_speed(self) -> float:
        return float(np.mean([vehicle.vx for vehicle in self.road.vehicles])) if self.road else 0.0

    def snapshot(self) -> np.ndarray:
        return np.asarray(
            [
                [
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.vx,
                    vehicle.vy,
                    vehicle.length,
                    vehicle.width,
                    vehicle.desired_speed,
                    float(vehicle.is_ego),
                    float(vehicle.crashed),
                ]
                for vehicle in self.road.vehicles
            ],
            dtype=float,
        )

    def _forward_distance(self, x_from: float, x_to: float) -> float:
        return float((x_to - x_from) % float(self.config["road_length"]))

    def _signed_distance(self, x_from: float, x_to: float) -> float:
        road_length = float(self.config["road_length"])
        return float(((x_to - x_from + 0.5 * road_length) % road_length) - 0.5 * road_length)

    def _distance_to_ego(self, vehicle: LaneFreeVehicle) -> float:
        dx = self._signed_distance(self.vehicle.position[0], vehicle.position[0])
        dy = float(vehicle.position[1] - self.vehicle.position[1])
        return float(np.sqrt(dx * dx + dy * dy))


_install_lane_free_road_renderer()
register_lane_free_env()


__all__ = ["LaneFreeTrafficEnv", "LaneFreeVehicle", "LaneFreeVehicleState", "register_lane_free_env"]
