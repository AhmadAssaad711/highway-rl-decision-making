"""
Erratic-drivers highway environment used for unstructured traffic evaluation.

This file intentionally contains only the environment definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle


class NormalVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (0, 255, 0)


class AggressiveVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (255, 0, 0)
        self.target_speed = 35
        self.TIME_WANTED = 0.5
        self.MAX_ACCELERATION = 5.0
        self.COMFORT_ACC_MAX = 4.0
        self.COMFORT_ACC_MIN = -6.0
        self.POLITENESS = 0.0


class LazyVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (0, 0, 255)
        self.target_speed = 15
        self.TIME_WANTED = 2.5
        self.POLITENESS = 0.8

    def act(self, action=None):
        super().act(action)
        if np.random.rand() < 0.02:
            self.change_lane_policy()


class EgoVehicle(ControlledVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (255, 255, 0)


class ErraticDriversEnv(HighwayEnv):
    def _create_vehicles(self) -> None:
        self.road.vehicles = []

        for _ in range(self.config["vehicles_count"]):
            random_value = np.random.rand()
            if random_value < 0.2:
                vehicle = AggressiveVehicle.create_random(self.road)
            elif random_value < 0.4:
                vehicle = LazyVehicle.create_random(self.road)
            else:
                vehicle = NormalVehicle.create_random(self.road)

            self.road.vehicles.append(vehicle)

        ego_vehicle = EgoVehicle.create_random(self.road)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle


# Backward-compatible alias with the original class name.
CustomHighwayEnv = ErraticDriversEnv


def make_erratic_drivers_env(
    render_mode: str = "human",
    config: dict[str, Any] | None = None,
) -> ErraticDriversEnv:
    env = ErraticDriversEnv(render_mode=render_mode)
    if config:
        env.configure(config)
    return env
