"""
Custom highway-env extension for evaluating the Kourani DQN in unstructured traffic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import near_split

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
DQN_MODULE_DIR = PROJECT_ROOT / "src" / "deep_learning" / "DQN"

if str(DQN_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DQN_MODULE_DIR))

from ttc_reward_wrapper import build_ttc_config, wrap_env_with_ttc


class NominalVehicle(IDMVehicle):
    """Traffic participant with close-to-default highway-env behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (0, 180, 0)
        self.target_speed = 27
        self.TIME_WANTED = 1.4
        self.POLITENESS = 0.3


class AggressiveVehicle(IDMVehicle):
    """Faster and less polite vehicle to create unstable traffic pockets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (220, 30, 30)
        self.target_speed = 34
        self.TIME_WANTED = 0.7
        self.MAX_ACCELERATION = 6.0
        self.COMFORT_ACC_MAX = 4.5
        self.COMFORT_ACC_MIN = -6.0
        self.POLITENESS = 0.0


class CautiousVehicle(IDMVehicle):
    """Slower and more conservative vehicle for mixed-flow conditions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (40, 90, 220)
        self.target_speed = 20
        self.TIME_WANTED = 2.2
        self.COMFORT_ACC_MAX = 2.0
        self.COMFORT_ACC_MIN = -4.0
        self.POLITENESS = 0.7


class UnstructuredKouraniHighwayEnv(HighwayEnv):
    """
    HighwayEnv variant that preserves the Kourani DQN action/observation contract
    while allowing mixed traffic populations and scenario-specific density settings.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "collision_reward": -5.0,
                "high_speed_reward": 0.3,
                "right_lane_reward": 0.15,
                "lane_change_reward": -0.01,
                "reward_speed_range": [20, 30],
                "aggressive_vehicle_ratio": 0.2,
                "cautious_vehicle_ratio": 0.2,
            }
        )
        return config

    def _create_vehicles(self) -> None:
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        self.road.vehicles = []

        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle_class = self._sample_vehicle_class()
                traffic_vehicle = vehicle_class.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                )
                self.road.vehicles.append(traffic_vehicle)

    def _sample_vehicle_class(self) -> type[IDMVehicle]:
        aggressive_ratio = float(self.config.get("aggressive_vehicle_ratio", 0.2))
        cautious_ratio = float(self.config.get("cautious_vehicle_ratio", 0.2))

        aggressive_ratio = max(0.0, aggressive_ratio)
        cautious_ratio = max(0.0, cautious_ratio)
        total_ratio = aggressive_ratio + cautious_ratio
        if total_ratio > 1.0:
            aggressive_ratio /= total_ratio
            cautious_ratio /= total_ratio

        draw = float(self.np_random.random())
        if draw < aggressive_ratio:
            return AggressiveVehicle
        if draw < aggressive_ratio + cautious_ratio:
            return CautiousVehicle
        return NominalVehicle

    def _info(self, obs, action: int | None = None) -> dict[str, Any]:
        info = super()._info(obs, action)
        lane = self.vehicle.lane_index[2] if self.vehicle and self.vehicle.lane_index else -1
        info.update(
            {
                "lane_index": lane,
                "x_position": float(self.vehicle.position[0]),
                "y_position": float(self.vehicle.position[1]),
            }
        )
        return info


def make_unstructured_kourani_env(
    render_mode: str = "rgb_array",
    config: dict[str, Any] | None = None,
    ttc_config: dict[str, Any] | None = None,
):
    env = UnstructuredKouraniHighwayEnv(render_mode=render_mode)
    if config:
        env.configure(config)
    wrapped_env = wrap_env_with_ttc(env, ttc_config=build_ttc_config(ttc_config))
    if config:
        wrapped_env.reset()
    return wrapped_env
