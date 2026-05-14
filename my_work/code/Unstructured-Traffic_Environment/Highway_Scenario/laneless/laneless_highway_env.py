from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import LaneIndex, Road, RoadNetwork, Route
from highway_env.utils import Vector
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle


class LanelessIDMVehicle(IDMVehicle):
    """
    Background vehicle for an unmarked highway corridor.

    Longitudinal control is IDM. Lateral behavior chooses the best lateral target
    from sampled candidates inside the drivable corridor.
    """

    LATERAL_SAMPLES = 21
    CORRIDOR_WIDTH = 2.8
    BOUNDARY_MARGIN = 0.5 * Vehicle.WIDTH
    L_SAFE = 2 * Vehicle.LENGTH
    LATERAL_SAFE_MARGIN = 0.5
    LATERAL_CLEARANCE = 0.5
    MAX_LATERAL_DECISION_DISTANCE = CORRIDOR_WIDTH
    LANE_CHANGE_DELAY = 1.0

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        road_width: float | None = None,
    ):
        super().__init__(
            road,
            position,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
            enable_lane_change=enable_lane_change,
            timer=timer,
        )
        self.road_width = road_width or self.lane.width_at(0)
        self.target_y = self.lane.local_coordinates(self.position)[1]
        self.lane_change_delay = self.LANE_CHANGE_DELAY
        if timer is None:
            self.timer = self.road.np_random.uniform(0, self.lane_change_delay)

    def act(self, action: dict | str = None):
        if self.crashed:
            return

        if self.enable_lane_change and utils.do_every(
            self.lane_change_delay, self.timer
        ):
            self.timer -= self.lane_change_delay
            self.target_y = self.choose_lateral_target()
        _, current_y = self.lane.local_coordinates(self.position)
        current_front_vehicle = self.front_vehicle_at(current_y)
        target_front_vehicle = self.front_vehicle_at(self.target_y)
        current_acceleration = self.acceleration(self, current_front_vehicle)
        target_acceleration = self.acceleration(self, target_front_vehicle)

        action = {
            "steering": self.steering_to_lateral_target(self.target_y),
            "acceleration": min(current_acceleration, target_acceleration),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        Vehicle.act(self, action)

    def choose_lateral_target(self) -> float:
        low = -self.road_width / 2 + self.BOUNDARY_MARGIN
        high = self.road_width / 2 - self.BOUNDARY_MARGIN
        _, current_y = self.lane.local_coordinates(self.position)
        candidates = [
            float(candidate)
            for candidate in np.linspace(low, high, self.LATERAL_SAMPLES)
            if abs(candidate - current_y) <= self.MAX_LATERAL_DECISION_DISTANCE
        ]

        old_preceding, old_following = self.neighbour_vehicles_at(current_y)
        self_a = self.acceleration(self, old_preceding)

        def mobil_gain(y: float) -> float | None:
            if self.has_lateral_conflict(current_y, y):
                return None

            new_preceding, new_following = self.neighbour_vehicles_at(y)
            new_following_a = self.acceleration(new_following, new_preceding)
            new_following_pred_a = self.acceleration(new_following, self)

            if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return None

            self_pred_a = self.acceleration(self, new_preceding)
            old_following_a = self.acceleration(old_following, self)
            old_following_pred_a = self.acceleration(old_following, old_preceding)
            return (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )

        valid_candidates = [
            (candidate, gain)
            for candidate in candidates
            for gain in [mobil_gain(candidate)]
            if gain is not None and gain >= self.LANE_CHANGE_MIN_ACC_GAIN
        ]
        if not valid_candidates:
            return self.target_y

        best_y, _ = max(valid_candidates, key=lambda item: item[1])
        return best_y

    def has_lateral_conflict(self, current_y: float, target_y: float) -> bool:
        if np.isclose(current_y, target_y):
            return False

        ego_s, _ = self.lane.local_coordinates(self.position)
        swept_left = min(current_y, target_y) - self.WIDTH / 2 - self.LATERAL_SAFE_MARGIN
        swept_right = max(current_y, target_y) + self.WIDTH / 2 + self.LATERAL_SAFE_MARGIN

        for other in self.road.vehicles:
            if other is self:
                continue
            other_s, other_y = self.lane.local_coordinates(other.position)
            longitudinal_clearance = (
                self.LENGTH / 2 + other.LENGTH / 2 + self.L_SAFE
            )
            if abs(other_s - ego_s) > longitudinal_clearance:
                continue
            other_left = other_y - other.WIDTH / 2
            other_right = other_y + other.WIDTH / 2
            if swept_left <= other_right and other_left <= swept_right:
                return True

        return False

    def neighbour_vehicles_at(
        self, lateral_target: float
    ) -> tuple[Vehicle | None, Vehicle | None]:
        ego_s, _ = self.lane.local_coordinates(self.position)
        front_vehicle = None
        rear_vehicle = None
        front_gap = np.inf
        rear_gap = np.inf

        for other in self.road.vehicles:
            if other is self:
                continue
            other_s, other_y = self.lane.local_coordinates(other.position)
            gap = other_s - ego_s
            lateral_overlap = (
                abs(other_y - lateral_target)
                <= self.WIDTH / 2 + other.WIDTH / 2 + self.LATERAL_CLEARANCE
            )
            if not lateral_overlap:
                continue
            if 0 < gap < front_gap:
                front_vehicle = other
                front_gap = gap
            elif gap < 0 and -gap < rear_gap:
                rear_vehicle = other
                rear_gap = -gap

        return front_vehicle, rear_vehicle

    def front_vehicle_at(self, lateral_target: float) -> Vehicle | None:
        front_vehicle, _ = self.neighbour_vehicles_at(lateral_target)
        return front_vehicle

    def steering_to_lateral_target(self, lateral_target: float) -> float:
        s, y = self.lane.local_coordinates(self.position)
        future_heading = self.lane.heading_at(s + self.speed * self.TAU_PURSUIT)

        lateral_error = lateral_target - y
        lateral_speed_command = self.KP_LATERAL * lateral_error
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        return float(np.arctan(2 * np.tan(slip_angle)))


class LanelessHighwayEnv(AbstractEnv):
    """
    Straight unmarked highway corridor.

    The road is one wide hidden lane used only for coordinates and boundaries.
    Background traffic ignores lane changes and selects lateral gaps directly.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-8, 8],
                        "vx": [-40, 40],
                        "vy": [-10, 10],
                    },
                },
                "action": {"type": "ContinuousAction"},
                "road_width": 12.0,
                "road_length": 10000.0,
                "lateral_bins_count": 10,
                "boundary_buffer": 0.5,
                "nominal_lane_width": 4.0,
                "vehicles_count": 60,
                "vehicles_density": 1.0,
                "ego_spacing": 2,
                "duration": 40,
                "speed_limit": 30,
                "target_speed_range": [0.8, 1.1],
                "collision_reward": -1.0,
                "offroad_reward": -1.0,
                "high_speed_reward": 0.4,
                "reward_speed_range": [20, 30],
                "comfort_reward": -0.05,
                "normalize_reward": True,
                "offroad_terminal": True,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        network = RoadNetwork()
        network.add_lane(
            "0",
            "1",
            StraightLane(
                [0, 0],
                [self.config["road_length"], 0],
                width=self.config["road_width"],
                line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
                speed_limit=self.config["speed_limit"],
            ),
        )
        self.road = Road(
            network=network,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        lane = self.road.network.get_lane(("0", "1", 0))
        lateral_bins = self._lateral_spawn_bins()

        ego_x = self._random_longitudinal_position(
            lane,
            speed=25,
            spacing=self.config["ego_spacing"],
        )
        ego = self.action_type.vehicle_class(
            self.road,
            lane.position(ego_x, 0),
            lane.heading_at(ego_x),
            speed=25,
        )
        self.vehicle = ego
        self.road.vehicles.append(ego)

        for _ in range(self.config["vehicles_count"]):
            speed = self._sample_spawn_speed(lane)
            x = self._random_longitudinal_position(
                lane,
                speed=speed,
                spacing=1 / self.config["vehicles_density"],
            )
            y = lateral_bins[int(self.np_random.integers(len(lateral_bins)))]
            vehicle = LanelessIDMVehicle(
                self.road,
                lane.position(x, y),
                heading=lane.heading_at(x),
                speed=speed,
                target_lane_index=("0", "1", 0),
                target_speed=self._sample_target_speed(lane),
                road_width=self.config["road_width"],
            )
            self.road.vehicles.append(vehicle)

    def _sample_spawn_speed(self, lane: StraightLane) -> float:
        if lane.speed_limit is not None:
            return float(
                self.np_random.uniform(0.7 * lane.speed_limit, 0.8 * lane.speed_limit)
            )
        return float(
            self.np_random.uniform(
                Vehicle.DEFAULT_INITIAL_SPEEDS[0],
                Vehicle.DEFAULT_INITIAL_SPEEDS[1],
            )
        )

    def _sample_target_speed(self, lane: StraightLane) -> float:
        speed_limit = lane.speed_limit or self.config["speed_limit"]
        low, high = self.config["target_speed_range"]
        return float(self.np_random.uniform(low * speed_limit, high * speed_limit))

    def _random_longitudinal_position(
        self,
        lane: StraightLane,
        speed: float,
        spacing: float,
    ) -> float:
        default_spacing = 12 + speed
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * self._nominal_lanes_count())
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in self.road.vehicles])
            if self.road.vehicles
            else 3 * offset
        )
        return float(x0 + offset * self.np_random.uniform(0.9, 1.1))

    def _nominal_lanes_count(self) -> int:
        return max(1, round(self.config["road_width"] / self.config["nominal_lane_width"]))

    def _lateral_spawn_bins(self) -> np.ndarray:
        """Safe lateral center positions for laneless traffic spawning."""
        road_left = -self.config["road_width"] / 2
        road_right = self.config["road_width"] / 2
        center_margin = Vehicle.WIDTH / 2 + self.config["boundary_buffer"]
        safe_left = road_left + center_margin
        safe_right = road_right - center_margin

        if safe_left > safe_right:
            raise ValueError(
                "Road width is too small for vehicle width and boundary buffer."
            )

        return np.linspace(
            safe_left,
            safe_right,
            int(self.config["lateral_bins_count"]),
        )

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = (
            self.config["collision_reward"] * rewards["collision_reward"]
            + self.config["offroad_reward"] * (1 - rewards["on_road_reward"])
            + self.config["high_speed_reward"] * rewards["high_speed_reward"]
            + self.config["comfort_reward"] * rewards["comfort_reward"]
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["offroad_reward"],
                    self.config["high_speed_reward"],
                ],
                [0, 1],
            )
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        action_array = np.array(action if action is not None else [0, 0])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "comfort_reward": float(np.linalg.norm(action_array)),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or (
            self.config["offroad_terminal"] and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
