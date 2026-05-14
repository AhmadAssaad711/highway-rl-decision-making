"""
Standalone PPO reproduction of the highway decision-making formulation in
"Decision-making for Autonomous Vehicles on Highway: Deep Reinforcement
Learning with Continuous Action Horizon" (arXiv:2008.11852).

This script is intentionally isolated from the existing PPO baselines in the
repository. It matches the paper where the paper is explicit:

- continuous acceleration and steering actions
- 3-lane highway with 5 surrounding vehicles per lane
- 50 s episodes
- 20 Hz simulation and 1 Hz decision frequency
- reward composed of collision, speed, and lane-preference terms
- PPO hyperparameters gamma=0.8, lambda=0.92, clip=0.2, lr=0.01,
  total_timesteps=51200, batch_size=64

The paper leaves several implementation details unspecified, notably the
network architecture, PPO epochs, and exact state tensor packing. This script
documents those approximations in code instead of silently inheriting them
from the repo's other PPO experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway-env modules
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "artifacts" / "ppo" / "paper_ppo_reproduction"
MODELS_DIR = RUN_ROOT / "models"
TB_DIR = RUN_ROOT / "tensorboard"
LOGS_DIR = RUN_ROOT / "logs"


def merge_nested_dicts(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not updates:
        return merged

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class PaperHighwayEnv(HighwayEnv):
    """
    Custom highway environment aligned to the paper's formulation.

    The main modeling choices are:
    - ego vehicle uses true continuous acceleration and steering
    - surrounding traffic uses IDM + MOBIL behavior
    - all surrounding vehicles start ahead of the ego vehicle so the task is
      explicitly an overtaking problem
    - reward follows the paper's equation and is linearly normalized to [0, 1]
      using the formula's theoretical min/max over the 3-lane setup
    """

    PAPER_MIN_RAW_REWARD = -120.0
    PAPER_MAX_RAW_REWARD = 40.0
    DEFAULT_STEERING_RANGE = 0.02
    OVERTAKE_PROGRESS_BONUS = 4.0
    SUCCESS_BONUS = 20.0
    STEERING_PENALTY = 2.0

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features": ["x", "y", "vx", "vy"],
                    "vehicles_count": 16,  # ego + 15 surrounding vehicles
                    "absolute": False,
                    "normalize": True,
                    "clip": False,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "acceleration_range": [-5.0, 5.0],
                    # A 1 Hz decision rate makes raw steering extremely sensitive in
                    # highway-env. Use a tighter steering envelope so lane changes are
                    # feasible without immediately leaving the road.
                    "steering_range": [-cls.DEFAULT_STEERING_RANGE, cls.DEFAULT_STEERING_RANGE],
                    "speed_range": [0.0, 30.0],
                },
                "lanes_count": 3,
                "vehicles_per_lane": 5,
                "vehicles_count": 15,
                "controlled_vehicles": 1,
                "initial_lane_id": 2,  # right lane in highway-env indexing
                "duration": 50,  # [s]
                "simulation_frequency": 20,  # [Hz]
                "policy_frequency": 1,  # [Hz]
                "ego_spacing": 2.0,
                "road_length": 2500.0,
                "speed_limit": 30.0,
                "ego_speed_range": [23.0, 25.0],
                "other_speed_range": [23.0, 25.0],
                "spawn_lead_range": [80.0, 120.0],
                "spawn_gap_range": [28.0, 55.0],
                "offroad_terminal": True,
                "paper_normalize_reward": True,
            }
        )
        return config

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        resolved_config = merge_nested_dicts(self.default_config(), config)
        self._paper_other_vehicles: list[IDMVehicle] = []
        self._reward_components: dict[str, float] = {}
        self._episode_reward = 0.0
        self._episode_raw_reward = 0.0
        self._episode_speeds: list[float] = []
        self._episode_lane_numbers: list[float] = []
        self._last_overtaken_count = 0
        super().__init__(config=resolved_config, render_mode=render_mode)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._reset_episode_tracking()
        return obs, info

    def step(self, action):
        filtered_action = self._filter_boundary_steering(action)
        obs, reward, terminated, truncated, info = super().step(filtered_action)
        info["applied_action"] = np.asarray(filtered_action, dtype=np.float32).tolist()
        return obs, reward, terminated, truncated, info

    def _reset_episode_tracking(self) -> None:
        self._reward_components = {}
        self._episode_reward = 0.0
        self._episode_raw_reward = 0.0
        self._episode_speeds = []
        self._episode_lane_numbers = []
        self._last_overtaken_count = 0

    def _filter_boundary_steering(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        filtered_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if filtered_action.size < 2:
            return filtered_action

        lane_index = int(self.vehicle.lane_index[2])
        leftmost_lane = 0
        rightmost_lane = int(self.config["lanes_count"]) - 1

        # Prevent exploratory steering that points directly out of the road from
        # an edge lane. The agent can still steer back inward or change lanes.
        if lane_index == leftmost_lane and filtered_action[1] < 0.0:
            filtered_action[1] = 0.0
        elif lane_index == rightmost_lane and filtered_action[1] > 0.0:
            filtered_action[1] = 0.0
        return filtered_action

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                length=float(self.config["road_length"]),
                speed_limit=float(self.config["speed_limit"]),
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        self.controlled_vehicles = []
        self._paper_other_vehicles = []

        ego_lane_id = int(self.config["initial_lane_id"])
        ego_lane = self.road.network.get_lane(("0", "1", ego_lane_id))
        ego_speed = float(
            self.np_random.uniform(
                low=float(self.config["ego_speed_range"][0]),
                high=float(self.config["ego_speed_range"][1]),
            )
        )
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(0.0, 0.0),
            ego_lane.heading_at(0.0),
            ego_speed,
        )
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)

        for lane_id in range(int(self.config["lanes_count"])):
            lane = self.road.network.get_lane(("0", "1", lane_id))
            longitudinal_positions = self._sample_lane_positions(
                count=int(self.config["vehicles_per_lane"])
            )
            for longitudinal_position in longitudinal_positions:
                vehicle_speed = float(
                    self.np_random.uniform(
                        low=float(self.config["other_speed_range"][0]),
                        high=float(self.config["other_speed_range"][1]),
                    )
                )
                vehicle = IDMVehicle(
                    self.road,
                    lane.position(float(longitudinal_position), 0.0),
                    heading=lane.heading_at(float(longitudinal_position)),
                    speed=vehicle_speed,
                    target_lane_index=("0", "1", lane_id),
                    target_speed=float(self.config["speed_limit"]),
                    enable_lane_change=True,
                )
                self._configure_other_vehicle(vehicle)
                self.road.vehicles.append(vehicle)
                self._paper_other_vehicles.append(vehicle)

    def _sample_lane_positions(self, count: int) -> list[float]:
        lead_low, lead_high = self.config["spawn_lead_range"]
        gap_low, gap_high = self.config["spawn_gap_range"]
        positions: list[float] = []
        current_x = float(self.np_random.uniform(lead_low, lead_high))
        for _ in range(count):
            positions.append(current_x)
            current_x += float(self.np_random.uniform(gap_low, gap_high))
        return positions

    def _configure_other_vehicle(self, vehicle: IDMVehicle) -> None:
        vehicle.ACC_MAX = 6.0
        vehicle.COMFORT_ACC_MAX = 6.0
        vehicle.COMFORT_ACC_MIN = -5.0
        vehicle.DELTA = 4.0
        vehicle.TIME_WANTED = 1.5
        vehicle.DISTANCE_WANTED = 10.0
        vehicle.POLITENESS = 0.001
        vehicle.LANE_CHANGE_MIN_ACC_GAIN = 0.2
        vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0
        vehicle.target_speed = float(self.config["speed_limit"])

    def _paper_lane_number(self) -> int:
        lane_index = int(self.vehicle.lane_index[2])
        return int(self.config["lanes_count"]) - lane_index

    def _has_overtaken_all_vehicles(self) -> bool:
        return self._overtaken_vehicle_count() == len(self._paper_other_vehicles)

    def _overtaken_vehicle_count(self) -> int:
        ego_x = float(self.vehicle.position[0])
        return sum(ego_x > float(other.position[0]) for other in self._paper_other_vehicles)

    def _is_success(self) -> bool:
        return self._has_overtaken_all_vehicles() and not self.vehicle.crashed and self.vehicle.on_road

    def _is_truncated(self) -> bool:
        return bool(super()._is_truncated() or self._has_overtaken_all_vehicles())

    def _reward(self, action) -> float:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else np.zeros(2)
        steering_signal = float(np.clip(action_arr[1], -1.0, 1.0)) if action_arr.size > 1 else 0.0
        collision = 1.0 if self.vehicle.crashed else 0.0
        offroad = 1.0 if not self.vehicle.on_road else 0.0
        failure = collision or offroad
        speed = float(np.clip(self.vehicle.speed, 0.0, float(self.config["speed_limit"])))
        lane_number = float(self._paper_lane_number())
        overtaken_count = self._overtaken_vehicle_count()
        progress_delta = max(0, overtaken_count - self._last_overtaken_count)

        collision_term = -100.0 * collision
        offroad_term = -100.0 * offroad
        speed_term = 40.0 * (speed / float(self.config["speed_limit"])) ** 2
        lane_term = -10.0 * (lane_number - 1.0)
        progress_term = self.OVERTAKE_PROGRESS_BONUS * float(progress_delta)
        steering_term = -self.STEERING_PENALTY * abs(steering_signal)
        success_term = self.SUCCESS_BONUS if self._is_success() else 0.0
        raw_reward = collision_term + offroad_term + speed_term + lane_term + progress_term + steering_term + success_term

        # Make catastrophic terminal outcomes clearly unattractive instead of
        # letting them keep a positive speed reward.
        if failure:
            raw_reward = self.PAPER_MIN_RAW_REWARD
        self._last_overtaken_count = overtaken_count

        if self.config["paper_normalize_reward"]:
            reward = np.interp(
                raw_reward,
                [self.PAPER_MIN_RAW_REWARD, self.PAPER_MAX_RAW_REWARD],
                [0.0, 1.0],
            )
            reward = float(np.clip(reward, 0.0, 1.0))
        else:
            reward = float(raw_reward)

        self._reward_components = {
            "raw_reward": float(raw_reward),
            "reward": float(reward),
            "collision_term": float(collision_term),
            "offroad_term": float(offroad_term),
            "speed_term": float(speed_term),
            "lane_term": float(lane_term),
            "progress_term": float(progress_term),
            "steering_term": float(steering_term),
            "success_term": float(success_term),
            "speed_mps": float(speed),
            "paper_lane_number": float(lane_number),
            "overtaken_count": float(overtaken_count),
        }
        self._episode_reward += float(reward)
        self._episode_raw_reward += float(raw_reward)
        self._episode_speeds.append(float(speed))
        self._episode_lane_numbers.append(float(lane_number))
        return float(reward)

    def _info(self, obs: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
        info = super()._info(obs, action)
        info.update(self._reward_components)
        info["overtaken_all"] = float(self._has_overtaken_all_vehicles())
        info["success"] = float(self._is_success())
        info["offroad"] = float(not self.vehicle.on_road)

        if self._is_terminated() or self._is_truncated():
            decision_steps = float(
                self.steps * float(self.config["policy_frequency"]) / float(self.config["simulation_frequency"])
            )
            elapsed_seconds = float(self.steps) / float(self.config["simulation_frequency"])
            info["episode_metrics"] = {
                "episode_reward": float(self._episode_reward),
                "episode_raw_reward": float(self._episode_raw_reward),
                "episode_length": float(self.steps),
                "episode_length_decisions": decision_steps,
                "episode_length_seconds": elapsed_seconds,
                "collision": float(self.vehicle.crashed),
                "offroad": float(not self.vehicle.on_road),
                "success": float(self._is_success()),
                "overtaken_all": float(self._has_overtaken_all_vehicles()),
                "mean_speed_mps": float(np.mean(self._episode_speeds)) if self._episode_speeds else 0.0,
                "mean_lane_number": (
                    float(np.mean(self._episode_lane_numbers)) if self._episode_lane_numbers else 0.0
                ),
                "safe_completion": float(not self.vehicle.crashed and self.vehicle.on_road),
                "overtaken_count": float(self._overtaken_vehicle_count()),
            }
        return info


def make_paper_env(
    render_mode: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> gym.Env:
    return PaperHighwayEnv(config=config_overrides, render_mode=render_mode)


def scenario_overrides(
    lanes_count: int,
    vehicles_per_lane: int,
    observation_vehicles_count: int = 16,
) -> dict[str, Any]:
    return {
        "lanes_count": int(lanes_count),
        "vehicles_per_lane": int(vehicles_per_lane),
        "vehicles_count": int(lanes_count * vehicles_per_lane),
        "initial_lane_id": int(lanes_count - 1),
        "observation": {
            # Keep the observation tensor shape fixed so a policy trained on the
            # default paper scenario can still be evaluated on altered traffic
            # layouts during adaptability tests.
            "vehicles_count": int(observation_vehicles_count),
        },
    }


class TimestepProgressCallback(BaseCallback):
    """Print lightweight progress updates during training."""

    def __init__(self, total_timesteps: int, every_n_steps: int = 4096, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.every_n_steps = max(1, int(every_n_steps))
        self._next_print = self.every_n_steps
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            elapsed = time.time() - self._start_time
            progress = min(100.0, 100.0 * self.num_timesteps / self.total_timesteps)
            print(
                f"[train] timesteps={self.num_timesteps}/{self.total_timesteps} "
                f"({progress:.1f}%) elapsed={elapsed:.1f}s"
            )
            while self._next_print <= self.num_timesteps:
                self._next_print += self.every_n_steps
        return True


def evaluate_model(
    model: PPO,
    *,
    episodes: int,
    base_seed: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    episode_rewards: list[float] = []
    episode_raw_rewards: list[float] = []
    episode_lengths: list[float] = []
    episode_decision_lengths: list[float] = []
    episode_second_lengths: list[float] = []
    collisions: list[float] = []
    offroads: list[float] = []
    successes: list[float] = []
    overtakes: list[float] = []
    mean_speeds: list[float] = []
    mean_lanes: list[float] = []
    safe_completions: list[float] = []
    overtaken_counts: list[float] = []

    for episode_idx in range(int(episodes)):
        env = make_paper_env(config_overrides=config_overrides)
        obs, _ = env.reset(seed=base_seed + episode_idx)
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            final_info = info

        env.close()
        metrics = final_info.get("episode_metrics", {})
        episode_rewards.append(float(metrics.get("episode_reward", 0.0)))
        episode_raw_rewards.append(float(metrics.get("episode_raw_reward", 0.0)))
        episode_lengths.append(float(metrics.get("episode_length", 0.0)))
        episode_decision_lengths.append(float(metrics.get("episode_length_decisions", 0.0)))
        episode_second_lengths.append(float(metrics.get("episode_length_seconds", 0.0)))
        collisions.append(float(metrics.get("collision", 0.0)))
        offroads.append(float(metrics.get("offroad", 0.0)))
        successes.append(float(metrics.get("success", 0.0)))
        overtakes.append(float(metrics.get("overtaken_all", 0.0)))
        mean_speeds.append(float(metrics.get("mean_speed_mps", 0.0)))
        mean_lanes.append(float(metrics.get("mean_lane_number", 0.0)))
        safe_completions.append(float(metrics.get("safe_completion", 0.0)))
        overtaken_counts.append(float(metrics.get("overtaken_count", 0.0)))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_raw_reward": float(np.mean(episode_raw_rewards)),
        "collision_rate": float(np.mean(collisions)),
        "offroad_rate": float(np.mean(offroads)),
        "success_rate": float(np.mean(successes)),
        "overtake_rate": float(np.mean(overtakes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_episode_decisions": float(np.mean(episode_decision_lengths)),
        "mean_episode_seconds": float(np.mean(episode_second_lengths)),
        "mean_speed_mps": float(np.mean(mean_speeds)),
        "mean_lane_number": float(np.mean(mean_lanes)),
        "safe_completion_rate": float(np.mean(safe_completions)),
        "mean_overtaken_count": float(np.mean(overtaken_counts)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-matched PPO reproduction on highway-env")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="PPO rollout steps per update",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO training epochs per update",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Post-training evaluation episodes")
    parser.add_argument("--n-envs", type=int, default=3, help="Parallel training environments")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluate every N timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--progress-every", type=int, default=4096, help="Progress print frequency")
    parser.add_argument(
        "--run-adaptability-tests",
        action="store_true",
        help="Evaluate the trained policy on the paper's two altered lane/vehicle scenarios",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        make_paper_env,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        seed=args.seed,
        device=args.device,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
    )

    eval_env = Monitor(make_paper_env())
    best_model_dir = MODELS_DIR / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(LOGS_DIR),
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        deterministic=True,
        render=False,
    )

    print(
        "Starting paper reproduction PPO training with "
        f"timesteps={args.timesteps}, gamma={args.gamma}, lr={args.learning_rate}, "
        f"lambda={args.gae_lambda}, clip={args.clip_range}, n_envs={args.n_envs}"
    )
    start_time = time.time()
    progress_callback = TimestepProgressCallback(
        total_timesteps=args.timesteps,
        every_n_steps=args.progress_every,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[progress_callback, eval_callback],
        tb_log_name="paper_ppo_reproduction",
        progress_bar=False,
    )
    elapsed = time.time() - start_time

    model_path = MODELS_DIR / "paper_ppo_reproduction"
    model.save(str(model_path))

    best_model_path = best_model_dir / "best_model.zip"
    evaluation_model = PPO.load(str(best_model_path if best_model_path.exists() else model_path.with_suffix(".zip")))

    evaluation = evaluate_model(
        evaluation_model,
        episodes=args.eval_episodes,
        base_seed=args.seed + 10_000,
    )
    evaluation["elapsed_seconds"] = float(elapsed)
    evaluation["model_path"] = str(model_path.with_suffix(".zip"))
    evaluation["best_model_path"] = str(best_model_path) if best_model_path.exists() else None

    if args.run_adaptability_tests:
        evaluation["adaptability_scenario_1"] = evaluate_model(
            evaluation_model,
            episodes=args.eval_episodes,
            base_seed=args.seed + 20_000,
            config_overrides=scenario_overrides(lanes_count=4, vehicles_per_lane=5),
        )
        evaluation["adaptability_scenario_2"] = evaluate_model(
            evaluation_model,
            episodes=args.eval_episodes,
            base_seed=args.seed + 30_000,
            config_overrides=scenario_overrides(lanes_count=2, vehicles_per_lane=10),
        )

    evaluation_path = MODELS_DIR / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print(f"Training finished in {elapsed:.2f}s")
    print(json.dumps(evaluation, indent=2))

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
