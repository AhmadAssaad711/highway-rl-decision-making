from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAB_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ppo" / "overtake_lab"
LAB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


@dataclass
class RewardConfig:
    collision_penalty: float = -120.0
    offroad_penalty: float = -120.0
    speed_weight: float = 28.0
    progress_bonus: float = 20.0
    success_bonus: float = 80.0
    steering_penalty: float = 0.4
    blocked_in_right_penalty: float = -3.0
    blocked_overtake_bonus: float = 3.0
    keep_right_bonus: float = 1.5
    unsafe_headway_penalty: float = -8.0
    normalize_reward: bool = False
    reward_clip_low: float = -150.0
    reward_clip_high: float = 120.0


@dataclass
class ScenarioConfig:
    lanes_count: int = 3
    vehicles_per_lane: int = 5
    observation_vehicles: int = 16
    duration: int = 50
    simulation_frequency: int = 20
    policy_frequency: int = 2
    road_length: float = 2500.0
    speed_limit: float = 30.0
    steering_range: float = 0.01
    ego_speed_range: tuple[float, float] = (24.0, 26.0)
    other_speed_range: tuple[float, float] = (18.0, 23.0)
    spawn_lead_range: tuple[float, float] = (60.0, 110.0)
    spawn_gap_range: tuple[float, float] = (25.0, 45.0)
    initial_lane_id: int = 2
    offroad_terminal: bool = True


@dataclass
class PPOConfig:
    timesteps: int = 6_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    batch_size: int = 128
    n_steps: int = 64
    n_epochs: int = 5
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 24
    eval_freq: int = 1_536
    eval_episodes: int = 3
    seed: int = 42
    device: str = "auto"
    progress_every: int = 1_536
    policy_pi: tuple[int, int] = (256, 256)
    policy_vf: tuple[int, int] = (256, 256)


@dataclass
class ExperimentConfig:
    name: str = "default_overtake"
    reward: RewardConfig = field(default_factory=RewardConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


class TimestepProgressCallback(BaseCallback):
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


class NotebookOvertakeEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict[str, Any]:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features": ["x", "y", "vx", "vy"],
                    "vehicles_count": 16,
                    "absolute": False,
                    "normalize": True,
                    "clip": False,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "acceleration_range": [-5.0, 5.0],
                    "steering_range": [-0.01, 0.01],
                    "speed_range": [0.0, 30.0],
                },
                "lanes_count": 3,
                "vehicles_per_lane": 5,
                "vehicles_count": 15,
                "controlled_vehicles": 1,
                "initial_lane_id": 2,
                "duration": 50,
                "simulation_frequency": 20,
                "policy_frequency": 2,
                "ego_spacing": 2.0,
                "road_length": 2500.0,
                "speed_limit": 30.0,
                "ego_speed_range": [24.0, 26.0],
                "other_speed_range": [18.0, 23.0],
                "spawn_lead_range": [60.0, 110.0],
                "spawn_gap_range": [25.0, 45.0],
                "offroad_terminal": True,
                "reward": asdict(RewardConfig()),
            }
        )
        return config

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        resolved_config = merge_nested_dicts(self.default_config(), config)
        self._other_vehicles: list[IDMVehicle] = []
        self._reward_components: dict[str, float] = {}
        self._episode_reward = 0.0
        self._episode_raw_reward = 0.0
        self._episode_speeds: list[float] = []
        self._episode_lane_numbers: list[float] = []
        self._last_overtaken_count = 0
        super().__init__(config=resolved_config, render_mode=render_mode)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._reward_components = {}
        self._episode_reward = 0.0
        self._episode_raw_reward = 0.0
        self._episode_speeds = []
        self._episode_lane_numbers = []
        self._last_overtaken_count = 0
        return obs, info

    def step(self, action):
        filtered_action = self._filter_boundary_steering(action)
        obs, reward, terminated, truncated, info = super().step(filtered_action)
        info["applied_action"] = np.asarray(filtered_action, dtype=np.float32).tolist()
        return obs, reward, terminated, truncated, info

    def _filter_boundary_steering(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        filtered_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if filtered_action.size < 2:
            return filtered_action

        lane_index = int(self.vehicle.lane_index[2])
        if lane_index == 0 and filtered_action[1] < 0.0:
            filtered_action[1] = 0.0
        elif lane_index == int(self.config["lanes_count"]) - 1 and filtered_action[1] > 0.0:
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
        self._other_vehicles = []

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
            current_x = float(
                self.np_random.uniform(
                    low=float(self.config["spawn_lead_range"][0]),
                    high=float(self.config["spawn_lead_range"][1]),
                )
            )
            for _ in range(int(self.config["vehicles_per_lane"])):
                vehicle_speed = float(
                    self.np_random.uniform(
                        low=float(self.config["other_speed_range"][0]),
                        high=float(self.config["other_speed_range"][1]),
                    )
                )
                vehicle = IDMVehicle(
                    self.road,
                    lane.position(current_x, 0.0),
                    heading=lane.heading_at(current_x),
                    speed=vehicle_speed,
                    target_lane_index=("0", "1", lane_id),
                    target_speed=float(self.config["speed_limit"]),
                    enable_lane_change=True,
                )
                self._configure_other_vehicle(vehicle)
                self.road.vehicles.append(vehicle)
                self._other_vehicles.append(vehicle)
                current_x += float(
                    self.np_random.uniform(
                        low=float(self.config["spawn_gap_range"][0]),
                        high=float(self.config["spawn_gap_range"][1]),
                    )
                )

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

    def _overtaken_vehicle_count(self) -> int:
        ego_x = float(self.vehicle.position[0])
        return sum(ego_x > float(other.position[0]) for other in self._other_vehicles)

    def _has_overtaken_all_vehicles(self) -> bool:
        return self._overtaken_vehicle_count() == len(self._other_vehicles)

    def _is_success(self) -> bool:
        return self._has_overtaken_all_vehicles() and not self.vehicle.crashed and self.vehicle.on_road

    def _is_truncated(self) -> bool:
        return bool(super()._is_truncated() or self._has_overtaken_all_vehicles())

    def _front_vehicle_same_lane(self) -> IDMVehicle | None:
        ego_x = float(self.vehicle.position[0])
        ego_lane = int(self.vehicle.lane_index[2])
        candidates = [
            vehicle
            for vehicle in self._other_vehicles
            if int(vehicle.lane_index[2]) == ego_lane and float(vehicle.position[0]) > ego_x
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda vehicle: float(vehicle.position[0]))

    def _reward(self, action) -> float:
        reward_cfg = self.config["reward"]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else np.zeros(2)
        steering_signal = float(np.clip(action_arr[1], -1.0, 1.0)) if action_arr.size > 1 else 0.0

        collision = 1.0 if self.vehicle.crashed else 0.0
        offroad = 1.0 if not self.vehicle.on_road else 0.0
        speed = float(np.clip(self.vehicle.speed, 0.0, float(self.config["speed_limit"])))
        lane_number = float(self._paper_lane_number())
        overtaken_count = self._overtaken_vehicle_count()
        progress_delta = max(0, overtaken_count - self._last_overtaken_count)

        front_vehicle = self._front_vehicle_same_lane()
        front_gap = float(front_vehicle.position[0] - self.vehicle.position[0]) if front_vehicle is not None else np.inf
        front_speed = float(front_vehicle.speed) if front_vehicle is not None else float(self.config["speed_limit"])
        blocked = front_vehicle is not None and front_gap < 35.0 and front_speed < speed - 0.5

        collision_term = reward_cfg["collision_penalty"] * collision
        offroad_term = reward_cfg["offroad_penalty"] * offroad
        speed_term = reward_cfg["speed_weight"] * (speed / float(self.config["speed_limit"]))
        progress_term = reward_cfg["progress_bonus"] * float(progress_delta)
        success_term = reward_cfg["success_bonus"] if self._is_success() else 0.0
        steering_term = -reward_cfg["steering_penalty"] * abs(steering_signal)

        if blocked:
            if lane_number <= 1.0:
                lane_strategy_term = reward_cfg["blocked_in_right_penalty"]
            else:
                lane_strategy_term = reward_cfg["blocked_overtake_bonus"] * (lane_number - 1.0)
        else:
            lane_strategy_term = reward_cfg["keep_right_bonus"] if lane_number == 1.0 else -0.5 * (lane_number - 1.0)

        if front_gap < 18.0:
            headway_term = reward_cfg["unsafe_headway_penalty"] * (1.0 - max(front_gap, 0.0) / 18.0)
        else:
            headway_term = 0.0

        raw_reward = (
            collision_term
            + offroad_term
            + speed_term
            + progress_term
            + success_term
            + steering_term
            + lane_strategy_term
            + headway_term
        )

        if collision or offroad:
            raw_reward = min(raw_reward, reward_cfg["collision_penalty"])

        if reward_cfg["normalize_reward"]:
            raw_reward = float(
                np.clip(raw_reward, reward_cfg["reward_clip_low"], reward_cfg["reward_clip_high"])
            )
            reward = float(
                np.interp(
                    raw_reward,
                    [reward_cfg["reward_clip_low"], reward_cfg["reward_clip_high"]],
                    [0.0, 1.0],
                )
            )
        else:
            reward = float(raw_reward)

        self._last_overtaken_count = overtaken_count
        self._reward_components = {
            "raw_reward": float(raw_reward),
            "reward": float(reward),
            "collision_term": float(collision_term),
            "offroad_term": float(offroad_term),
            "speed_term": float(speed_term),
            "progress_term": float(progress_term),
            "success_term": float(success_term),
            "steering_term": float(steering_term),
            "lane_strategy_term": float(lane_strategy_term),
            "headway_term": float(headway_term),
            "speed_mps": float(speed),
            "paper_lane_number": float(lane_number),
            "overtaken_count": float(overtaken_count),
            "front_gap": float(front_gap if np.isfinite(front_gap) else -1.0),
            "blocked": float(blocked),
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
                "mean_lane_number": float(np.mean(self._episode_lane_numbers)) if self._episode_lane_numbers else 0.0,
                "safe_completion": float(not self.vehicle.crashed and self.vehicle.on_road),
                "overtaken_count": float(self._overtaken_vehicle_count()),
            }
        return info


def build_env_config(experiment: ExperimentConfig) -> dict[str, Any]:
    scenario = experiment.scenario
    reward = experiment.reward
    return {
        "observation": {
            "vehicles_count": scenario.observation_vehicles,
        },
        "action": {
            "steering_range": [-float(scenario.steering_range), float(scenario.steering_range)],
            "speed_range": [0.0, float(scenario.speed_limit)],
        },
        "lanes_count": int(scenario.lanes_count),
        "vehicles_per_lane": int(scenario.vehicles_per_lane),
        "vehicles_count": int(scenario.lanes_count * scenario.vehicles_per_lane),
        "initial_lane_id": int(scenario.initial_lane_id),
        "duration": int(scenario.duration),
        "simulation_frequency": int(scenario.simulation_frequency),
        "policy_frequency": int(scenario.policy_frequency),
        "road_length": float(scenario.road_length),
        "speed_limit": float(scenario.speed_limit),
        "ego_speed_range": list(scenario.ego_speed_range),
        "other_speed_range": list(scenario.other_speed_range),
        "spawn_lead_range": list(scenario.spawn_lead_range),
        "spawn_gap_range": list(scenario.spawn_gap_range),
        "offroad_terminal": bool(scenario.offroad_terminal),
        "reward": asdict(reward),
    }


def make_overtake_env(
    experiment: ExperimentConfig,
    render_mode: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> gym.Env:
    base_config = build_env_config(experiment)
    merged_config = merge_nested_dicts(base_config, config_overrides)
    return NotebookOvertakeEnv(config=merged_config, render_mode=render_mode)


def evaluate_model(
    model: PPO,
    experiment: ExperimentConfig,
    *,
    episodes: int | None = None,
    base_seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    eval_episodes = int(episodes if episodes is not None else experiment.ppo.eval_episodes)
    seed = int(base_seed if base_seed is not None else experiment.ppo.seed + 10_000)

    episode_rewards: list[float] = []
    collisions: list[float] = []
    offroads: list[float] = []
    successes: list[float] = []
    overtaken_counts: list[float] = []
    safe_completions: list[float] = []
    episode_seconds: list[float] = []

    for episode_idx in range(eval_episodes):
        env = make_overtake_env(experiment, config_overrides=config_overrides)
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            final_info = info

        metrics = final_info.get("episode_metrics", {})
        episode_rewards.append(float(metrics.get("episode_reward", 0.0)))
        collisions.append(float(metrics.get("collision", 0.0)))
        offroads.append(float(metrics.get("offroad", 0.0)))
        successes.append(float(metrics.get("success", 0.0)))
        overtaken_counts.append(float(metrics.get("overtaken_count", 0.0)))
        safe_completions.append(float(metrics.get("safe_completion", 0.0)))
        episode_seconds.append(float(metrics.get("episode_length_seconds", 0.0)))
        env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "collision_rate": float(np.mean(collisions)),
        "offroad_rate": float(np.mean(offroads)),
        "success_rate": float(np.mean(successes)),
        "safe_completion_rate": float(np.mean(safe_completions)),
        "mean_overtaken_count": float(np.mean(overtaken_counts)),
        "mean_episode_seconds": float(np.mean(episode_seconds)),
    }


def train_overtake_agent(experiment: ExperimentConfig) -> tuple[PPO, dict[str, Any]]:
    run_dir = LAB_OUTPUT_DIR / experiment.name
    best_model_dir = run_dir / "best_model"
    logs_dir = run_dir / "logs"
    tb_dir = run_dir / "tb"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = experiment.ppo
    rollout_size = int(ppo_cfg.n_envs) * int(ppo_cfg.n_steps)
    env = make_vec_env(
        lambda: make_overtake_env(experiment),
        n_envs=int(ppo_cfg.n_envs),
        seed=int(ppo_cfg.seed),
    )
    eval_env = Monitor(make_overtake_env(experiment))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(ppo_cfg.learning_rate),
        n_steps=int(ppo_cfg.n_steps),
        batch_size=int(ppo_cfg.batch_size),
        n_epochs=int(ppo_cfg.n_epochs),
        gamma=float(ppo_cfg.gamma),
        gae_lambda=float(ppo_cfg.gae_lambda),
        clip_range=float(ppo_cfg.clip_range),
        ent_coef=float(ppo_cfg.ent_coef),
        vf_coef=float(ppo_cfg.vf_coef),
        max_grad_norm=float(ppo_cfg.max_grad_norm),
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=int(ppo_cfg.seed),
        device=str(ppo_cfg.device),
        policy_kwargs={"net_arch": {"pi": list(ppo_cfg.policy_pi), "vf": list(ppo_cfg.policy_vf)}},
    )

    progress_callback = TimestepProgressCallback(
        total_timesteps=int(ppo_cfg.timesteps),
        every_n_steps=int(ppo_cfg.progress_every),
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(logs_dir),
        eval_freq=max(1, int(ppo_cfg.eval_freq) // max(1, int(ppo_cfg.n_envs))),
        deterministic=True,
        render=False,
    )

    print(
        f"Training {experiment.name} with "
        f"timesteps={ppo_cfg.timesteps}, lr={ppo_cfg.learning_rate}, "
        f"gamma={ppo_cfg.gamma}, n_envs={ppo_cfg.n_envs}, "
        f"n_steps={ppo_cfg.n_steps}, rollout_size={rollout_size}"
    )
    if int(ppo_cfg.timesteps) < rollout_size:
        print(
            f"Warning: total timesteps {ppo_cfg.timesteps} is smaller than one PPO rollout "
            f"({rollout_size}), so SB3 will still collect about {rollout_size} steps."
        )
    start = time.time()
    model.learn(
        total_timesteps=int(ppo_cfg.timesteps),
        callback=[progress_callback, eval_callback],
        tb_log_name=experiment.name,
        progress_bar=False,
    )
    elapsed = time.time() - start

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))

    best_model_path = best_model_dir / "best_model.zip"
    eval_model = PPO.load(str(best_model_path if best_model_path.exists() else final_model_path.with_suffix(".zip")))
    evaluation = evaluate_model(eval_model, experiment)
    evaluation["elapsed_seconds"] = float(elapsed)
    evaluation["final_model_path"] = str(final_model_path.with_suffix(".zip"))
    evaluation["best_model_path"] = str(best_model_path) if best_model_path.exists() else None

    (run_dir / "experiment.json").write_text(json.dumps(asdict(experiment), indent=2), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    eval_env.close()
    env.close()
    return eval_model, evaluation


def render_human_episodes(
    model: PPO,
    experiment: ExperimentConfig,
    *,
    episodes: int = 3,
    base_seed: int | None = None,
    sleep: float = 0.05,
    config_overrides: dict[str, Any] | None = None,
) -> None:
    seed = int(base_seed if base_seed is not None else experiment.ppo.seed + 20_000)
    for episode_idx in range(int(episodes)):
        env = make_overtake_env(experiment, render_mode="human", config_overrides=config_overrides)
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            env.render()
            if sleep > 0:
                time.sleep(sleep)

        metrics = final_info.get("episode_metrics", {})
        print(
            f"Episode {episode_idx + 1}: "
            f"reward={total_reward:.2f}, "
            f"collision={int(metrics.get('collision', 0.0))}, "
            f"offroad={int(metrics.get('offroad', 0.0))}, "
            f"success={int(metrics.get('success', 0.0))}, "
            f"overtaken_count={metrics.get('overtaken_count', 0.0):.0f}"
        )
        env.close()


def lane_density_override(lanes_count: int, vehicles_per_lane: int, observation_vehicles: int = 16) -> dict[str, Any]:
    return {
        "lanes_count": int(lanes_count),
        "vehicles_per_lane": int(vehicles_per_lane),
        "vehicles_count": int(lanes_count * vehicles_per_lane),
        "initial_lane_id": int(lanes_count - 1),
        "observation": {"vehicles_count": int(observation_vehicles)},
    }
