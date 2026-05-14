from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate the project root.")


PROJECT_ROOT = find_project_root()
HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from dqn_notebook_utils import (  # noqa: E402
    build_dqn_args,
    build_env_config,
    evaluate_saved_model,
    load_dqn_backend,
    train_and_display,
)


def common_parts() -> dict[str, Any]:
    environment_overrides = {
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
        "ego_spacing": 2.0,
        "vehicles_density": 1.0,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "initial_lane_id": None,
        "offroad_terminal": False,
    }
    return {
        "environment_profile": "structured_baseline",
        "eval_environment_profile": "structured_baseline",
        "environment_overrides": environment_overrides,
        "eval_environment_overrides": dict(environment_overrides),
        "observation_config": {
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action_config": {"type": "DiscreteMetaAction"},
        "reward_config": {
            "collision_reward": -1.0,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0.0,
            "normalize_reward": True,
        },
        "speed_config": {"reward_speed_range": [20.0, 30.0]},
    }


def tuned_hyperparameters(n_envs: int) -> dict[str, Any]:
    train_freq = 4
    return {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 64,
        "gamma": 0.95,
        "target_update_interval": 1_000,
        "train_freq": train_freq,
        "gradient_steps": train_freq * n_envs,
        "exploration_fraction": 0.70,
        "exploration_final_eps": 0.10,
        "progress_every": 1_000,
        "verbose": 0,
    }


def build_experiment(
    *,
    results_dir: Path,
    run_name: str,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    n_envs: int,
    device: str,
    adaptive_longitudinal: dict[str, Any],
    rear_flow: dict[str, Any] | None = None,
    traffic_flow_reward: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = common_parts()
    training_env_config = build_env_config(
        profile_name=parts["environment_profile"],
        profile_overrides=parts["environment_overrides"],
        observation=parts["observation_config"],
        action=parts["action_config"],
        reward=parts["reward_config"],
        speed=parts["speed_config"],
        adaptive_longitudinal=adaptive_longitudinal,
        rear_flow=rear_flow,
        traffic_flow_reward=traffic_flow_reward,
    )
    saved_model_eval_env_config = build_env_config(
        profile_name=parts["eval_environment_profile"],
        profile_overrides=parts["eval_environment_overrides"],
        observation=parts["observation_config"],
        action=parts["action_config"],
        reward=parts["reward_config"],
        speed=parts["speed_config"],
        adaptive_longitudinal=adaptive_longitudinal,
        rear_flow=rear_flow,
        traffic_flow_reward=traffic_flow_reward,
    )
    args = build_dqn_args(
        results_dir=results_dir,
        run_name=run_name,
        timesteps=timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        num_envs=n_envs,
        device=device,
        hyperparameters=tuned_hyperparameters(n_envs),
    )
    return {
        "run_name": run_name,
        "args": args,
        "training_env_config": training_env_config,
        "saved_model_eval_env_config": saved_model_eval_env_config,
    }


def adaptive_base_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "delta",
        "ttc_midpoint": 4.0,
        "ttc_temperature": 1.0,
        "ttc_cap": 10.0,
        "min_target_speed": 10.0,
        "max_target_speed": 35.0,
        "faster_max_delta": 1.25,
        "slower_min_delta": 1.25,
        "slower_max_delta": 2.5,
    }


def rear_flow_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "spawn_on_reset": True,
        "spawn_during_episode": True,
        "vehicles_per_lane": 1,
        "lanes": "ego_and_adjacent",
        "distance_range": [25.0, 70.0],
        "speed_offset_range": [2.0, 6.0],
        "absolute_speed_range": [23.0, 34.0],
        "min_lane_gap": 18.0,
        "spawn_probability": 0.35,
        "cooldown_policy_steps": 3,
        "min_ego_progress": 80.0,
        "max_extra_vehicles": 12,
    }


def traffic_flow_reward_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "penalty_weight": 0.12,
        "speed_tolerance": 2.0,
        "max_penalty": 0.8,
        "front_ttc_safe": 4.0,
        "rear_ttc_pressure": 5.0,
        "ttc_cap": 10.0,
        "rear_pressure_floor": 0.25,
        "flow_radius": 120.0,
        "lanes": "ego_and_adjacent",
    }


def run_one(
    *,
    trainer,
    experiment: dict[str, Any],
    label: str,
    saved_eval_episodes: int,
    saved_eval_seed: int,
) -> dict[str, Any]:
    print(f"\n=== {label} ===", flush=True)
    summary = train_and_display(
        trainer,
        experiment["args"],
        experiment["training_env_config"],
        label=label,
    )
    saved_summary = evaluate_saved_model(
        trainer,
        summary_path=Path(experiment["args"].results_root) / experiment["run_name"] / "summary.json",
        env_config=experiment["saved_model_eval_env_config"],
        episodes=saved_eval_episodes,
        seed=saved_eval_seed,
        name=f"saved_model_eval_{saved_eval_episodes}_episodes",
        label=label,
    )
    return {"summary": summary, "saved_eval_summary": saved_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline and adaptive DQN notebook experiments.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-eval-episodes", type=int, default=5)
    parser.add_argument("--saved-eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=["all", "baseline", "rear_flow", "rear_reward", "safe_speed"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    n_envs = min(4, os.cpu_count() or 1)

    baseline_trainer, _, _, baseline_results_dir, default_device = load_dqn_backend(
        backend_module="elurant_dqn",
        notebook_subdir="baseline_dqn",
        results_subdir="baseline_dqn",
    )
    adaptive_trainer, _, _, adaptive_results_dir, _ = load_dqn_backend(
        backend_module="elurant_dqn",
        notebook_subdir="adaptive_lower_controller",
        results_subdir="adaptive_lower_controller",
    )

    runs: list[tuple[str, Any, dict[str, Any], str, int]] = []
    baseline = build_experiment(
        results_dir=baseline_results_dir,
        run_name="baseline_dqn_tuned_20k",
        timesteps=cli.timesteps,
        eval_episodes=cli.train_eval_episodes,
        seed=cli.seed,
        n_envs=n_envs,
        device=default_device,
        adaptive_longitudinal={"enabled": False},
        rear_flow={"enabled": False},
        traffic_flow_reward={"enabled": False},
    )
    runs.append(("baseline", baseline_trainer, baseline, "Baseline DQN", cli.seed + 10000))

    rear_flow = build_experiment(
        results_dir=adaptive_results_dir,
        run_name="adaptive_rear_flow_env_20k",
        timesteps=cli.timesteps,
        eval_episodes=cli.train_eval_episodes,
        seed=cli.seed,
        n_envs=n_envs,
        device=default_device,
        adaptive_longitudinal={**adaptive_base_config(), "mode": "delta"},
        rear_flow=rear_flow_config(),
        traffic_flow_reward={"enabled": False},
    )
    runs.append(("rear_flow", adaptive_trainer, rear_flow, "Adaptive DQN - Rear Flow Env", cli.seed + 10000))

    rear_reward = build_experiment(
        results_dir=adaptive_results_dir,
        run_name="adaptive_rear_flow_reward_20k",
        timesteps=cli.timesteps,
        eval_episodes=cli.train_eval_episodes,
        seed=cli.seed + 100,
        n_envs=n_envs,
        device=default_device,
        adaptive_longitudinal={**adaptive_base_config(), "mode": "delta"},
        rear_flow=rear_flow_config(),
        traffic_flow_reward=traffic_flow_reward_config(),
    )
    runs.append(("rear_reward", adaptive_trainer, rear_reward, "Adaptive DQN - Rear Flow + Reward", cli.seed + 10100))

    safe_speed_adaptive = {
        **adaptive_base_config(),
        "mode": "safe_speed_limiter",
        "min_target_speed": 18.0,
        "cruise_speed": 28.0,
        "action_speed_delta": 3.0,
    }
    safe_speed = build_experiment(
        results_dir=adaptive_results_dir,
        run_name="adaptive_safe_speed_controller_20k",
        timesteps=cli.timesteps,
        eval_episodes=cli.train_eval_episodes,
        seed=cli.seed + 200,
        n_envs=n_envs,
        device=default_device,
        adaptive_longitudinal=safe_speed_adaptive,
        rear_flow=rear_flow_config(),
        traffic_flow_reward=traffic_flow_reward_config(),
    )
    runs.append(("safe_speed", adaptive_trainer, safe_speed, "Adaptive DQN - Safe-Speed Controller", cli.seed + 10200))

    completed: dict[str, Any] = {}
    for key, trainer, experiment, label, saved_seed in runs:
        if cli.only != "all" and cli.only != key:
            continue
        completed[key] = run_one(
            trainer=trainer,
            experiment=experiment,
            label=label,
            saved_eval_episodes=cli.saved_eval_episodes,
            saved_eval_seed=saved_seed,
        )

    print("\nCompleted runs:")
    for key, result in completed.items():
        summary = result["summary"]
        saved_eval = result["saved_eval_summary"]
        print(
            f"{key}: summary={Path(summary['model_path']).parents[1] / 'summary.json'} "
            f"saved_eval={saved_eval['evaluation_metrics_path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
