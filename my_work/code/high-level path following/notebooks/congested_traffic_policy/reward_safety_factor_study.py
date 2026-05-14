from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate the project root.")


PROJECT_ROOT = find_project_root()
HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import dqn_notebook_utils as _dqn_notebook_utils  # noqa: E402

_dqn_notebook_utils = importlib.reload(_dqn_notebook_utils)
build_dqn_args = _dqn_notebook_utils.build_dqn_args
build_env_config = _dqn_notebook_utils.build_env_config
evaluate_saved_model = _dqn_notebook_utils.evaluate_saved_model
load_dqn_backend = _dqn_notebook_utils.load_dqn_backend
train_and_display = _dqn_notebook_utils.train_and_display


RESULTS_SUBDIR = "congested_reward_safety_factor_study"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "dqn" / RESULTS_SUBDIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENVIRONMENT_PROFILE = "structured_baseline"
SAFETY_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMESTEPS = 100_000
SAVED_MODEL_EVAL_EPISODES = 1_000
TRAINING_EVAL_EPISODES = 20
N_ENVS = min(4, os.cpu_count() or 1)
SEED = 42
TRAIN_FREQ = 4
GRADIENT_STEPS = 4
SAVED_MODEL_EVAL_SEED = SEED + 10_000
SAVED_MODEL_EVAL_NAME = f"saved_model_eval_{SAVED_MODEL_EVAL_EPISODES}_episodes"


CONGESTED_ENVIRONMENT_OVERRIDES = {
    "lanes_count": 3,
    "vehicles_count": 30,
    "duration": 40,
    "ego_spacing": 1.8,
    "vehicles_density": 1.2,
    "simulation_frequency": 15,
    "policy_frequency": 3,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "initial_lane_id": None,
    "offroad_terminal": False,
}

CONGESTED_OBSERVATION_CONFIG = {
    "vehicles_count": 12,
    "features": ["presence", "x", "y", "vx", "vy"],
    "absolute": False,
    "see_behind": True,
}

CONGESTED_TTC_OBSERVATION_CONFIG = {
    "enabled": True,
    "feature_name": "front_ttc",
    "ttc_cap": 10.0,
    "lane_y_threshold": 0.50,
    "front_only": True,
    "normalize": True,
}

ACTION_CONFIG = {"type": "DiscreteMetaAction"}
SPEED_CONFIG = {"reward_speed_range": [15.0, 28.0]}

DRIVER_AGGRESSIVENESS_CONFIG = {
    "enabled": True,
    "distribution": "uniform",
    "min_score": 0.0,
    "max_score": 100.0,
    "fixed_score": None,
    "normal_mean": 50.0,
    "normal_std": 20.0,
    "conservative": {
        "target_speed": 18.0,
        "acc_max": 4.0,
        "comfort_acc_max": 2.0,
        "comfort_acc_min": -4.0,
        "delta": 4.5,
        "time_wanted": 2.4,
        "distance_wanted": 14.0,
        "politeness": 0.8,
        "lane_change_min_acc_gain": 0.8,
        "lane_change_max_braking_imposed": 1.0,
        "lane_change_delay": 1.5,
    },
    "aggressive": {
        "target_speed": 30.0,
        "acc_max": 7.0,
        "comfort_acc_max": 5.5,
        "comfort_acc_min": -6.5,
        "delta": 3.5,
        "time_wanted": 0.6,
        "distance_wanted": 6.0,
        "politeness": 0.0,
        "lane_change_min_acc_gain": 0.05,
        "lane_change_max_braking_imposed": 3.5,
        "lane_change_delay": 0.5,
    },
}

DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG = {
    "enabled": True,
    "feature_name": "driver_aggressiveness",
    "normalize": True,
    "ego_value": 0.0,
    "missing_value": 0.0,
}

ADAPTIVE_CONFIG = {
    "enabled": True,
    "mode": "ttc_safety_override",
    "ttc_midpoint": 4.0,
    "ttc_temperature": 0.75,
    "ttc_cap": 10.0,
    "safety_ttc_threshold": 2.5,
    "unsafe_action_penalty": 1.0,
    "min_target_speed": 12.0,
    "max_target_speed": 30.0,
    "faster_max_delta": 6.0,
    "slower_min_delta": 1.0,
    "slower_max_delta": 8.0,
    "cruise_speed": 24.0,
    "action_speed_delta": 2.0,
}

ATTENTION_CONFIG = {
    "features_dim": 128,
    "attention_heads": 2,
    "attention_dropout": 0.0,
    "presence_feature_idx": 0,
    "embedding_arch": "128,128",
    "net_arch": "128,128",
}

HYPERPARAMETERS = {
    "learning_rate": 1e-4,
    "buffer_size": 200_000,
    "learning_starts": 10_000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 2_000,
    "train_freq": TRAIN_FREQ,
    "gradient_steps": GRADIENT_STEPS,
    "exploration_fraction": 0.40,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "progress_every": 5_000,
    "verbose": 0,
}


def clamp_safety_factor(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def make_reward_config(reward_safety_factor: float) -> dict[str, Any]:
    factor = clamp_safety_factor(reward_safety_factor)
    return {
        "collision_reward": -5.0 * (1.0 + factor),
        "right_lane_reward": 0.03,
        "high_speed_reward": 0.25 * (1.0 - 0.40 * factor),
        "lane_change_reward": -0.02 - 0.03 * factor,
        "normalize_reward": False,
    }


def make_safety_ttc_flow_reward_config(reward_safety_factor: float) -> dict[str, Any]:
    factor = clamp_safety_factor(reward_safety_factor)
    if factor <= 0.0:
        return {"enabled": False}
    return {
        "enabled": True,
        "ttc_safe_threshold": 4.0,
        "ttc_target": 6.0,
        "ttc_cap": 10.0,
        "low_ttc_penalty_weight": 2.0 * factor,
        "max_low_ttc_penalty": 2.0 * factor,
        "safe_ttc_bonus_weight": 0.05 * factor,
        "max_safe_ttc_bonus": 0.10 * factor,
        # As safety increases, allow the agent to slow down more without being
        # punished for lagging behind the local flow.
        "lag_penalty_weight": 0.08 * max(0.25, 1.0 - factor),
        "speed_tolerance": 2.0 + 2.0 * factor,
        "max_lag_penalty": 0.5 * max(0.25, 1.0 - factor),
        "rear_ttc_pressure": 5.0,
        "rear_pressure_floor": 0.15,
        "flow_radius": 120.0,
        "lanes": "ego_and_adjacent",
    }


def make_env_config(
    reward_safety_factor: float,
    *,
    adaptive: bool = True,
    aggressiveness_state: bool = False,
    ttc_observation: bool = True,
) -> dict[str, Any]:
    return build_env_config(
        profile_name=ENVIRONMENT_PROFILE,
        profile_overrides=CONGESTED_ENVIRONMENT_OVERRIDES,
        observation=CONGESTED_OBSERVATION_CONFIG,
        action=ACTION_CONFIG,
        reward=make_reward_config(reward_safety_factor),
        speed=SPEED_CONFIG,
        adaptive_longitudinal=ADAPTIVE_CONFIG if adaptive else {"enabled": False},
        rear_flow={"enabled": False},
        traffic_flow_reward={"enabled": False},
        safety_ttc_flow_reward=make_safety_ttc_flow_reward_config(reward_safety_factor),
        driver_aggressiveness=DRIVER_AGGRESSIVENESS_CONFIG,
        driver_aggressiveness_observation=(
            DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG
            if aggressiveness_state
            else {"enabled": False}
        ),
        ttc_observation=CONGESTED_TTC_OBSERVATION_CONFIG if ttc_observation else {"enabled": False},
    )


def factor_tag(reward_safety_factor: float) -> str:
    return f"{int(round(100 * clamp_safety_factor(reward_safety_factor))):03d}"


def run_one_factor(reward_safety_factor: float, *, timesteps: int = TIMESTEPS) -> dict[str, Any]:
    factor = clamp_safety_factor(reward_safety_factor)
    tag = factor_tag(factor)
    run_name = f"attention_adaptive_ttc_safety_factor_{tag}_{timesteps // 1000}k"
    label = f"Attention + Adaptive TTC | safety factor {factor:.2f}"
    trainer, _, _, results_dir, default_device = load_dqn_backend(
        backend_module="attention_dqn",
        notebook_subdir="congested_traffic_policy",
        results_subdir=RESULTS_SUBDIR,
    )
    env_config = make_env_config(factor)
    args = build_dqn_args(
        results_dir=results_dir,
        run_name=run_name,
        timesteps=timesteps,
        eval_episodes=TRAINING_EVAL_EPISODES,
        seed=SEED + 2_000 + int(round(1000 * factor)),
        num_envs=N_ENVS,
        device=default_device,
        hyperparameters=HYPERPARAMETERS,
        extra=ATTENTION_CONFIG,
    )

    print(f"\n=== {label} ===")
    print(json.dumps(
        {
            "run_name": run_name,
            "reward_safety_factor": factor,
            "reward": make_reward_config(factor),
            "safety_ttc_flow_reward": make_safety_ttc_flow_reward_config(factor),
        },
        indent=2,
    ))
    summary = train_and_display(trainer, args, env_config, label=label)
    saved_eval_summary = evaluate_saved_model(
        trainer,
        summary_path=results_dir / run_name / "summary.json",
        env_config=env_config,
        episodes=SAVED_MODEL_EVAL_EPISODES,
        seed=SAVED_MODEL_EVAL_SEED + 2_000 + int(round(1000 * factor)),
        name=SAVED_MODEL_EVAL_NAME,
        label=label,
    )
    output = {
        "run_name": run_name,
        "label": label,
        "reward_safety_factor": factor,
        "summary": summary,
        "saved_eval_summary": saved_eval_summary,
    }
    (results_dir / run_name / "reward_safety_factor.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output


def run_baseline_aggressiveness_state(
    *,
    timesteps: int = TIMESTEPS,
    reward_safety_factor: float = 0.0,
    ttc_observation: bool = False,
) -> dict[str, Any]:
    factor = clamp_safety_factor(reward_safety_factor)
    tag = factor_tag(factor)
    ttc_tag = "_ttc_state" if ttc_observation else ""
    run_name = f"baseline_aggr_state{ttc_tag}_{tag}_{timesteps // 1000}k"
    label = "Baseline DQN + Driver Aggressiveness State"
    trainer, _, _, results_dir, default_device = load_dqn_backend(
        backend_module="elurant_dqn",
        notebook_subdir="congested_traffic_policy",
        results_subdir=RESULTS_SUBDIR,
    )
    env_config = make_env_config(
        factor,
        adaptive=False,
        aggressiveness_state=True,
        ttc_observation=ttc_observation,
    )
    args = build_dqn_args(
        results_dir=results_dir,
        run_name=run_name,
        timesteps=timesteps,
        eval_episodes=TRAINING_EVAL_EPISODES,
        seed=SEED + 1_000 + int(round(1000 * factor)),
        num_envs=N_ENVS,
        device=default_device,
        hyperparameters=HYPERPARAMETERS,
        extra={"disable_tensorboard": True},
    )

    print(f"\n=== {label} ===")
    print(json.dumps(
        {
            "run_name": run_name,
            "reward_safety_factor": factor,
            "reward": make_reward_config(factor),
            "driver_aggressiveness": DRIVER_AGGRESSIVENESS_CONFIG,
            "driver_aggressiveness_observation": DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG,
            "ttc_observation": CONGESTED_TTC_OBSERVATION_CONFIG if ttc_observation else {"enabled": False},
        },
        indent=2,
    ))
    summary = train_and_display(trainer, args, env_config, label=label)
    saved_eval_summary = evaluate_saved_model(
        trainer,
        summary_path=results_dir / run_name / "summary.json",
        env_config=env_config,
        episodes=SAVED_MODEL_EVAL_EPISODES,
        seed=SAVED_MODEL_EVAL_SEED + 1_000 + int(round(1000 * factor)),
        name=SAVED_MODEL_EVAL_NAME,
        label=label,
    )
    output = {
        "run_name": run_name,
        "label": label,
        "backend_module": "elurant_dqn",
        "attention": False,
        "adaptive": False,
        "aggressiveness_state": True,
        "ttc_observation": bool(ttc_observation),
        "reward_safety_factor": factor,
        "summary": summary,
        "saved_eval_summary": saved_eval_summary,
    }
    (results_dir / run_name / "baseline_aggressiveness_state.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output


def build_comparison(outputs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for output in outputs:
        row = {
            "run_name": output["run_name"],
            "label": output["label"],
            "reward_safety_factor": output["reward_safety_factor"],
        }
        for record in output["saved_eval_summary"].get("metric_summary", []):
            metric = (
                record.get("metric", "")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_per_")
                .lower()
            )
            row[f"{metric}_mean"] = record.get("mean")
            row[f"{metric}_std"] = record.get("std")
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    comparison_path = RESULTS_DIR / "reward_safety_factor_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("Comparison saved to:", comparison_path)
    return comparison_df


def run_study(
    factors: Optional[list[float]] = None,
    *,
    timesteps: int = TIMESTEPS,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    selected_factors = list(SAFETY_FACTORS if factors is None else factors)
    if max_runs is not None:
        selected_factors = selected_factors[: int(max_runs)]
    outputs = [run_one_factor(factor, timesteps=timesteps) for factor in selected_factors]
    return build_comparison(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run congested reward-safety-factor study.")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--factors", nargs="*", type=float, default=SAFETY_FACTORS)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--baseline-aggressiveness-state",
        action="store_true",
        help="Run only the plain baseline DQN with mixed driver aggressiveness exposed in the observation.",
    )
    parser.add_argument(
        "--baseline-ttc-observation",
        action="store_true",
        help="Also append the front-TTC feature during the baseline aggressiveness-state run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline_aggressiveness_state:
        output = run_baseline_aggressiveness_state(
            timesteps=args.timesteps,
            ttc_observation=args.baseline_ttc_observation,
        )
        print(json.dumps(output["saved_eval_summary"].get("metric_summary", []), indent=2))
        return
    comparison_df = run_study(args.factors, timesteps=args.timesteps, max_runs=args.max_runs)
    print(comparison_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
