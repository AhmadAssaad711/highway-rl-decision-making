from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


SETUP_TEMPLATE = """
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate the project root.")


PROJECT_ROOT = find_project_root()
HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
helper_dir_str = str(HELPER_DIR)
if helper_dir_str not in sys.path:
    sys.path.insert(0, helper_dir_str)

from dqn_notebook_utils import (
    build_dqn_args,
    build_env_config,
    evaluate_saved_model,
    load_dqn_backend,
    show_policy_panel,
    train_and_display,
)

trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = load_dqn_backend(
    backend_module="[[BACKEND_MODULE]]",
    notebook_subdir="[[NOTEBOOK_SUBDIR]]",
    results_subdir="[[RESULTS_SUBDIR]]",
)
"""


COMMON_TRAIN_CELL = """
summary = train_and_display(
    trainer,
    args,
    training_env_config,
    label="[[LABEL]]",
)
"""


COMMON_EVAL_CELL = """
saved_eval_summary = evaluate_saved_model(
    trainer,
    summary_path=RESULTS_DIR / run_name / "summary.json",
    env_config=saved_model_eval_env_config,
    episodes=saved_model_eval_episodes,
    seed=saved_model_eval_seed,
    name=saved_model_eval_name,
    label="[[LABEL]]",
)
"""


COMMON_VISUAL_CELL = """
policy_panel_rows = show_policy_panel(
    trainer,
    summary_path=RESULTS_DIR / run_name / "summary.json",
    env_config=saved_model_eval_env_config,
    episodes=visualization_episodes,
    max_steps=visualization_max_steps,
    seed=visualization_seed,
    stochastic=visualization_stochastic,
)
"""


ADAPTIVE_BOOTSTRAP_CELL = """
if "configure_adaptive_experiment" not in globals():
    import os
    import sys
    from pathlib import Path

    import pandas as pd

    if "build_env_config" not in globals() or "RESULTS_DIR" not in globals():
        def find_project_root() -> Path:
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "src").exists() and (candidate / ".git").exists():
                    return candidate
            raise RuntimeError("Could not locate the project root.")

        PROJECT_ROOT = find_project_root()
        HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
        helper_dir_str = str(HELPER_DIR)
        if helper_dir_str not in sys.path:
            sys.path.insert(0, helper_dir_str)

        from dqn_notebook_utils import (
            build_dqn_args,
            build_env_config,
            evaluate_saved_model,
            load_dqn_backend,
            train_and_display,
        )

        trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = load_dqn_backend(
            backend_module="elurant_dqn",
            notebook_subdir="adaptive_lower_controller",
            results_subdir="adaptive_lower_controller",
        )

    environment_profile = "structured_baseline"
    eval_environment_profile = environment_profile

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
    # Evaluation environment: choose a profile independently from training.
    # Leave overrides empty unless you intentionally want to override that
    # eval profile's defaults.
    eval_environment_overrides = {}

    observation_config = {
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    }
    action_config = {
        "type": "DiscreteMetaAction",
    }
    reward_config = {
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0.0,
        "normalize_reward": True,
    }
    min_reward_speed = 20.0
    max_reward_speed = 30.0
    speed_config = {
        "reward_speed_range": [min_reward_speed, max_reward_speed],
    }

    timesteps = 20_000
    n_envs = min(4, os.cpu_count() or 1)
    eval_episodes = 5
    seed = 42
    train_freq = 4
    gradient_steps = train_freq * n_envs

    base_adaptive_longitudinal_config = {
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
    rear_flow_config = {
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
    traffic_flow_reward_config = {
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
    hyperparameters = {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 64,
        "gamma": 0.95,
        "target_update_interval": 1_000,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": 0.70,
        "exploration_final_eps": 0.10,
        "progress_every": 1_000,
        "verbose": 0,
    }

    saved_model_eval_episodes = 1000

    def configure_adaptive_experiment(
        *,
        run_name: str,
        adaptive_overrides: dict | None = None,
        rear_flow: dict | None = None,
        traffic_flow_reward: dict | None = None,
        ttc_observation: dict | None = None,
        seed_offset: int = 0,
    ) -> dict:
        adaptive_config = {
            **base_adaptive_longitudinal_config,
            **dict(adaptive_overrides or {}),
        }
        training_env_config = build_env_config(
            profile_name=environment_profile,
            profile_overrides=environment_overrides,
            observation=observation_config,
            action=action_config,
            reward=reward_config,
            speed=speed_config,
            adaptive_longitudinal=adaptive_config,
            rear_flow=rear_flow,
            traffic_flow_reward=traffic_flow_reward,
            ttc_observation=ttc_observation,
        )
        saved_model_eval_env_config = build_env_config(
            profile_name=eval_environment_profile,
            profile_overrides=eval_environment_overrides,
            observation=observation_config,
            action=action_config,
            reward=reward_config,
            speed=speed_config,
            adaptive_longitudinal=adaptive_config,
            rear_flow=rear_flow,
            traffic_flow_reward=traffic_flow_reward,
            ttc_observation=ttc_observation,
        )
        args = build_dqn_args(
            results_dir=RESULTS_DIR,
            run_name=run_name,
            timesteps=timesteps,
            eval_episodes=eval_episodes,
            seed=seed + seed_offset,
            num_envs=n_envs,
            device=DEFAULT_DEVICE,
            hyperparameters=hyperparameters,
        )
        return {
            "run_name": run_name,
            "args": args,
            "training_env_config": training_env_config,
            "saved_model_eval_env_config": saved_model_eval_env_config,
            "saved_model_eval_seed": seed + 10000 + seed_offset,
            "saved_model_eval_name": f"saved_model_eval_{saved_model_eval_episodes}_episodes",
        }

    def run_adaptive_experiment(experiment: dict, label: str) -> dict:
        display(pd.DataFrame({
            "training": pd.Series(experiment["training_env_config"]),
            "saved_eval": pd.Series(experiment["saved_model_eval_env_config"]),
        }))
        summary = train_and_display(
            trainer,
            experiment["args"],
            experiment["training_env_config"],
            label=label,
        )
        saved_eval_summary = evaluate_saved_model(
            trainer,
            summary_path=RESULTS_DIR / experiment["run_name"] / "summary.json",
            env_config=experiment["saved_model_eval_env_config"],
            episodes=saved_model_eval_episodes,
            seed=experiment["saved_model_eval_seed"],
            name=experiment["saved_model_eval_name"],
            label=label,
        )
        return {"summary": summary, "saved_eval_summary": saved_eval_summary}
"""


def render(template: str, **values: str) -> str:
    text = dedent(template).strip("\n")
    for key, value in values.items():
        text = text.replace(f"[[{key}]]", value)
    return text


def join_code_blocks(*blocks: str) -> str:
    return "\n\n".join(render(block) for block in blocks if block.strip())


def baseline_cells() -> list[dict]:
    config = """
    if "build_env_config" not in globals() or "RESULTS_DIR" not in globals():
        from pathlib import Path
        import sys

        def find_project_root() -> Path:
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "src").exists() and (candidate / ".git").exists():
                    return candidate
            raise RuntimeError("Could not locate the project root.")

        PROJECT_ROOT = find_project_root()
        HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
        helper_dir_str = str(HELPER_DIR)
        if helper_dir_str not in sys.path:
            sys.path.insert(0, helper_dir_str)

        from dqn_notebook_utils import build_dqn_args, build_env_config, load_dqn_backend

        trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = load_dqn_backend(
            backend_module="elurant_dqn",
            notebook_subdir="baseline_dqn",
            results_subdir="baseline_dqn",
        )

    environment_profile = "structured_baseline"
    eval_environment_profile = environment_profile

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
    # Evaluation environment: choose a profile independently from training.
    # Leave overrides empty unless you intentionally want to override that
    # eval profile's defaults.
    eval_environment_overrides = {}

    observation_config = {
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    }
    action_config = {
        "type": "DiscreteMetaAction",
    }
    reward_config = {
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0.0,
        "normalize_reward": True,
    }
    min_reward_speed = 20.0
    max_reward_speed = 30.0
    speed_config = {
        "reward_speed_range": [min_reward_speed, max_reward_speed],
    }
    target_speed_range = {
        "min_target_speed": 10.0,
        "max_target_speed": 35.0,
    }
    adaptive_longitudinal_config = {
        "enabled": False,
        "ttc_midpoint": 4.0,
        "ttc_temperature": 1.0,
        "ttc_cap": 10.0,
        **target_speed_range,
        "faster_max_delta": 1.25,
        "slower_min_delta": 1.25,
        "slower_max_delta": 2.5,
    }

    timesteps = 20_000
    n_envs = 4
    eval_episodes = 5
    seed = 42
    run_name = "baseline_dqn_tuned_20k"
    train_freq = 4
    gradient_steps = train_freq * n_envs

    hyperparameters = {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 64,
        "gamma": 0.95,
        "target_update_interval": 1_000,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": 0.70,
        "exploration_final_eps": 0.10,
        "progress_every": 1_000,
        "verbose": 0,
    }

    training_env_config = build_env_config(
        profile_name=environment_profile,
        profile_overrides=environment_overrides,
        observation=observation_config,
        action=action_config,
        reward=reward_config,
        speed=speed_config,
        adaptive_longitudinal=adaptive_longitudinal_config,
    )
    saved_model_eval_env_config = build_env_config(
        profile_name=eval_environment_profile,
        profile_overrides=eval_environment_overrides,
        observation=observation_config,
        action=action_config,
        reward=reward_config,
        speed=speed_config,
        adaptive_longitudinal=adaptive_longitudinal_config,
    )
    args = build_dqn_args(
        results_dir=RESULTS_DIR,
        run_name=run_name,
        timesteps=timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        num_envs=n_envs,
        device=DEFAULT_DEVICE,
        hyperparameters=hyperparameters,
    )

    saved_model_eval_episodes = 1000
    saved_model_eval_seed = seed + 10000
    saved_model_eval_name = f"saved_model_eval_{saved_model_eval_episodes}_episodes"
    visualization_episodes = 5
    visualization_max_steps = 300
    visualization_seed = seed + 20000
    visualization_stochastic = False

    display(pd.DataFrame({"training": pd.Series(training_env_config), "saved_eval": pd.Series(saved_model_eval_env_config)}))
    """
    congestion_diagnostics = """
    import json
    from pathlib import Path

    from stable_baselines3 import DQN

    from congestion_diagnostics import evaluate_congestion_diagnostics, save_congestion_diagnostics

    congestion_diagnostic_episodes = 100
    congestion_diagnostic_seed = seed + 50_000
    congestion_diagnostic_config = {
        "ttc_cap": 10.0,
        "front_ttc_safe": 4.0,
        "front_ttc_critical": 1.5,
        "rear_ttc_safe": 4.0,
        "rear_ttc_critical": 1.5,
        "lane_gap_safe": 12.0,
        "bad_action_margin": 0.35,
        "no_good_action_risk": 0.85,
        "wrong_lane_quality_margin": 0.18,
        "wrong_lane_lookback_steps": 6,
        "final_lookback_steps": 4,
    }

    congestion_summary_path = RESULTS_DIR / run_name / "summary.json"
    if not congestion_summary_path.exists():
        raise RuntimeError("Run the baseline training cell once so a saved model exists.")

    congestion_saved_summary = json.loads(congestion_summary_path.read_text(encoding="utf-8"))
    congestion_model = DQN.load(congestion_saved_summary["model_path"], device=DEFAULT_DEVICE)
    congestion_output_dir = RESULTS_DIR / run_name / f"congestion_diagnostics_{congestion_diagnostic_episodes}_episodes"

    congestion_df, congestion_traces = evaluate_congestion_diagnostics(
        congestion_model,
        trainer.make_env,
        env_config=saved_model_eval_env_config,
        episodes=congestion_diagnostic_episodes,
        seed=congestion_diagnostic_seed,
        diagnostic_config=congestion_diagnostic_config,
    )
    congestion_paths = save_congestion_diagnostics(congestion_df, congestion_traces, congestion_output_dir)

    label_columns = [
        "collision",
        "agent_chose_badly",
        "no_good_discrete_action",
        "wrong_lane_earlier",
        "unavoidable_rear_pressure",
    ]
    label_rates = (
        congestion_df[label_columns]
        .astype(float)
        .mean()
        .mul(100.0)
        .rename("rate_percent")
        .reset_index()
        .rename(columns={"index": "label"})
    )
    collision_breakdown = (
        congestion_df["collision_type"]
        .value_counts(dropna=False)
        .rename_axis("collision_type")
        .reset_index(name="episodes")
    )

    print(json.dumps(congestion_paths, indent=2))
    display(label_rates.round(2))
    display(collision_breakdown)
    display(congestion_df.head(20))
    """
    return [
        markdown_cell("# Baseline DQN\n\nThin runner for the Leurent/rl-agents-style DQN baseline."),
        code_cell(render(SETUP_TEMPLATE, BACKEND_MODULE="elurant_dqn", NOTEBOOK_SUBDIR="baseline_dqn", RESULTS_SUBDIR="baseline_dqn")),
        markdown_cell("## Config"),
        code_cell(config),
        markdown_cell("## Train"),
        code_cell(render(COMMON_TRAIN_CELL, LABEL="Baseline DQN")),
        markdown_cell("## Saved-Model Evaluation"),
        code_cell(render(COMMON_EVAL_CELL, LABEL="Baseline DQN")),
        markdown_cell(
            "## Congestion Failure Diagnostics\n\n"
            "Label each evaluation episode with congestion failure modes: bad action, no good discrete action, "
            "wrong lane earlier, and unavoidable rear pressure. The per-step traces are saved so individual "
            "failures can be inspected afterward."
        ),
        code_cell(congestion_diagnostics),
        markdown_cell("## Policy Panel"),
        code_cell(COMMON_VISUAL_CELL),
    ]


def adaptive_cells() -> list[dict]:
    config = """
    import os
    import sys
    from pathlib import Path

    if "build_env_config" not in globals() or "RESULTS_DIR" not in globals():
        def find_project_root() -> Path:
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "src").exists() and (candidate / ".git").exists():
                    return candidate
            raise RuntimeError("Could not locate the project root.")

        PROJECT_ROOT = find_project_root()
        HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
        helper_dir_str = str(HELPER_DIR)
        if helper_dir_str not in sys.path:
            sys.path.insert(0, helper_dir_str)

        from dqn_notebook_utils import (
            build_dqn_args,
            build_env_config,
            evaluate_saved_model,
            load_dqn_backend,
            train_and_display,
        )

        trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = load_dqn_backend(
            backend_module="elurant_dqn",
            notebook_subdir="adaptive_lower_controller",
            results_subdir="adaptive_lower_controller",
        )

    environment_profile = "structured_baseline"
    eval_environment_profile = environment_profile

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
    # Evaluation environment: choose a profile independently from training.
    # Leave overrides empty unless you intentionally want to override that
    # eval profile's defaults.
    eval_environment_overrides = {}

    observation_config = {
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    }
    action_config = {
        "type": "DiscreteMetaAction",
    }
    reward_config = {
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0.0,
        "normalize_reward": True,
    }
    min_reward_speed = 20.0
    max_reward_speed = 30.0
    speed_config = {
        "reward_speed_range": [min_reward_speed, max_reward_speed],
    }

    timesteps = 20_000
    n_envs = min(4, os.cpu_count() or 1)
    eval_episodes = 5
    seed = 42
    train_freq = 4
    gradient_steps = train_freq * n_envs

    base_adaptive_longitudinal_config = {
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
    rear_flow_config = {
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
    traffic_flow_reward_config = {
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
    hyperparameters = {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 64,
        "gamma": 0.95,
        "target_update_interval": 1_000,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": 0.70,
        "exploration_final_eps": 0.10,
        "progress_every": 1_000,
        "verbose": 0,
    }

    saved_model_eval_episodes = 1000
    render_trial_episodes = 5
    render_trial_seed = seed + 30_000
    render_trial_env_config = build_env_config(
        profile_name=environment_profile,
        profile_overrides=environment_overrides,
        observation=observation_config,
        action=action_config,
        reward=reward_config,
        speed=speed_config,
        adaptive_longitudinal={"enabled": False},
        rear_flow=rear_flow_config,
        traffic_flow_reward={"enabled": False},
    )

    def configure_adaptive_experiment(
        *,
        run_name: str,
        adaptive_overrides: dict | None = None,
        rear_flow: dict | None = None,
        traffic_flow_reward: dict | None = None,
        ttc_observation: dict | None = None,
        seed_offset: int = 0,
    ) -> dict:
        adaptive_config = {
            **base_adaptive_longitudinal_config,
            **dict(adaptive_overrides or {}),
        }
        training_env_config = build_env_config(
            profile_name=environment_profile,
            profile_overrides=environment_overrides,
            observation=observation_config,
            action=action_config,
            reward=reward_config,
            speed=speed_config,
            adaptive_longitudinal=adaptive_config,
            rear_flow=rear_flow,
            traffic_flow_reward=traffic_flow_reward,
            ttc_observation=ttc_observation,
        )
        saved_model_eval_env_config = build_env_config(
            profile_name=eval_environment_profile,
            profile_overrides=eval_environment_overrides,
            observation=observation_config,
            action=action_config,
            reward=reward_config,
            speed=speed_config,
            adaptive_longitudinal=adaptive_config,
            rear_flow=rear_flow,
            traffic_flow_reward=traffic_flow_reward,
            ttc_observation=ttc_observation,
        )
        args = build_dqn_args(
            results_dir=RESULTS_DIR,
            run_name=run_name,
            timesteps=timesteps,
            eval_episodes=eval_episodes,
            seed=seed + seed_offset,
            num_envs=n_envs,
            device=DEFAULT_DEVICE,
            hyperparameters=hyperparameters,
        )
        return {
            "run_name": run_name,
            "args": args,
            "training_env_config": training_env_config,
            "saved_model_eval_env_config": saved_model_eval_env_config,
            "saved_model_eval_seed": seed + 10000 + seed_offset,
            "saved_model_eval_name": f"saved_model_eval_{saved_model_eval_episodes}_episodes",
        }

    def run_adaptive_experiment(experiment: dict, label: str) -> dict:
        display(pd.DataFrame({
            "training": pd.Series(experiment["training_env_config"]),
            "saved_eval": pd.Series(experiment["saved_model_eval_env_config"]),
        }))
        summary = train_and_display(
            trainer,
            experiment["args"],
            experiment["training_env_config"],
            label=label,
        )
        saved_eval_summary = evaluate_saved_model(
            trainer,
            summary_path=RESULTS_DIR / experiment["run_name"] / "summary.json",
            env_config=experiment["saved_model_eval_env_config"],
            episodes=saved_model_eval_episodes,
            seed=experiment["saved_model_eval_seed"],
            name=experiment["saved_model_eval_name"],
            label=label,
        )
        return {"summary": summary, "saved_eval_summary": saved_eval_summary}
    """

    adaptive_congestion_diagnostics = """
    import json
    from pathlib import Path

    from stable_baselines3 import DQN

    from congestion_diagnostics import evaluate_congestion_diagnostics, save_congestion_diagnostics

    adaptive_diagnostic_episodes = 100
    adaptive_diagnostic_seed = seed + 60_000
    adaptive_diagnostic_config = {
        "ttc_cap": 10.0,
        "front_ttc_safe": 4.0,
        "front_ttc_critical": 1.5,
        "rear_ttc_safe": 4.0,
        "rear_ttc_critical": 1.5,
        "lane_gap_safe": 12.0,
        "bad_action_margin": 0.35,
        "no_good_action_risk": 0.85,
        "wrong_lane_quality_margin": 0.18,
        "wrong_lane_lookback_steps": 6,
        "final_lookback_steps": 4,
    }

    adaptive_diagnostic_run_names = [
        "adaptive_rear_flow_env_20k",
        "adaptive_rear_flow_reward_20k",
        "adaptive_safe_speed_controller_20k",
        "adaptive_wide_band_delta_20k",
        "adaptive_ttc_observation_20k",
        "adaptive_ttc_safety_override_20k",
    ]

    adaptive_diagnostic_outputs = []
    adaptive_diagnostic_frames = []
    for diagnostic_run_name in adaptive_diagnostic_run_names:
        summary_path = RESULTS_DIR / diagnostic_run_name / "summary.json"
        if not summary_path.exists():
            print(f"Skipping missing run: {summary_path}")
            continue

        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diagnostic_model = DQN.load(saved_summary["model_path"], device=DEFAULT_DEVICE)
        diagnostic_env_config = saved_summary["env_config"]
        output_dir = RESULTS_DIR / diagnostic_run_name / f"congestion_diagnostics_{adaptive_diagnostic_episodes}_episodes"

        diagnostic_df, diagnostic_traces = evaluate_congestion_diagnostics(
            diagnostic_model,
            trainer.make_env,
            env_config=diagnostic_env_config,
            episodes=adaptive_diagnostic_episodes,
            seed=adaptive_diagnostic_seed,
            diagnostic_config=adaptive_diagnostic_config,
        )
        diagnostic_paths = save_congestion_diagnostics(diagnostic_df, diagnostic_traces, output_dir)
        diagnostic_df.insert(0, "run_name", diagnostic_run_name)
        adaptive_diagnostic_frames.append(diagnostic_df)
        adaptive_diagnostic_outputs.append({"run_name": diagnostic_run_name, **diagnostic_paths})

    if not adaptive_diagnostic_frames:
        raise RuntimeError("No saved adaptive lower-controller runs were found for congestion diagnostics.")

    adaptive_congestion_df = pd.concat(adaptive_diagnostic_frames, ignore_index=True)
    label_columns = [
        "collision",
        "agent_chose_badly",
        "no_good_discrete_action",
        "wrong_lane_earlier",
        "unavoidable_rear_pressure",
    ]
    adaptive_label_rates = (
        adaptive_congestion_df.groupby("run_name")[label_columns]
        .mean()
        .mul(100.0)
        .round(2)
        .reset_index()
    )
    adaptive_collision_breakdown = (
        adaptive_congestion_df.groupby(["run_name", "collision_type"])
        .size()
        .reset_index(name="episodes")
    )

    print(json.dumps(adaptive_diagnostic_outputs, indent=2))
    display(adaptive_label_rates)
    display(adaptive_collision_breakdown)
    display(adaptive_congestion_df.head(20))
    """
    return [
        markdown_cell("# Adaptive Lower Controller\n\nThree focused adaptive DQN experiments: rear-flow pressure, traffic-conformity reward, and a safe-speed controller formulation."),
        code_cell(render(SETUP_TEMPLATE, BACKEND_MODULE="elurant_dqn", NOTEBOOK_SUBDIR="adaptive_lower_controller", RESULTS_SUBDIR="adaptive_lower_controller")),
        markdown_cell("## Common Config"),
        code_cell(config),
        markdown_cell("## RENDER TRIAL"),
        code_cell(
            """
            import json
            import sys
            from pathlib import Path

            import pandas as pd

            from highway_env.vehicle.behavior import IDMVehicle


            if "build_env_config" not in globals() or "RESULTS_DIR" not in globals():
                def find_project_root() -> Path:
                    for candidate in [Path.cwd(), *Path.cwd().parents]:
                        if (candidate / "src").exists() and (candidate / ".git").exists():
                            return candidate
                    raise RuntimeError("Could not locate the project root.")

                PROJECT_ROOT = find_project_root()
                HELPER_DIR = PROJECT_ROOT / "notebooks" / "_shared"
                helper_dir_str = str(HELPER_DIR)
                if helper_dir_str not in sys.path:
                    sys.path.insert(0, helper_dir_str)

                from dqn_notebook_utils import build_env_config, load_dqn_backend

                trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = load_dqn_backend(
                    backend_module="elurant_dqn",
                    notebook_subdir="adaptive_lower_controller",
                    results_subdir="adaptive_lower_controller",
                )


            def handoff_ego_to_idm(env) -> None:
                ego_vehicle = env.unwrapped.vehicle
                if ego_vehicle is None:
                    raise RuntimeError("No ego vehicle found to convert into IDM behavior.")
                npc_ego = IDMVehicle.create_from(ego_vehicle)
                npc_ego.randomize_behavior()

                road_vehicles = env.unwrapped.road.vehicles
                for idx, vehicle in enumerate(road_vehicles):
                    if vehicle is ego_vehicle:
                        road_vehicles[idx] = npc_ego
                        break
                env.unwrapped.controlled_vehicles = [npc_ego]
                env.unwrapped.vehicle = npc_ego
                env.unwrapped.action_type.controlled_vehicle = npc_ego
                env.unwrapped.observation_type.observer_vehicle = npc_ego


            if "render_trial_env_config" not in globals():
                render_trial_episodes = 5
                render_trial_seed = 42 + 30_000
                render_trial_env_config = build_env_config(
                    profile_name="structured_baseline",
                    profile_overrides={
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
                    },
                    observation={
                        "vehicles_count": 5,
                        "features": ["presence", "x", "y", "vx", "vy"],
                        "absolute": False,
                    },
                    action={"type": "DiscreteMetaAction"},
                    reward={
                        "collision_reward": -1.0,
                        "right_lane_reward": 0.1,
                        "high_speed_reward": 0.4,
                        "lane_change_reward": 0.0,
                        "normalize_reward": True,
                    },
                    speed={"reward_speed_range": [20.0, 30.0]},
                    adaptive_longitudinal={"enabled": False},
                    rear_flow={
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
                    },
                    traffic_flow_reward={"enabled": False},
                )

            print("Render trial env config:")
            print(json.dumps(render_trial_env_config, indent=2))

            render_env = trainer.make_env(render_mode="human", config=render_trial_env_config)
            idle_action = render_env.unwrapped.action_type.actions_indexes["IDLE"]
            render_trial_rows = []

            try:
                for episode in range(render_trial_episodes):
                    obs, info = render_env.reset(seed=render_trial_seed + episode)
                    handoff_ego_to_idm(render_env)
                    render_env.render()

                    done = False
                    truncated = False
                    step_count = 0
                    rear_spawn_total = int(info.get("rear_flow_spawned_count", 0))
                    final_info = dict(info)

                    while not (done or truncated):
                        obs, reward, done, truncated, info = render_env.step(idle_action)
                        render_env.render()
                        step_count += 1
                        rear_spawn_total += int(info.get("rear_flow_spawned_count", 0))
                        final_info = dict(info)

                    row = {
                        "episode": episode + 1,
                        "steps": step_count,
                        "collision": bool(final_info.get("crashed", getattr(render_env.unwrapped.vehicle, "crashed", False))),
                        "rear_flow_spawned_total": rear_spawn_total,
                        "final_speed": float(getattr(render_env.unwrapped.vehicle, "speed", 0.0)),
                    }
                    render_trial_rows.append(row)
                    print(row)
            finally:
                render_env.close()

            display(pd.DataFrame(render_trial_rows))
            """
        ),
        markdown_cell("## Experiment 1: Rear Flow Environment"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                rear_flow_only = configure_adaptive_experiment(
                run_name="adaptive_rear_flow_env_20k",
                adaptive_overrides={"mode": "delta"},
                rear_flow=rear_flow_config,
                traffic_flow_reward={"enabled": False},
                seed_offset=0,
                )
                rear_flow_only_results = run_adaptive_experiment(
                rear_flow_only,
                label="Adaptive DQN - Rear Flow Env",
                )
                """,
            )
        ),
        markdown_cell("## Experiment 2: Rear Flow + Traffic Reward"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                rear_flow_reward = configure_adaptive_experiment(
                run_name="adaptive_rear_flow_reward_20k",
                adaptive_overrides={"mode": "delta"},
                rear_flow=rear_flow_config,
                traffic_flow_reward=traffic_flow_reward_config,
                seed_offset=100,
                )
                rear_flow_reward_results = run_adaptive_experiment(
                rear_flow_reward,
                label="Adaptive DQN - Rear Flow + Reward",
                )
                """,
            )
        ),
        markdown_cell("## Experiment 3: Safe-Speed Controller"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                safe_speed_controller = configure_adaptive_experiment(
                run_name="adaptive_safe_speed_controller_20k",
                adaptive_overrides={
                    "mode": "safe_speed_limiter",
                    "min_target_speed": 18.0,
                    "cruise_speed": 28.0,
                    "action_speed_delta": 3.0,
                },
                rear_flow=rear_flow_config,
                traffic_flow_reward=traffic_flow_reward_config,
                seed_offset=200,
                )
                safe_speed_controller_results = run_adaptive_experiment(
                safe_speed_controller,
                label="Adaptive DQN - Safe-Speed Controller",
                )
                """,
            )
        ),
        markdown_cell("## Experiment 4: Wide-Band Delta Controller"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                # Wider-band controller: keep continuous target-speed control, but restore the
                # highway-env-sized action bandwidth. FASTER can add up to +10 m/s and SLOWER
                # can range from 0 to -10 m/s, with TTC controlling how much of that command is used.
                wide_band_adaptive_overrides = {
                    "mode": "delta",
                    "min_target_speed": 20.0,
                    "max_target_speed": 30.0,
                    "faster_max_delta": 10.0,
                    "slower_min_delta": 0.0,
                    "slower_max_delta": 10.0,
                    "ttc_temperature": 0.75,
                }

                wide_band_delta_controller = configure_adaptive_experiment(
                    run_name="adaptive_wide_band_delta_20k",
                    adaptive_overrides=wide_band_adaptive_overrides,
                    rear_flow=rear_flow_config,
                    traffic_flow_reward=traffic_flow_reward_config,
                    seed_offset=300,
                )
                wide_band_delta_results = run_adaptive_experiment(
                    wide_band_delta_controller,
                    label="Adaptive DQN - Wide-Band Delta Controller",
                )
                """,
            )
        ),
        markdown_cell("## Experiment 5: TTC-Augmented State"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                # Refresh helper/backend definitions so this cell works after earlier stale notebook cells.
                import importlib

                import dqn_notebook_utils as dqn_utils

                dqn_utils = importlib.reload(dqn_utils)
                build_dqn_args = dqn_utils.build_dqn_args
                build_env_config = dqn_utils.build_env_config
                evaluate_saved_model = dqn_utils.evaluate_saved_model
                train_and_display = dqn_utils.train_and_display
                trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = dqn_utils.load_dqn_backend(
                    backend_module="elurant_dqn",
                    notebook_subdir="adaptive_lower_controller",
                    results_subdir="adaptive_lower_controller",
                )

                _required_adaptive_names = [
                    "environment_profile",
                    "eval_environment_profile",
                    "environment_overrides",
                    "eval_environment_overrides",
                    "observation_config",
                    "action_config",
                    "reward_config",
                    "speed_config",
                    "base_adaptive_longitudinal_config",
                    "hyperparameters",
                    "timesteps",
                    "eval_episodes",
                    "saved_model_eval_episodes",
                    "n_envs",
                    "seed",
                ]
                _missing_adaptive_names = [name for name in _required_adaptive_names if name not in globals()]
                if _missing_adaptive_names:
                    raise RuntimeError(f"Run the Common Config cell first. Missing: {_missing_adaptive_names}")


                def configure_adaptive_experiment(
                    *,
                    run_name: str,
                    adaptive_overrides: dict | None = None,
                    rear_flow: dict | None = None,
                    traffic_flow_reward: dict | None = None,
                    ttc_observation: dict | None = None,
                    seed_offset: int = 0,
                ) -> dict:
                    adaptive_config = {
                        **base_adaptive_longitudinal_config,
                        **dict(adaptive_overrides or {}),
                    }
                    training_env_config = build_env_config(
                        profile_name=environment_profile,
                        profile_overrides=environment_overrides,
                        observation=observation_config,
                        action=action_config,
                        reward=reward_config,
                        speed=speed_config,
                        adaptive_longitudinal=adaptive_config,
                        rear_flow=rear_flow,
                        traffic_flow_reward=traffic_flow_reward,
                        ttc_observation=ttc_observation,
                    )
                    saved_model_eval_env_config = build_env_config(
                        profile_name=eval_environment_profile,
                        profile_overrides=eval_environment_overrides,
                        observation=observation_config,
                        action=action_config,
                        reward=reward_config,
                        speed=speed_config,
                        adaptive_longitudinal=adaptive_config,
                        rear_flow=rear_flow,
                        traffic_flow_reward=traffic_flow_reward,
                        ttc_observation=ttc_observation,
                    )
                    args = build_dqn_args(
                        results_dir=RESULTS_DIR,
                        run_name=run_name,
                        timesteps=timesteps,
                        eval_episodes=eval_episodes,
                        seed=seed + seed_offset,
                        num_envs=n_envs,
                        device=DEFAULT_DEVICE,
                        hyperparameters=hyperparameters,
                    )
                    return {
                        "run_name": run_name,
                        "args": args,
                        "training_env_config": training_env_config,
                        "saved_model_eval_env_config": saved_model_eval_env_config,
                        "saved_model_eval_seed": seed + 10000 + seed_offset,
                        "saved_model_eval_name": f"saved_model_eval_{saved_model_eval_episodes}_episodes",
                    }

                # Add TTC as an explicit continuous feature in the kinematic observation matrix.
                ttc_observation_config = {
                    "enabled": True,
                    "feature_name": "ttc",
                    "ttc_cap": 10.0,
                    "lane_y_threshold": 0.35,
                    "front_only": True,
                    "normalize": True,
                }

                ttc_observation_adaptive_overrides = globals().get(
                    "wide_band_adaptive_overrides",
                    {
                        "mode": "delta",
                        "min_target_speed": 20.0,
                        "max_target_speed": 30.0,
                        "faster_max_delta": 10.0,
                        "slower_min_delta": 0.0,
                        "slower_max_delta": 10.0,
                        "ttc_temperature": 0.75,
                    },
                )

                ttc_observation_controller = configure_adaptive_experiment(
                    run_name="adaptive_ttc_observation_20k",
                    adaptive_overrides=ttc_observation_adaptive_overrides,
                    rear_flow=rear_flow_config,
                    traffic_flow_reward=traffic_flow_reward_config,
                    ttc_observation=ttc_observation_config,
                    seed_offset=400,
                )
                ttc_observation_results = run_adaptive_experiment(
                    ttc_observation_controller,
                    label="Adaptive DQN - Wide-Band + TTC Observation",
                )
                """,
            )
        ),
        markdown_cell("## Experiment 6: TTC Safety Override Controller"),
        code_cell(
            join_code_blocks(
                ADAPTIVE_BOOTSTRAP_CELL,
                """
                # Refresh helper/backend definitions so this cell works after earlier stale notebook cells.
                import importlib

                import dqn_notebook_utils as dqn_utils

                dqn_utils = importlib.reload(dqn_utils)
                build_dqn_args = dqn_utils.build_dqn_args
                build_env_config = dqn_utils.build_env_config
                evaluate_saved_model = dqn_utils.evaluate_saved_model
                train_and_display = dqn_utils.train_and_display
                trainer, PROJECT_ROOT, NOTEBOOK_DIR, RESULTS_DIR, DEFAULT_DEVICE = dqn_utils.load_dqn_backend(
                    backend_module="elurant_dqn",
                    notebook_subdir="adaptive_lower_controller",
                    results_subdir="adaptive_lower_controller",
                )

                _required_adaptive_names = [
                    "environment_profile",
                    "eval_environment_profile",
                    "environment_overrides",
                    "eval_environment_overrides",
                    "observation_config",
                    "action_config",
                    "reward_config",
                    "speed_config",
                    "base_adaptive_longitudinal_config",
                    "hyperparameters",
                    "timesteps",
                    "eval_episodes",
                    "saved_model_eval_episodes",
                    "n_envs",
                    "seed",
                ]
                _missing_adaptive_names = [name for name in _required_adaptive_names if name not in globals()]
                if _missing_adaptive_names:
                    raise RuntimeError(f"Run the Common Config cell first. Missing: {_missing_adaptive_names}")


                def configure_adaptive_experiment(
                    *,
                    run_name: str,
                    adaptive_overrides: dict | None = None,
                    rear_flow: dict | None = None,
                    traffic_flow_reward: dict | None = None,
                    ttc_observation: dict | None = None,
                    seed_offset: int = 0,
                ) -> dict:
                    adaptive_config = {
                        **base_adaptive_longitudinal_config,
                        **dict(adaptive_overrides or {}),
                    }
                    training_env_config = build_env_config(
                        profile_name=environment_profile,
                        profile_overrides=environment_overrides,
                        observation=observation_config,
                        action=action_config,
                        reward=reward_config,
                        speed=speed_config,
                        adaptive_longitudinal=adaptive_config,
                        rear_flow=rear_flow,
                        traffic_flow_reward=traffic_flow_reward,
                        ttc_observation=ttc_observation,
                    )
                    saved_model_eval_env_config = build_env_config(
                        profile_name=eval_environment_profile,
                        profile_overrides=eval_environment_overrides,
                        observation=observation_config,
                        action=action_config,
                        reward=reward_config,
                        speed=speed_config,
                        adaptive_longitudinal=adaptive_config,
                        rear_flow=rear_flow,
                        traffic_flow_reward=traffic_flow_reward,
                        ttc_observation=ttc_observation,
                    )
                    args = build_dqn_args(
                        results_dir=RESULTS_DIR,
                        run_name=run_name,
                        timesteps=timesteps,
                        eval_episodes=eval_episodes,
                        seed=seed + seed_offset,
                        num_envs=n_envs,
                        device=DEFAULT_DEVICE,
                        hyperparameters=hyperparameters,
                    )
                    return {
                        "run_name": run_name,
                        "args": args,
                        "training_env_config": training_env_config,
                        "saved_model_eval_env_config": saved_model_eval_env_config,
                        "saved_model_eval_seed": seed + 10000 + seed_offset,
                        "saved_model_eval_name": f"saved_model_eval_{saved_model_eval_episodes}_episodes",
                    }

                # Safety override: the lower controller computes target speed from TTC directly.
                # High-level FASTER/SLOWER magnitude is ignored; unsafe FASTER requests below 1s TTC get an extra penalty.
                ttc_safety_override_config = {
                    "mode": "ttc_safety_override",
                    "min_target_speed": 20.0,
                    "max_target_speed": 30.0,
                    "ttc_midpoint": 4.0,
                    "ttc_temperature": 0.75,
                    "ttc_cap": 10.0,
                    "safety_ttc_threshold": 1.0,
                    "unsafe_action_penalty": 1.0,
                }

                ttc_safety_override_controller = configure_adaptive_experiment(
                    run_name="adaptive_ttc_safety_override_20k",
                    adaptive_overrides=ttc_safety_override_config,
                    rear_flow=rear_flow_config,
                    traffic_flow_reward=traffic_flow_reward_config,
                    seed_offset=500,
                )
                ttc_safety_override_results = run_adaptive_experiment(
                    ttc_safety_override_controller,
                    label="Adaptive DQN - TTC Safety Override Controller",
                )
                """,
            )
        ),
        markdown_cell(
            "## Congestion Failure Diagnostics\n\n"
            "Apply the congestion taxonomy to saved adaptive lower-controller experiments and compare the "
            "failure modes across controller formulations."
        ),
        code_cell(adaptive_congestion_diagnostics),
    ]


def attention_cells() -> list[dict]:
    config = """
    import os

    environment_profile = "structured_baseline"
    eval_environment_profile = environment_profile

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
    # Evaluation environment: choose a profile independently from training.
    # Leave overrides empty unless you intentionally want to override that
    # eval profile's defaults.
    eval_environment_overrides = {}

    observation_config = {
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    }
    action_config = {
        "type": "DiscreteMetaAction",
    }
    reward_config = {
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0.0,
        "normalize_reward": True,
    }
    min_reward_speed = 20.0
    max_reward_speed = 30.0
    speed_config = {
        "reward_speed_range": [min_reward_speed, max_reward_speed],
    }

    timesteps = 20_000
    n_envs = min(4, os.cpu_count() or 1)
    eval_episodes = 5
    seed = 42
    run_name = "attention_dqn_tuned_20k"
    train_freq = 4
    gradient_steps = train_freq * n_envs

    hyperparameters = {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 64,
        "gamma": 0.95,
        "target_update_interval": 1_000,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": 0.70,
        "exploration_final_eps": 0.10,
        "progress_every": 1_000,
        "verbose": 0,
    }
    attention_config = {
        "features_dim": 64,
        "attention_heads": 2,
        "attention_dropout": 0.0,
        "presence_feature_idx": 0,
        "embedding_arch": "64,64",
        "net_arch": "64,64",
    }

    training_env_config = build_env_config(
        profile_name=environment_profile,
        profile_overrides=environment_overrides,
        observation=observation_config,
        action=action_config,
        reward=reward_config,
        speed=speed_config,
    )
    saved_model_eval_env_config = build_env_config(
        profile_name=eval_environment_profile,
        profile_overrides=eval_environment_overrides,
        observation=observation_config,
        action=action_config,
        reward=reward_config,
        speed=speed_config,
    )
    args = build_dqn_args(
        results_dir=RESULTS_DIR,
        run_name=run_name,
        timesteps=timesteps,
        eval_episodes=eval_episodes,
        seed=seed,
        num_envs=n_envs,
        device=DEFAULT_DEVICE,
        hyperparameters=hyperparameters,
        extra=attention_config,
    )

    saved_model_eval_episodes = 1000
    saved_model_eval_seed = seed + 10000
    saved_model_eval_name = f"saved_model_eval_{saved_model_eval_episodes}_episodes"
    visualization_episodes = 5
    visualization_max_steps = 300
    visualization_seed = seed + 20000
    visualization_stochastic = False

    display(pd.DataFrame({"training": pd.Series(training_env_config), "saved_eval": pd.Series(saved_model_eval_env_config)}))
    """

    attention_congestion_diagnostics = """
    import json
    from pathlib import Path

    from stable_baselines3 import DQN

    from congestion_diagnostics import evaluate_congestion_diagnostics, save_congestion_diagnostics

    attention_diagnostic_episodes = 100
    attention_diagnostic_seed = seed + 50_000
    attention_diagnostic_config = {
        "ttc_cap": 10.0,
        "front_ttc_safe": 4.0,
        "front_ttc_critical": 1.5,
        "rear_ttc_safe": 4.0,
        "rear_ttc_critical": 1.5,
        "lane_gap_safe": 12.0,
        "bad_action_margin": 0.35,
        "no_good_action_risk": 0.85,
        "wrong_lane_quality_margin": 0.18,
        "wrong_lane_lookback_steps": 6,
        "final_lookback_steps": 4,
    }

    attention_diagnostic_run_names = [run_name]
    for candidate_run_name in [globals().get("aggressive_safety_run_name"), "attention_dqn_aggressive_ttc_safety_20k"]:
        if candidate_run_name and candidate_run_name not in attention_diagnostic_run_names:
            attention_diagnostic_run_names.append(candidate_run_name)

    attention_diagnostic_outputs = []
    attention_diagnostic_frames = []
    for diagnostic_run_name in attention_diagnostic_run_names:
        summary_path = RESULTS_DIR / diagnostic_run_name / "summary.json"
        if not summary_path.exists():
            print(f"Skipping missing run: {summary_path}")
            continue

        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diagnostic_model = DQN.load(saved_summary["model_path"], device=DEFAULT_DEVICE)
        diagnostic_env_config = saved_summary["env_config"]
        output_dir = RESULTS_DIR / diagnostic_run_name / f"congestion_diagnostics_{attention_diagnostic_episodes}_episodes"

        diagnostic_df, diagnostic_traces = evaluate_congestion_diagnostics(
            diagnostic_model,
            trainer.make_env,
            env_config=diagnostic_env_config,
            episodes=attention_diagnostic_episodes,
            seed=attention_diagnostic_seed,
            diagnostic_config=attention_diagnostic_config,
        )
        diagnostic_paths = save_congestion_diagnostics(diagnostic_df, diagnostic_traces, output_dir)
        diagnostic_df.insert(0, "run_name", diagnostic_run_name)
        attention_diagnostic_frames.append(diagnostic_df)
        attention_diagnostic_outputs.append({"run_name": diagnostic_run_name, **diagnostic_paths})

    if not attention_diagnostic_frames:
        raise RuntimeError("No saved Attention DQN runs were found for congestion diagnostics.")

    attention_congestion_df = pd.concat(attention_diagnostic_frames, ignore_index=True)
    label_columns = [
        "collision",
        "agent_chose_badly",
        "no_good_discrete_action",
        "wrong_lane_earlier",
        "unavoidable_rear_pressure",
    ]
    attention_label_rates = (
        attention_congestion_df.groupby("run_name")[label_columns]
        .mean()
        .mul(100.0)
        .round(2)
        .reset_index()
    )
    attention_collision_breakdown = (
        attention_congestion_df.groupby(["run_name", "collision_type"])
        .size()
        .reset_index(name="episodes")
    )

    print(json.dumps(attention_diagnostic_outputs, indent=2))
    display(attention_label_rates)
    display(attention_collision_breakdown)
    display(attention_congestion_df.head(20))
    """
    return [
        markdown_cell("# Attention DQN\n\nDQN runner using the ego-attention feature extractor."),
        code_cell(render(SETUP_TEMPLATE, BACKEND_MODULE="attention_dqn", NOTEBOOK_SUBDIR="attention_dqn", RESULTS_SUBDIR="attention_dqn")),
        markdown_cell("## Config"),
        code_cell(config),
        markdown_cell("## Train"),
        code_cell(render(COMMON_TRAIN_CELL, LABEL="Attention DQN")),
        markdown_cell("## Saved-Model Evaluation"),
        code_cell(render(COMMON_EVAL_CELL, LABEL="Attention DQN")),

        markdown_cell(
            "## Congestion Failure Diagnostics\n\n"
            "Run the same congestion taxonomy on saved Attention DQN policies and save both per-episode "
            "labels and per-step traces."
        ),
        code_cell(attention_congestion_diagnostics),
                markdown_cell("## Aggressive TTC Safety Reward"),
        code_cell(
            """
            # Retrain Attention DQN with a TTC safety term tuned to preserve aggressive flow.
            aggressive_safety_observation_config = {
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "see_behind": True,
            }
            aggressive_rear_flow_config = {
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
            aggressive_ttc_safety_reward_config = {
                "enabled": True,
                "ttc_safe_threshold": 3.0,
                "ttc_target": 4.5,
                "ttc_cap": 10.0,
                "low_ttc_penalty_weight": 0.45,
                "max_low_ttc_penalty": 0.7,
                "safe_ttc_bonus_weight": 0.03,
                "max_safe_ttc_bonus": 0.05,
                "lag_penalty_weight": 0.25,
                "speed_tolerance": 1.0,
                "max_lag_penalty": 1.0,
                "rear_ttc_pressure": 5.0,
                "rear_pressure_floor": 0.35,
                "flow_radius": 120.0,
                "lanes": "ego_and_adjacent",
            }

            aggressive_safety_run_name = "attention_dqn_aggressive_ttc_safety_20k"
            aggressive_safety_training_env_config = build_env_config(
                profile_name=environment_profile,
                profile_overrides=environment_overrides,
                observation=aggressive_safety_observation_config,
                action=action_config,
                reward=reward_config,
                speed=speed_config,
                rear_flow=aggressive_rear_flow_config,
                traffic_flow_reward={"enabled": False},
                safety_ttc_flow_reward=aggressive_ttc_safety_reward_config,
            )
            aggressive_safety_eval_env_config = build_env_config(
                profile_name=eval_environment_profile,
                profile_overrides=eval_environment_overrides,
                observation=aggressive_safety_observation_config,
                action=action_config,
                reward=reward_config,
                speed=speed_config,
                rear_flow=aggressive_rear_flow_config,
                traffic_flow_reward={"enabled": False},
                safety_ttc_flow_reward=aggressive_ttc_safety_reward_config,
            )
            aggressive_safety_args = build_dqn_args(
                results_dir=RESULTS_DIR,
                run_name=aggressive_safety_run_name,
                timesteps=timesteps,
                eval_episodes=eval_episodes,
                seed=seed + 400,
                num_envs=n_envs,
                device=DEFAULT_DEVICE,
                hyperparameters=hyperparameters,
                extra=attention_config,
            )

            display(pd.DataFrame({
                "training": pd.Series(aggressive_safety_training_env_config),
                "saved_eval": pd.Series(aggressive_safety_eval_env_config),
            }))

            aggressive_safety_summary = train_and_display(
                trainer,
                aggressive_safety_args,
                aggressive_safety_training_env_config,
                label="Attention DQN + Aggressive TTC Safety Reward",
            )
            aggressive_safety_saved_eval_summary = evaluate_saved_model(
                trainer,
                summary_path=RESULTS_DIR / aggressive_safety_run_name / "summary.json",
                env_config=aggressive_safety_eval_env_config,
                episodes=saved_model_eval_episodes,
                seed=saved_model_eval_seed + 400,
                name=saved_model_eval_name,
                label="Attention DQN + Aggressive TTC Safety Reward",
            )
            """
        ),
        markdown_cell("## Policy Panel"),
        code_cell(COMMON_VISUAL_CELL),
    ]


def main() -> None:
    targets = [
        (NOTEBOOKS_DIR / "baseline_dqn" / "baseline_dqn.ipynb", baseline_cells()),
        (NOTEBOOKS_DIR / "adaptive_lower_controller" / "adaptive_lower_controller.ipynb", adaptive_cells()),
        (NOTEBOOKS_DIR / "attention_dqn" / "attention_dqn.ipynb", attention_cells()),
    ]
    for path, cells in targets:
        write_notebook(path, cells)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
