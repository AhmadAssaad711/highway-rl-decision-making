"""
Curriculum-learning study for the unstructured Kourani DQN.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from functools import partial
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_with_project_venv() -> None:
    """
    Re-run with the repo virtualenv when the current interpreter lacks dependencies.
    """

    if os.environ.get("HYP2_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import gymnasium  # noqa: F401
        import highway_env  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        child_env = dict(os.environ)
        child_env["HYP2_SKIP_VENV_REEXEC"] = "1"
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            env=child_env,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from curriculum_env import make_curriculum_env
from scenario_sets import (
    DEFAULT_STAGE_SPLIT,
    DEFAULT_STAGE_WEIGHTS,
    DEFAULT_TOTAL_TIMESTEPS,
    build_scenario_sets_payload,
    filter_named_scenarios,
    get_curriculum_stages,
    get_evaluation_scenarios,
)

DEFAULT_RUNS_ROOT = CURRENT_DIR / "runs"
DEFAULT_PROGRESS_EVERY = 10_000
DEFAULT_EPISODES_PER_SCENARIO = 200
DEFAULT_MAX_STEPS = 300
DEFAULT_POLICY_KWARGS = {"net_arch": [256, 256]}
DEFAULT_VIDEO_FPS = 5.0


class TimestepProgressCallback(BaseCallback):
    def __init__(
        self,
        stage_name: str,
        total_timesteps: int,
        log_dir: Path,
        scenario_names: list[str],
        every_n_steps: int = DEFAULT_PROGRESS_EVERY,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.stage_name = stage_name
        self.total_timesteps = max(1, int(total_timesteps))
        self.log_dir = Path(log_dir)
        self.scenario_names = list(scenario_names)
        self.every_n_steps = max(1, int(every_n_steps))
        self._stage_start_timesteps = 0
        self._next_print = self.every_n_steps
        self._start_time = 0.0
        self._writer: SummaryWriter | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._stage_start_timesteps = int(self.model.num_timesteps)
        self._next_print = self.every_n_steps
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._writer.add_text("stage/name", self.stage_name, 0)
        self._writer.add_text("stage/scenarios", ", ".join(self.scenario_names), 0)

    def _on_step(self) -> bool:
        stage_timesteps = int(self.model.num_timesteps) - self._stage_start_timesteps
        if stage_timesteps >= self._next_print:
            elapsed = time.time() - self._start_time
            progress = min(100.0, 100.0 * stage_timesteps / self.total_timesteps)
            if self._writer is not None:
                self._writer.add_scalar("train/stage_timesteps", stage_timesteps, stage_timesteps)
                self._writer.add_scalar("train/progress_pct", progress, stage_timesteps)
                self._writer.add_scalar("train/elapsed_seconds", elapsed, stage_timesteps)
                self._writer.flush()
            print(
                f"[{self.stage_name}] timesteps={stage_timesteps}/{self.total_timesteps} "
                f"({progress:.1f}%) elapsed={elapsed:.1f}s"
            )
            while self._next_print <= stage_timesteps:
                self._next_print += self.every_n_steps
        return True

    def _on_training_end(self) -> None:
        stage_timesteps = int(self.model.num_timesteps) - self._stage_start_timesteps
        elapsed = time.time() - self._start_time
        if self._writer is not None:
            self._writer.add_scalar("train/stage_timesteps_final", stage_timesteps, stage_timesteps)
            self._writer.add_scalar("train/elapsed_seconds_final", elapsed, stage_timesteps)
            self._writer.flush()
            self._writer.close()
            self._writer = None


def add_train_args(
    parser: argparse.ArgumentParser,
    include_seed: bool = True,
    include_device: bool = True,
) -> None:
    parser.add_argument("--run-name", default=None, help="Optional label appended to the run id.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument(
        "--stage-timesteps",
        type=int,
        nargs=3,
        default=None,
        metavar=("STAGE1", "STAGE2", "STAGE3"),
        help="Optional explicit timesteps for the 3 curriculum stages.",
    )
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--buffer-size", type=int, default=15000)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Defaults to --n-envs when omitted.",
    )
    parser.add_argument("--target-update-interval", type=int, default=50)
    if include_seed:
        parser.add_argument("--seed", type=int, default=42)
    if include_device:
        parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--verbose", type=int, default=1)


def add_eval_args(
    parser: argparse.ArgumentParser,
    require_model_path: bool,
    include_seed: bool = True,
    include_device: bool = True,
) -> None:
    parser.add_argument("--model-path", required=require_model_path, default=None)
    parser.add_argument(
        "--output-run-dir",
        default=None,
        help="Optional explicit run directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--split",
        choices=("seen", "unseen", "all"),
        default="all",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Optional scenario name filter. Can be passed multiple times.",
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=DEFAULT_EPISODES_PER_SCENARIO)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    if include_seed:
        parser.add_argument("--seed", type=int, default=42)
    if include_device:
        parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--record-videos", action="store_true")
    parser.add_argument("--stochastic", action="store_true")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curriculum-learning study for the unstructured Kourani DQN."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run the 3-stage curriculum training.")
    add_train_args(train_parser)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained curriculum checkpoint on seen and/or unseen scenarios.",
    )
    add_eval_args(eval_parser, require_model_path=True)

    full_parser = subparsers.add_parser(
        "full",
        help="Train the curriculum model and then evaluate the final checkpoint.",
    )
    add_train_args(full_parser)
    add_eval_args(
        full_parser,
        require_model_path=False,
        include_seed=False,
        include_device=False,
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("Please choose one of: train, evaluate, full")
    return args


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def make_run_id(run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("r%Y%m%d_%H%M%S")
    if not run_name:
        return timestamp
    short_name = sanitize_name(run_name)[:12]
    return f"{short_name}_{timestamp}"


def to_windows_long_path(path: Path) -> str:
    resolved = path.resolve()
    path_string = str(resolved)
    if os.name != "nt" or path_string.startswith("\\\\?\\"):
        return path_string
    if path_string.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_string[2:]
    return "\\\\?\\" + path_string


def ensure_directory(path: Path) -> Path:
    resolved = path.resolve()
    os.makedirs(to_windows_long_path(resolved), exist_ok=True)
    return resolved


def resolve_stage_timesteps(total_timesteps: int, explicit_stage_timesteps: list[int] | None) -> list[int]:
    if explicit_stage_timesteps is not None:
        values = [int(value) for value in explicit_stage_timesteps]
        if any(value <= 0 for value in values):
            raise ValueError("--stage-timesteps values must all be positive")
        return values

    if total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")

    weight_total = sum(DEFAULT_STAGE_WEIGHTS)
    allocated: list[int] = []
    remaining = int(total_timesteps)
    for index, weight in enumerate(DEFAULT_STAGE_WEIGHTS):
        if index == len(DEFAULT_STAGE_WEIGHTS) - 1:
            value = remaining
        else:
            value = int(total_timesteps * weight / weight_total)
            remaining -= value
        allocated.append(value)

    if allocated == [0, 0, total_timesteps]:
        raise ValueError("Total timesteps are too small for the 3-stage curriculum")
    if any(value <= 0 for value in allocated):
        raise ValueError("Resolved stage timesteps must all be positive")
    return allocated


def build_curriculum_plan(stage_timesteps: list[int]) -> list[dict]:
    stages = get_curriculum_stages()
    if len(stages) != len(stage_timesteps):
        raise ValueError("Expected exactly one timestep value per curriculum stage")

    plan = []
    for index, stage in enumerate(stages):
        stage_copy = dict(stage)
        stage_copy["scenario_names"] = list(stage["scenario_names"])
        stage_copy["scenarios"] = [dict(scenario) for scenario in stage["scenarios"]]
        stage_copy["timesteps"] = int(stage_timesteps[index])
        plan.append(stage_copy)
    return plan


def scenario_sets_json_payload(stage_timesteps: list[int]) -> dict:
    payload = build_scenario_sets_payload(stage_timesteps=stage_timesteps)
    payload["default_total_timesteps"] = DEFAULT_TOTAL_TIMESTEPS
    payload["default_stage_split"] = list(DEFAULT_STAGE_SPLIT)
    return payload


def write_json(path: Path, payload: Any) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def action_lane_change_ids(env) -> set[int]:
    action_indexes = getattr(env.unwrapped.action_type, "actions_indexes", {})
    lane_change_actions = set()
    for action_name in ("LANE_LEFT", "LANE_RIGHT"):
        if action_name in action_indexes:
            lane_change_actions.add(int(action_indexes[action_name]))
    return lane_change_actions


def build_training_env(scenarios: list[dict], split_name: str, render_mode: str = "rgb_array"):
    return make_curriculum_env(
        scenarios=scenarios,
        render_mode=render_mode,
        split_name=split_name,
    )


def create_training_vec_env(
    scenarios: list[dict],
    split_name: str,
    n_envs: int,
    seed: int,
    monitor_dir: Path,
):
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env_factory = partial(
        build_training_env,
        scenarios=scenarios,
        split_name=split_name,
        render_mode="rgb_array",
    )
    return make_vec_env(
        env_id=env_factory,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
        vec_env_cls=vec_env_cls,
        monitor_kwargs={"info_keywords": ("scenario_name", "split")},
    )


def resolve_gradient_steps(n_envs: int, gradient_steps: int | None) -> int:
    return int(n_envs if gradient_steps is None else gradient_steps)


def save_study_config(
    run_root: Path,
    command_name: str,
    args: argparse.Namespace,
    stage_plan: list[dict] | None,
    model_path: Path | None = None,
) -> Path:
    payload = {
        "command": command_name,
        "created_at": datetime.now().isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if stage_plan is not None:
        payload["stage_plan"] = [
            {
                "name": stage["name"],
                "label": stage["label"],
                "description": stage["description"],
                "scenario_names": stage["scenario_names"],
                "timesteps": stage["timesteps"],
            }
            for stage in stage_plan
        ]
        payload["total_timesteps"] = int(sum(stage["timesteps"] for stage in stage_plan))
    if model_path is not None:
        payload["model_path"] = str(model_path)
    return write_json(run_root / "study_config.json", payload)


def record_video_frames(video_path: Path, frames: list[np.ndarray], fps: float = DEFAULT_VIDEO_FPS) -> None:
    if not frames:
        return

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for --record-videos but is not installed."
        ) from exc

    ensure_directory(video_path.parent)
    frame_height, frame_width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_width, frame_height),
    )
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def evaluate_scenario(
    model_path: Path,
    scenario: dict,
    split_name: str,
    args: argparse.Namespace,
    scenario_index: int,
    video_root: Path | None,
) -> dict:
    env = make_curriculum_env(
        scenarios=[scenario],
        render_mode="rgb_array",
        split_name=split_name,
    )
    model = DQN.load(str(model_path), env=env, device=args.device)
    lane_change_ids = action_lane_change_ids(env)
    policy_frequency = float(env.unwrapped.config["policy_frequency"])
    fps = float(env.metadata.get("render_fps", DEFAULT_VIDEO_FPS))

    rewards: list[float] = []
    avg_speeds: list[float] = []
    episode_lengths: list[int] = []
    distances: list[float] = []
    lane_changes: list[int] = []
    crash_flags: list[int] = []

    try:
        for episode_idx in range(args.episodes_per_scenario):
            obs, info = env.reset(seed=args.seed + scenario_index * 1000 + episode_idx)
            terminated = False
            truncated = False
            total_reward = 0.0
            speed_trace: list[float] = []
            distance_m = 0.0
            step_count = 0
            lane_change_count = 0
            last_info = info
            video_frames: list[np.ndarray] = []

            if video_root is not None and episode_idx == 0:
                frame = np.asarray(env.render(), dtype=np.uint8)
                video_frames.append(frame)

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action_value = int(np.asarray(action).item())
                if action_value in lane_change_ids:
                    lane_change_count += 1

                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                step_count += 1

                speed_mps = float(last_info.get("speed", 0.0))
                speed_trace.append(speed_mps)
                distance_m += speed_mps / policy_frequency

                if video_root is not None and episode_idx == 0:
                    frame = np.asarray(env.render(), dtype=np.uint8)
                    video_frames.append(frame)

                if args.max_steps and step_count >= args.max_steps:
                    truncated = True

            if video_root is not None and episode_idx == 0:
                video_path = video_root / split_name / f"{scenario['name']}.mp4"
                record_video_frames(video_path=video_path, frames=video_frames, fps=fps)
                print(f"[video] {split_name}/{scenario['name']} -> {video_path}")

            avg_speed = float(np.mean(speed_trace)) if speed_trace else 0.0
            crashed = int(bool(last_info.get("crashed", False)))

            rewards.append(total_reward)
            avg_speeds.append(avg_speed)
            episode_lengths.append(step_count)
            distances.append(distance_m)
            lane_changes.append(lane_change_count)
            crash_flags.append(crashed)

            if (
                episode_idx == 0
                or (episode_idx + 1) % max(1, args.progress_interval) == 0
                or (episode_idx + 1) == args.episodes_per_scenario
            ):
                print(
                    f"[{split_name}:{scenario['name']}] episode "
                    f"{episode_idx + 1}/{args.episodes_per_scenario} "
                    f"| reward={total_reward:.2f} speed={avg_speed:.2f} "
                    f"| crash={bool(crashed)} steps={step_count}"
                )
    finally:
        env.close()

    return {
        "split": split_name,
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "episodes": args.episodes_per_scenario,
        "mean_reward": float(np.mean(rewards)),
        "collision_pct": float(100.0 * np.mean(crash_flags)),
        "success_pct": float(100.0 * (1.0 - np.mean(crash_flags))),
        "avg_speed_mps": float(np.mean(avg_speeds)),
        "episode_length_steps": float(np.mean(episode_lengths)),
        "avg_distance_m": float(np.mean(distances)),
        "avg_lane_changes": float(np.mean(lane_changes)),
        "model_path": str(model_path),
    }


def summarize_by_split(per_scenario_rows: list[dict]) -> list[dict]:
    rows_by_split: dict[str, list[dict]] = {}
    for row in per_scenario_rows:
        rows_by_split.setdefault(str(row["split"]), []).append(row)

    summary_rows: list[dict] = []
    for split_name, rows in sorted(rows_by_split.items()):
        summary_rows.append(
            {
                "split": split_name,
                "scenario_count": len(rows),
                "episodes_per_scenario": int(rows[0]["episodes"]) if rows else 0,
                "mean_reward": float(np.mean([row["mean_reward"] for row in rows])) if rows else 0.0,
                "collision_pct": float(np.mean([row["collision_pct"] for row in rows])) if rows else 0.0,
                "success_pct": float(np.mean([row["success_pct"] for row in rows])) if rows else 0.0,
                "avg_speed_mps": float(np.mean([row["avg_speed_mps"] for row in rows])) if rows else 0.0,
                "episode_length_steps": float(
                    np.mean([row["episode_length_steps"] for row in rows])
                )
                if rows
                else 0.0,
                "avg_distance_m": float(np.mean([row["avg_distance_m"] for row in rows])) if rows else 0.0,
                "avg_lane_changes": float(np.mean([row["avg_lane_changes"] for row in rows])) if rows else 0.0,
                "model_path": rows[0]["model_path"] if rows else "",
            }
        )
    return summary_rows


def write_evaluation_outputs(run_root: Path, per_scenario_rows: list[dict]) -> None:
    results_dir = ensure_directory(run_root / "results")
    per_scenario_rows = sorted(
        per_scenario_rows,
        key=lambda row: (str(row["split"]), str(row["scenario_name"])),
    )
    split_summary_rows = summarize_by_split(per_scenario_rows)

    write_json(results_dir / "per_scenario_metrics.json", per_scenario_rows)
    write_csv(
        results_dir / "per_scenario_metrics.csv",
        per_scenario_rows,
        fieldnames=[
            "split",
            "scenario_name",
            "description",
            "episodes",
            "mean_reward",
            "collision_pct",
            "success_pct",
            "avg_speed_mps",
            "episode_length_steps",
            "avg_distance_m",
            "avg_lane_changes",
            "model_path",
        ],
    )

    write_json(results_dir / "split_summary.json", split_summary_rows)
    write_csv(
        results_dir / "split_summary.csv",
        split_summary_rows,
        fieldnames=[
            "split",
            "scenario_count",
            "episodes_per_scenario",
            "mean_reward",
            "collision_pct",
            "success_pct",
            "avg_speed_mps",
            "episode_length_steps",
            "avg_distance_m",
            "avg_lane_changes",
            "model_path",
        ],
    )


def infer_evaluation_run_root(model_path: Path, explicit_output_run_dir: str | None) -> Path:
    if explicit_output_run_dir:
        return ensure_directory(Path(explicit_output_run_dir).expanduser().resolve())

    if model_path.parent.name == "models" and model_path.parent.parent.exists():
        return ensure_directory(model_path.parent.parent)

    return ensure_directory(DEFAULT_RUNS_ROOT / make_run_id("evaluation"))


def train_curriculum(args: argparse.Namespace, run_root: Path) -> tuple[Path, list[dict]]:
    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    stage_timesteps = resolve_stage_timesteps(args.total_timesteps, args.stage_timesteps)
    stage_plan = build_curriculum_plan(stage_timesteps)
    models_dir = ensure_directory(run_root / "models")
    tensorboard_dir = ensure_directory(run_root / "tensorboard")
    scenario_sets_path = run_root / "scenario_sets.json"
    write_json(scenario_sets_path, scenario_sets_json_payload(stage_timesteps))

    current_env = None
    model: DQN | None = None

    try:
        for stage_index, stage in enumerate(stage_plan):
            stage_name = stage["name"]
            stage_monitor_dir = ensure_directory(run_root / "monitor" / stage_name)
            stage_seed = int(args.seed + stage_index * 1000)
            reset_num_timesteps = stage_index == 0
            stage_tensorboard_dir = ensure_directory(tensorboard_dir / stage_name)

            print(
                f"\n=== {stage_name} | timesteps={stage['timesteps']} "
                f"| scenarios={len(stage['scenario_names'])} ==="
            )
            print(", ".join(stage["scenario_names"]))

            next_env = create_training_vec_env(
                scenarios=stage["scenarios"],
                split_name=stage_name,
                n_envs=args.n_envs,
                seed=stage_seed,
                monitor_dir=stage_monitor_dir,
            )

            if model is None:
                model = DQN(
                    policy="MlpPolicy",
                    env=next_env,
                    policy_kwargs=DEFAULT_POLICY_KWARGS,
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    train_freq=args.train_freq,
                    gradient_steps=resolve_gradient_steps(args.n_envs, args.gradient_steps),
                    target_update_interval=args.target_update_interval,
                    tensorboard_log=None,
                    seed=args.seed,
                    device=args.device,
                    verbose=args.verbose,
                )
            else:
                previous_env = current_env
                model.set_env(next_env)
                current_env = next_env
                if previous_env is not None:
                    previous_env.close()
            if current_env is None:
                current_env = next_env

            progress_callback = TimestepProgressCallback(
                stage_name=stage_name,
                total_timesteps=stage["timesteps"],
                log_dir=stage_tensorboard_dir,
                scenario_names=stage["scenario_names"],
                every_n_steps=args.progress_every,
            )

            save_study_config(
                run_root=run_root,
                command_name="train",
                args=args,
                stage_plan=stage_plan,
            )

            model.learn(
                total_timesteps=stage["timesteps"],
                callback=progress_callback,
                reset_num_timesteps=reset_num_timesteps,
            )

            stage_model_path = models_dir / stage_name
            model.save(str(stage_model_path))
            print(f"[saved] {stage_model_path}.zip")

        assert model is not None
        final_model_path = models_dir / "curriculum_final"
        model.save(str(final_model_path))
        final_zip_path = final_model_path.with_suffix(".zip")
        save_study_config(
            run_root=run_root,
            command_name="train",
            args=args,
            stage_plan=stage_plan,
            model_path=final_zip_path,
        )
        print(f"[saved] final curriculum model -> {final_zip_path}")
        return final_zip_path, stage_plan
    finally:
        if current_env is not None:
            current_env.close()


def evaluate_curriculum(
    model_path: Path,
    args: argparse.Namespace,
    run_root: Path,
) -> list[dict]:
    scenarios_by_split = get_evaluation_scenarios(split=args.split)
    scenarios_by_split = filter_named_scenarios(scenarios_by_split, args.scenario)
    write_json(run_root / "scenario_sets.json", scenario_sets_json_payload(list(DEFAULT_STAGE_SPLIT)))

    video_root = ensure_directory(run_root / "videos") if args.record_videos else None
    per_scenario_rows: list[dict] = []
    scenario_counter = 0

    for split_name, scenarios in scenarios_by_split.items():
        for scenario in scenarios:
            scenario_counter += 1
            print(
                f"\nEvaluating {scenario['name']} ({split_name}) "
                f"with {model_path.name}"
            )
            per_scenario_rows.append(
                evaluate_scenario(
                    model_path=model_path,
                    scenario=scenario,
                    split_name=split_name,
                    args=args,
                    scenario_index=scenario_counter,
                    video_root=video_root,
                )
            )

    write_evaluation_outputs(run_root=run_root, per_scenario_rows=per_scenario_rows)
    save_study_config(
        run_root=run_root,
        command_name="evaluate",
        args=args,
        stage_plan=None,
        model_path=model_path,
    )
    return per_scenario_rows


def resolve_model_path(model_path_value: str | None) -> Path:
    if not model_path_value:
        raise ValueError("--model-path is required for evaluate")
    model_path = Path(model_path_value).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    return model_path


def print_evaluation_summary(per_scenario_rows: list[dict]) -> None:
    print("\nEvaluation complete.")
    for row in per_scenario_rows:
        print(
            f"{row['split']}:{row['scenario_name']} "
            f"| reward={row['mean_reward']:.3f} "
            f"| collisions={row['collision_pct']:.1f}% "
            f"| success={row['success_pct']:.1f}% "
            f"| speed={row['avg_speed_mps']:.2f} m/s"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "train":
        run_root = ensure_directory(DEFAULT_RUNS_ROOT / make_run_id(args.run_name))
        final_model_path, _ = train_curriculum(args=args, run_root=run_root)
        print(f"\nTraining complete. Run folder: {run_root}")
        print(f"Final model: {final_model_path}")
        return

    if args.command == "evaluate":
        model_path = resolve_model_path(args.model_path)
        run_root = infer_evaluation_run_root(model_path, args.output_run_dir)
        per_scenario_rows = evaluate_curriculum(
            model_path=model_path,
            args=args,
            run_root=run_root,
        )
        print(f"\nEvaluation artifacts written to: {run_root}")
        print_evaluation_summary(per_scenario_rows)
        return

    if args.command == "full":
        run_root = ensure_directory(DEFAULT_RUNS_ROOT / make_run_id(args.run_name))
        final_model_path, stage_plan = train_curriculum(args=args, run_root=run_root)
        save_study_config(
            run_root=run_root,
            command_name="full",
            args=args,
            stage_plan=stage_plan,
            model_path=final_model_path,
        )
        per_scenario_rows = evaluate_curriculum(
            model_path=final_model_path,
            args=args,
            run_root=run_root,
        )
        print(f"\nFull study complete. Run folder: {run_root}")
        print(f"Final model: {final_model_path}")
        print_evaluation_summary(per_scenario_rows)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
