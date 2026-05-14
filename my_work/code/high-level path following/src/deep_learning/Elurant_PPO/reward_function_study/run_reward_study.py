from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_with_project_venv() -> None:
    """
    If the script is launched with a Python interpreter that does not have the
    study dependencies, transparently re-run it with the repo's virtualenv.
    """
    if os.environ.get("PPO_REWARD_STUDY_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from study_env import DEFAULT_REWARD_CONFIG, HighwayRewardStudyEnv, make_study_env


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = PROJECT_ROOT / "artifacts" / "ppo" / "reward_function_study"
MODELS_DIR = RUN_ROOT / "models"
LOGS_DIR = RUN_ROOT / "logs"
TB_DIR = RUN_ROOT / "tensorboard"
VIDEOS_DIR = RUN_ROOT / "videos"
SUMMARY_DIR = RUN_ROOT / "summary"
SUMMARY_CSV = SUMMARY_DIR / "reward_grid_summary.csv"

COLLISION_VALUES = (-0.75, -1.0, -1.5)
HIGH_SPEED_VALUES = (0.25, 0.4, 0.6)
RIGHT_LANE_VALUES = (0.0, 0.05, 0.1)
EXTRA_BASELINE_SEEDS = (142, 143, 144)


@dataclass(frozen=True)
class TrialSpec:
    trial_index: int
    run_name: str
    group: str
    seed: int
    collision_reward: float
    high_speed_reward: float
    right_lane_reward: float

    def reward_config(self) -> dict[str, Any]:
        config = dict(DEFAULT_REWARD_CONFIG)
        config.update(
            {
                "collision_reward": self.collision_reward,
                "offroad_penalty": self.collision_reward,
                "high_speed_reward": self.high_speed_reward,
                "right_lane_reward": self.right_lane_reward,
            }
        )
        return config


class TimestepProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, every_n_steps: int = 10000, verbose: int = 0):
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


class EpisodeMetricTensorboardCallback(BaseCallback):
    def __init__(self, window_size: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = max(1, int(window_size))
        self.metric_windows: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if dones is None:
            return True

        updated = False
        for done, info in zip(dones, infos):
            if not done:
                continue
            metrics = info.get("episode_metrics")
            if not metrics:
                continue
            updated = True
            for key, value in metrics.items():
                self.metric_windows[key].append(float(value))

        if updated:
            for key, values in self.metric_windows.items():
                self.logger.record(f"study/{key}", float(np.mean(values)))
        return True


def format_float_tag(value: float) -> str:
    sign = "m" if value < 0 else ""
    body = f"{int(round(abs(value) * 100)):03d}"
    return f"{sign}{body}"


def build_default_trials(base_seed: int = 42) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    trial_index = 1

    for collision_reward, high_speed_reward, right_lane_reward in itertools.product(
        COLLISION_VALUES,
        HIGH_SPEED_VALUES,
        RIGHT_LANE_VALUES,
    ):
        run_name = (
            f"t{trial_index:02d}_cr{format_float_tag(collision_reward)}"
            f"_hs{format_float_tag(high_speed_reward)}"
            f"_rl{format_float_tag(right_lane_reward)}"
        )
        trials.append(
            TrialSpec(
                trial_index=trial_index,
                run_name=run_name,
                group="grid",
                seed=base_seed + trial_index - 1,
                collision_reward=collision_reward,
                high_speed_reward=high_speed_reward,
                right_lane_reward=right_lane_reward,
            )
        )
        trial_index += 1

    for repeat_index, seed in enumerate(EXTRA_BASELINE_SEEDS, start=1):
        run_name = f"t{trial_index:02d}_base_r{repeat_index:02d}"
        trials.append(
            TrialSpec(
                trial_index=trial_index,
                run_name=run_name,
                group="baseline_repeat",
                seed=seed,
                collision_reward=-1.0,
                high_speed_reward=0.4,
                right_lane_reward=0.1,
            )
        )
        trial_index += 1

    return trials


def save_trial_manifest(run_dir: Path, trial: TrialSpec) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "trial_config.json"
    payload = asdict(trial)
    payload["reward_config"] = trial.reward_config()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_trial_manifest(manifest_path: Path) -> TrialSpec:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return TrialSpec(
        trial_index=int(payload["trial_index"]),
        run_name=str(payload["run_name"]),
        group=str(payload["group"]),
        seed=int(payload["seed"]),
        collision_reward=float(payload["collision_reward"]),
        high_speed_reward=float(payload["high_speed_reward"]),
        right_lane_reward=float(payload["right_lane_reward"]),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential PPO sweep for the native highway-v0 reward."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the reward-function study runs")
    train_parser.add_argument("--timesteps", type=int, default=300000)
    train_parser.add_argument("--n-envs", type=int, default=24)
    train_parser.add_argument("--eval-freq", type=int, default=10000)
    train_parser.add_argument("--eval-episodes", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--n-steps", type=int, default=4096)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--progress-every", type=int, default=10000)
    train_parser.add_argument("--metric-window", type=int, default=50)
    train_parser.add_argument("--start-index", type=int, default=1)
    train_parser.add_argument("--max-runs", type=int, default=30)
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument(
        "--auto-video",
        action="store_true",
        help="Record one evaluation episode immediately after each trained run.",
    )
    train_parser.add_argument("--video-seed-offset", type=int, default=5000)

    video_parser = subparsers.add_parser(
        "video",
        help="Record one evaluation episode for every saved model in the study folder.",
    )
    video_parser.add_argument("--seed-offset", type=int, default=5000)
    video_parser.add_argument("--overwrite", action="store_true")

    return parser


def normalize_cli_args(argv: list[str] | None = None) -> list[str]:
    """
    Allow the script to be run directly without an explicit subcommand.

    Examples:
      python run_reward_study.py
      python run_reward_study.py --timesteps 100000
      python run_reward_study.py video
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["train"]

    known_commands = {"train", "video"}
    if args[0] not in known_commands:
        return ["train", *args]
    return args


def make_eval_env(trial: TrialSpec) -> Monitor:
    return Monitor(make_study_env(render_mode="rgb_array", reward_config=trial.reward_config()))


def evaluate_trained_model(
    model_path: Path,
    trial: TrialSpec,
    n_eval_episodes: int,
) -> dict[str, float]:
    env = make_study_env(render_mode="rgb_array", reward_config=trial.reward_config())
    model = PPO.load(str(model_path), env=env)

    episode_rewards: list[float] = []
    episode_metrics: list[dict[str, float]] = []

    for episode_idx in range(n_eval_episodes):
        obs, info = env.reset(seed=trial.seed + 1000 + episode_idx)
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        metrics = dict(info.get("episode_metrics", {}))
        metrics["episode_reward"] = float(total_reward)
        episode_metrics.append(metrics)
        episode_rewards.append(float(total_reward))

    env.close()

    summary: dict[str, float] = {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
    }
    if episode_metrics:
        metric_names = sorted(episode_metrics[0].keys())
        for metric_name in metric_names:
            values = [float(metrics[metric_name]) for metrics in episode_metrics]
            summary[f"eval_{metric_name}"] = float(np.mean(values))

    return summary


def record_trial_video(
    model_path: Path,
    trial: TrialSpec,
    seed_offset: int,
    overwrite: bool = False,
) -> str:
    run_video_dir = VIDEOS_DIR / trial.run_name
    run_video_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = sorted(run_video_dir.glob("*.mp4"))
    if existing_videos and not overwrite:
        return str(existing_videos[0])

    env = make_study_env(render_mode="rgb_array", reward_config=trial.reward_config())
    video_env = RecordVideo(
        env,
        video_folder=str(run_video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=trial.run_name,
    )
    model = PPO.load(str(model_path), env=video_env)

    obs, info = video_env.reset(seed=trial.seed + seed_offset)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)

    video_env.close()
    generated_videos = sorted(run_video_dir.glob("*.mp4"))
    return str(generated_videos[0]) if generated_videos else ""


def write_run_evaluation(run_dir: Path, payload: dict[str, Any]) -> Path:
    output_path = run_dir / "evaluation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def refresh_summary_csv() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for manifest_path in sorted(MODELS_DIR.glob("*/trial_config.json")):
        run_dir = manifest_path.parent
        evaluation_path = run_dir / "evaluation.json"
        if not evaluation_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        row = {
            "trial_index": manifest["trial_index"],
            "run_name": manifest["run_name"],
            "group": manifest["group"],
            "seed": manifest["seed"],
            "collision_reward": manifest["collision_reward"],
            "high_speed_reward": manifest["high_speed_reward"],
            "right_lane_reward": manifest["right_lane_reward"],
            "model_path": evaluation.get("model_path", ""),
            "video_path": evaluation.get("video_path", ""),
        }
        for key, value in evaluation.items():
            if key in {"model_path", "video_path"}:
                continue
            row[key] = value
        rows.append(row)

    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_trial(args: argparse.Namespace, trial: TrialSpec) -> None:
    run_dir = MODELS_DIR / trial.run_name
    log_dir = LOGS_DIR / trial.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    save_trial_manifest(run_dir, trial)

    model_path = run_dir / "ppo_highway"
    model_zip_path = model_path.with_suffix(".zip")
    eval_json_path = run_dir / "evaluation.json"

    if model_zip_path.exists() and not args.overwrite:
        print(f"[reuse] {trial.run_name} already has a trained model. Refreshing outputs only.")
        evaluation = evaluate_trained_model(
            model_path=model_zip_path,
            trial=trial,
            n_eval_episodes=args.eval_episodes,
        )
        evaluation["model_path"] = str(model_zip_path)
        if args.auto_video:
            video_path = record_trial_video(
                model_path=model_zip_path,
                trial=trial,
                seed_offset=args.video_seed_offset,
                overwrite=False,
            )
            evaluation["video_path"] = video_path
            print(f"[video] {trial.run_name} -> {video_path}")
        write_run_evaluation(run_dir, evaluation)
        refresh_summary_csv()
        return

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        HighwayRewardStudyEnv,
        n_envs=args.n_envs,
        seed=trial.seed,
        monitor_dir=str(log_dir),
        env_kwargs={"reward_config": trial.reward_config()},
        vec_env_cls=vec_env_cls,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        device=args.device,
        seed=trial.seed,
    )

    eval_env = make_eval_env(trial)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        deterministic=True,
        render=False,
    )
    progress_callback = TimestepProgressCallback(
        total_timesteps=args.timesteps,
        every_n_steps=args.progress_every,
    )
    metric_callback = EpisodeMetricTensorboardCallback(window_size=args.metric_window)

    print(
        f"\n=== Starting {trial.run_name} | "
        f"collision={trial.collision_reward} "
        f"high_speed={trial.high_speed_reward} "
        f"right_lane={trial.right_lane_reward} "
        f"seed={trial.seed} ===\n"
    )
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, progress_callback, metric_callback],
        tb_log_name=trial.run_name,
    )
    elapsed = time.time() - start_time
    model.save(str(model_path))
    eval_env.close()
    env.close()
    del model

    evaluation = evaluate_trained_model(
        model_path=model_zip_path,
        trial=trial,
        n_eval_episodes=args.eval_episodes,
    )
    evaluation["elapsed_seconds"] = float(elapsed)
    evaluation["model_path"] = str(model_zip_path)

    if args.auto_video:
        video_path = record_trial_video(
            model_path=model_zip_path,
            trial=trial,
            seed_offset=args.video_seed_offset,
            overwrite=args.overwrite,
        )
        evaluation["video_path"] = video_path
        print(f"[video] {trial.run_name} -> {video_path}")

    write_run_evaluation(run_dir, evaluation)
    refresh_summary_csv()

    print(
        f"[done] {trial.run_name} saved to {model_zip_path} "
        f"in {elapsed:.2f}s"
    )


def run_train_command(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_trials = build_default_trials(base_seed=args.seed)
    selected_trials = [
        trial
        for trial in all_trials
        if args.start_index <= trial.trial_index < args.start_index + args.max_runs
    ]

    print(f"TensorBoard logdir: {TB_DIR}")
    print(f"Running {len(selected_trials)} study runs sequentially...")
    for trial in selected_trials:
        train_trial(args, trial)

    refresh_summary_csv()
    print(f"Summary CSV written to {SUMMARY_CSV}")


def run_video_command(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = sorted(MODELS_DIR.glob("*/trial_config.json"))
    if not manifests:
        raise FileNotFoundError(f"No trained run manifests found under {MODELS_DIR}")

    for manifest_path in manifests:
        trial = load_trial_manifest(manifest_path)
        run_dir = manifest_path.parent
        model_zip_path = run_dir / "ppo_highway.zip"
        if not model_zip_path.exists():
            print(f"[skip] Missing model for {trial.run_name}")
            continue

        video_path = record_trial_video(
            model_path=model_zip_path,
            trial=trial,
            seed_offset=args.seed_offset,
            overwrite=args.overwrite,
        )
        evaluation_path = run_dir / "evaluation.json"
        if evaluation_path.exists():
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        else:
            payload = {"model_path": str(model_zip_path)}
        payload["video_path"] = video_path
        write_run_evaluation(run_dir, payload)
        print(f"[video] {trial.run_name} -> {video_path}")

    refresh_summary_csv()
    print(f"Summary CSV written to {SUMMARY_CSV}")


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(normalize_cli_args(argv))

    if args.command == "train":
        run_train_command(args)
    elif args.command == "video":
        run_video_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
