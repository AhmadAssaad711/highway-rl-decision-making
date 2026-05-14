from __future__ import annotations

import argparse
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
    if os.environ.get("PPO_OFFROAD_PENALTY_SKIP_VENV_REEXEC") == "1":
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from study_env import DEFAULT_REWARD_CONFIG, DiffRewardStudyEnv


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = PROJECT_ROOT / "artifacts" / "ppo" / "diff_reward_structure"
MODEL_DIR = RUN_ROOT / "m"
LOG_DIR = RUN_ROOT / "l"
TB_DIR = RUN_ROOT / "t"
SUMMARY_PATH = RUN_ROOT / "summary.json"

TIMESTEPS = 200_000
N_ENVS = 24
LEARNING_RATE = 1e-4
N_STEPS = 4096
BATCH_SIZE = 128
GAMMA = 0.99
CLIP_RANGE = 0.2
ENT_COEF = 0.01
DEVICE = "auto"
METRIC_WINDOW = 50
PROGRESS_EVERY = 10_000
DEFAULT_SEED = 42
RUN_NAME = "base"

BASELINE_REWARD_CONFIG: dict[str, Any] = {
    "collision_reward": -1.0,
    "offroad_penalty": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True,
    "offroad_terminal": True,
}


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    seed: int
    description: str
    reward_config: dict[str, Any]


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
                self.logger.record(f"study/{key}", float(sum(values) / len(values)))
        return True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PPO on the baseline reward plus an off-road penalty."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_run_spec(seed: int) -> RunSpec:
    reward_config = dict(DEFAULT_REWARD_CONFIG)
    reward_config.update(BASELINE_REWARD_CONFIG)
    return RunSpec(
        run_name=RUN_NAME,
        seed=seed,
        description=(
            "Baseline highway reward with normalized off-road terminal penalty "
            "equal to collision reward."
        ),
        reward_config=reward_config,
    )


def save_manifest(spec: RunSpec) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "description": spec.description,
        "reward_config": spec.reward_config,
        "training_config": {
            "timesteps": TIMESTEPS,
            "n_envs": N_ENVS,
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "device": DEVICE,
            "metric_window": METRIC_WINDOW,
            "progress_every": PROGRESS_EVERY,
        },
    }
    manifest_path = MODEL_DIR / "trial_config.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def write_summary(
    spec: RunSpec,
    model_zip_path: Path,
    elapsed_seconds: float,
    tensorboard_logdir: Path,
) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "description": spec.description,
        "reward_config": spec.reward_config,
        "training_config": {
            "timesteps": TIMESTEPS,
            "n_envs": N_ENVS,
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "device": DEVICE,
            "metric_window": METRIC_WINDOW,
            "progress_every": PROGRESS_EVERY,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "model_path": str(model_zip_path),
        "tensorboard_logdir": str(tensorboard_logdir),
        "monitor_dir": str(LOG_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(spec: RunSpec, overwrite: bool) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    save_manifest(spec)

    model_path = MODEL_DIR / "ppo_highway"
    model_zip_path = model_path.with_suffix(".zip")

    if model_zip_path.exists() and not overwrite:
        print(f"[reuse] Existing model found at {model_zip_path}")
        return

    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    env = make_vec_env(
        DiffRewardStudyEnv,
        n_envs=N_ENVS,
        seed=spec.seed,
        env_kwargs={"reward_config": spec.reward_config},
        monitor_dir=str(LOG_DIR),
        vec_env_cls=vec_env_cls,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        tensorboard_log=str(TB_DIR),
        device=DEVICE,
        seed=spec.seed,
    )

    callbacks = [
        TimestepProgressCallback(TIMESTEPS, every_n_steps=PROGRESS_EVERY),
        EpisodeMetricTensorboardCallback(window_size=METRIC_WINDOW),
    ]

    next_tb_run_id = get_latest_run_id(str(TB_DIR), spec.run_name) + 1
    tb_run_dir = TB_DIR / f"{spec.run_name}_{next_tb_run_id}"
    start_time = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        progress_bar=False,
        tb_log_name=spec.run_name,
    )
    elapsed = time.time() - start_time
    model.save(str(model_path))
    env.close()

    write_summary(
        spec,
        model_zip_path=model_zip_path,
        elapsed_seconds=elapsed,
        tensorboard_logdir=tb_run_dir,
    )
    print(f"[done] model saved to {model_zip_path}")
    print(f"[done] tensorboard logs at {tb_run_dir}")


def main() -> None:
    args = build_argument_parser().parse_args()
    spec = build_run_spec(seed=args.seed)
    train(spec, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
