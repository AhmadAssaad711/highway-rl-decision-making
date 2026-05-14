"""
Run a sequential PPO beta sweep on CustomHighwayEnv.

By default, this script trains 5 PPO runs back-to-back. Each run multiplies
the reward-scale beta by 1.5 relative to the previous run, and all runs log to
the same TensorBoard root so they can be compared in one board.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.custom_highway_env import CustomHighwayEnv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "artifacts" / "ppo" / "ppo_highway_beta_sweep"
MODELS_DIR = RUN_ROOT / "models"
LOGS_DIR = RUN_ROOT / "logs"
TB_DIR = RUN_ROOT / "tensorboard"
DEFAULT_N_ENVS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential PPO beta sweep")
    parser.add_argument("--runs", type=int, default=5, help="Number of consecutive beta runs")
    parser.add_argument("--base-beta", type=float, default=0.005, help="Initial beta value")
    parser.add_argument(
        "--beta-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to beta after each run",
    )
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps per run")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Parallel training environments (default: 24, matching the Kourani DQN setup)",
    )
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=4096, help="Rollout steps per update")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", default="auto", help="Torch device for PPO")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print training progress every N timesteps",
    )
    return parser.parse_args()


def beta_for_run(base_beta: float, beta_multiplier: float, run_index: int) -> float:
    return float(base_beta * (beta_multiplier ** run_index))


def format_beta(beta: float) -> str:
    return f"{beta:.8f}".rstrip("0").rstrip(".")


def make_env(beta: float):
    return CustomHighwayEnv(beta=beta)


class TimestepProgressCallback(BaseCallback):
    """Print lightweight progress updates during PPO training."""

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


def train_one_run(args: argparse.Namespace, run_index: int) -> None:
    beta = beta_for_run(args.base_beta, args.beta_multiplier, run_index)
    beta_tag = format_beta(beta).replace(".", "p")
    run_name = f"run_{run_index + 1:02d}_beta_{beta_tag}"

    run_model_dir = MODELS_DIR / run_name
    run_log_dir = LOGS_DIR / run_name
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    # Match the Kourani DQN approach by spawning 24 subprocess environments
    # for parallel rollout collection on CPU.
    print(f"Spawning {args.n_envs} parallel PPO environments for {run_name}")
    env = make_vec_env(
        lambda: make_env(beta),
        n_envs=args.n_envs,
        seed=args.seed + run_index,
        monitor_dir=str(run_log_dir),
        vec_env_cls=SubprocVecEnv,
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
        seed=args.seed + run_index,
    )

    eval_env = Monitor(make_env(beta))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best_model"),
        log_path=str(run_log_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    progress_callback = TimestepProgressCallback(
        total_timesteps=args.timesteps,
        every_n_steps=args.progress_every,
    )

    print(f"\n=== Starting {run_name} with beta={beta} ===\n")
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, progress_callback],
        tb_log_name=run_name,
    )
    elapsed = time.time() - start_time

    final_model_path = run_model_dir / "ppo_highway"
    model.save(str(final_model_path))
    print(f"\nCompleted {run_name} in {elapsed:.2f}s. Model saved to {final_model_path}.zip")

    eval_env.close()
    env.close()


def main() -> None:
    args = parse_args()
    for run_index in range(args.runs):
        train_one_run(args, run_index)


if __name__ == "__main__":
    main()
