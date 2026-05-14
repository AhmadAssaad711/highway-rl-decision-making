from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "DQN" / "elurant_dqn.py"
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "dqn" / "dqn_highway_sweep"


def resolve_python_executable() -> str:
    if PROJECT_VENV_PYTHON.exists():
        return str(PROJECT_VENV_PYTHON)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-configuration DQN sweep on highway-v0")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Training timesteps per run",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per run",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 8) // 3)),
        help="Parallel environment workers per run",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for SB3 (default: cpu)",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory for sweep artifacts",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run summaries instead of retraining completed runs",
    )
    return parser.parse_args()


def sweep_configs() -> list[dict]:
    return [
        {
            "run_name": "baseline_lr5e4_gamma080_bs32_buf15000_tu50",
            "learning_rate": 5e-4,
            "gamma": 0.80,
            "batch_size": 32,
            "buffer_size": 15000,
            "learning_starts": 200,
            "target_update_interval": 50,
            "seed": 42,
        },
        {
            "run_name": "lr3e4_gamma090_bs64_buf30000_tu100",
            "learning_rate": 3e-4,
            "gamma": 0.90,
            "batch_size": 64,
            "buffer_size": 30000,
            "learning_starts": 500,
            "target_update_interval": 100,
            "seed": 43,
        },
        {
            "run_name": "lr1e3_gamma085_bs64_buf50000_tu250",
            "learning_rate": 1e-3,
            "gamma": 0.85,
            "batch_size": 64,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "target_update_interval": 250,
            "seed": 44,
        },
        {
            "run_name": "lr3e4_gamma095_bs128_buf50000_tu250",
            "learning_rate": 3e-4,
            "gamma": 0.95,
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "target_update_interval": 250,
            "seed": 45,
        },
    ]


def run_one(config: dict, args: argparse.Namespace, results_root: Path) -> dict:
    summary_path = results_root / config["run_name"] / "summary.json"
    if args.reuse_existing and summary_path.exists():
        print(f"\n=== Reusing {config['run_name']} ===")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    command = [
        resolve_python_executable(),
        str(TRAIN_SCRIPT),
        "--timesteps",
        str(args.timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--num-envs",
        str(args.num_envs),
        "--device",
        args.device,
        "--results-root",
        str(results_root),
        "--run-name",
        config["run_name"],
        "--learning-rate",
        str(config["learning_rate"]),
        "--gamma",
        str(config["gamma"]),
        "--batch-size",
        str(config["batch_size"]),
        "--buffer-size",
        str(config["buffer_size"]),
        "--learning-starts",
        str(config["learning_starts"]),
        "--target-update-interval",
        str(config["target_update_interval"]),
        "--seed",
        str(config["seed"]),
    ]
    print(f"\n=== Running {config['run_name']} ===")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_leaderboard(results: list[dict], results_root: Path) -> None:
    ordered = sorted(results, key=lambda item: item["mean_reward"], reverse=True)
    leaderboard_json = results_root / "leaderboard.json"
    leaderboard_csv = results_root / "leaderboard.csv"
    leaderboard_json.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    with leaderboard_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "run_name",
                "mean_reward",
                "std_reward",
                "timesteps",
                "eval_episodes",
                "num_envs",
                "device",
                "learning_rate",
                "gamma",
                "batch_size",
                "buffer_size",
                "learning_starts",
                "target_update_interval",
                "seed",
                "model_path",
                "tensorboard_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(ordered)

    print(f"\nLeaderboard written to {leaderboard_json}")
    print(f"CSV leaderboard written to {leaderboard_csv}")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config in sweep_configs():
        summaries.append(run_one(config, args, results_root))

    write_leaderboard(summaries, results_root)

    print("\nTop runs by mean reward:")
    for idx, item in enumerate(sorted(summaries, key=lambda x: x["mean_reward"], reverse=True), start=1):
        print(
            f"{idx}. {item['run_name']}: "
            f"mean={item['mean_reward']:.2f}, std={item['std_reward']:.2f}, "
            f"model={item['model_path']}"
        )


if __name__ == "__main__":
    main()
