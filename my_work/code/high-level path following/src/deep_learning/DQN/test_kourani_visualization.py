"""
Visual test runner for the trained Kourani DQN with a live policy panel.

This thin wrapper delegates to the shared visualization helper in
`DQN_Kourani.py` so notebook and script rollouts stay in sync.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("KOURANI_VISUAL_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import highway_env  # noqa: F401
        import matplotlib  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        child_env = dict(os.environ)
        child_env["KOURANI_VISUAL_SKIP_VENV_REEXEC"] = "1"
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            env=child_env,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

from DQN_Kourani import MODEL_PATH, resolve_kourani_model_path, run_policy_panel_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained Kourani DQN with a live state-action value panel."
    )
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_kourani_model_path(args.model_path)
    run_policy_panel_visualization(
        model_path=model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        stochastic=args.stochastic,
    )


if __name__ == "__main__":
    main()
