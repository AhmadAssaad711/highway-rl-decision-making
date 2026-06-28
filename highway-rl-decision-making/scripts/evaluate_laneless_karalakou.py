from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import Any


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        source = "".join(notebook["cells"][cell_index].get("source", []))
        print(f"[eval-runner] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run laneless Karalakou final evaluations out of process.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}

    base_cells = [2, 4, 6, 7, 9]
    cbf_cells = [31, 33, 35, 37, 39, 41]
    exec_notebook_cells(notebook_path, base_cells + (cbf_cells if args.variant == "ddpg-cbf" else []), namespace)

    namespace["DEVICE"] = args.device
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "ppo":
        model_path = namespace["MODEL_PATH"]
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["PPO"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating PPO", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
        )
    elif args.variant == "ddpg":
        model_path = namespace["DDPG_MODEL_PATH"]
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
        )
    else:
        model_path = namespace["DDPG_CBF_MODEL_PATH"]
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG-CBF", flush=True)
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
        )

    metrics.to_csv(output_path, index=False)
    print(f"[eval-runner] wrote {output_path}", flush=True)
    print(metrics.drop(columns=["episode"]).mean().to_frame("mean"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
