from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import Any

from laneless_training_registry import archive_training_outputs, make_run_tag


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
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cell(notebook: dict[str, Any], notebook_path: Path, cell_index: int, namespace: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][cell_index].get("source", []))
    print(f"[notebook-task] executing notebook cell {cell_index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        exec_notebook_cell(notebook, notebook_path, cell_index, namespace)


def apply_overrides(namespace: dict[str, Any], args: argparse.Namespace, task: dict[str, Any]) -> None:
    namespace["DEVICE"] = args.device
    namespace[task["flag"]] = True

    if args.timesteps is not None:
        timesteps = int(args.timesteps)
        namespace[str(task["timesteps_key"])] = timesteps
        if args.task == "guided-ddpg-cbf-train":
            namespace["DDPG_CBF_TOTAL_TIMESTEPS"] = timesteps
    if args.n_envs is not None:
        namespace["DDPG_NUM_ENVS"] = int(args.n_envs)
        namespace["DDPG_CBF_NUM_ENVS"] = int(args.n_envs)
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)


TASKS = {
    "ppo-train": {
        "deps": [2, 4, 6, 7, 9],
        "cell": 11,
        "flag": "RUN_PPO_TRAIN",
        "timesteps_key": "TOTAL_TIMESTEPS",
    },
    "ddpg-train": {
        "deps": [2, 4, 6, 7, 9],
        "cell": 22,
        "flag": "RUN_DDPG_TRAIN",
        "timesteps_key": "DDPG_TOTAL_TIMESTEPS",
    },
    "ddpg-cbf-train": {
        "deps": [2, 4, 6, 7, 9, 31, 33, 35, 37, 39, 41],
        "cell": 43,
        "flag": "RUN_DDPG_CBF_TRAIN",
        "timesteps_key": "DDPG_CBF_TOTAL_TIMESTEPS",
    },
    "guided-ddpg-cbf-train": {
        "deps": [2, 4, 6, 7, 9, 31, 33, 35, 37, 39, 41],
        "cell": 51,
        "flag": "RUN_GUIDED_DDPG_CBF_TRAIN",
        "timesteps_key": "GUIDED_DDPG_CBF_TOTAL_TIMESTEPS",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crash-prone laneless notebook training cells out of process.")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()
    if args.run_tag is None:
        args.run_tag = make_run_tag()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    task = TASKS[args.task]
    namespace: dict[str, Any] = {"__name__": "__main__"}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in task["deps"]:
        exec_notebook_cell(notebook, notebook_path, cell_index, namespace)
        if cell_index == 33:
            apply_overrides(namespace, args, task)
    apply_overrides(namespace, args, task)

    print(
        "[notebook-task] starting",
        {
            "task": args.task,
            "device": namespace["DEVICE"],
            "timesteps": namespace.get(str(task["timesteps_key"])),
            "n_envs": namespace.get("DDPG_NUM_ENVS"),
            "lambda_filter": namespace.get("CBF_FILTER_REWARD_LAMBDA"),
            "k0": namespace.get("CBF_K0"),
            "k1": namespace.get("CBF_K1"),
        },
        flush=True,
    )
    exec_notebook_cell(notebook, notebook_path, int(task["cell"]), namespace)
    archived = archive_training_outputs(
        namespace=namespace,
        task_name=args.task,
        run_tag=args.run_tag,
        command=sys.argv,
    )
    if archived is not None:
        if archived["complete"]:
            print(
                "[notebook-task] linked latest run",
                {
                    "variant": archived["label"],
                    "run_dir": archived["run_dir"],
                    "max_timestep": archived["max_timestep"],
                },
                flush=True,
            )
        else:
            print(
                "[notebook-task] archived partial run without updating latest",
                {
                    "variant": archived["label"],
                    "run_dir": archived["run_dir"],
                    "max_timestep": archived["max_timestep"],
                    "expected_timesteps": archived["expected_timesteps"],
                },
                flush=True,
            )
    print(f"[notebook-task] completed {args.task}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
