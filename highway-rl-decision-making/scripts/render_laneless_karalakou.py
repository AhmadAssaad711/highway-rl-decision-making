from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
from typing import Any

from laneless_script_config import active_traffic_model, env_config_from_args


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


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        source = "".join(notebook["cells"][cell_index].get("source", []))
        print(f"[render-runner] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render laneless Karalakou policies out of the notebook kernel.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf"], required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    return parser.parse_args()


def with_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


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
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])

    if args.variant == "ppo":
        model_path = args.model_path or namespace["MODEL_PATH"]
        if args.model_path is None and traffic_model != "force":
            model_path = with_stem_suffix(Path(model_path), f"traffic_{traffic_model}")
        model = namespace["PPO"].load(str(model_path), device=args.device)
        env = namespace["make_single_env"](seed=args.seed, render_mode="human", env_config=namespace["ENV_CONFIG"])
    elif args.variant == "ddpg":
        model_path = args.model_path or namespace["DDPG_MODEL_PATH"]
        if args.model_path is None and traffic_model != "force":
            model_path = with_stem_suffix(Path(model_path), f"traffic_{traffic_model}")
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        env = namespace["make_single_env"](seed=args.seed, render_mode="human", env_config=namespace["ENV_CONFIG"])
    else:
        model_path = args.model_path or namespace["DDPG_CBF_MODEL_PATH"]
        if args.model_path is None and traffic_model != "force":
            model_path = with_stem_suffix(Path(model_path), f"traffic_{traffic_model}")
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        env = namespace["make_cbf_single_env"](
            seed=args.seed,
            render_mode="human",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            env_config=namespace["ENV_CONFIG"],
        )

    print(f"[render-runner] loaded {model_path}", flush=True)
    print(f"[render-runner] rendering {args.variant} for {args.steps:,} steps", flush=True)
    try:
        obs, _ = env.reset(seed=args.seed)
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    print("[render-runner] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
