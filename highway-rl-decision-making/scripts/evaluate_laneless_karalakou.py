from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
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
    print(f"[eval-runner] executing notebook cell {cell_index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def apply_cbf_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def exec_notebook_cells(
    notebook_path: Path,
    cell_indices: list[int],
    namespace: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        exec_notebook_cell(notebook, notebook_path, cell_index, namespace)
        if cell_index == 33:
            apply_cbf_overrides(namespace, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run laneless Karalakou final evaluations out of process.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf", "guided-ddpg-cbf"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    return parser.parse_args()


def with_stem_suffix(path: Path, suffix: str) -> Path:
    suffixed = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if os.name != "nt" or len(str(suffixed)) < 260:
        return suffixed

    digest = hashlib.sha1(suffixed.stem.encode("utf-8")).hexdigest()[:8]
    overflow = len(str(suffixed)) - 248
    keep = max(24, len(suffixed.stem) - overflow - len(digest) - 1)
    short_stem = suffixed.stem[:keep].rstrip("._-")
    return suffixed.with_name(f"{short_stem}_{digest}{suffixed.suffix}")


def normalize_artifact_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    suffix = suffix.strip("._-")
    if len(suffix) > 16:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        suffix = f"{suffix[:7].strip('._-')}_{digest}"
    return suffix or None


def artifact_path(path: Path, traffic_model: str, artifact_suffix: str | None) -> Path:
    suffix = normalize_artifact_suffix(artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    return with_stem_suffix(Path(path), suffix) if suffix else Path(path)


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}

    base_cells = [2, 4, 6, 7, 9]
    cbf_cells = [32, 34, 36, 38, 40, 42]
    guided_cells = [52]
    needs_cbf = args.variant in {"ddpg-cbf", "guided-ddpg-cbf"}
    exec_notebook_cells(
        notebook_path,
        base_cells + (cbf_cells if needs_cbf else []) + (guided_cells if args.variant == "guided-ddpg-cbf" else []),
        namespace,
        args,
    )
    apply_cbf_overrides(namespace, args)

    namespace["DEVICE"] = args.device
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "ppo":
        model_path = args.model_path or artifact_path(namespace["MODEL_PATH"], traffic_model, args.artifact_suffix)
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["PPO"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating PPO", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
        )
    elif args.variant == "ddpg":
        model_path = args.model_path or artifact_path(namespace["DDPG_MODEL_PATH"], traffic_model, args.artifact_suffix)
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
        )
    elif args.variant == "ddpg-cbf":
        model_path = args.model_path or artifact_path(namespace["DDPG_CBF_MODEL_PATH"], traffic_model, args.artifact_suffix)
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG-CBF", flush=True)
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )
    else:
        model_path = args.model_path or artifact_path(namespace["GUIDED_DDPG_CBF_MODEL_PATH"], traffic_model, args.artifact_suffix)
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["GuidedCBFDDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating guided DDPG-CBF", flush=True)
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )

    metrics.to_csv(output_path, index=False)
    print(f"[eval-runner] wrote {output_path}", flush=True)
    print(metrics.drop(columns=["episode"]).mean().to_frame("mean"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
