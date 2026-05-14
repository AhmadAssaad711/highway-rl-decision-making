from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate the project root.")


def main() -> int:
    project_root = find_project_root()
    notebook_path = project_root / "notebooks" / "congested_traffic_policy" / "congested_traffic_policy_v2.ipynb"
    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    os.chdir(project_root)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }

    print(f"[runner] executing {notebook_path}", flush=True)
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"[runner] cell {cell_index}", flush=True)
        try:
            exec(compile(source, f"{notebook_path}:cell_{cell_index}", "exec"), namespace)
        except Exception:
            print(f"[runner] failed in cell {cell_index}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return 1

    print("[runner] completed congested traffic v2 notebook", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
