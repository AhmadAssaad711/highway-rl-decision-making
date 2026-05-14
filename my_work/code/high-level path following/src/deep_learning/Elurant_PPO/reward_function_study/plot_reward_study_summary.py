from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_with_project_venv() -> None:
    """Re-run with the repo virtualenv if plotting dependencies are missing."""
    if os.environ.get("PPO_REWARD_STUDY_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = PROJECT_ROOT / "artifacts" / "ppo" / "reward_function_study" / "summary"
DEFAULT_CSV = SUMMARY_DIR / "reward_grid_summary.csv"
DEFAULT_OUTPUT_DIR = SUMMARY_DIR / "plots"

GRID_GROUP = "grid"
BASELINE_GROUP = "baseline_repeat"


@dataclass(frozen=True)
class FactorSpec:
    column: str
    label: str
    baseline_value: float


@dataclass(frozen=True)
class ResponseSpec:
    column: str
    label: str


FACTORS = (
    FactorSpec("collision_reward", "Collision Reward", -1.0),
    FactorSpec("high_speed_reward", "High-Speed Reward", 0.4),
    FactorSpec("right_lane_reward", "Right-Lane Reward", 0.1),
)

MAIN_EFFECT_RESPONSES = (
    ResponseSpec("eval_episode_reward", "Eval Episode Reward"),
    ResponseSpec("eval_collision", "Collision Rate"),
    ResponseSpec("eval_offroad", "Off-road Rate"),
    ResponseSpec("eval_mean_forward_speed", "Mean Forward Speed"),
)

INTERACTION_RESPONSES = (
    ResponseSpec("eval_episode_reward", "Eval Episode Reward"),
    ResponseSpec("eval_collision", "Collision Rate"),
    ResponseSpec("eval_offroad", "Off-road Rate"),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate main-effect and interaction plots for the PPO reward study."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    numeric_columns = [
        "collision_reward",
        "high_speed_reward",
        "right_lane_reward",
        "mean_reward",
        "std_reward",
        "eval_collision",
        "eval_episode_length",
        "eval_episode_reward",
        "eval_mean_abs_delta_steering",
        "eval_mean_abs_delta_throttle",
        "eval_mean_forward_speed",
        "eval_offroad",
        "eval_right_lane_ratio",
        "eval_steering_var",
        "eval_throttle_var",
        "elapsed_seconds",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def format_reward_value(value: float) -> str:
    text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def factor_values(df: pd.DataFrame, factor: FactorSpec) -> list[float]:
    values = sorted(df[factor.column].dropna().unique())
    return [float(value) for value in values]


def select_level(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    mask = np.isclose(df[column].to_numpy(dtype=float), value)
    return df.loc[mask].copy()


def baseline_setting_rows(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.copy()
    for factor in FACTORS:
        subset = select_level(subset, factor.column, factor.baseline_value)
    return subset


def baseline_stats(
    grid_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    response: ResponseSpec,
) -> tuple[float, float] | None:
    if not baseline_df.empty and response.column in baseline_df.columns:
        values = baseline_df[response.column].dropna().to_numpy(dtype=float)
        if values.size:
            mean = float(np.mean(values))
            std = float(np.std(values))
            return mean, std

    baseline_rows = baseline_setting_rows(grid_df)
    if response.column in baseline_rows.columns:
        values = baseline_rows[response.column].dropna().to_numpy(dtype=float)
        if values.size:
            return float(np.mean(values)), float(np.std(values))

    return None


def direction_note(response: ResponseSpec) -> str:
    if response.column in {"eval_collision", "eval_offroad"}:
        return "negative is better than baseline"
    return "positive is better than baseline"


def make_interaction_pairs() -> list[tuple[FactorSpec, FactorSpec, FactorSpec]]:
    return [
        (FACTORS[0], FACTORS[1], FACTORS[2]),
        (FACTORS[0], FACTORS[2], FACTORS[1]),
        (FACTORS[1], FACTORS[2], FACTORS[0]),
    ]


def plot_baseline_ablations(
    grid_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(MAIN_EFFECT_RESPONSES),
        len(FACTORS),
        figsize=(15, 11),
        constrained_layout=True,
    )

    fig.suptitle(
        "Baseline-Centered Ablations: Change One Reward Weight at a Time\n"
        "For each column, the other two reward weights stay at the baseline setting.\n"
        "Y-axis shows change vs baseline-repeat mean; red band is baseline +/- 1 std.",
        fontsize=15,
        fontweight="bold",
    )

    for row_idx, response in enumerate(MAIN_EFFECT_RESPONSES):
        stats = baseline_stats(grid_df, baseline_df, response)
        if stats is None:
            continue
        baseline_mean, baseline_std = stats

        for col_idx, factor in enumerate(FACTORS):
            ax = axes[row_idx, col_idx]
            subset = grid_df.copy()
            for other_factor in FACTORS:
                if other_factor.column == factor.column:
                    continue
                subset = select_level(subset, other_factor.column, other_factor.baseline_value)

            subset = subset.sort_values(factor.column).dropna(subset=[response.column])
            x_values = subset[factor.column].to_numpy(dtype=float)
            y_raw = subset[response.column].to_numpy(dtype=float)
            y_delta = y_raw - baseline_mean

            ax.axhspan(
                -baseline_std,
                baseline_std,
                color="#d62728",
                alpha=0.12,
                label="Baseline +/- 1 std" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.axhline(
                0.0,
                color="#d62728",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="Baseline mean" if row_idx == 0 and col_idx == 0 else None,
            )

            ax.plot(
                x_values,
                y_delta,
                marker="o",
                color="#1f77b4",
                linewidth=2,
                markersize=6,
                label="Ablation run" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.scatter(
                [factor.baseline_value],
                [0.0],
                marker="*",
                s=180,
                color="#111111",
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
                label="Baseline setting" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.axvline(
                factor.baseline_value,
                color="#d62728",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )

            if row_idx == 0:
                ax.set_title(factor.label)
            if col_idx == 0:
                ax.set_ylabel(f"{response.label}\nDelta vs baseline")
            if row_idx == len(MAIN_EFFECT_RESPONSES) - 1:
                ax.set_xlabel(factor.label)

            ax.text(
                0.03,
                0.95,
                direction_note(response),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
            )
            ax.set_xticks(x_values)
            ax.set_xticklabels([format_reward_value(value) for value in x_values])
            ax.grid(alpha=0.25)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-", color="#1f77b4", label="Ablation run"),
        Line2D([0], [0], marker="*", linestyle="", color="#111111", label="Baseline setting"),
        Line2D([0], [0], linestyle="--", color="#d62728", label="Baseline mean"),
        Line2D([0], [0], linewidth=8, color="#d62728", alpha=0.12, label="Baseline +/- 1 std"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=4, frameon=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_figure(
    grid_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    response: ResponseSpec,
    output_path: Path,
    dpi: int,
) -> None:
    stats = baseline_stats(grid_df, baseline_df, response)
    if stats is None:
        return
    baseline_mean, baseline_std = stats

    pairings = make_interaction_pairs()
    fig, axes = plt.subplots(
        len(pairings),
        3,
        figsize=(16, 12),
        constrained_layout=True,
        sharey="row",
    )

    fig.suptitle(
        f"Interaction Plots Relative to Baseline: {response.label}\n"
        "Y-axis shows change vs baseline-repeat mean; red band is baseline +/- 1 std.\n"
        "Each row compares two reward weights while the third weight is fixed by panel.",
        fontsize=15,
        fontweight="bold",
    )

    for row_idx, (x_factor, line_factor, fixed_factor) in enumerate(pairings):
        fixed_values = factor_values(grid_df, fixed_factor)
        line_values = factor_values(grid_df, line_factor)
        x_values = factor_values(grid_df, x_factor)
        colors = plt.get_cmap("tab10")(np.linspace(0.05, 0.75, len(line_values)))

        for col_idx, fixed_value in enumerate(fixed_values):
            ax = axes[row_idx, col_idx]
            fixed_df = select_level(grid_df, fixed_factor.column, fixed_value)
            is_baseline_slice = np.isclose(fixed_value, fixed_factor.baseline_value)

            ax.axhspan(-baseline_std, baseline_std, color="#d62728", alpha=0.12)
            ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.9)
            ax.axvline(
                x_factor.baseline_value,
                color="#d62728",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

            for color, line_value in zip(colors, line_values):
                line_df = (
                    select_level(fixed_df, line_factor.column, line_value)
                    .sort_values(x_factor.column)
                    .dropna(subset=[response.column])
                )
                if line_df.empty:
                    continue

                y_delta = line_df[response.column].to_numpy(dtype=float) - baseline_mean
                is_baseline_line = np.isclose(line_value, line_factor.baseline_value)
                ax.plot(
                    line_df[x_factor.column],
                    y_delta,
                    marker="o",
                    linewidth=2.8 if is_baseline_line else 2.0,
                    color=color,
                    alpha=1.0 if is_baseline_line else 0.85,
                    label=f"{line_factor.label} = {format_reward_value(line_value)}",
                )

            if is_baseline_slice:
                ax.scatter(
                    [x_factor.baseline_value],
                    [0.0],
                    marker="*",
                    s=180,
                    color="#111111",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )

            if row_idx == 0:
                title = f"{fixed_factor.label} fixed at {format_reward_value(fixed_value)}"
                if is_baseline_slice:
                    title += " [baseline slice]"
                ax.set_title(
                    title
                )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{response.label}\nDelta vs baseline\nX: {x_factor.label}\nLines: {line_factor.label}"
                )
            if row_idx == len(pairings) - 1:
                ax.set_xlabel(x_factor.label)

            ax.text(
                0.03,
                0.95,
                direction_note(response),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
            )
            ax.set_xticks(x_values)
            ax.set_xticklabels([format_reward_value(value) for value in x_values])
            ax.grid(alpha=0.25)

        axes[row_idx, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(df: pd.DataFrame) -> None:
    grid_df = df[df["group"] == GRID_GROUP].copy()
    baseline_df = df[df["group"] == BASELINE_GROUP].copy()
    if grid_df.empty:
        return

    stats = baseline_stats(
        grid_df=grid_df,
        baseline_df=baseline_df,
        response=ResponseSpec("eval_episode_reward", "Eval Episode Reward"),
    )
    if stats is None:
        return
    baseline_mean, baseline_std = stats

    print("\nBaseline-centered ablation deltas for eval_episode_reward:")
    for factor in FACTORS:
        subset = grid_df.copy()
        for other_factor in FACTORS:
            if other_factor.column == factor.column:
                continue
            subset = select_level(subset, other_factor.column, other_factor.baseline_value)
        grouped = subset.sort_values(factor.column)[[factor.column, "eval_episode_reward"]]
        parts = [
            f"{format_reward_value(level)} -> {value - baseline_mean:+.2f}"
            for level, value in zip(grouped[factor.column], grouped["eval_episode_reward"])
        ]
        print(f"  {factor.label}: " + ", ".join(parts))
    print(f"  Baseline repeat mean: {baseline_mean:.2f}")
    print(f"  Baseline repeat std: {baseline_std:.2f}")


def main() -> None:
    args = build_argument_parser().parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.outdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)
    grid_df = df[df["group"] == GRID_GROUP].copy()
    baseline_df = df[df["group"] == BASELINE_GROUP].copy()
    if grid_df.empty:
        raise ValueError(f"No '{GRID_GROUP}' rows found in {csv_path}")

    main_effects_output = output_dir / "reward_grid_baseline_ablations.png"
    plot_baseline_ablations(
        grid_df=grid_df,
        baseline_df=baseline_df,
        output_path=main_effects_output,
        dpi=args.dpi,
    )
    print(f"Saved: {main_effects_output}")

    for response in INTERACTION_RESPONSES:
        output_path = output_dir / f"reward_grid_interaction_delta_{response.column}.png"
        plot_interaction_figure(
            grid_df=grid_df,
            baseline_df=baseline_df,
            response=response,
            output_path=output_path,
            dpi=args.dpi,
        )
        print(f"Saved: {output_path}")

    print_console_summary(df)


if __name__ == "__main__":
    main()
