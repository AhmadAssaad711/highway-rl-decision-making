"""
Evaluate the trained Kourani DQN on unstructured highway scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("KOURANI_UNSTRUCTURED_EVAL_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import cv2  # noqa: F401
        import highway_env  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        child_env = dict(os.environ)
        child_env["KOURANI_UNSTRUCTURED_EVAL_SKIP_VENV_REEXEC"] = "1"
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            env=child_env,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

import cv2
import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.tensorboard import SummaryWriter

from highway_env_extension import make_unstructured_kourani_env
from scenarios import SCENARIOS

DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "artifacts" / "dqn" / "kourani_ttc" / "model_ttc.zip",
    PROJECT_ROOT / "artifacts" / "dqn" / "kourani_ttc" / "model.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Kourani DQN across unstructured highway-env scenarios."
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit path to the trained DQN zip file.")
    parser.add_argument("--episodes-per-scenario", type=int, default=1000, help="Evaluation episodes to run per scenario.")
    parser.add_argument("--max-steps", type=int, default=300, help="Safety cap on episode steps.")
    parser.add_argument("--device", default="auto", help="Torch device passed to SB3 when loading the model.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for repeatable scenario evaluation.")
    parser.add_argument("--skip-videos", action="store_true", help="Disable video recording for faster evaluation runs.")
    parser.add_argument("--scenario", action="append", default=None, help="Optional scenario name filter. Can be passed multiple times.")
    parser.add_argument("--progress-interval", type=int, default=25, help="Print progress every N episodes per scenario.")
    return parser.parse_args()


def make_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def resolve_model_path(explicit_path: str | None) -> Path:
    if explicit_path:
        model_path = Path(explicit_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return model_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a DQN model automatically. Checked: "
        + ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
    )


def select_scenarios(selected_names: list[str] | None) -> list[dict]:
    if not selected_names:
        return SCENARIOS
    selected = set(selected_names)
    filtered = [scenario for scenario in SCENARIOS if scenario["name"] in selected]
    if not filtered:
        raise ValueError(f"No scenarios matched: {sorted(selected)}")
    return filtered


def action_lane_change_ids(env) -> set[int]:
    action_indexes = getattr(env.unwrapped.action_type, "actions_indexes", {})
    lane_change_actions = set()
    for action_name in ("LANE_LEFT", "LANE_RIGHT"):
        if action_name in action_indexes:
            lane_change_actions.add(int(action_indexes[action_name]))
    return lane_change_actions


def predict_q_values(model: DQN, observation: np.ndarray) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(observation, dtype=np.float32))
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy()[0]
    return q_values


def action_probabilities_from_q(q_values: np.ndarray) -> np.ndarray:
    shifted = q_values - np.max(q_values)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    return probs


def make_policy_panel(
    frame_width: int,
    action_labels: dict[int, str],
    q_values: np.ndarray,
    chosen_action: int,
) -> np.ndarray:
    panel_height = 180
    panel = np.zeros((panel_height, frame_width, 3), dtype=np.uint8)
    action_count = len(q_values)
    cell_width = max(frame_width // action_count, 1)

    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    if np.isclose(q_min, q_max):
        q_min -= 1.0
        q_max += 1.0

    probabilities = action_probabilities_from_q(q_values)
    best_action = int(np.argmax(q_values))

    for action in range(action_count):
        left = action * cell_width
        right = frame_width if action == action_count - 1 else min((action + 1) * cell_width, frame_width)

        normalized = float(np.clip((q_values[action] - q_min) / (q_max - q_min), 0.0, 1.0))
        color = cv2.applyColorMap(
            np.uint8([[round(255 * normalized)]]),
            cv2.COLORMAP_VIRIDIS,
        )[0, 0].tolist()
        panel[:, left:right] = color

        border_color = (100, 255, 120) if action == best_action else (255, 255, 255)
        border_width = 5 if action == chosen_action else 2
        cv2.rectangle(panel, (left, 0), (right - 1, panel_height - 1), border_color, border_width)

        text_lines = [
            action_labels[action],
            f"Q={q_values[action]:.2f}",
            f"p={probabilities[action]:.2f}",
        ]
        for line_index, text in enumerate(text_lines):
            cv2.putText(
                panel,
                text,
                (left + 10, 28 + line_index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

    footer = "Green border = argmax Q, thicker border = executed action"
    cv2.putText(
        panel,
        footer,
        (12, panel_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def compose_visualization_frame(
    env,
    action_labels: dict[int, str],
    q_values: np.ndarray,
    chosen_action: int,
) -> np.ndarray:
    road_frame = np.asarray(env.render(), dtype=np.uint8)
    policy_panel = make_policy_panel(
        frame_width=road_frame.shape[1],
        action_labels=action_labels,
        q_values=q_values,
        chosen_action=chosen_action,
    )
    return np.vstack([road_frame, policy_panel])


def create_video_writer(video_root: Path, scenario_name: str, fps: float):
    video_root.mkdir(parents=True, exist_ok=True)
    video_path = video_root / f"{scenario_name}_policy.mp4"
    return video_path, None


def evaluate_scenario(
    model_path: Path,
    scenario: dict,
    args: argparse.Namespace,
    scenario_index: int,
    run_root: Path,
    video_run_root: Path | None,
) -> dict:
    scenario_name = scenario["name"]
    config = dict(scenario["config"])

    video_root = video_run_root / scenario_name if video_run_root else None
    # Keep eval tensorboard paths short enough for Windows event filenames.
    tensorboard_root = run_root / "tb" / f"s{scenario_index + 1:02d}"
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_root))

    env = make_unstructured_kourani_env(render_mode="rgb_array", config=config)

    model = DQN.load(str(model_path), env=env, device=args.device)
    lane_change_ids = action_lane_change_ids(env)
    policy_frequency = float(env.unwrapped.config["policy_frequency"])
    action_labels = dict(env.unwrapped.action_type.actions)

    rewards: list[float] = []
    avg_speeds: list[float] = []
    episode_lengths: list[int] = []
    distances: list[float] = []
    lane_changes: list[int] = []
    crash_flags: list[int] = []
    episode_avg_ttc: list[float] = []
    episode_avg_ttc_penalty: list[float] = []

    try:
        for episode_idx in range(args.episodes_per_scenario):
            obs, info = env.reset(seed=args.seed + scenario_index * 100 + episode_idx)
            terminated = False
            truncated = False
            total_reward = 0.0
            speed_trace: list[float] = []
            distance_m = 0.0
            step_count = 0
            lane_change_count = 0
            ttc_trace: list[float] = []
            ttc_penalty_trace: list[float] = []
            last_info = info
            video_writer = None
            video_path = None
            try:
                while not (terminated or truncated):
                    q_values = predict_q_values(model, obs)
                    action = int(np.argmax(q_values))
                    if episode_idx == 0 and not args.skip_videos:
                        composed_frame = compose_visualization_frame(
                            env=env,
                            action_labels=action_labels,
                            q_values=q_values,
                            chosen_action=action,
                        )
                        if video_writer is None:
                            assert video_root is not None
                            video_path, _ = create_video_writer(
                                video_root=video_root,
                                scenario_name=scenario_name,
                                fps=float(env.metadata.get("render_fps", 5)),
                            )
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(
                                str(video_path),
                                fourcc,
                                float(env.metadata.get("render_fps", 5)),
                                (composed_frame.shape[1], composed_frame.shape[0]),
                            )
                        video_writer.write(cv2.cvtColor(composed_frame, cv2.COLOR_RGB2BGR))
                    action_value = int(action)
                    if action_value in lane_change_ids:
                        lane_change_count += 1

                    obs, reward, terminated, truncated, last_info = env.step(action)
                    step_count += 1
                    total_reward += float(reward)

                    speed_mps = float(last_info.get("speed", 0.0))
                    speed_trace.append(speed_mps)
                    distance_m += speed_mps / policy_frequency
                    ttc_trace.append(float(last_info.get("ttc_current", np.nan)))
                    ttc_penalty_trace.append(float(last_info.get("ttc_penalty", 0.0)))

                    if args.max_steps and step_count >= args.max_steps:
                        truncated = True
            finally:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    if video_path is not None:
                        print(f"[{scenario_name}] saved policy video: {video_path}")

            avg_speed = float(np.mean(speed_trace)) if speed_trace else 0.0
            crashed = int(bool(last_info.get("crashed", False)))
            avg_ttc = float(np.nanmean(ttc_trace)) if ttc_trace else float("nan")
            avg_ttc_penalty = float(np.mean(ttc_penalty_trace)) if ttc_penalty_trace else 0.0

            rewards.append(total_reward)
            avg_speeds.append(avg_speed)
            episode_lengths.append(step_count)
            distances.append(distance_m)
            lane_changes.append(lane_change_count)
            crash_flags.append(crashed)
            episode_avg_ttc.append(avg_ttc)
            episode_avg_ttc_penalty.append(avg_ttc_penalty)

            writer.add_scalar("episodes/reward", total_reward, episode_idx)
            writer.add_scalar("episodes/avg_speed_mps", avg_speed, episode_idx)
            writer.add_scalar("episodes/episode_length_steps", step_count, episode_idx)
            writer.add_scalar("episodes/distance_m", distance_m, episode_idx)
            writer.add_scalar("episodes/lane_changes", lane_change_count, episode_idx)
            writer.add_scalar("episodes/collision", crashed, episode_idx)
            if not np.isnan(avg_ttc):
                writer.add_scalar("episodes/avg_ttc_current", avg_ttc, episode_idx)
            writer.add_scalar("episodes/avg_ttc_penalty", avg_ttc_penalty, episode_idx)
            if (
                episode_idx == 0
                or (episode_idx + 1) % max(1, args.progress_interval) == 0
                or (episode_idx + 1) == args.episodes_per_scenario
            ):
                print(
                    f"[{scenario_name}] episode {episode_idx + 1}/{args.episodes_per_scenario} "
                    f"| reward={total_reward:.2f} speed={avg_speed:.2f} "
                    f"| crash={bool(crashed)} steps={step_count} "
                    f"| avg_ttc={avg_ttc:.2f} penalty={avg_ttc_penalty:.3f}"
                )
                writer.flush()
    finally:
        writer.flush()
        writer.close()
        env.close()

    summary = {
        "scenario_name": scenario_name,
        "description": scenario["description"],
        "model_path": str(model_path),
        "episodes": args.episodes_per_scenario,
        "avg_speed_mps": float(np.mean(avg_speeds)),
        "collision_pct": float(100.0 * np.mean(crash_flags)),
        "episode_length_steps": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(rewards)),
        "avg_distance_m": float(np.mean(distances)),
        "success_pct": float(100.0 * (1.0 - np.mean(crash_flags))),
        "avg_lane_changes": float(np.mean(lane_changes)),
        "avg_ttc_current": float(np.nanmean(episode_avg_ttc)),
        "avg_ttc_penalty": float(np.mean(episode_avg_ttc_penalty)),
        "config": config,
        "tensorboard_dir": str(tensorboard_root),
        "video_dir": None if video_root is None else str(video_root),
    }

    writer = SummaryWriter(log_dir=str(tensorboard_root))
    writer.add_text("scenario/description", scenario["description"], 0)
    writer.add_scalar("aggregate/avg_speed_mps", summary["avg_speed_mps"], 0)
    writer.add_scalar("aggregate/collision_pct", summary["collision_pct"], 0)
    writer.add_scalar("aggregate/episode_length_steps", summary["episode_length_steps"], 0)
    writer.add_scalar("aggregate/mean_reward", summary["mean_reward"], 0)
    writer.add_scalar("aggregate/avg_distance_m", summary["avg_distance_m"], 0)
    writer.add_scalar("aggregate/success_pct", summary["success_pct"], 0)
    writer.add_scalar("aggregate/avg_lane_changes", summary["avg_lane_changes"], 0)
    if not np.isnan(summary["avg_ttc_current"]):
        writer.add_scalar("aggregate/avg_ttc_current", summary["avg_ttc_current"], 0)
    writer.add_scalar("aggregate/avg_ttc_penalty", summary["avg_ttc_penalty"], 0)
    writer.flush()
    writer.close()

    return summary


def write_results(summaries: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "scenario_metrics.json"
    csv_path = results_dir / "scenario_metrics.csv"

    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    fieldnames = [
        "scenario_name",
        "description",
        "episodes",
        "avg_speed_mps",
        "collision_pct",
        "episode_length_steps",
        "mean_reward",
        "avg_distance_m",
        "success_pct",
        "avg_lane_changes",
        "avg_ttc_current",
        "avg_ttc_penalty",
        "model_path",
        "tensorboard_dir",
        "video_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in summaries)


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    output_root = CURRENT_DIR
    scenarios = select_scenarios(args.scenario)
    run_id = make_run_id()
    run_root = output_root / "runs" / run_id
    video_run_root = None if args.skip_videos else run_root / "videos"

    summaries = []
    for scenario_index, scenario in enumerate(scenarios):
        print(f"Running {scenario['name']} with model {model_path.name}")
        summaries.append(
            evaluate_scenario(
                model_path=model_path,
                scenario=scenario,
                args=args,
                scenario_index=scenario_index,
                run_root=run_root,
                video_run_root=video_run_root,
            )
        )

    write_results(summaries, run_root / "results")
    print(f"\nScenario sweep complete. Run folder: {run_root}")
    for summary in summaries:
        print(
            f"{summary['scenario_name']}: "
            f"speed={summary['avg_speed_mps']:.2f} m/s, "
            f"collisions={summary['collision_pct']:.1f}%, "
            f"length={summary['episode_length_steps']:.1f} steps, "
            f"reward={summary['mean_reward']:.3f}, "
            f"ttc={summary['avg_ttc_current']:.2f}, "
            f"ttc_penalty={summary['avg_ttc_penalty']:.3f}"
        )


if __name__ == "__main__":
    main()
