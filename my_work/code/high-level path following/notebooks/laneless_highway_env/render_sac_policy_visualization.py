from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


NOTEBOOK_RELATIVE_PATH = Path("notebooks") / "laneless_highway_env" / "laneless_highway_env.ipynb"


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []
    for root in [script_path.parent, cwd, *script_path.parents, *cwd.parents, Path.home()]:
        candidates.extend(
            [
                root,
                root / "my_work" / "code" / "high-level path following",
                root / "code" / "high-level path following",
                root / "high-level path following",
                root
                / "OneDrive - American University of Beirut"
                / "AGV research"
                / "my_work"
                / "code"
                / "high-level path following",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "Unstructured-Traffic_Environment").exists() and (candidate / NOTEBOOK_RELATIVE_PATH).exists():
            return candidate

    raise RuntimeError("Could not locate the high-level path following project root.")


def load_notebook_namespace(project_root: Path) -> dict[str, Any]:
    notebook_path = project_root / NOTEBOOK_RELATIVE_PATH
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(notebook_path)}

    # These cells define imports/config, driver profiles, RL env wrappers, and training paths.
    setup_cell_indices = [2, 6, 8, 12]
    with contextlib.redirect_stdout(io.StringIO()):
        for index in setup_cell_indices:
            exec("".join(notebook["cells"][index]["source"]), namespace)
    return namespace


def model_path_candidates(results_dir: Path, run_name: str) -> list[Path]:
    run_dir = results_dir / run_name
    candidates: list[Path] = []
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("model_path"):
                candidates.append(Path(summary["model_path"]))
        except Exception:
            pass
    candidates.extend([run_dir / "model.zip", run_dir / "best_model" / "best_model.zip"])
    return candidates


def read_run_config(results_dir: Path, run_name: str) -> dict[str, Any] | None:
    summary_path = results_dir / run_name / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    env_config = summary.get("env_config")
    return dict(env_config) if isinstance(env_config, dict) else None


def load_sac_model(namespace: dict[str, Any], requested_run_name: str | None) -> tuple[Any, str, dict[str, Any], Path]:
    sac_cls = namespace["SAC"]
    results_dir = Path(namespace["RESULTS_DIR"])
    device = namespace["DEFAULT_DEVICE"]
    default_config = dict(namespace["SAC_LANELESS_RL_CONFIG"])
    configured_run_name = namespace["LANELESS_SAC_RUN_NAME"]

    run_names = [
        requested_run_name,
        configured_run_name,
        "laneless_sac_300k_timesteps",
        "laneless_sac_20k_timesteps",
        "laneless_sac_congested_overtake_reward_20k",
        "laneless_sac_congested_overtake_reward_rear_pressure_20k",
    ]
    ordered_run_names: list[str] = []
    for run_name in run_names:
        if run_name and run_name not in ordered_run_names:
            ordered_run_names.append(run_name)

    checked_paths: list[str] = []
    for run_name in ordered_run_names:
        env_config = read_run_config(results_dir, run_name) or default_config
        for candidate in model_path_candidates(results_dir, run_name):
            checked_paths.append(str(candidate))
            if candidate.exists():
                model = sac_cls.load(str(candidate), device=device)
                return model, run_name, env_config, candidate

    raise FileNotFoundError(
        "No saved SAC model found. Checked:\n" + "\n".join(checked_paths)
    )


class SACPolicyInspector:
    def __init__(self, model: Any, env: Any, grid_size: int):
        self.model = model
        self.env = env
        self.device = model.device
        self.grid_size = int(grid_size)
        self.low = np.asarray(env.action_space.low, dtype=np.float32)
        self.high = np.asarray(env.action_space.high, dtype=np.float32)

    def scale_actions_for_critic(self, actions: np.ndarray) -> np.ndarray:
        if hasattr(self.model.policy, "scale_action"):
            return self.model.policy.scale_action(actions)
        return actions

    def critic_values(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        obs_tensor, _ = self.model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
        obs_batch = obs_tensor.repeat((actions.shape[0],) + (1,) * (obs_tensor.ndim - 1))
        action_tensor = torch.as_tensor(self.scale_actions_for_critic(actions), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_outputs = self.model.policy.critic(obs_batch, action_tensor)
            q_stack = torch.cat(q_outputs, dim=1) if isinstance(q_outputs, tuple) else q_outputs.reshape(-1, 1)
            q_values = torch.min(q_stack, dim=1).values
        return q_values.detach().cpu().numpy()

    def action_grid(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        speed_values = np.linspace(self.low[0], self.high[0], self.grid_size, dtype=np.float32)
        lateral_values = np.linspace(self.low[1], self.high[1], self.grid_size, dtype=np.float32)
        speed_grid, lateral_grid = np.meshgrid(speed_values, lateral_values, indexing="ij")
        actions = np.column_stack([speed_grid.ravel(), lateral_grid.ravel()]).astype(np.float32)
        q_grid = self.critic_values(obs, actions).reshape(self.grid_size, self.grid_size)
        return speed_values, lateral_values, q_grid

    def vehicle_info(self) -> dict[str, float | bool]:
        vehicle = self.env.unwrapped.vehicle
        _, lateral_position = vehicle.lane.local_coordinates(vehicle.position)
        return {
            "speed": float(vehicle.speed),
            "y": float(lateral_position),
            "target_y": float(getattr(vehicle, "target_y", np.nan)),
            "crashed": bool(getattr(vehicle, "crashed", False)),
        }


def capture_frame(env: Any) -> np.ndarray:
    try:
        frame = env.render()
    except AttributeError:
        env.unwrapped.viewer = None
        frame = env.render()
    env.unwrapped.enable_auto_render = False
    return np.asarray(frame, dtype=np.uint8)


def font(size: int = 14) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_policy_panel(
    inspector: SACPolicyInspector,
    obs: np.ndarray,
    action: np.ndarray,
    reward_so_far: float,
    step: int,
    episode: int,
    width: int,
    height: int = 260,
) -> np.ndarray:
    speed_values, lateral_values, q_grid = inspector.action_grid(obs)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    q_value = float(inspector.critic_values(obs, action)[0])
    vehicle_info = inspector.vehicle_info()
    action_type = inspector.env.unwrapped.action_type
    speed_delta = float(action[0] * getattr(action_type, "speed_delta", 1.0))
    lateral_delta = float(action[1] * getattr(action_type, "max_lateral_delta", 1.0))

    panel = Image.new("RGB", (width, height), (12, 12, 14))
    draw = ImageDraw.Draw(panel)
    title_font = font(16)
    body_font = font(14)
    small_font = font(12)

    plot_left, plot_top = 70, 28
    plot_width = min(430, max(220, width - 360))
    plot_height = 170

    q_min = float(np.min(q_grid))
    q_max = float(np.max(q_grid))
    if np.isclose(q_min, q_max):
        q_min -= 1.0
        q_max += 1.0
    normalized = (q_grid - q_min) / (q_max - q_min)
    heatmap = matplotlib.colormaps.get_cmap("viridis")(normalized[::-1, :], bytes=True)[:, :, :3]
    heatmap_img = Image.fromarray(heatmap, mode="RGB").resize((plot_width, plot_height), Image.Resampling.NEAREST)
    panel.paste(heatmap_img, (plot_left, plot_top))

    marker_x = plot_left + np.interp(action[1], [lateral_values[0], lateral_values[-1]], [0, plot_width])
    marker_y = plot_top + np.interp(action[0], [speed_values[0], speed_values[-1]], [plot_height, 0])
    draw.ellipse((marker_x - 7, marker_y - 7, marker_x + 7, marker_y + 7), outline=(100, 255, 120), width=3)

    draw.rectangle((plot_left, plot_top, plot_left + plot_width, plot_top + plot_height), outline=(220, 220, 220), width=1)
    draw.text((plot_left, plot_top + plot_height + 6), "lateral -1", fill=(230, 230, 230), font=small_font)
    draw.text((plot_left + plot_width - 70, plot_top + plot_height + 6), "lateral +1", fill=(230, 230, 230), font=small_font)
    draw.text((8, plot_top), "speed +1", fill=(230, 230, 230), font=small_font)
    draw.text((8, plot_top + plot_height - 14), "speed -1", fill=(230, 230, 230), font=small_font)

    text_left = plot_left + plot_width + 26
    lines = [
        f"SAC policy panel | ep {episode}, step {step}",
        f"action = [{action[0]:+.3f}, {action[1]:+.3f}]",
        f"speed delta = {speed_delta:+.2f} m/s",
        f"lateral delta = {lateral_delta:+.2f} m",
        f"critic Q(s,a) = {q_value:.3f}",
        f"reward so far = {reward_so_far:.2f}",
        f"ego speed = {vehicle_info['speed']:.2f} m/s",
        f"ego y = {vehicle_info['y']:+.2f} m, target y = {vehicle_info['target_y']:+.2f} m",
    ]
    draw.text((text_left, 24), lines[0], fill=(255, 255, 255), font=title_font)
    for index, line in enumerate(lines[1:], start=1):
        draw.text((text_left, 30 + index * 24), line, fill=(225, 225, 225), font=body_font)

    draw.text(
        (14, height - 24),
        "Heatmap: min critic Q(s,a); green circle: actor action",
        fill=(210, 210, 210),
        font=small_font,
    )
    return np.asarray(panel, dtype=np.uint8)


def compose_frame(sim_frame: np.ndarray, panel_frame: np.ndarray) -> np.ndarray:
    sim_img = Image.fromarray(sim_frame, mode="RGB")
    panel_img = Image.fromarray(panel_frame, mode="RGB")
    if panel_img.width != sim_img.width:
        panel_img = panel_img.resize((sim_img.width, panel_img.height), Image.Resampling.BILINEAR)
    combined = Image.new("RGB", (sim_img.width, sim_img.height + panel_img.height), (0, 0, 0))
    combined.paste(sim_img, (0, 0))
    combined.paste(panel_img, (0, sim_img.height))
    return np.asarray(combined, dtype=np.uint8)


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError(f"No frames to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (int(width), int(height)),
    )
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def render_policy_visualization(
    namespace: dict[str, Any],
    model: Any,
    run_name: str,
    env_config: dict[str, Any],
    output_dir: Path,
    *,
    episodes: int,
    max_steps: int,
    grid_size: int,
    seed: int,
    fps: int,
) -> list[dict[str, Any]]:
    env_cls = namespace["LanelessContinuousRLTargetEnv"]
    config = dict(env_config)
    config["real_time_rendering"] = False
    env = env_cls(config=config, render_mode="rgb_array")
    inspector = SACPolicyInspector(model, env, grid_size=grid_size)
    rows: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            obs, _ = env.reset(seed=seed + episode_index)
            terminated = truncated = False
            total_reward = 0.0
            steps = 0
            frames: list[np.ndarray] = []

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                sim_frame = capture_frame(env)
                panel_frame = make_policy_panel(
                    inspector,
                    obs,
                    action,
                    reward_so_far=total_reward,
                    step=steps,
                    episode=episode_index + 1,
                    width=sim_frame.shape[1],
                )
                frames.append(compose_frame(sim_frame, panel_frame))

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if steps >= max_steps:
                    truncated = True

            video_path = output_dir / f"episode_{episode_index + 1:02d}_sac_policy_panel.mp4"
            write_video(frames, video_path, fps=fps)
            preview_path = output_dir / f"episode_{episode_index + 1:02d}_preview.png"
            Image.fromarray(frames[0]).save(preview_path)
            row = {
                "episode": episode_index + 1,
                "seed": seed + episode_index,
                "steps": steps,
                "reward": total_reward,
                "collision": bool(getattr(env.unwrapped.vehicle, "crashed", False)),
                "video_path": str(video_path),
                "preview_path": str(preview_path),
                "model_run": run_name,
            }
            rows.append(row)
            print(
                f"[saved] episode={row['episode']} steps={steps} reward={total_reward:.2f} "
                f"collision={row['collision']} video={video_path}"
            )
    finally:
        env.close()

    return rows


def save_metrics(rows: list[dict[str, Any]], output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "render_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "render_summary.json").write_text(json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save laneless SAC continuous policy visualizations to disk.")
    parser.add_argument("--run-name", default=None, help="Preferred SAC run name. Falls back to available SAC runs.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--seed", type=int, default=32042)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    sys.path.insert(0, str(project_root))
    namespace = load_notebook_namespace(project_root)
    model, run_name, env_config, model_path = load_sac_model(namespace, args.run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(namespace["RESULTS_DIR"])
        / "sac_continuous_policy_visualization"
        / f"{run_name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Loaded SAC model: {model_path}")
    print(f"Saving visualizations to: {output_dir}")
    rows = render_policy_visualization(
        namespace,
        model,
        run_name,
        env_config,
        output_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        grid_size=args.grid_size,
        seed=args.seed,
        fps=args.fps,
    )
    save_metrics(
        rows,
        output_dir,
        {
            "project_root": str(project_root),
            "model_run": run_name,
            "model_path": str(model_path),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "grid_size": args.grid_size,
            "seed": args.seed,
            "fps": args.fps,
        },
    )
    print(f"Done. Summary: {output_dir / 'render_summary.json'}")


if __name__ == "__main__":
    main()
