"""
Run model_DQN_15 on the erratic-drivers environment with rendering.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from stable_baselines3 import DQN

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from erratic_drivers_env import make_erratic_drivers_env

LOCAL_MODEL_ZIP = CURRENT_DIR / "model_DQN_15.zip"
LOCAL_MODEL_DIR = CURRENT_DIR / "model_DQN_15"
EXTERNAL_MODEL_DIR = CURRENT_DIR.parent.parent / "model_DQN_15"

MODEL_COMPATIBLE_CONFIG = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "DiscreteMetaAction"},
}


def zip_model_directory(source_dir: Path, destination_zip: Path) -> Path:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
    return destination_zip


def ensure_model_zip() -> Path:
    if LOCAL_MODEL_ZIP.exists():
        return LOCAL_MODEL_ZIP

    for candidate_dir in (LOCAL_MODEL_DIR, EXTERNAL_MODEL_DIR):
        if candidate_dir.exists() and candidate_dir.is_dir():
            return zip_model_directory(candidate_dir, LOCAL_MODEL_ZIP)

    raise FileNotFoundError(
        "Could not find model_DQN_15. Expected either "
        f"{LOCAL_MODEL_ZIP} or an extracted model directory at "
        f"{LOCAL_MODEL_DIR} / {EXTERNAL_MODEL_DIR}."
    )


def load_model(env, model_path: str | Path | None = None, device: str = "auto") -> DQN:
    resolved_model_path = Path(model_path) if model_path else ensure_model_zip()
    return DQN.load(str(resolved_model_path), env=env, device=device)


def run_visual_policy(
    episodes: int = 5,
    max_steps: int = 300,
    deterministic: bool = True,
    render_mode: str = "human",
    device: str = "auto",
    model_path: str | Path | None = None,
) -> None:
    env = make_erratic_drivers_env(
        render_mode=render_mode,
        config=MODEL_COMPATIBLE_CONFIG,
    )
    model = load_model(env=env, model_path=model_path, device=device)

    rewards: list[float] = []
    crash_count = 0

    try:
        for episode in range(episodes):
            observation, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0

            while not (done or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                step_count += 1

                if render_mode == "human":
                    env.render()

                if max_steps and step_count >= max_steps:
                    truncated = True

            crashed = bool(info.get("crashed", False))
            crash_count += int(crashed)
            rewards.append(total_reward)
            print(
                f"Episode {episode + 1}: reward={total_reward:.2f}, "
                f"steps={step_count}, crashed={crashed}"
            )
    finally:
        env.close()

    mean_reward = sum(rewards) / len(rewards)
    print(
        f"\nErratic-drivers summary: mean_reward={mean_reward:.2f}, "
        f"episodes={episodes}, crashes={crash_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model_DQN_15 on the erratic-drivers environment."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.headless else "human"
    run_visual_policy(
        episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        render_mode=render_mode,
        device=args.device,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
