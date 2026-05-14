"""
Test model_DQN_15 on the erratic-drivers environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from erratic_drivers_dqn import MODEL_COMPATIBLE_CONFIG, load_model
from erratic_drivers_env import make_erratic_drivers_env


def run_test(
    episodes: int = 10,
    max_steps: int = 300,
    deterministic: bool = True,
    render_mode: str = "human",
    device: str = "auto",
    model_path: str | None = None,
) -> None:
    env = make_erratic_drivers_env(
        render_mode=render_mode,
        config=MODEL_COMPATIBLE_CONFIG,
    )
    model = load_model(env=env, model_path=model_path, device=device)

    rewards: list[float] = []
    crash_count = 0
    step_counts: list[int] = []

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
            rewards.append(total_reward)
            step_counts.append(step_count)
            crash_count += int(crashed)
            print(
                f"Test episode {episode + 1}: reward={total_reward:.2f}, "
                f"steps={step_count}, crashed={crashed}"
            )
    finally:
        env.close()

    mean_reward = sum(rewards) / len(rewards)
    reward_variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    reward_std = reward_variance ** 0.5
    mean_steps = sum(step_counts) / len(step_counts)
    crash_rate = crash_count / len(rewards)

    print("\nErratic-drivers test summary")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Reward std: {reward_std:.2f}")
    print(f"Mean steps: {mean_steps:.1f}")
    print(f"Crash rate: {crash_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test model_DQN_15 on the erratic-drivers environment."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.headless else "human"
    run_test(
        episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        render_mode=render_mode,
        device=args.device,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
