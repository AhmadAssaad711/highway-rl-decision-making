from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from paper_ppo_reproduction import MODELS_DIR, make_paper_env, scenario_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-render evaluation for the paper PPO agent")
    parser.add_argument("--episodes", type=int, default=5, help="Number of visual episodes to run")
    parser.add_argument("--seed", type=int, default=100, help="Base seed for evaluation episodes")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "best_model" / "best_model.zip",
        help="Path to the PPO zip model",
    )
    parser.add_argument(
        "--scenario",
        choices=("base", "adapt1", "adapt2"),
        default="base",
        help="Which evaluation scenario to render",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Delay in seconds after each decision step",
    )
    return parser.parse_args()


def resolve_config_overrides(scenario: str):
    if scenario == "adapt1":
        return scenario_overrides(lanes_count=4, vehicles_per_lane=5)
    if scenario == "adapt2":
        return scenario_overrides(lanes_count=2, vehicles_per_lane=10)
    return None


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not model_path.exists():
        fallback_path = MODELS_DIR / "paper_ppo_reproduction.zip"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Could not find model at {args.model_path} or {fallback_path}")

    config_overrides = resolve_config_overrides(args.scenario)
    model = PPO.load(str(model_path))

    print(f"Loaded model: {model_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}\n")

    for episode_idx in range(args.episodes):
        env = make_paper_env(render_mode="human", config_overrides=config_overrides)
        obs, _ = env.reset(seed=args.seed + episode_idx)
        terminated = False
        truncated = False
        total_reward = 0.0
        final_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            env.render()
            if args.sleep > 0:
                time.sleep(args.sleep)

        metrics = final_info.get("episode_metrics", {})
        print(
            f"Episode {episode_idx + 1}: "
            f"reward={total_reward:.2f}, "
            f"collision={int(metrics.get('collision', 0.0))}, "
            f"offroad={int(metrics.get('offroad', 0.0))}, "
            f"safe_completion={int(metrics.get('safe_completion', 0.0))}, "
            f"overtaken_count={metrics.get('overtaken_count', 0.0):.0f}, "
            f"seconds={metrics.get('episode_length_seconds', 0.0):.1f}"
        )
        env.close()


if __name__ == "__main__":
    main()
