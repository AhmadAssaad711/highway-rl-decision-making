"""
Baseline DQN training script using Stable-Baselines3 with highway-env.
Copied from the project Leurent DQN baseline and scoped to this folder.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = Path(__file__).resolve().parent / "elurant_dqn.zip"


def make_env(render_mode: str = "rgb_array"):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
    }
    return gym.make("highway-v0", render_mode=render_mode, config=config)


def train(total_timesteps: int = 200000) -> Path:
    env = make_env(render_mode="rgb_array")
    env.reset()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        target_update_interval=50,
        verbose=1,
    )

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    model.save(str(MODEL_PATH.with_suffix("")))
    print(f"Model saved to {MODEL_PATH}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(
        f"Evaluation over 10 episodes: mean reward = {mean_reward:.2f}, "
        f"std = {std_reward:.2f}"
    )

    env.close()
    return MODEL_PATH


def main() -> None:
    train()


if __name__ == "__main__":
    main()
