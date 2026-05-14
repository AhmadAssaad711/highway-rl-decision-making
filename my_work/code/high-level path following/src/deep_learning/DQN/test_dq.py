"""
Visual and numerical evaluation of the trained Kourani DQN agent.
Run: python src/deep_learning/DQN/test_dq.py
"""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


KOURANI_MODEL_PATH = Path(
    r"C:\Users\aha173\OneDrive - American University of Beirut\AGV research\high-level path following\logs\model.zip"
)


def main():
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
    }

    env = gym.make("highway-v0", render_mode="human", config=config)

    model = DQN.load(str(KOURANI_MODEL_PATH), env=env)

    # Visual evaluation
    print("Running 5 visual episodes...\n")
    for ep in range(5):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    # Numerical evaluation
    print("\nRunning numerical evaluation (10 episodes)...")
    mean, std = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean:.2f} +- {std:.2f}")

    env.close()


if __name__ == "__main__":
    main()
