from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPO_SRC_DIR = PROJECT_ROOT / "src" / "deep_learning" / "Elurant_PPO"
if str(PPO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PPO_SRC_DIR))

from ppo_overtake_lab import (
    ExperimentConfig,
    PPOConfig,
    RewardConfig,
    ScenarioConfig,
    train_overtake_agent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickstart runner for the PPO overtaking lab")
    parser.add_argument("--name", default="overtake_quickstart", help="Experiment name")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Torch device")
    return parser.parse_args()


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        name=args.name,
        reward=RewardConfig(
            collision_penalty=-120.0,
            offroad_penalty=-120.0,
            speed_weight=28.0,
            progress_bonus=24.0,
            success_bonus=100.0,
            steering_penalty=0.35,
            blocked_in_right_penalty=-4.0,
            blocked_overtake_bonus=4.5,
            keep_right_bonus=1.5,
            unsafe_headway_penalty=-8.0,
            normalize_reward=False,
        ),
        scenario=ScenarioConfig(
            lanes_count=3,
            vehicles_per_lane=5,
            observation_vehicles=16,
            duration=50,
            simulation_frequency=20,
            policy_frequency=2,
            steering_range=0.01,
            ego_speed_range=(24.0, 26.0),
            other_speed_range=(18.0, 22.0),
            spawn_lead_range=(60.0, 110.0),
            spawn_gap_range=(25.0, 45.0),
            initial_lane_id=2,
        ),
        ppo=PPOConfig(
            timesteps=args.timesteps,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            batch_size=256,
            n_steps=512,
            n_epochs=10,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            device=args.device,
            policy_pi=(256, 256),
            policy_vf=(256, 256),
        ),
    )


def main() -> None:
    args = parse_args()
    experiment = build_experiment(args)
    _, evaluation = train_overtake_agent(experiment)
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
