from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym


ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ENV_ROOT = ROOT / "HighwayEnv"

if str(HIGHWAY_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGHWAY_ENV_ROOT))

import highway_env  # noqa: F401
from highway_env.road.lane import LineType


def remove_lane_markings(env) -> None:
    """Hide lane markings, keeping only the outer road boundaries visible."""
    lanes = env.unwrapped.road.network.lanes_list()
    for lane in lanes:
        lane.line_types = [LineType.NONE, LineType.NONE]
    if lanes:
        lanes[0].line_types[0] = LineType.CONTINUOUS_LINE
        lanes[-1].line_types[1] = LineType.CONTINUOUS_LINE


def main() -> None:
    episodes = 5
    config = {
        "lanes_count": 3,
        "vehicles_count": 50,
    }
    env = gym.make("highway-v0", config=config, render_mode="human")

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=episode)
            remove_lane_markings(env)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
