from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ENV_ROOT = ROOT / "HighwayEnv"

for path in (ROOT, HIGHWAY_ENV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Highway_Scenario.laneless.laneless_highway_env import (
    LanelessHighwayEnv,
    LanelessIDMVehicle,
)


def make_ego_npc(env: LanelessHighwayEnv) -> None:
    """Replace the externally controlled ego with the laneless traffic model."""
    ego = env.vehicle
    npc_ego = LanelessIDMVehicle(
        env.road,
        ego.position,
        heading=ego.heading,
        speed=ego.speed,
        target_lane_index=ego.lane_index,
        target_speed=env.config["speed_limit"],
        road_width=env.config["road_width"],
    )

    ego_index = env.road.vehicles.index(ego)
    env.road.vehicles[ego_index] = npc_ego
    env.vehicle = npc_ego


def main() -> None:
    episodes = 3
    fps = 10
    policy_fps = 1
    env = LanelessHighwayEnv(
        config={
            "policy_frequency": policy_fps,
            "simulation_frequency": fps,
            "real_time_rendering": True,
        },
        render_mode="human",
    )

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=episode)
            make_ego_npc(env)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(None)
    finally:
        env.close()


if __name__ == "__main__":
    main()
