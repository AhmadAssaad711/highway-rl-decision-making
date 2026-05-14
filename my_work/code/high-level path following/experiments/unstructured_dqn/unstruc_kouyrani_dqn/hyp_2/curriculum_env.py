"""
Curriculum environment helpers for the hyp_2 Kourani DQN study.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence
import sys

import gymnasium as gym
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from highway_env_extension import make_unstructured_kourani_env


def clone_scenarios(scenarios: Sequence[dict]) -> list[dict]:
    return [deepcopy(scenario) for scenario in scenarios]


class CurriculumScenarioWrapper(gym.Wrapper):
    """
    Sample one scenario uniformly on reset and apply its config before the episode.
    """

    def __init__(
        self,
        env: gym.Env,
        scenarios: Sequence[dict],
        split_name: str,
    ) -> None:
        super().__init__(env)
        if not scenarios:
            raise ValueError("CurriculumScenarioWrapper requires at least one scenario")

        self.scenarios = clone_scenarios(scenarios)
        self.split_name = str(split_name)
        self.current_scenario: dict | None = None
        self.current_scenario_index: int | None = None
        self._scenario_rng = np.random.default_rng()

    def set_scenarios(self, scenarios: Sequence[dict], split_name: str | None = None) -> None:
        if not scenarios:
            raise ValueError("Scenario list must not be empty")
        self.scenarios = clone_scenarios(scenarios)
        if split_name is not None:
            self.split_name = str(split_name)

    def _decorate_info(self, info: dict | None) -> dict:
        payload = dict(info or {})
        scenario = self.current_scenario or {}
        payload.update(
            {
                "scenario_name": scenario.get("name"),
                "scenario_description": scenario.get("description"),
                "split": self.split_name,
            }
        )
        return payload

    def _sample_scenario(self) -> dict:
        self.current_scenario_index = int(self._scenario_rng.integers(len(self.scenarios)))
        self.current_scenario = deepcopy(self.scenarios[self.current_scenario_index])
        return self.current_scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._scenario_rng = np.random.default_rng(seed)

        scenario = self._sample_scenario()
        self.env.unwrapped.configure(dict(scenario["config"]))
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, self._decorate_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self._decorate_info(info)


def make_curriculum_env(
    scenarios: Sequence[dict],
    render_mode: str = "rgb_array",
    split_name: str = "curriculum",
):
    env = make_unstructured_kourani_env(render_mode=render_mode)
    return CurriculumScenarioWrapper(
        env=env,
        scenarios=scenarios,
        split_name=split_name,
    )
