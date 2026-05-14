"""
Scenario sets for the hyp_2 curriculum-learning study.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from scenarios import SCENARIOS

DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_STAGE_SPLIT = (20_000, 30_000, 50_000)
DEFAULT_STAGE_WEIGHTS = (2, 3, 5)

_STAGE_DEFINITIONS = [
    {
        "name": "stage_1_easy",
        "label": "easy",
        "description": "Open and moderate-flow traffic before denser mixtures are introduced.",
        "scenario_names": [
            "01_open_flow",
            "02_balanced_medium",
            "05_sparse_five_lane",
        ],
    },
    {
        "name": "stage_2_medium",
        "label": "medium",
        "description": "Carry forward easy cases and add denser and longer mixed-traffic settings.",
        "scenario_names": [
            "01_open_flow",
            "02_balanced_medium",
            "05_sparse_five_lane",
            "03_dense_commuter",
            "06_busy_three_lane",
            "07_long_horizon_mix",
            "09_cautious_platoon",
        ],
    },
    {
        "name": "stage_3_hard",
        "label": "hard",
        "description": "Full curriculum with all available training scenarios.",
        "scenario_names": [scenario["name"] for scenario in SCENARIOS],
    },
]

_HOLDOUT_SCENARIOS = [
    {
        "name": "11_medium_three_lane_mix",
        "description": "Three-lane medium-density mix not seen during curriculum training.",
        "config": {
            "lanes_count": 3,
            "vehicles_count": 40,
            "vehicles_density": 1.10,
            "ego_spacing": 2.2,
            "duration": 50,
            "aggressive_vehicle_ratio": 0.15,
            "cautious_vehicle_ratio": 0.25,
        },
    },
    {
        "name": "12_dense_five_lane_aggressive",
        "description": "Wide, dense, aggressive traffic blend reserved for holdout testing.",
        "config": {
            "lanes_count": 5,
            "vehicles_count": 65,
            "vehicles_density": 1.45,
            "ego_spacing": 1.6,
            "duration": 45,
            "aggressive_vehicle_ratio": 0.35,
            "cautious_vehicle_ratio": 0.10,
        },
    },
    {
        "name": "13_open_four_lane_cautious_long",
        "description": "Longer open-flow episode with a cautious-dominant population.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 35,
            "vehicles_density": 0.85,
            "ego_spacing": 2.4,
            "duration": 55,
            "aggressive_vehicle_ratio": 0.10,
            "cautious_vehicle_ratio": 0.35,
        },
    },
    {
        "name": "14_long_three_lane_aggressive",
        "description": "Compressed three-lane road with a longer horizon and more aggressive flow.",
        "config": {
            "lanes_count": 3,
            "vehicles_count": 50,
            "vehicles_density": 1.25,
            "ego_spacing": 1.7,
            "duration": 60,
            "aggressive_vehicle_ratio": 0.30,
            "cautious_vehicle_ratio": 0.10,
        },
    },
    {
        "name": "15_balanced_five_lane_long",
        "description": "Balanced five-lane long-horizon holdout combination.",
        "config": {
            "lanes_count": 5,
            "vehicles_count": 55,
            "vehicles_density": 1.15,
            "ego_spacing": 1.9,
            "duration": 60,
            "aggressive_vehicle_ratio": 0.20,
            "cautious_vehicle_ratio": 0.20,
        },
    },
]

SEEN_SCENARIOS = deepcopy(SCENARIOS)
HOLDOUT_SCENARIOS = deepcopy(_HOLDOUT_SCENARIOS)
ALL_EVALUATION_SCENARIOS = deepcopy(SEEN_SCENARIOS) + deepcopy(HOLDOUT_SCENARIOS)

_SCENARIO_LOOKUP = {
    scenario["name"]: deepcopy(scenario)
    for scenario in ALL_EVALUATION_SCENARIOS
}


def _resolve_named_scenarios(names: list[str]) -> list[dict]:
    missing = sorted(name for name in names if name not in _SCENARIO_LOOKUP)
    if missing:
        raise KeyError(f"Unknown scenario names: {missing}")
    return [deepcopy(_SCENARIO_LOOKUP[name]) for name in names]


def get_curriculum_stages() -> list[dict]:
    return [
        {
            "name": stage["name"],
            "label": stage["label"],
            "description": stage["description"],
            "scenario_names": list(stage["scenario_names"]),
            "scenarios": _resolve_named_scenarios(stage["scenario_names"]),
        }
        for stage in _STAGE_DEFINITIONS
    ]


CURRICULUM_STAGES = get_curriculum_stages()


def get_seen_scenarios() -> list[dict]:
    return deepcopy(SEEN_SCENARIOS)


def get_holdout_scenarios() -> list[dict]:
    return deepcopy(HOLDOUT_SCENARIOS)


def get_evaluation_scenarios(split: str = "all") -> dict[str, list[dict]]:
    if split == "seen":
        return {"seen": get_seen_scenarios()}
    if split == "unseen":
        return {"unseen": get_holdout_scenarios()}
    if split == "all":
        return {
            "seen": get_seen_scenarios(),
            "unseen": get_holdout_scenarios(),
        }
    raise ValueError(f"Unsupported split: {split}")


def filter_named_scenarios(
    scenarios_by_split: dict[str, list[dict]],
    selected_names: list[str] | None,
) -> dict[str, list[dict]]:
    if not selected_names:
        return scenarios_by_split

    selected = set(selected_names)
    filtered = {
        split_name: [
            scenario
            for scenario in scenarios
            if scenario["name"] in selected
        ]
        for split_name, scenarios in scenarios_by_split.items()
    }
    if not any(filtered.values()):
        raise ValueError(f"No scenarios matched filter: {sorted(selected)}")
    return {split_name: scenarios for split_name, scenarios in filtered.items() if scenarios}


def build_scenario_sets_payload(stage_timesteps: list[int] | tuple[int, ...] | None = None) -> dict:
    payload = {
        "seen_scenarios": get_seen_scenarios(),
        "holdout_scenarios": get_holdout_scenarios(),
        "curriculum_stages": [],
    }

    stages = get_curriculum_stages()
    for index, stage in enumerate(stages):
        stage_payload = {
            "name": stage["name"],
            "label": stage["label"],
            "description": stage["description"],
            "scenario_names": stage["scenario_names"],
            "scenarios": stage["scenarios"],
        }
        if stage_timesteps is not None:
            stage_payload["timesteps"] = int(stage_timesteps[index])
        payload["curriculum_stages"].append(stage_payload)

    return payload
