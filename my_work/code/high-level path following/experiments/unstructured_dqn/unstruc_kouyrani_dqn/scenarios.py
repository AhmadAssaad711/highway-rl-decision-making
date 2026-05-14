"""
Scenario definitions for Kourani DQN generalization evaluation.
"""

from __future__ import annotations

SCENARIOS: list[dict] = [
    {
        "name": "01_open_flow",
        "description": "Light traffic with comfortable spacing on a 4-lane highway.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 30,
            "vehicles_density": 0.7,
            "ego_spacing": 2.5,
            "duration": 40,
            "aggressive_vehicle_ratio": 0.15,
            "cautious_vehicle_ratio": 0.15,
        },
    },
    {
        "name": "02_balanced_medium",
        "description": "Moderate mixed traffic close to the default highway flow.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 45,
            "vehicles_density": 1.0,
            "ego_spacing": 2.0,
            "duration": 40,
            "aggressive_vehicle_ratio": 0.20,
            "cautious_vehicle_ratio": 0.20,
        },
    },
    {
        "name": "03_dense_commuter",
        "description": "Dense commuter traffic with reduced spacing.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 60,
            "vehicles_density": 1.4,
            "ego_spacing": 1.6,
            "duration": 45,
            "aggressive_vehicle_ratio": 0.25,
            "cautious_vehicle_ratio": 0.15,
        },
    },
    {
        "name": "04_congested_heavy",
        "description": "Heavier congestion with more conservative traffic mixed in.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 75,
            "vehicles_density": 1.8,
            "ego_spacing": 1.2,
            "duration": 50,
            "aggressive_vehicle_ratio": 0.20,
            "cautious_vehicle_ratio": 0.30,
        },
    },
    {
        "name": "05_sparse_five_lane",
        "description": "Wide highway with lower density but more freedom to change lanes.",
        "config": {
            "lanes_count": 5,
            "vehicles_count": 35,
            "vehicles_density": 0.65,
            "ego_spacing": 2.5,
            "duration": 40,
            "aggressive_vehicle_ratio": 0.25,
            "cautious_vehicle_ratio": 0.10,
        },
    },
    {
        "name": "06_busy_three_lane",
        "description": "Narrower road where the same traffic volume feels more compressed.",
        "config": {
            "lanes_count": 3,
            "vehicles_count": 55,
            "vehicles_density": 1.5,
            "ego_spacing": 1.4,
            "duration": 45,
            "aggressive_vehicle_ratio": 0.20,
            "cautious_vehicle_ratio": 0.20,
        },
    },
    {
        "name": "07_long_horizon_mix",
        "description": "Longer episode horizon with a strongly mixed traffic population.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.1,
            "ego_spacing": 2.0,
            "duration": 60,
            "aggressive_vehicle_ratio": 0.30,
            "cautious_vehicle_ratio": 0.10,
        },
    },
    {
        "name": "08_aggressive_surge",
        "description": "Traffic dominated by faster and less polite vehicles.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.2,
            "ego_spacing": 1.8,
            "duration": 40,
            "aggressive_vehicle_ratio": 0.40,
            "cautious_vehicle_ratio": 0.05,
        },
    },
    {
        "name": "09_cautious_platoon",
        "description": "Traffic dominated by slower, higher-headway vehicles.",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.3,
            "ego_spacing": 1.8,
            "duration": 45,
            "aggressive_vehicle_ratio": 0.05,
            "cautious_vehicle_ratio": 0.40,
        },
    },
    {
        "name": "10_heavy_five_lane_mix",
        "description": "A wide but very busy highway with large population variance.",
        "config": {
            "lanes_count": 5,
            "vehicles_count": 70,
            "vehicles_density": 1.6,
            "ego_spacing": 1.5,
            "duration": 50,
            "aggressive_vehicle_ratio": 0.25,
            "cautious_vehicle_ratio": 0.20,
        },
    },
]
