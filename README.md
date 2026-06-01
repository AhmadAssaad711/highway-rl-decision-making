<div align="center">

# Highway RL Decision Making

**Baselines, improvements, and extensions for structured, congested, and laneless highway environments.**

[Project Workspace](highway-rl-decision-making/) | [Notebook Map](highway-rl-decision-making/notebooks/) | [Paper](highway-rl-decision-making/docs/paper/highway-rl-decision-making-paper.pdf) | [Setup](#setup)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Notebook-first](https://img.shields.io/badge/Notebook--first-research-111111?style=flat-square)
![RL](https://img.shields.io/badge/RL-highway_decision_making-0B7285?style=flat-square)
![Paper](https://img.shields.io/badge/Paper-included-495057?style=flat-square)

</div>

---

## Focus

This repository presents work on high-level decision making for autonomous driving:

- lane-based highway policies
- dense traffic behavior
- safety-aware reward design
- laneless and unstructured highway environments
- planning comparisons for decision-level behavior

The repo is intentionally notebook-first. Each notebook captures a specific experiment, baseline, reproduction, or environment study.

## Research Flow

1. Establish baselines in structured highway environments.
2. Improve those baselines with attention models, PPO variants, and reward-safety studies.
3. Extend and test the ideas in harder settings: congested traffic and laneless highway environments.

Associated paper: [`highway-rl-decision-making-paper.pdf`](highway-rl-decision-making/docs/paper/highway-rl-decision-making-paper.pdf)

## Research Map

| Area | Core Question | Entry Points |
| --- | --- | --- |
| Structured highway RL | Baseline DQN and PPO behavior in lane-based highway settings. | [`baseline_dqn`](highway-rl-decision-making/notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb), [`attention_dqn`](highway-rl-decision-making/notebooks/structured_highway/attention_dqn/attention_dqn.ipynb), [`PPO_trials`](highway-rl-decision-making/notebooks/structured_highway/ppo/PPO_trials.ipynb) |
| Attention and hybrid PPO | Improvements over baseline policy structure. | [`Attention_PPO_baseline`](highway-rl-decision-making/notebooks/structured_highway/ppo/Attention_PPO_baseline.ipynb), [`Hybrid_PPO_baseline`](highway-rl-decision-making/notebooks/structured_highway/ppo/Hybrid_PPO_baseline.ipynb) |
| Congested traffic | Baseline extensions under dense traffic and safety constraints. | [`congested_traffic_policy`](highway-rl-decision-making/notebooks/congested_traffic/congested_traffic_policy.ipynb), [`congested_traffic_policy_v2`](highway-rl-decision-making/notebooks/congested_traffic/congested_traffic_policy_v2.ipynb), [`potential_field_reward_test`](highway-rl-decision-making/notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb) |
| Laneless environments | Baseline extensions when lane assumptions break down. | [`laneless_highway_env`](highway-rl-decision-making/notebooks/laneless_unstructured/laneless_highway_env.ipynb) |
| Planning comparison | How do planning-based methods compare as decision baselines? | [`CEM_planning_trials`](highway-rl-decision-making/notebooks/planning/CEM_planning_trials.ipynb) |

## Repository Layout

```text
highway-rl-decision-making/
  README.md
  requirements.txt
  docs/
    paper/
  notebooks/
    structured_highway/
    congested_traffic/
    laneless_unstructured/
    planning/
```

## What Is Included

- clean notebook portfolio
- associated paper
- reproducible environment requirements
- grouped experiments by research theme
- public-facing structure around Highway RL Decision Making

## What Is Excluded

- unrelated practice problems
- old experimental folders
- vendored external repositories
- generated logs, videos, checkpoints, and artifacts
- material outside decision-level highway RL

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r highway-rl-decision-making\requirements.txt
```

Then open the notebooks from:

```text
highway-rl-decision-making/notebooks/
```
