<div align="center">

# High-Level Highway RL

**Reinforcement learning for autonomous highway decision making in structured, congested, and laneless traffic.**

[Project Workspace](high-level-highway-rl/) | [Notebook Map](high-level-highway-rl/notebooks/) | [Setup](#setup)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Notebook-first](https://img.shields.io/badge/Notebook--first-research-111111?style=flat-square)
![RL](https://img.shields.io/badge/RL-highway_decision_making-0B7285?style=flat-square)

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

## Research Map

| Area | Core Question | Entry Points |
| --- | --- | --- |
| Structured highway RL | How do DQN and PPO policies behave in lane-based highway settings? | [`baseline_dqn`](high-level-highway-rl/notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb), [`attention_dqn`](high-level-highway-rl/notebooks/structured_highway/attention_dqn/attention_dqn.ipynb), [`PPO_trials`](high-level-highway-rl/notebooks/structured_highway/ppo/PPO_trials.ipynb) |
| Attention and hybrid PPO | Can policy structure improve highway decision quality? | [`Attention_PPO_baseline`](high-level-highway-rl/notebooks/structured_highway/ppo/Attention_PPO_baseline.ipynb), [`Hybrid_PPO_baseline`](high-level-highway-rl/notebooks/structured_highway/ppo/Hybrid_PPO_baseline.ipynb) |
| Congested traffic | How should policies react under dense traffic and safety constraints? | [`congested_traffic_policy`](high-level-highway-rl/notebooks/congested_traffic/congested_traffic_policy.ipynb), [`congested_traffic_policy_v2`](high-level-highway-rl/notebooks/congested_traffic/congested_traffic_policy_v2.ipynb), [`reward_safety_factor_study`](high-level-highway-rl/notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb) |
| Laneless environments | How does decision making change when lane assumptions break down? | [`laneless_highway_env`](high-level-highway-rl/notebooks/laneless_unstructured/laneless_highway_env.ipynb) |
| Planning comparison | How do planning-based methods compare as decision baselines? | [`CEM_planning_trials`](high-level-highway-rl/notebooks/planning/CEM_planning_trials.ipynb) |

## Repository Layout

```text
high-level-highway-rl/
  README.md
  requirements.txt
  notebooks/
    structured_highway/
    congested_traffic/
    laneless_unstructured/
    planning/
```

## What Is Included

- clean notebook portfolio
- reproducible environment requirements
- grouped experiments by research theme
- public-facing structure around high-level highway RL

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
python -m pip install -r high-level-highway-rl\requirements.txt
```

Then open the notebooks from:

```text
high-level-highway-rl/notebooks/
```
