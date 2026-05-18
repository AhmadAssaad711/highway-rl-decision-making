# High-Level Highway RL

Notebook workspace for reinforcement-learning experiments in autonomous highway decision making.

## Scope

This work focuses on decision-level behavior:

- structured highway policies
- congestion-aware decision making
- reward and safety-factor studies
- laneless and unstructured highway environments
- planning baselines for comparison

## Notebooks

| Folder | Purpose |
| --- | --- |
| [`structured_highway/`](notebooks/structured_highway/) | DQN, attention DQN, PPO, hybrid PPO, and reproduction notebooks. |
| [`congested_traffic/`](notebooks/congested_traffic/) | Dense traffic policy experiments and reward-safety studies. |
| [`laneless_unstructured/`](notebooks/laneless_unstructured/) | Laneless highway environment experiments. |
| [`planning/`](notebooks/planning/) | CEM planning trials used as decision-level comparisons. |

Full notebook list: [`notebooks/README.md`](notebooks/README.md)

## Install

```powershell
python -m pip install -r requirements.txt
```

## Notes

This is a notebook-only public version. Generated outputs, source scripts, old practice work, and vendor copies are intentionally excluded.
