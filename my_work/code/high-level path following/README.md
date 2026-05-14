# High-Level Path Following

Cleaned repo layout for the highway driving experiments:

- `notebooks/baseline_dqn/`: compact baseline DQN notebook based on the Leurent/rl-agents setup.
- `notebooks/adaptive_lower_controller/`: DQN notebook for testing the TTC-adaptive longitudinal controller.
- `notebooks/attention_dqn/`: DQN notebook for attention experiments.
- `notebooks/ppo/`: interactive PPO notebooks.
- `notebooks/planning/`: interactive planning notebooks such as CEM.
- `src/basic/`: tabular Q-learning baselines.
- `src/deep_learning/`: DQN and PPO training code.
- `scripts/ppo/`: small command-line entry points.
- `experiments/unstructured_dqn/`: unstructured-traffic experiment code.
- `artifacts/`: generated models, logs, videos, summaries, and plots.

Conventions:

- Source code lives under `src/`.
- Notebooks live under `notebooks/`.
- Generated outputs should go under `artifacts/` and stay out of git.
- Rebuild the DQN experiment notebooks with `python notebooks/build_dqn_experiment_notebooks.py`.
