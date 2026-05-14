# Notebooks

Notebook folders are organized by experiment purpose:

- `baseline_dqn/`: compact baseline DQN runner based on the Leurent/rl-agents setup.
- `adaptive_lower_controller/`: baseline DQN with the TTC-adaptive longitudinal controller enabled.
- `attention_dqn/`: DQN with the ego-attention feature extractor.
- `ppo/`: PPO experiments.
- `planning/`: planning experiments such as CEM.

Generated models, plots, logs, and videos should go under `artifacts/`.
