# AGV Research

Research workspace for AGV and autonomous-driving reinforcement learning experiments. The repository collects current work, exploratory practice problems, adapted traffic environments, and previous project material.

## Repository Layout

```text
.
+-- my_work/
|   +-- code/
|       +-- high-level path following/        # Main highway/path-following experiments
|       +-- path-following codes/             # Racetrack path-following RL comparisons
|       +-- RL practice problems/             # Small RL exercises and custom environments
|       +-- Unstructured-Traffic_Environment/ # Highway-env based traffic environment work
+-- previous_work/
|   +-- fall 2025/
|       +-- pendulum#/                        # Earlier pendulum RL/control experiments
+-- external_work/                            # External references, ignored by git
```

## Main Areas

### High-Level Path Following

Path: `my_work/code/high-level path following/`

The main cleaned workspace for highway driving experiments. It includes:

- `src/basic/`: tabular Q-learning baselines.
- `src/deep_learning/`: DQN and PPO training code.
- `notebooks/`: experiment notebooks for DQN, PPO, planning, attention, congestion, and lower-controller studies.
- `experiments/unstructured_dqn/`: unstructured-traffic DQN experiments.
- `scripts/ppo/`: command-line PPO entry points.

See the local README in that folder for the subproject conventions.

### Path-Following Codes

Path: `my_work/code/path-following codes/`

Compares two tabular Q-learning steering controllers on a `highway-env` racetrack:

- `2_basic_rl_agent/basic_rl_agent.py`
- `3_physics_informed_agent/physics_informed_agent.py`

The physics-informed variant adds lane curvature to the state representation.

### RL Practice Problems

Path: `my_work/code/RL practice problems/`

Small reinforcement learning exercises and prototypes, including Frozen Lake, Blackjack, Flappy Bird, custom Gymnasium environments, and highway-env trials.

### Previous Work

Path: `previous_work/fall 2025/pendulum#/`

Earlier pendulum RL/control experiments, including Q-learning, policy gradient, model-based control, LQR tests, plotting, animation, and web/pygame launch scripts.

## Setup

Most code is Python-based. A typical setup from the repository root is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r "my_work\code\high-level path following\requirements.txt"
```

Some older or standalone scripts may need extra packages depending on the experiment being run.

## Running Examples

Run the high-level path-following scripts from their subproject folder:

```powershell
cd "my_work\code\high-level path following"
python "scripts\ppo\run_overtake_quickstart.py"
```

Run the racetrack path-following comparison scripts:

```powershell
cd "my_work\code\path-following codes"
python "2_basic_rl_agent\basic_rl_agent.py"
python "3_physics_informed_agent\physics_informed_agent.py"
```

Run the pendulum work:

```powershell
cd "previous_work\fall 2025\pendulum#"
python q_learning.py
python policy_gradient.py
```

## Generated Outputs

Generated experiment files should stay out of git. The `.gitignore` already excludes common output locations and file types, including:

- `artifacts/`, `results/`, `runs/`, `wandb/`, `checkpoints/`, `models/`
- model files such as `.pt`, `.pth`, `.ckpt`, `.pkl`
- videos, images, spreadsheets, and large numeric arrays

Use each subproject's `artifacts/` or `results/` directory for plots, trained models, logs, videos, and summaries.

## Notes

- Prefer working from the relevant subproject folder when running scripts.
- Keep source code under the existing `src/`, `scripts/`, `notebooks/`, or experiment directories.
- Subproject READMEs contain more specific commands and conventions where available.
