# Different Reward Structure Study

This folder now contains a single focused experiment:

- `off_road_penalty.py`

What it does:

- trains PPO for `200000` timesteps
- uses the baseline highway reward coefficients
- adds an explicit normalized off-road terminal penalty equal to the collision penalty
- writes TensorBoard logs, model artifacts, monitor logs, and a small JSON summary

Reward configuration used:

- `collision_reward = -1.0`
- `offroad_penalty = -1.0`
- `high_speed_reward = 0.4`
- `right_lane_reward = 0.1`
- `normalize_reward = True`
- `offroad_terminal = True`

Important behavior:

- the first off-road step gets a negative reward contribution in the normalized reward formula
- the episode terminates immediately after that step
- there is no video recording in this experiment

Artifacts are written under:

- `r/m/` for the model and manifest
- `r/l/` for monitor logs
- `r/t/` for TensorBoard
- `r/summary.json` for the run summary

Run it with:

```powershell
python .\src\deep_learning\Elurant_PPO\diff_reward_structure\off_road_penalty.py
```
