"""Minimal guided DDPG-CBF replay buffer and actor loss.

The notebook still defines the environment, reward, and CBF shield. This module
overrides only the guided reward-plus-loss pieces used by Python scripts.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update


class CBFGuidedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    safe_actions: th.Tensor
    interventions: th.Tensor
    projection_jacobians: th.Tensor


class CBFGuidedReplayBuffer(ReplayBuffer):
    """Replay buffer with actor-scale CBF targets and local projection Jacobians."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        action_shape = (self.buffer_size, self.n_envs, self.action_dim)
        scalar_shape = (self.buffer_size, self.n_envs, 1)
        jacobian_shape = (self.buffer_size, self.n_envs, self.action_dim, self.action_dim)
        self.safe_actions = np.zeros(action_shape, dtype=np.float32)
        self.interventions = np.zeros(scalar_shape, dtype=np.float32)
        self.projection_jacobians = np.broadcast_to(
            np.eye(self.action_dim, dtype=np.float32),
            jacobian_shape,
        ).copy()

    @staticmethod
    def _read_safe_action(info: dict[str, Any]) -> Optional[np.ndarray]:
        if "safe_action_phys" in info:
            action = np.asarray(info["safe_action_phys"], dtype=np.float32).reshape(-1)
        elif "cbf_a_safe_x" in info and "cbf_a_safe_y" in info:
            action = np.asarray([info["cbf_a_safe_x"], info["cbf_a_safe_y"]], dtype=np.float32)
        else:
            return None
        if action.size < 2 or not np.all(np.isfinite(action[:2])):
            return None
        return action[:2].astype(np.float32)

    def _to_actor_scale(self, action_phys: np.ndarray) -> np.ndarray:
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)[: self.action_dim]
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)[: self.action_dim]
        action_phys = np.asarray(action_phys, dtype=np.float32).reshape(-1)[: self.action_dim]
        scaled = 2.0 * ((np.clip(action_phys, low, high) - low) / np.maximum(high - low, 1e-6)) - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _read_event_intervention(info: dict[str, Any]) -> bool:
        if "cbf_event_intervened" in info:
            return bool(info["cbf_event_intervened"])
        if "intervention" in info:
            return bool(info["intervention"])
        correction = info.get("cbf_correction_norm", info.get("correction_norm"))
        if correction is None:
            return False
        threshold = float(info.get("cbf_event_intervention_threshold", 0.03))
        return bool(float(correction) > threshold)

    def _read_projection_jacobian(
        self,
        info: dict[str, Any],
        raw_action_scaled: np.ndarray,
        safe_action_scaled: np.ndarray,
        intervened: bool,
    ) -> np.ndarray:
        identity = np.eye(self.action_dim, dtype=np.float32)

        for key in ["cbf_projection_jacobian_scaled", "projection_jacobian_scaled"]:
            if key not in info:
                continue
            candidate = np.asarray(info[key], dtype=np.float32)
            if candidate.shape == identity.shape and np.all(np.isfinite(candidate)):
                return candidate

        for key in ["cbf_active_constraint_rows_scaled", "active_constraint_rows_scaled"]:
            if key not in info:
                continue
            rows = np.asarray(info[key], dtype=np.float32).reshape(-1, self.action_dim)
            if rows.size == 0 or not np.all(np.isfinite(rows)):
                continue
            gram = rows @ rows.T
            try:
                projection = identity - rows.T @ np.linalg.pinv(gram, rcond=1e-5) @ rows
            except np.linalg.LinAlgError:
                continue
            projection = 0.5 * (projection + projection.T)
            if projection.shape == identity.shape and np.all(np.isfinite(projection)):
                return projection.astype(np.float32)

        if not intervened:
            return identity

        correction = np.asarray(raw_action_scaled - safe_action_scaled, dtype=np.float32).reshape(-1)[: self.action_dim]
        correction_norm = float(np.linalg.norm(correction))
        if not np.isfinite(correction_norm) or correction_norm <= 1e-6:
            return identity
        normal = correction / correction_norm
        projection = identity - np.outer(normal, normal)
        return projection.astype(np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        slot = self.pos
        raw_actions_scaled = np.asarray(action, dtype=np.float32).reshape((self.n_envs, self.action_dim))
        self.safe_actions[slot] = raw_actions_scaled

        for env_idx, info in enumerate(infos):
            safe_action_scaled = raw_actions_scaled[env_idx]
            safe_phys = self._read_safe_action(info)
            if safe_phys is not None:
                safe_action_scaled = self._to_actor_scale(safe_phys)
                self.safe_actions[slot, env_idx] = safe_action_scaled
            intervened = self._read_event_intervention(info)
            self.interventions[slot, env_idx, 0] = float(intervened)
            self.projection_jacobians[slot, env_idx] = self._read_projection_jacobian(
                info,
                raw_actions_scaled[env_idx],
                safe_action_scaled,
                intervened,
            )

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> CBFGuidedReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.safe_actions[batch_inds, env_indices, :],
            self.interventions[batch_inds, env_indices, :],
            self.projection_jacobians[batch_inds, env_indices, :, :],
        )
        return CBFGuidedReplayBufferSamples(*tuple(map(self.to_torch, data)))


class GuidedCBFDDPG(DDPG):
    """DDPG with standard critic loss and a minimal CBF safe-action actor term."""

    def __init__(
        self,
        *args,
        lambda_bc: float = 0.10,
        bc_delta: float = 0.03,
        bc_action_scale: float = 1.0,
        bc_weight_max: float = 5.0,
        use_projected_q: bool = True,
        **kwargs,
    ) -> None:
        if kwargs.get("replay_buffer_class") is None:
            kwargs["replay_buffer_class"] = CBFGuidedReplayBuffer
        self.lambda_bc = float(lambda_bc)
        self.bc_delta = float(bc_delta)
        self.bc_action_scale = float(max(bc_action_scale, 1e-6))
        self.bc_weight_max = float(bc_weight_max)
        self.use_projected_q = bool(use_projected_q)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, actor_rl_losses, bc_losses, critic_losses = [], [], [], []
        bc_mask_rates, bc_weight_means = [], []
        projection_trace_means, projection_active_rates, projected_action_gaps = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                a_pred = self.actor(replay_data.observations)
                actor_q_action = a_pred
                if self.use_projected_q and hasattr(replay_data, "projection_jacobians"):
                    projection_jacobians = replay_data.projection_jacobians.detach()
                    local_delta = (a_pred - replay_data.actions).unsqueeze(-1)
                    actor_q_action = replay_data.safe_actions.detach() + th.bmm(projection_jacobians, local_delta).squeeze(-1)
                    actor_q_action = actor_q_action.clamp(-1.0, 1.0)
                    projection_identity = th.eye(
                        actor_q_action.shape[1],
                        device=actor_q_action.device,
                        dtype=actor_q_action.dtype,
                    ).unsqueeze(0)
                    projection_delta = projection_jacobians - projection_identity
                    projection_active = th.norm(projection_delta, dim=(1, 2)) > 1e-6
                    projection_active_rates.append(projection_active.float().mean().item())
                    projection_trace_means.append(th.diagonal(projection_jacobians, dim1=1, dim2=2).sum(dim=1).mean().item())
                    projected_action_gaps.append(th.norm(actor_q_action - a_pred, dim=1).mean().item())

                rl_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_q_action).mean()

                correction = th.norm(replay_data.safe_actions - replay_data.actions, dim=1, keepdim=True)
                mask_float = (replay_data.interventions > 0.5).float()
                weights = 1.0 + th.clamp(
                    correction / self.bc_action_scale,
                    min=0.0,
                    max=self.bc_weight_max,
                )
                bc_per_sample = ((a_pred - replay_data.safe_actions) ** 2).sum(dim=1, keepdim=True)
                bc_loss = (mask_float * weights * bc_per_sample).sum() / (mask_float.sum() + 1e-6)
                actor_loss = rl_actor_loss + self.lambda_bc * bc_loss

                actor_losses.append(actor_loss.item())
                actor_rl_losses.append(rl_actor_loss.item())
                bc_losses.append(bc_loss.item())
                bc_mask_rates.append(mask_float.mean().item())
                if mask_float.sum().item() > 0.0:
                    bc_weight_means.append((mask_float * weights).sum().item() / (mask_float.sum().item() + 1e-6))
                else:
                    bc_weight_means.append(0.0)

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/actor_rl_loss", np.mean(actor_rl_losses))
            self.logger.record("train/cbf_bc_loss", np.mean(bc_losses))
            self.logger.record("train/cbf_bc_mask_rate", np.mean(bc_mask_rates))
            self.logger.record("train/cbf_bc_weight", np.mean(bc_weight_means))
            if projection_trace_means:
                self.logger.record("train/cbf_projection_trace", np.mean(projection_trace_means))
                self.logger.record("train/cbf_projection_active_rate", np.mean(projection_active_rates))
                self.logger.record("train/cbf_projected_action_gap", np.mean(projected_action_gaps))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


def install_minimal_guided_cbf(namespace: dict[str, Any]) -> None:
    """Install minimal guided-CBF definitions into a notebook-derived namespace."""

    namespace.setdefault("GUIDED_CBF_LAMBDA_BC", 0.10)
    namespace.setdefault("GUIDED_CBF_BC_DELTA", 0.03)
    namespace.setdefault("GUIDED_CBF_ACTION_SCALE", 1.0)
    namespace.setdefault("GUIDED_CBF_WEIGHT_MAX", 5.0)
    namespace.setdefault("GUIDED_CBF_USE_PROJECTED_Q", True)
    namespace.setdefault("GUIDED_DDPG_CBF_TOTAL_TIMESTEPS", namespace.get("DDPG_CBF_TOTAL_TIMESTEPS"))
    namespace.setdefault(
        "GUIDED_DDPG_CBF_MODEL_PATH",
        namespace["ARTIFACT_DIR"] / "guided_ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou.zip",
    )
    namespace.setdefault(
        "GUIDED_DDPG_CBF_HISTORY_PATH",
        namespace["ARTIFACT_DIR"] / "guided_ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou_eval_history.csv",
    )

    def make_guided_cbf_single_env(
        seed: int | None = None,
        render_mode: Optional[str] = None,
        lambda_filter: float | None = None,
        eps_side: float | None = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
    ) -> gym.Env:
        env = gym.make(
            "lane-free-v0",
            render_mode=render_mode,
            config=env_config or namespace["ENV_CONFIG"],
        )
        env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config or namespace["REWARD_CONFIG"])
        env = namespace["SafetyFilteredAccelerationWrapper"](
            env,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"] if lambda_filter is None else lambda_filter,
            eps_side=namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side,
            k0=namespace["CBF_K0"],
            k1=namespace["CBF_K1"],
        )
        normalize = namespace["NORMALIZE_RL_OBSERVATIONS"] if normalize_observation is None else normalize_observation
        if normalize:
            env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
        env = Monitor(env)
        env.reset(seed=namespace["SEED"] if seed is None else seed)
        return env

    def make_guided_cbf_training_env(
        seed: int | None = None,
        lambda_filter: float | None = None,
        eps_side: float | None = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
        n_envs: int = 1,
        use_subproc: bool = False,
    ):
        def _single_env(env_seed: int) -> gym.Env:
            return make_guided_cbf_single_env(
                seed=env_seed,
                render_mode=None,
                lambda_filter=lambda_filter,
                eps_side=eps_side,
                env_config=env_config,
                reward_config=reward_config,
                normalize_observation=normalize_observation,
            )

        return namespace["_make_vectorized_env"](
            _single_env,
            seed=namespace["SEED"] if seed is None else seed,
            n_envs=n_envs,
            use_subproc=use_subproc,
            start_method=namespace["DDPG_SUBPROC_START_METHOD"],
        )

    namespace.update(
        {
            "CBFGuidedReplayBufferSamples": CBFGuidedReplayBufferSamples,
            "CBFGuidedReplayBuffer": CBFGuidedReplayBuffer,
            "GuidedCBFDDPG": GuidedCBFDDPG,
            "make_guided_cbf_single_env": make_guided_cbf_single_env,
            "make_guided_cbf_training_env": make_guided_cbf_training_env,
        }
    )
