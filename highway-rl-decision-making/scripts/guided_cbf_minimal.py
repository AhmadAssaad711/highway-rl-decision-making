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


class CBFGuidedReplayBuffer(ReplayBuffer):
    """Replay buffer with one extra actor-scale CBF target: ``safe_actions``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        action_shape = (self.buffer_size, self.n_envs, self.action_dim)
        scalar_shape = (self.buffer_size, self.n_envs, 1)
        self.safe_actions = np.zeros(action_shape, dtype=np.float32)
        self.interventions = np.zeros(scalar_shape, dtype=np.float32)

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

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        slot = self.pos
        raw_actions_scaled = np.asarray(action, dtype=np.float32).reshape((self.n_envs, self.action_dim))
        self.safe_actions[slot] = raw_actions_scaled

        for env_idx, info in enumerate(infos):
            safe_phys = self._read_safe_action(info)
            if safe_phys is not None:
                self.safe_actions[slot, env_idx] = self._to_actor_scale(safe_phys)
            self.interventions[slot, env_idx, 0] = float(self._read_event_intervention(info))

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
        **kwargs,
    ) -> None:
        if kwargs.get("replay_buffer_class") is None:
            kwargs["replay_buffer_class"] = CBFGuidedReplayBuffer
        self.lambda_bc = float(lambda_bc)
        self.bc_delta = float(bc_delta)
        self.bc_action_scale = float(max(bc_action_scale, 1e-6))
        self.bc_weight_max = float(bc_weight_max)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, actor_rl_losses, bc_losses, critic_losses = [], [], [], []
        bc_mask_rates, bc_weight_means = [], []

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
                rl_actor_loss = -self.critic.q1_forward(replay_data.observations, a_pred).mean()

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
        self.logger.record("train/critic_loss", np.mean(critic_losses))


def install_minimal_guided_cbf(namespace: dict[str, Any]) -> None:
    """Install minimal guided-CBF definitions into a notebook-derived namespace."""

    namespace.setdefault("GUIDED_CBF_LAMBDA_BC", 0.10)
    namespace.setdefault("GUIDED_CBF_BC_DELTA", 0.03)
    namespace.setdefault("GUIDED_CBF_ACTION_SCALE", 1.0)
    namespace.setdefault("GUIDED_CBF_WEIGHT_MAX", 5.0)
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
