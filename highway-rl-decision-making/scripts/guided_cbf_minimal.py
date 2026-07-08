"""Minimal guided DDPG-CBF replay buffer and actor loss.

The notebook still defines the environment, reward, and CBF shield. This module
overrides only the guided reward-plus-loss pieces used by Python scripts.
"""

from __future__ import annotations

import warnings
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


def _projection_from_active_rows(rows: np.ndarray, action_dim: int) -> np.ndarray:
    identity = np.eye(action_dim, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, action_dim)
    if rows.size == 0 or not np.all(np.isfinite(rows)):
        return identity
    try:
        projection = identity - rows.T @ np.linalg.pinv(rows @ rows.T, rcond=1e-5) @ rows
    except np.linalg.LinAlgError:
        return identity
    projection = 0.5 * (projection + projection.T)
    if not np.all(np.isfinite(projection)):
        return identity
    return projection.astype(np.float32)


def _install_cbf_projection_reporting(namespace: dict[str, Any]) -> None:
    original_filter = namespace.get("cbf_filter_2d")
    pairwise_constraint = namespace.get("pairwise_hocbf_constraint")
    if original_filter is None or pairwise_constraint is None:
        return
    if getattr(original_filter, "_guided_projection_reporting", False):
        return

    def cbf_filter_2d_with_projection(
        a_rl,
        ego: dict[str, float],
        neighbors: list[dict[str, float]],
        road_width: float,
        ax_bounds: tuple[float, float] | None = None,
        ay_bounds: tuple[float, float] | None = None,
        eps_side: float | None = None,
        k0: float | None = None,
        k1: float | None = None,
        max_neighbor_constraints: Optional[int] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ax_bounds = namespace["CBF_AX_BOUNDS"] if ax_bounds is None else ax_bounds
        ay_bounds = namespace["CBF_AY_BOUNDS"] if ay_bounds is None else ay_bounds
        eps_side = float(namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side)
        k0 = float(namespace["CBF_K0"] if k0 is None else k0)
        k1 = float(namespace["CBF_K1"] if k1 is None else k1)
        if max_neighbor_constraints is None:
            max_neighbor_constraints = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")

        a_rl = np.asarray(a_rl, dtype=float).reshape(-1)
        if a_rl.size < 2:
            raise ValueError("a_rl must contain [ax, ay].")
        a_rl = a_rl[:2]

        active_neighbors = list(neighbors)
        if max_neighbor_constraints is not None:
            active_neighbors = active_neighbors[: int(max_neighbor_constraints)]

        rows: list[np.ndarray] = []
        bounds: list[float] = []
        min_h = np.inf
        min_center_distance = np.inf
        min_required_distance = np.inf
        neighbor_constraints = 0
        for neighbor in active_neighbors:
            finite_values = [
                ego.get("x", 0.0),
                ego.get("y", 0.0),
                ego.get("vx", 0.0),
                ego.get("vy", 0.0),
                ego.get("length", 0.0),
                ego.get("width", 0.0),
                neighbor.get("x", 0.0),
                neighbor.get("y", 0.0),
                neighbor.get("vx", 0.0),
                neighbor.get("vy", 0.0),
                neighbor.get("length", 0.0),
                neighbor.get("width", 0.0),
                neighbor.get("ax", 0.0),
                neighbor.get("ay", 0.0),
            ]
            if not np.all(np.isfinite(np.asarray(finite_values, dtype=float))):
                continue
            other_acc = np.asarray(
                [float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))],
                dtype=float,
            )
            try:
                row, bound, h_ij, center_distance, required_distance = pairwise_constraint(
                    ego,
                    neighbor,
                    eps_side=eps_side,
                    k0=k0,
                    k1=k1,
                    other_acc=other_acc,
                )
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
                continue
            if not np.all(np.isfinite(np.asarray(row, dtype=float))) or not np.isfinite(float(bound)):
                continue
            rows.append(np.asarray(row, dtype=float).reshape(-1)[:2])
            bounds.append(float(bound))
            min_h = min(min_h, float(h_ij))
            min_center_distance = min(min_center_distance, float(center_distance))
            min_required_distance = min(min_required_distance, float(required_distance))
            neighbor_constraints += 1

        ego_y = float(ego["y"])
        ego_vy = float(ego["vy"])
        ego_half_width = 0.5 * float(ego["width"])
        h_left = ego_y - ego_half_width
        h_right = float(road_width) - ego_half_width - ego_y
        rows.extend([np.asarray([0.0, -1.0], dtype=float), np.asarray([0.0, 1.0], dtype=float)])
        bounds.extend([float(k1 * ego_vy + k0 * h_left), float(-k1 * ego_vy + k0 * h_right)])

        low = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=np.float32)
        high = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=np.float32)
        half_range = 0.5 * np.maximum(high - low, 1e-6)
        rows_arr = np.asarray(rows, dtype=np.float32).reshape(-1, 2)
        bounds_arr = np.asarray(bounds, dtype=np.float32).reshape(-1)
        rl_constraint_values = rows_arr @ a_rl.astype(np.float32) - bounds_arr
        raw_is_feasible = (
            bool(np.all(rl_constraint_values <= float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-6))))
            and bool(np.all(a_rl >= low - float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-6))))
            and bool(np.all(a_rl <= high + float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-6))))
        )
        qp_error = ""
        if raw_is_feasible:
            qp_success = True
            a_safe = a_rl.copy()
        else:
            solution = None
            try:
                sparse = namespace["sparse"]
                solve_qp = namespace["solve_qp"]
                p_matrix = sparse.csc_matrix(2.0 * np.eye(2, dtype=float))
                q_vector = -2.0 * a_rl
                g_matrix = sparse.csc_matrix(rows_arr.astype(float))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"OSQP exited.*")
                    solution = solve_qp(
                        p_matrix,
                        q_vector,
                        G=g_matrix,
                        h=bounds_arr.astype(float),
                        lb=low.astype(float),
                        ub=high.astype(float),
                        solver=namespace.get("CBF_QP_SOLVER", "osqp"),
                        verbose=False,
                    )
            except Exception as exc:
                qp_error = repr(exc)
            qp_success = solution is not None and bool(np.all(np.isfinite(solution)))
            if qp_success:
                a_safe = np.clip(np.asarray(solution, dtype=float).reshape(-1)[:2], low, high)
            elif "_least_violating_bounded_action" in namespace:
                a_safe = namespace["_least_violating_bounded_action"](
                    a_rl,
                    rows,
                    bounds,
                    ax_bounds,
                    ay_bounds,
                )
            else:
                a_safe = np.asarray([float(ax_bounds[0]), 0.0], dtype=float)

        safe = np.asarray(a_safe, dtype=np.float32).reshape(-1)[:2]
        safe_constraint_values = rows_arr @ safe - bounds_arr
        correction_norm = float(np.linalg.norm(safe - a_rl.astype(np.float32)))
        if not np.isfinite(min_h):
            min_h = np.nan
        if not np.isfinite(min_center_distance):
            min_center_distance = np.nan
        if not np.isfinite(min_required_distance):
            min_required_distance = np.nan
        values = rows_arr @ safe - bounds_arr
        active_tol = float(namespace.get("CBF_QP_ACTIVE_TOL", 1e-3))
        active_rows = [row for row, value in zip(rows_arr, values) if float(value) >= -active_tol]

        if safe[0] <= low[0] + active_tol:
            active_rows.append(np.asarray([-1.0, 0.0], dtype=np.float32))
        if safe[0] >= high[0] - active_tol:
            active_rows.append(np.asarray([1.0, 0.0], dtype=np.float32))
        if safe[1] <= low[1] + active_tol:
            active_rows.append(np.asarray([0.0, -1.0], dtype=np.float32))
        if safe[1] >= high[1] - active_tol:
            active_rows.append(np.asarray([0.0, 1.0], dtype=np.float32))

        fallback_used = bool(not qp_success)
        if fallback_used or not active_rows:
            active_scaled = np.zeros((0, 2), dtype=np.float32)
        else:
            active_scaled = np.asarray(active_rows, dtype=np.float32) * half_range.reshape(1, -1)
        projection = _projection_from_active_rows(active_scaled, 2)

        info = {
            "a_safe": safe.astype(np.float32),
            "correction_norm": correction_norm,
            "raw_feasible": bool(raw_is_feasible),
            "max_constraint_violation_rl": float(np.max(rl_constraint_values)) if len(rl_constraint_values) else 0.0,
            "max_constraint_violation_safe": float(np.max(safe_constraint_values)) if len(safe_constraint_values) else 0.0,
            "min_h": float(min_h),
            "min_center_distance": float(min_center_distance),
            "min_required_distance": float(min_required_distance),
            "eps_side": float(eps_side),
            "k0": float(k0),
            "k1": float(k1),
            "qp_success": bool(qp_success),
            "fallback_used": fallback_used,
            "qp_error": qp_error,
            "num_neighbor_constraints": int(neighbor_constraints),
            "left_boundary_h": float(h_left),
            "right_boundary_h": float(h_right),
            "min_boundary_h": float(min(h_left, h_right)),
            "active_constraint_rows_scaled": active_scaled.astype(np.float32),
            "cbf_active_constraint_rows_scaled": active_scaled.astype(np.float32),
            "projection_jacobian_scaled": projection,
            "cbf_projection_jacobian_scaled": projection,
            "cbf_active_constraint_count": int(active_scaled.shape[0]),
        }
        return np.asarray(a_safe, dtype=np.float32), info

    cbf_filter_2d_with_projection._guided_projection_reporting = True  # type: ignore[attr-defined]
    namespace["cbf_filter_2d"] = cbf_filter_2d_with_projection

    wrapper_cls = namespace.get("SafetyFilteredAccelerationWrapper")
    if wrapper_cls is None or getattr(wrapper_cls.step, "_guided_projection_reporting", False):
        return

    def step_with_projection_info(self, action):
        a_rl = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        ego = namespace["get_ego_state"](self)
        neighbors = namespace["get_neighbor_states"](self, neighbor_range=self.neighbor_range)
        road_width = float(namespace["_lane_free_base"](self).config["road_width"])
        filter_kwargs = {
            "ax_bounds": self.ax_bounds,
            "ay_bounds": self.ay_bounds,
            "eps_side": self.eps_side,
            "k0": self.k0,
            "k1": self.k1,
        }
        if hasattr(self, "max_neighbor_constraints"):
            filter_kwargs["max_neighbor_constraints"] = self.max_neighbor_constraints
        a_safe, filter_info = namespace["cbf_filter_2d"](a_rl, ego, neighbors, road_width, **filter_kwargs)
        normalized_action = namespace["_physical_to_normalized_action"](self, a_safe)
        obs, reward, terminated, truncated, info = self.env.step(normalized_action)
        correction_norm = float(filter_info["correction_norm"])
        correction_penalty = float(self.lambda_filter * correction_norm**2)
        reward = float(reward) - correction_penalty

        info = dict(info)
        info.update(
            {
                "cbf_a_rl_x": float(a_rl[0]),
                "cbf_a_rl_y": float(a_rl[1]),
                "cbf_a_safe_x": float(a_safe[0]),
                "cbf_a_safe_y": float(a_safe[1]),
                "cbf_correction_norm": correction_norm,
                "cbf_intervened": bool(correction_norm > 1e-6),
                "cbf_raw_feasible": bool(filter_info.get("raw_feasible", False)),
                "cbf_max_constraint_violation_rl": float(filter_info.get("max_constraint_violation_rl", 0.0)),
                "cbf_max_constraint_violation_safe": float(filter_info.get("max_constraint_violation_safe", 0.0)),
                "cbf_min_h": float(filter_info["min_h"]),
                "cbf_min_center_distance": float(filter_info["min_center_distance"]),
                "cbf_min_required_distance": float(filter_info["min_required_distance"]),
                "cbf_eps_side": float(filter_info["eps_side"]),
                "cbf_k0": float(filter_info.get("k0", self.k0)),
                "cbf_k1": float(filter_info.get("k1", self.k1)),
                "cbf_qp_success": bool(filter_info["qp_success"]),
                "cbf_fallback_used": bool(filter_info.get("fallback_used", not filter_info["qp_success"])),
                "cbf_qp_error": str(filter_info["qp_error"]),
                "cbf_num_neighbor_constraints": int(filter_info["num_neighbor_constraints"]),
                "cbf_min_boundary_h": float(filter_info["min_boundary_h"]),
                "cbf_left_boundary_h": float(filter_info["left_boundary_h"]),
                "cbf_right_boundary_h": float(filter_info["right_boundary_h"]),
                "cbf_filter_reward_penalty": correction_penalty,
                "cbf_active_constraint_count": int(filter_info.get("cbf_active_constraint_count", 0)),
            }
        )
        for key in (
            "active_constraint_rows_scaled",
            "cbf_active_constraint_rows_scaled",
            "projection_jacobian_scaled",
            "cbf_projection_jacobian_scaled",
        ):
            if key in filter_info:
                info[key] = filter_info[key]
        return obs, reward, terminated, truncated, info

    step_with_projection_info._guided_projection_reporting = True  # type: ignore[attr-defined]
    wrapper_cls.step = step_with_projection_info


class GuidedCBFDDPG(DDPG):
    """DDPG with standard critic loss and a minimal CBF safe-action actor term."""

    def __init__(
        self,
        *args,
        lambda_bc: float = 0.10,
        lambda_bc_final: float | None = None,
        lambda_bc_decay_steps: int = 0,
        bc_delta: float = 0.03,
        bc_action_scale: float = 1.0,
        bc_weight_max: float = 5.0,
        use_projected_q: bool = True,
        projected_q_weight: float = 0.0,
        critic_action_mode: str = "raw",
        actor_action_mode: str = "raw",
        cbf_projection_steps: int = 2,
        cbf_projection_fd_step: float = 1e-3,
        cbf_road_width: float = 10.2,
        cbf_sensing_range: float = 90.0,
        cbf_obs_vmax: float = 24.0,
        cbf_obs_vymax: float = 7.2,
        cbf_eps_side: float = 0.10,
        cbf_k0: float = 5.29,
        cbf_k1: float = 3.68,
        **kwargs,
    ) -> None:
        if kwargs.get("replay_buffer_class") is None:
            kwargs["replay_buffer_class"] = CBFGuidedReplayBuffer
        self.lambda_bc = float(lambda_bc)
        self.lambda_bc_initial = float(lambda_bc)
        self.lambda_bc_final = None if lambda_bc_final is None else float(lambda_bc_final)
        self.lambda_bc_decay_steps = int(max(lambda_bc_decay_steps, 0))
        self.bc_delta = float(bc_delta)
        self.bc_action_scale = float(max(bc_action_scale, 1e-6))
        self.bc_weight_max = float(bc_weight_max)
        self.use_projected_q = bool(use_projected_q)
        self.projected_q_weight = float(np.clip(projected_q_weight, 0.0, 1.0))
        self.critic_action_mode = str(critic_action_mode).strip().lower()
        self.actor_action_mode = str(actor_action_mode).strip().lower()
        if self.critic_action_mode not in {"raw", "safe"}:
            raise ValueError(f"critic_action_mode must be 'raw' or 'safe', got {critic_action_mode!r}")
        if self.actor_action_mode not in {"raw", "diff_cbf"}:
            raise ValueError(f"actor_action_mode must be 'raw' or 'diff_cbf', got {actor_action_mode!r}")
        self.cbf_projection_steps = int(max(cbf_projection_steps, 1))
        self.cbf_projection_fd_step = float(max(cbf_projection_fd_step, 1e-5))
        self.cbf_road_width = float(cbf_road_width)
        self.cbf_sensing_range = float(cbf_sensing_range)
        self.cbf_obs_vmax = float(cbf_obs_vmax)
        self.cbf_obs_vymax = float(cbf_obs_vymax)
        self.cbf_eps_side = float(cbf_eps_side)
        self.cbf_k0 = float(cbf_k0)
        self.cbf_k1 = float(cbf_k1)
        super().__init__(*args, **kwargs)

    def _action_bounds_tensors(self, action: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        low = th.as_tensor(self.action_space.low, device=action.device, dtype=action.dtype).reshape(1, -1)[:, : action.shape[1]]
        high = th.as_tensor(self.action_space.high, device=action.device, dtype=action.dtype).reshape(1, -1)[:, : action.shape[1]]
        return low, high

    def _actor_to_physical_action(self, action_scaled: th.Tensor) -> th.Tensor:
        low, high = self._action_bounds_tensors(action_scaled)
        return low + 0.5 * (action_scaled + 1.0) * (high - low)

    def _physical_to_actor_action(self, action_phys: th.Tensor) -> th.Tensor:
        low, high = self._action_bounds_tensors(action_phys)
        return (2.0 * (action_phys - low) / th.clamp(high - low, min=1e-6) - 1.0).clamp(-1.0, 1.0)

    def _ellipse_clearance_from_obs(
        self,
        px: th.Tensor,
        py: th.Tensor,
        ego_length: th.Tensor,
        ego_width: th.Tensor,
        other_length: th.Tensor,
        other_width: th.Tensor,
    ) -> th.Tensor:
        eps = th.as_tensor(1e-6, device=px.device, dtype=px.dtype)
        radius = th.sqrt(px.square() + py.square() + eps)
        phi = th.atan2(py, px)

        def inflated_radius(length: th.Tensor, width: th.Tensor) -> th.Tensor:
            a = length / np.sqrt(2.0) + 2.0 * self.cbf_eps_side
            b = width / np.sqrt(2.0) + 2.0 * self.cbf_eps_side
            cos_phi = th.cos(phi)
            sin_phi = th.sin(phi)
            denom = th.sqrt((b * cos_phi).square() + (a * sin_phi).square() + eps)
            return a * b / th.clamp(denom, min=1e-6)

        required = inflated_radius(ego_length, ego_width) + inflated_radius(other_length, other_width)
        return radius - required

    def _neighbor_constraint_from_obs(
        self,
        obs_rows: th.Tensor,
        neighbor_index: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        ego = obs_rows[:, 0, :]
        other = obs_rows[:, neighbor_index, :]
        dtype = obs_rows.dtype
        device = obs_rows.device
        fd = th.as_tensor(self.cbf_projection_fd_step, device=device, dtype=dtype)

        ego_y = 0.5 * self.cbf_road_width * (ego[:, 0] + 1.0)
        ego_vx = ego[:, 2] * self.cbf_obs_vmax
        ego_vy = ego[:, 3] * self.cbf_obs_vymax
        ego_length = th.clamp(ego[:, 4] * 5.15, min=1e-3)
        ego_width = th.clamp(ego[:, 5] * 1.84, min=1e-3)

        dx = other[:, 0] * self.cbf_sensing_range
        dy = other[:, 1] * self.cbf_road_width
        other_vx = other[:, 2] * self.cbf_obs_vmax
        other_vy = other[:, 3] * self.cbf_obs_vymax
        other_length = th.clamp(other[:, 4] * 5.15, min=1e-3)
        other_width = th.clamp(other[:, 5] * 1.84, min=1e-3)
        valid = (other[:, 4] > 1e-4) & (other[:, 5] > 1e-4)

        def h_at(x: th.Tensor, y: th.Tensor) -> th.Tensor:
            return self._ellipse_clearance_from_obs(x, y, ego_length, ego_width, other_length, other_width)

        h0 = h_at(dx, dy)
        h_px = h_at(dx + fd, dy)
        h_mx = h_at(dx - fd, dy)
        h_py = h_at(dx, dy + fd)
        h_my = h_at(dx, dy - fd)
        h_pp = h_at(dx + fd, dy + fd)
        h_pm = h_at(dx + fd, dy - fd)
        h_mp = h_at(dx - fd, dy + fd)
        h_mm = h_at(dx - fd, dy - fd)

        grad_x = (h_px - h_mx) / (2.0 * fd)
        grad_y = (h_py - h_my) / (2.0 * fd)
        h_xx = (h_px - 2.0 * h0 + h_mx) / fd.square()
        h_yy = (h_py - 2.0 * h0 + h_my) / fd.square()
        h_xy = (h_pp - h_pm - h_mp + h_mm) / (4.0 * fd.square())

        dvx = other_vx - ego_vx
        dvy = other_vy - ego_vy
        h_dot = grad_x * dvx + grad_y * dvy
        v_h_v = h_xx * dvx.square() + 2.0 * h_xy * dvx * dvy + h_yy * dvy.square()
        bound = v_h_v + self.cbf_k1 * h_dot + self.cbf_k0 * h0
        row = th.stack([grad_x, grad_y], dim=1)
        row = th.where(valid.unsqueeze(1), row, th.zeros_like(row))
        bound = th.where(valid, bound, th.full_like(bound, 1e6))
        return row.detach(), bound.detach().unsqueeze(1), valid.detach().unsqueeze(1)

    def _constraints_from_observation(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = observations[:, :42]
        rows = obs.reshape(obs.shape[0], 6, 7)
        constraints: list[th.Tensor] = []
        bounds: list[th.Tensor] = []
        masks: list[th.Tensor] = []

        for neighbor_index in range(1, rows.shape[1]):
            row, bound, mask = self._neighbor_constraint_from_obs(rows, neighbor_index)
            constraints.append(row)
            bounds.append(bound)
            masks.append(mask)

        ego = rows[:, 0, :]
        ego_y = 0.5 * self.cbf_road_width * (ego[:, 0] + 1.0)
        ego_vy = ego[:, 3] * self.cbf_obs_vymax
        ego_width = th.clamp(ego[:, 5] * 1.84, min=1e-3)
        h_left = ego_y - 0.5 * ego_width
        h_right = self.cbf_road_width - 0.5 * ego_width - ego_y
        left_row = th.zeros((observations.shape[0], 2), device=observations.device, dtype=observations.dtype)
        right_row = th.zeros_like(left_row)
        left_row[:, 1] = -1.0
        right_row[:, 1] = 1.0
        constraints.extend([left_row.detach(), right_row.detach()])
        bounds.extend(
            [
                (self.cbf_k1 * ego_vy + self.cbf_k0 * h_left).detach().unsqueeze(1),
                (-self.cbf_k1 * ego_vy + self.cbf_k0 * h_right).detach().unsqueeze(1),
            ]
        )
        valid_boundary = th.ones((observations.shape[0], 1), device=observations.device, dtype=observations.dtype)
        masks.extend([valid_boundary, valid_boundary])

        return th.stack(constraints, dim=1), th.cat(bounds, dim=1), th.cat(masks, dim=1)

    def _diff_cbf_project_from_obs(self, observations: th.Tensor, action_scaled: th.Tensor) -> tuple[th.Tensor, dict[str, float]]:
        action_phys = self._actor_to_physical_action(action_scaled)
        low, high = self._action_bounds_tensors(action_phys)
        safe_phys = action_phys.clamp(low, high)
        constraint_rows, constraint_bounds, constraint_masks = self._constraints_from_observation(observations)

        for _ in range(self.cbf_projection_steps):
            for constraint_index in range(constraint_rows.shape[1]):
                row = constraint_rows[:, constraint_index, :]
                bound = constraint_bounds[:, constraint_index].unsqueeze(1)
                mask = constraint_masks[:, constraint_index].unsqueeze(1)
                violation = th.relu((row * safe_phys).sum(dim=1, keepdim=True) - bound) * mask
                denom = row.square().sum(dim=1, keepdim=True).clamp(min=1e-6)
                safe_phys = (safe_phys - violation * row / denom).clamp(low, high)

        safe_scaled = self._physical_to_actor_action(safe_phys)
        correction = th.norm(safe_scaled - action_scaled, dim=1)
        with th.no_grad():
            constraint_values = (constraint_rows * safe_phys.unsqueeze(1)).sum(dim=2) - constraint_bounds
            masked_values = th.where(constraint_masks > 0.5, constraint_values, th.full_like(constraint_values, -1e6))
            max_violation = th.relu(masked_values.max(dim=1).values)
        stats = {
            "diff_cbf_correction": float(correction.detach().mean().cpu().item()),
            "diff_cbf_active_rate": float((correction.detach() > 1e-6).float().mean().cpu().item()),
            "diff_cbf_max_violation": float(max_violation.detach().mean().cpu().item()),
        }
        return safe_scaled, stats

    def _current_lambda_bc(self) -> float:
        lambda_final = getattr(self, "lambda_bc_final", None)
        decay_steps = int(max(getattr(self, "lambda_bc_decay_steps", 0), 0))
        lambda_initial = float(getattr(self, "lambda_bc_initial", getattr(self, "lambda_bc", 0.0)))
        if lambda_final is None or decay_steps <= 0:
            return float(getattr(self, "lambda_bc", lambda_initial))
        progress = float(np.clip(float(getattr(self, "num_timesteps", 0)) / max(float(decay_steps), 1.0), 0.0, 1.0))
        return float(lambda_initial + progress * (float(lambda_final) - lambda_initial))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, actor_rl_losses, bc_losses, critic_losses = [], [], [], []
        actor_raw_q_losses, actor_projected_q_losses = [], []
        bc_mask_rates, bc_weight_means = [], []
        projection_trace_means, projection_active_rates, projected_action_gaps = [], [], []
        diff_cbf_corrections, diff_cbf_active_rates, diff_cbf_max_violations = [], [], []
        critic_raw_q_means, critic_safe_q_means = [], []

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

            critic_actions = replay_data.safe_actions if self.critic_action_mode == "safe" else replay_data.actions
            current_q_values = self.critic(replay_data.observations, critic_actions)
            with th.no_grad():
                critic_raw_q_means.append(self.critic.q1_forward(replay_data.observations, replay_data.actions).mean().item())
                critic_safe_q_means.append(self.critic.q1_forward(replay_data.observations, replay_data.safe_actions).mean().item())
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                a_pred = self.actor(replay_data.observations)
                actor_q_action = a_pred
                diff_cbf_actor_update = self.actor_action_mode == "diff_cbf"
                use_projected_update = (
                    self.use_projected_q
                    and self.projected_q_weight > 0.0
                    and hasattr(replay_data, "projection_jacobians")
                    and not diff_cbf_actor_update
                )
                if diff_cbf_actor_update:
                    actor_q_action, diff_stats = self._diff_cbf_project_from_obs(replay_data.observations, a_pred)
                    diff_cbf_corrections.append(diff_stats["diff_cbf_correction"])
                    diff_cbf_active_rates.append(diff_stats["diff_cbf_active_rate"])
                    diff_cbf_max_violations.append(diff_stats["diff_cbf_max_violation"])
                if use_projected_update:
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

                raw_q_actor_loss = -self.critic.q1_forward(replay_data.observations, a_pred).mean()
                projected_q_actor_loss = raw_q_actor_loss
                if diff_cbf_actor_update:
                    projected_q_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_q_action).mean()
                    rl_actor_loss = projected_q_actor_loss
                elif use_projected_update:
                    projected_q_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_q_action).mean()
                    rl_actor_loss = (
                        (1.0 - self.projected_q_weight) * raw_q_actor_loss
                        + self.projected_q_weight * projected_q_actor_loss
                    )
                else:
                    rl_actor_loss = raw_q_actor_loss

                if diff_cbf_actor_update:
                    correction = th.norm(actor_q_action - a_pred, dim=1, keepdim=True)
                    mask_float = (correction > 1e-6).float()
                    weights = 1.0 + th.clamp(
                        correction / self.bc_action_scale,
                        min=0.0,
                        max=self.bc_weight_max,
                    )
                    bc_per_sample = ((a_pred - actor_q_action) ** 2).sum(dim=1, keepdim=True)
                    bc_loss = bc_per_sample.mean()
                else:
                    correction = th.norm(replay_data.safe_actions - replay_data.actions, dim=1, keepdim=True)
                    mask_float = (replay_data.interventions > 0.5).float()
                    weights = 1.0 + th.clamp(
                        correction / self.bc_action_scale,
                        min=0.0,
                        max=self.bc_weight_max,
                    )
                    bc_per_sample = ((a_pred - replay_data.safe_actions) ** 2).sum(dim=1, keepdim=True)
                    bc_loss = (mask_float * weights * bc_per_sample).sum() / (mask_float.sum() + 1e-6)
                lambda_bc_current = self._current_lambda_bc()
                actor_loss = rl_actor_loss + lambda_bc_current * bc_loss

                actor_losses.append(actor_loss.item())
                actor_rl_losses.append(rl_actor_loss.item())
                actor_raw_q_losses.append(raw_q_actor_loss.item())
                actor_projected_q_losses.append(projected_q_actor_loss.item())
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
            self.logger.record("train/actor_raw_q_loss", np.mean(actor_raw_q_losses))
            self.logger.record("train/actor_projected_q_loss", np.mean(actor_projected_q_losses))
            self.logger.record("train/cbf_critic_action_mode_safe", float(self.critic_action_mode == "safe"))
            self.logger.record("train/cbf_actor_action_mode_diff", float(self.actor_action_mode == "diff_cbf"))
            self.logger.record("train/cbf_critic_q_raw_mean", np.mean(critic_raw_q_means))
            self.logger.record("train/cbf_critic_q_safe_mean", np.mean(critic_safe_q_means))
            self.logger.record("train/cbf_projected_q_weight", self.projected_q_weight)
            self.logger.record("train/cbf_lambda_bc_current", self._current_lambda_bc())
            self.logger.record("train/cbf_bc_loss", np.mean(bc_losses))
            self.logger.record("train/cbf_bc_mask_rate", np.mean(bc_mask_rates))
            self.logger.record("train/cbf_bc_weight", np.mean(bc_weight_means))
            if diff_cbf_corrections:
                self.logger.record("train/diff_cbf_correction", np.mean(diff_cbf_corrections))
                self.logger.record("train/diff_cbf_active_rate", np.mean(diff_cbf_active_rates))
                self.logger.record("train/diff_cbf_max_violation", np.mean(diff_cbf_max_violations))
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
    namespace.setdefault("GUIDED_CBF_PROJECTED_Q_WEIGHT", 0.0)
    namespace.setdefault("GUIDED_CBF_CRITIC_ACTION_MODE", "raw")
    namespace.setdefault("GUIDED_CBF_ACTOR_ACTION_MODE", "raw")
    namespace.setdefault("GUIDED_CBF_DIFF_PROJECTION_STEPS", 2)
    namespace.setdefault("GUIDED_CBF_DIFF_PROJECTION_FD_STEP", 1e-3)
    namespace["install_cbf_projection_reporting"] = lambda: _install_cbf_projection_reporting(namespace)
    if bool(namespace.get("GUIDED_CBF_ENABLE_PROJECTION_REPORTING", False)):
        _install_cbf_projection_reporting(namespace)
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
