"""
Attention-based PPO baseline for native highway-v0 discrete control.

This ports the legacy ego-attention feature extractor to the repo's current
stack:

- gymnasium reset/step API
- highway-env 1.10.x environment creation
- stable-baselines3 2.x custom feature extractor interface

Unlike the existing hybrid PPO baseline in this folder, this script keeps the
original highway-v0 discrete action space.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
import numpy as np
import torch
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "artifacts" / "ppo" / "attention_ppo"
MODELS_DIR = RUN_ROOT / "models"
TB_DIR = RUN_ROOT / "tensorboard"


def activation_factory(activation_type: str) -> nn.Module:
    activation_type = activation_type.upper()
    if activation_type == "RELU":
        return nn.ReLU()
    if activation_type == "TANH":
        return nn.Tanh()
    if activation_type == "ELU":
        return nn.ELU()
    raise ValueError(f"Unknown activation_type: {activation_type}")


class BaseModule(nn.Module):
    """Torch module with configurable linear-layer initialization."""

    def __init__(self, activation_type: str = "RELU", reset_type: str = "XAVIER"):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type

    def _init_weights(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Linear):
            return
        if self.reset_type == "XAVIER":
            nn.init.xavier_uniform_(module.weight)
        elif self.reset_type == "ZEROS":
            nn.init.constant_(module.weight, 0.0)
        else:
            raise ValueError(f"Unknown reset type: {self.reset_type}")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    def reset(self) -> None:
        self.apply(self._init_weights)


class MultiLayerPerceptron(BaseModule):
    def __init__(
        self,
        in_size: int,
        layer_sizes: list[int] | None = None,
        reshape: bool = True,
        out_size: int | None = None,
        activation: str = "RELU",
        **kwargs: Any,
    ):
        super().__init__(activation_type=activation, **kwargs)
        self.reshape = reshape
        self.layer_sizes = layer_sizes or [64, 64]
        self.out_size = out_size

        sizes = [in_size] + self.layer_sizes
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.predict = nn.Linear(sizes[-1], out_size) if out_size is not None else None
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reshape:
            x = x.reshape(x.shape[0], -1)
        x = x.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.predict is not None:
            x = self.predict(x)
        return x


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention for the ego query against all entities."""

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attention_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class EgoAttention(BaseModule):
    def __init__(self, feature_size: int = 64, heads: int = 4, dropout_factor: float = 0.0):
        super().__init__()
        if feature_size % heads != 0:
            raise ValueError("feature_size must be divisible by heads")
        self.feature_size = int(feature_size)
        self.heads = int(heads)
        self.dropout_factor = float(dropout_factor)
        self.features_per_head = self.feature_size // self.heads

        self.value_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.key_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_ego = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.reset()

    def forward(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.reshape(batch_size, 1, self.feature_size), others), dim=1)

        key_all = self.key_all(input_all).reshape(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        value_all = self.value_all(input_all).reshape(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        query_ego = self.query_ego(ego).reshape(
            batch_size, 1, self.heads, self.features_per_head
        )

        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)

        expanded_mask = None
        if mask is not None:
            expanded_mask = mask.reshape(batch_size, 1, 1, n_entities).repeat(1, self.heads, 1, 1)

        value, attention_matrix = scaled_dot_product_attention(
            query_ego,
            key_all,
            value_all,
            expanded_mask,
            nn.Dropout(self.dropout_factor),
        )
        attended_ego = self.attention_combine(value.reshape(batch_size, self.feature_size))
        result = 0.5 * (attended_ego + ego.squeeze(1))
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule):
    def __init__(
        self,
        in_size: int,
        presence_feature_idx: int = 0,
        embedding_layer_kwargs: dict[str, Any] | None = None,
        attention_layer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.presence_feature_idx = int(presence_feature_idx)

        embedding_kwargs = dict(embedding_layer_kwargs or {})
        embedding_kwargs.setdefault("in_size", in_size)
        embedding_kwargs.setdefault("reshape", False)
        self.ego_embedding = MultiLayerPerceptron(**embedding_kwargs)
        self.embedding = MultiLayerPerceptron(**embedding_kwargs)

        self.attention_layer = EgoAttention(**(attention_layer_kwargs or {}))

    def split_input(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            aux = self.presence_feature_idx
            mask = x[:, :, aux : aux + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ego, others, mask = self.split_input(x)
        ego = self.ego_embedding(ego)
        others = self.embedding(others)
        return self.attention_layer(ego, others, mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ego_embedded_att, _ = self.forward_attention(x)
        return ego_embedded_att

    def get_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


class CustomExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor that applies ego-centered self-attention."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        embedding_layer_kwargs: dict[str, Any] | None = None,
        attention_layer_kwargs: dict[str, Any] | None = None,
        presence_feature_idx: int = 0,
        **kwargs: Any,
    ):
        if len(observation_space.shape) != 2:
            raise ValueError(
                "CustomExtractor expects a 2D kinematics observation of shape "
                f"(entities, features), got {observation_space.shape}"
            )
        entities, features = observation_space.shape
        if presence_feature_idx >= features:
            raise ValueError(
                f"presence_feature_idx={presence_feature_idx} out of bounds for {features} features"
            )

        attention_layer_kwargs = dict(attention_layer_kwargs or {})
        feature_size = int(attention_layer_kwargs.get("feature_size", 64))
        super().__init__(observation_space, features_dim=feature_size)

        embedding_layer_kwargs = dict(embedding_layer_kwargs or {})
        embedding_layer_kwargs.setdefault("in_size", int(features))
        embedding_layer_kwargs.setdefault("reshape", False)
        self.extractor = EgoAttentionNetwork(
            in_size=int(features),
            presence_feature_idx=presence_feature_idx,
            embedding_layer_kwargs=embedding_layer_kwargs,
            attention_layer_kwargs=attention_layer_kwargs,
            **kwargs,
        )
        self.n_entities = int(entities)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(observations)


DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "lanes_count": 3,
    "vehicles_count": 15,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
    },
    "policy_frequency": 2,
    "duration": 40,
}


DEFAULT_EXTRACTOR_KWARGS: dict[str, Any] = {
    "embedding_layer_kwargs": {"layer_sizes": [64, 64], "reshape": False},
    "attention_layer_kwargs": {"feature_size": 64, "heads": 2, "dropout_factor": 0.0},
    "presence_feature_idx": 0,
}


def make_highway_env(config: dict[str, Any] | None = None, render_mode: str | None = None) -> gym.Env:
    return gym.make("highway-v0", render_mode=render_mode, config=dict(config or DEFAULT_ENV_CONFIG))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate an attention-based PPO agent")
    parser.add_argument(
        "--mode",
        choices=("train", "eval"),
        default="train",
        help="Whether to train a new model or evaluate an existing one",
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel training environments")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="PPO learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="auto", help="Torch device for PPO")
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "ppo_highway_attention"),
        help="Path prefix for model save/load",
    )
    parser.add_argument(
        "--tensorboard-log",
        default=str(TB_DIR),
        help="TensorBoard log directory for training",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array", "none"),
        default="human",
        help="Evaluation render mode",
    )
    return parser.parse_args()


def build_policy_kwargs() -> dict[str, Any]:
    return {
        "features_extractor_class": CustomExtractor,
        "features_extractor_kwargs": dict(DEFAULT_EXTRACTOR_KWARGS),
    }


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        make_highway_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={"config": DEFAULT_ENV_CONFIG, "render_mode": None},
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs=build_policy_kwargs(),
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(str(model_path))
    env.close()
    print(f"Saved model to {model_path}.zip")


def evaluate(args: argparse.Namespace) -> None:
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = Monitor(make_highway_env(config=DEFAULT_ENV_CONFIG, render_mode=render_mode))
    model = PPO.load(args.model_path, env=env, device=args.device)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: reward={total_reward:.2f}")

    env.close()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
