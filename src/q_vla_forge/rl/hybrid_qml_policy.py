"""Hybrid quantum-classical PPO-compatible actor-critic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from q_vla_forge.quantum.pqc import (
    DEFAULT_PQC_CONFIG,
    PQCConfig,
    VariationalQuantumCircuit,
)


@dataclass(frozen=True)
class HybridQMLPolicyConfig:
    """Frozen Sprint 4.7 hybrid policy configuration."""

    quantum_features: int = 4
    critic_hidden_dim: int = 32
    angle_scale: float = math.pi
    initial_log_std: float = -0.5
    pqc_seed: int = 42

    def __post_init__(self) -> None:
        if self.quantum_features != 4:
            raise ValueError(
                "Sprint 4.7 freezes the hybrid policy " "to four quantum features"
            )

        if self.critic_hidden_dim != 32:
            raise ValueError("Sprint 4.7 freezes critic hidden width to 32")

        if not math.isclose(
            self.angle_scale,
            math.pi,
        ):
            raise ValueError("Sprint 4.7 freezes angle scaling to pi")


DEFAULT_HYBRID_QML_POLICY_CONFIG = HybridQMLPolicyConfig()


class HybridQuantumActorCritic(nn.Module):
    """PPO-compatible actor with a PQC and classical critic."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: HybridQMLPolicyConfig = (DEFAULT_HYBRID_QML_POLICY_CONFIG),
        pqc_config: PQCConfig = DEFAULT_PQC_CONFIG,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")

        action_low = np.asarray(
            action_low,
            dtype=np.float32,
        )

        action_high = np.asarray(
            action_high,
            dtype=np.float32,
        )

        if action_low.ndim != 1:
            raise ValueError("action bounds must be rank-1")

        if action_low.shape != action_high.shape:
            raise ValueError("action bounds must have matching shapes")

        if np.any(action_high <= action_low):
            raise ValueError("every action upper bound must exceed lower bound")

        if pqc_config.input_features != config.quantum_features:
            raise ValueError(
                "PQC input dimension must match " "hybrid quantum feature count"
            )

        self.observation_dim = observation_dim

        self.action_dim = int(action_low.shape[0])

        self.config = config

        self.quantum_projection = nn.Linear(
            observation_dim,
            config.quantum_features,
        )

        self.quantum_circuit = VariationalQuantumCircuit(
            config=pqc_config,
            seed=config.pqc_seed,
        )

        self.action_mean_head = nn.Linear(
            config.quantum_features,
            self.action_dim,
        )

        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dim,),
                float(config.initial_log_std),
                dtype=torch.float32,
            )
        )

        self.critic_body = nn.Sequential(
            nn.Linear(
                observation_dim,
                config.critic_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                config.critic_hidden_dim,
                config.critic_hidden_dim,
            ),
            nn.Tanh(),
        )

        self.value_head = nn.Linear(
            config.critic_hidden_dim,
            1,
        )

        self.register_buffer(
            "action_low",
            torch.as_tensor(
                action_low,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_high",
            torch.as_tensor(
                action_high,
                dtype=torch.float32,
            ),
        )

    def quantum_angles(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Project observations to bounded rotation angles."""
        self._validate_observations(observations)

        projected = self.quantum_projection(observations)

        return self.config.angle_scale * torch.tanh(projected)

    def quantum_features(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return PQC expectation features."""
        angles = self.quantum_angles(observations)

        return self.quantum_circuit(angles)

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> Normal:
        """Return the raw Gaussian actor distribution."""
        features = self.quantum_features(observations)

        means = self.action_mean_head(features)

        standard_deviation = torch.exp(self.log_std).expand_as(means)

        return Normal(
            means,
            standard_deviation,
        )

    def value(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return classical state-value estimates."""
        self._validate_observations(observations)

        features = self.critic_body(observations)

        return self.value_head(features).squeeze(-1)

    def scale_raw_action(
        self,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """Map unconstrained Gaussian actions to environment bounds."""
        squashed = torch.tanh(raw_action)

        action_low = cast(
            torch.Tensor,
            self.action_low,
        )

        action_high = cast(
            torch.Tensor,
            self.action_high,
        )

        return action_low + 0.5 * (squashed + 1.0) * (action_high - action_low)

    def sample_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample one raw Gaussian action for PPO."""
        distribution = self.distribution(observation)

        raw_action = distribution.sample()

        log_probability = distribution.log_prob(raw_action).sum(dim=-1)

        value = self.value(observation)

        return (
            raw_action,
            log_probability,
            value,
        )

    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded action from Gaussian mean."""
        distribution = self.distribution(observation)

        return self.scale_raw_action(distribution.mean)

    @property
    def quantum_parameter_count(
        self,
    ) -> int:
        return self.quantum_circuit.quantum_parameter_count

    @property
    def projection_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel() for parameter in self.quantum_projection.parameters()
        )

    @property
    def action_head_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel() for parameter in self.action_mean_head.parameters()
        )

    @property
    def actor_parameter_count(
        self,
    ) -> int:
        return (
            self.projection_parameter_count
            + self.quantum_parameter_count
            + self.action_head_parameter_count
            + self.log_std.numel()
        )

    @property
    def critic_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel() for parameter in self.critic_body.parameters()
        ) + sum(parameter.numel() for parameter in self.value_head.parameters())

    @property
    def total_parameter_count(
        self,
    ) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_observations(
        self,
        observations: torch.Tensor,
    ) -> None:
        if observations.ndim not in (
            1,
            2,
        ):
            raise ValueError("observation tensor must have rank 1 or 2")

        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observation feature dimension mismatch")
