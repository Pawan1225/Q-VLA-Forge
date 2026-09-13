"""Parameter-matched classical PPO actor for Sprint 4.12."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class MatchedClassicalPolicyConfig:
    """Frozen Sprint 4.12 matched-classical configuration."""

    latent_features: int = 4
    critic_hidden_dim: int = 32
    projection_scale: float = math.pi
    initial_log_std: float = -0.5

    def __post_init__(self) -> None:
        if self.latent_features != 4:
            raise ValueError("Sprint 4.12 freezes latent width to 4")

        if self.critic_hidden_dim != 32:
            raise ValueError("Sprint 4.12 freezes critic width to 32")

        if not math.isclose(
            self.projection_scale,
            math.pi,
        ):
            raise ValueError("Sprint 4.12 freezes projection scale to pi")


DEFAULT_MATCHED_CLASSICAL_CONFIG = MatchedClassicalPolicyConfig()


class MatchedClassicalActorCritic(nn.Module):
    """Classical actor exactly parameter-matched to the hybrid QML actor."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: MatchedClassicalPolicyConfig = (DEFAULT_MATCHED_CLASSICAL_CONFIG),
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
            raise ValueError("every upper bound must exceed lower bound")

        self.observation_dim = observation_dim

        self.action_dim = int(action_low.shape[0])

        self.config = config

        self.classical_projection = nn.Linear(
            observation_dim,
            config.latent_features,
        )

        self.classical_core = nn.Linear(
            config.latent_features,
            config.latent_features,
            bias=False,
        )

        self.action_mean_head = nn.Linear(
            config.latent_features,
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

        self.action_low: torch.Tensor
        self.action_high: torch.Tensor

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

    def projected_features(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Match the hybrid actor's four-dimensional input projection."""

        self._validate_observations(observations)

        projected = self.classical_projection(observations)

        return self.config.projection_scale * torch.tanh(projected)

    def actor_features(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return four bounded classical latent features."""

        projected = self.projected_features(observations)

        return torch.tanh(self.classical_core(projected))

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> Normal:
        """Return the Gaussian action distribution."""

        features = self.actor_features(observations)

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
        """Return critic values."""

        self._validate_observations(observations)

        features = self.critic_body(observations)

        return self.value_head(features).squeeze(-1)

    def scale_raw_action(
        self,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """Squash and scale raw actions to environment bounds."""

        squashed = torch.tanh(raw_action)

        return self.action_low + 0.5 * (squashed + 1.0) * (
            self.action_high - self.action_low
        )

    def sample_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a raw Gaussian action with log-probability and value."""

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
        """Return bounded deterministic action."""

        distribution = self.distribution(observation)

        return self.scale_raw_action(distribution.mean)

    @property
    def projection_parameter_count(
        self,
    ) -> int:
        """Return projection parameter count."""

        return sum(
            parameter.numel() for parameter in self.classical_projection.parameters()
        )

    @property
    def core_parameter_count(
        self,
    ) -> int:
        """Return classical-core parameter count."""

        return sum(parameter.numel() for parameter in self.classical_core.parameters())

    @property
    def action_head_parameter_count(
        self,
    ) -> int:
        """Return action-head parameter count."""

        return sum(
            parameter.numel() for parameter in self.action_mean_head.parameters()
        )

    @property
    def actor_parameter_count(
        self,
    ) -> int:
        """Return total trainable actor parameter count."""

        return (
            self.projection_parameter_count
            + self.core_parameter_count
            + self.action_head_parameter_count
            + int(self.log_std.numel())
        )

    @property
    def critic_parameter_count(
        self,
    ) -> int:
        """Return total critic parameter count."""

        return sum(
            parameter.numel() for parameter in self.critic_body.parameters()
        ) + sum(parameter.numel() for parameter in self.value_head.parameters())

    @property
    def total_parameter_count(
        self,
    ) -> int:
        """Return all trainable model parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_observations(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Validate observation shape."""

        if observations.ndim not in (
            1,
            2,
        ):
            raise ValueError("observations must have rank 1 or 2")

        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observation feature dimension mismatch")
