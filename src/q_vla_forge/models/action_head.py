from __future__ import annotations

import torch
from torch import nn


class SharedActionHead(nn.Module):
    """Predict a shared three-dimensional continuous action."""

    def __init__(
        self,
        input_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")

        if action_dim <= 0:
            raise ValueError("action_dim must be greater than zero")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")

        self._input_dim = input_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
            nn.Tanh(),
        )

    @property
    def input_dim(self) -> int:
        """Return the expected latent input dimension."""
        return self._input_dim

    @property
    def action_dim(self) -> int:
        """Return the predicted action dimension."""
        return self._action_dim

    @property
    def hidden_dim(self) -> int:
        """Return the action head hidden dimension."""
        return self._hidden_dim

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a continuous action from a latent representation."""
        if latent.ndim != 2:
            raise ValueError("latent input must have shape [batch, features]")

        if latent.shape[1] != self._input_dim:
            raise ValueError(f"latent input must have {self._input_dim} features")

        if not torch.is_floating_point(latent):
            raise TypeError("latent input must use a floating-point dtype")

        return self.network(latent)
