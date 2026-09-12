from __future__ import annotations

import torch
from torch import nn


class SharedLatentRepresentation(nn.Module):
    """Project fused multimodal features into a shared latent space."""

    def __init__(
        self,
        input_dim: int = 64,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")

        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
            nn.Tanh(),
            nn.LayerNorm(latent_dim),
        )

    @property
    def input_dim(self) -> int:
        """Return the expected fused input dimension."""
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        """Return the shared latent dimension."""
        return self._latent_dim

    @property
    def hidden_dim(self) -> int:
        """Return the hidden projection dimension."""
        return self._hidden_dim

    def forward(
        self,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        """Project fused features into the shared latent space."""
        if fused.ndim != 2:
            raise ValueError("fused input must have shape [batch, features]")

        if fused.shape[1] != self._input_dim:
            raise ValueError(f"fused input must have {self._input_dim} features")

        if not torch.is_floating_point(fused):
            raise TypeError("fused input must use a floating-point dtype")

        return self.network(fused)
