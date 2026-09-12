from __future__ import annotations

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """Fuse vision, language, and state embeddings."""

    def __init__(
        self,
        vision_dim: int = 64,
        language_dim: int = 32,
        state_dim: int = 16,
        output_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if vision_dim <= 0:
            raise ValueError("vision_dim must be greater than zero")

        if language_dim <= 0:
            raise ValueError("language_dim must be greater than zero")

        if state_dim <= 0:
            raise ValueError("state_dim must be greater than zero")

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")

        self._vision_dim = vision_dim
        self._language_dim = language_dim
        self._state_dim = state_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

        self._input_dim = vision_dim + language_dim + state_dim

        self.network = nn.Sequential(
            nn.Linear(
                self._input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    @property
    def vision_dim(self) -> int:
        """Return the expected vision embedding dimension."""
        return self._vision_dim

    @property
    def language_dim(self) -> int:
        """Return the expected language embedding dimension."""
        return self._language_dim

    @property
    def state_dim(self) -> int:
        """Return the expected state embedding dimension."""
        return self._state_dim

    @property
    def input_dim(self) -> int:
        """Return the concatenated multimodal dimension."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Return the fused representation dimension."""
        return self._output_dim

    @property
    def hidden_dim(self) -> int:
        """Return the fusion hidden dimension."""
        return self._hidden_dim

    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse three modality embeddings."""
        self._validate_input(
            tensor=vision,
            name="vision",
            expected_dim=self._vision_dim,
        )

        self._validate_input(
            tensor=language,
            name="language",
            expected_dim=self._language_dim,
        )

        self._validate_input(
            tensor=state,
            name="state",
            expected_dim=self._state_dim,
        )

        batch_size = vision.shape[0]

        if language.shape[0] != batch_size:
            raise ValueError("vision and language batch sizes must match")

        if state.shape[0] != batch_size:
            raise ValueError("vision and state batch sizes must match")

        fused = torch.cat(
            [
                vision,
                language,
                state,
            ],
            dim=1,
        )

        return self.network(fused)

    @staticmethod
    def _validate_input(
        tensor: torch.Tensor,
        name: str,
        expected_dim: int,
    ) -> None:
        """Validate one modality tensor."""
        if tensor.ndim != 2:
            raise ValueError(f"{name} input must have shape [batch, features]")

        if tensor.shape[1] != expected_dim:
            raise ValueError(f"{name} input must have {expected_dim} features")

        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} input must use a floating-point dtype")
