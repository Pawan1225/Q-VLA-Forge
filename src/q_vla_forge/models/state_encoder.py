from __future__ import annotations

import torch
from torch import nn

from q_vla_forge.data import Domain


class DomainStateEncoder(nn.Module):
    """Encode domain-specific state vectors into a shared representation."""

    def __init__(
        self,
        output_dim: int = 16,
        driving_input_dim: int = 4,
        robotics_input_dim: int = 6,
        adapter_dim: int = 16,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if driving_input_dim <= 0:
            raise ValueError("driving_input_dim must be greater than zero")

        if robotics_input_dim <= 0:
            raise ValueError("robotics_input_dim must be greater than zero")

        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be greater than zero")

        self._output_dim = output_dim
        self._driving_input_dim = driving_input_dim
        self._robotics_input_dim = robotics_input_dim
        self._adapter_dim = adapter_dim

        self.driving_adapter = nn.Sequential(
            nn.Linear(
                driving_input_dim,
                adapter_dim,
            ),
            nn.ReLU(),
        )

        self.robotics_adapter = nn.Sequential(
            nn.Linear(
                robotics_input_dim,
                adapter_dim,
            ),
            nn.ReLU(),
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(
                adapter_dim,
                output_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    @property
    def output_dim(self) -> int:
        """Return the shared state embedding dimension."""
        return self._output_dim

    @property
    def driving_input_dim(self) -> int:
        """Return the expected driving state dimension."""
        return self._driving_input_dim

    @property
    def robotics_input_dim(self) -> int:
        """Return the expected robotics state dimension."""
        return self._robotics_input_dim

    @property
    def adapter_dim(self) -> int:
        """Return the intermediate adapter dimension."""
        return self._adapter_dim

    def forward(
        self,
        state: torch.Tensor,
        domain: Domain,
    ) -> torch.Tensor:
        """Encode a batch of state vectors for a given domain."""
        if state.ndim != 2:
            raise ValueError("state input must have shape [batch, features]")

        if not torch.is_floating_point(state):
            raise TypeError("state input must use a floating-point dtype")

        if domain == Domain.AUTONOMOUS_DRIVING:
            expected_dim = self._driving_input_dim

            if state.shape[1] != expected_dim:
                raise ValueError(f"driving state must have {expected_dim} features")

            adapted = self.driving_adapter(state)

        elif domain == Domain.ROBOTICS:
            expected_dim = self._robotics_input_dim

            if state.shape[1] != expected_dim:
                raise ValueError(f"robotics state must have {expected_dim} features")

            adapted = self.robotics_adapter(state)

        else:
            raise ValueError(f"unsupported domain: {domain}")

        return self.shared_encoder(adapted)
