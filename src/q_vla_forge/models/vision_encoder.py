from __future__ import annotations

import torch
from torch import nn


class SharedVisionEncoder(nn.Module):
    """Lightweight CNN shared by driving and robotics observations."""

    def __init__(
        self,
        output_dim: int = 64,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if input_channels <= 0:
            raise ValueError("input_channels must be greater than zero")

        self._output_dim = output_dim
        self._input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    @property
    def output_dim(self) -> int:
        """Return the vision embedding dimension."""
        return self._output_dim

    @property
    def input_channels(self) -> int:
        """Return the expected number of image channels."""
        return self._input_channels

    def forward(
        self,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of visual observations."""
        if visual.ndim != 4:
            raise ValueError(
                "visual input must have shape " "[batch, channels, height, width]"
            )

        if visual.shape[1] != self._input_channels:
            raise ValueError(
                f"expected {self._input_channels} visual channels, "
                f"received {visual.shape[1]}"
            )

        if not torch.is_floating_point(visual):
            raise TypeError("visual input must use a floating-point dtype")

        features = self.features(visual)

        return self.projection(features)
