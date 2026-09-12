from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from q_vla_forge.data import Domain
from q_vla_forge.fusion import MultimodalFusion
from q_vla_forge.models.action_head import SharedActionHead
from q_vla_forge.models.language_encoder import SharedLanguageEncoder
from q_vla_forge.models.latent_encoder import SharedLatentRepresentation
from q_vla_forge.models.state_encoder import DomainStateEncoder
from q_vla_forge.models.vision_encoder import SharedVisionEncoder


class SharedVLAModel(nn.Module):
    """Shared multimodal VLA baseline for driving and robotics."""

    def __init__(self) -> None:
        super().__init__()

        self.vision_encoder = SharedVisionEncoder(
            output_dim=64,
        )

        self.language_encoder = SharedLanguageEncoder(
            output_dim=32,
        )

        self.state_encoder = DomainStateEncoder(
            output_dim=16,
        )

        self.fusion = MultimodalFusion(
            vision_dim=64,
            language_dim=32,
            state_dim=16,
            output_dim=64,
        )

        self.latent = SharedLatentRepresentation(
            input_dim=64,
            latent_dim=32,
        )

        self.action_head = SharedActionHead(
            input_dim=32,
            action_dim=3,
        )

    def forward(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        language_goals: Sequence[str],
        domain: Domain,
    ) -> torch.Tensor:
        """Predict actions from multimodal observations."""
        if visual.shape[0] != state.shape[0]:
            raise ValueError("visual and state batch sizes must match")

        if visual.shape[0] != len(language_goals):
            raise ValueError("visual and language batch sizes must match")

        vision_embedding = self.vision_encoder(visual)

        language_embedding = self.language_encoder(language_goals)

        state_embedding = self.state_encoder(
            state,
            domain,
        )

        fused = self.fusion(
            vision_embedding,
            language_embedding,
            state_embedding,
        )

        latent = self.latent(fused)

        return self.action_head(latent)
