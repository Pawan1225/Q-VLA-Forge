from __future__ import annotations

from enum import Enum

from torch import nn


class CompressionTarget(str, Enum):
    """Architectural targets used by the compression ablation."""

    FUSION = "fusion"
    LATENT = "latent"
    ACTION = "action"
    FUSION_LATENT = "fusion_latent"
    ALL = "all"


def module_matches_target(
    module_name: str,
    target: CompressionTarget,
) -> bool:
    """Return whether a named module belongs to an ablation target."""
    name = module_name.lower()

    is_fusion = "fusion" in name
    is_latent = "latent" in name
    is_action = "action" in name or "action_head" in name

    if target == CompressionTarget.FUSION:
        return is_fusion

    if target == CompressionTarget.LATENT:
        return is_latent

    if target == CompressionTarget.ACTION:
        return is_action

    if target == CompressionTarget.FUSION_LATENT:
        return is_fusion or is_latent

    if target == CompressionTarget.ALL:
        return True

    raise ValueError(f"unsupported compression target: {target}")


def selected_linear_layer_names(
    model: nn.Module,
    target: CompressionTarget,
    *,
    minimum_weight_parameters: int = 0,
) -> tuple[str, ...]:
    """Return eligible Linear layer names for one target."""
    if minimum_weight_parameters < 0:
        raise ValueError("minimum_weight_parameters must be >= 0")

    selected: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        if not module_matches_target(
            name,
            target,
        ):
            continue

        if module.weight.numel() < minimum_weight_parameters:
            continue

        selected.append(name)

    return tuple(selected)
