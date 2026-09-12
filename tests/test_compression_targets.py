from __future__ import annotations

import pytest

from q_vla_forge.compression.targets import (
    CompressionTarget,
    module_matches_target,
    selected_linear_layer_names,
)
from q_vla_forge.models import SharedVLAModel


def test_fusion_name_matches() -> None:
    assert module_matches_target(
        "fusion.network.0",
        CompressionTarget.FUSION,
    )


def test_latent_name_matches() -> None:
    assert module_matches_target(
        "latent.network.0",
        CompressionTarget.LATENT,
    )


def test_action_name_matches() -> None:
    assert module_matches_target(
        "action_head.network.0",
        CompressionTarget.ACTION,
    )


def test_fusion_does_not_match_latent() -> None:
    assert not module_matches_target(
        "fusion.network.0",
        CompressionTarget.LATENT,
    )


def test_fusion_latent_matches_both() -> None:
    assert module_matches_target(
        "fusion.network.0",
        CompressionTarget.FUSION_LATENT,
    )

    assert module_matches_target(
        "latent.network.0",
        CompressionTarget.FUSION_LATENT,
    )


def test_all_matches_any_name() -> None:
    assert module_matches_target(
        "vision.projection",
        CompressionTarget.ALL,
    )


@pytest.mark.parametrize(
    "target",
    [
        CompressionTarget.FUSION,
        CompressionTarget.LATENT,
        CompressionTarget.ACTION,
        CompressionTarget.FUSION_LATENT,
        CompressionTarget.ALL,
    ],
)
def test_model_target_selection_returns_tuple(
    target: CompressionTarget,
) -> None:
    model = SharedVLAModel()

    result = selected_linear_layer_names(
        model,
        target,
    )

    assert isinstance(
        result,
        tuple,
    )


def test_all_contains_at_least_fusion_latent_action() -> None:
    model = SharedVLAModel()

    all_layers = set(
        selected_linear_layer_names(
            model,
            CompressionTarget.ALL,
        )
    )

    fusion_latent = set(
        selected_linear_layer_names(
            model,
            CompressionTarget.FUSION_LATENT,
        )
    )

    action = set(
        selected_linear_layer_names(
            model,
            CompressionTarget.ACTION,
        )
    )

    assert fusion_latent.issubset(all_layers)

    assert action.issubset(all_layers)


def test_minimum_parameter_filter() -> None:
    model = SharedVLAModel()

    unfiltered = selected_linear_layer_names(
        model,
        CompressionTarget.ACTION,
        minimum_weight_parameters=0,
    )

    filtered = selected_linear_layer_names(
        model,
        CompressionTarget.ACTION,
        minimum_weight_parameters=1024,
    )

    assert len(filtered) <= len(unfiltered)


def test_negative_minimum_rejected() -> None:
    model = SharedVLAModel()

    with pytest.raises(ValueError):
        selected_linear_layer_names(
            model,
            CompressionTarget.ALL,
            minimum_weight_parameters=-1,
        )
