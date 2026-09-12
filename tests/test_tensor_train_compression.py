"""Tests for Tensor Train / TT-SVD compression."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from q_vla_forge.compression import (
    compress_model_tt,
    detensorize_matrix,
    factor_dimension,
    interleaved_tensorize_matrix,
    reconstruct_tt,
    relative_tt_error,
    tt_svd,
)
from q_vla_forge.models import SharedVLAModel


def test_factor_dimension_64() -> None:
    factors = factor_dimension(
        64,
        order=3,
    )

    assert math.prod(factors) == 64

    assert len(factors) == 3


def test_factor_dimension_112() -> None:
    factors = factor_dimension(
        112,
        order=3,
    )

    assert math.prod(factors) == 112


def test_factor_dimension_rejects_invalid_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        factor_dimension(
            0,
            order=3,
        )


def test_tensorization_round_trip() -> None:
    torch.manual_seed(42)

    matrix = torch.randn(
        64,
        32,
    )

    row_factors = (
        4,
        4,
        4,
    )

    column_factors = (
        2,
        4,
        4,
    )

    tensor, _ = interleaved_tensorize_matrix(
        matrix,
        row_factors,
        column_factors,
    )

    restored = detensorize_matrix(
        tensor,
        row_factors,
        column_factors,
    )

    assert torch.equal(
        matrix,
        restored,
    )


def test_tt_svd_returns_correct_number_of_cores() -> None:
    torch.manual_seed(42)

    tensor = torch.randn(
        16,
        16,
        16,
    )

    decomposition = tt_svd(
        tensor,
        max_rank=4,
    )

    assert len(decomposition.cores) == 3

    assert len(decomposition.ranks) == 4


def test_tt_boundary_ranks_are_one() -> None:
    tensor = torch.randn(
        8,
        8,
        8,
    )

    decomposition = tt_svd(
        tensor,
        max_rank=4,
    )

    assert decomposition.ranks[0] == 1
    assert decomposition.ranks[-1] == 1


def test_tt_internal_ranks_respect_limit() -> None:
    tensor = torch.randn(
        16,
        16,
        16,
    )

    decomposition = tt_svd(
        tensor,
        max_rank=4,
    )

    assert all(rank <= 4 for rank in decomposition.ranks)


def test_high_rank_reconstruction_is_close() -> None:
    torch.manual_seed(42)

    tensor = torch.randn(
        4,
        4,
        4,
    )

    decomposition = tt_svd(
        tensor,
        max_rank=16,
    )

    restored = reconstruct_tt(decomposition)

    assert torch.allclose(
        tensor,
        restored,
        atol=1e-5,
        rtol=1e-5,
    )


def test_relative_tt_error_nonnegative() -> None:
    torch.manual_seed(42)

    tensor = torch.randn(
        8,
        8,
        8,
    )

    decomposition = tt_svd(
        tensor,
        max_rank=2,
    )

    restored = reconstruct_tt(decomposition)

    error = relative_tt_error(
        tensor,
        restored,
    )

    assert math.isfinite(error)

    assert error >= 0.0


def test_higher_tt_rank_not_worse() -> None:
    torch.manual_seed(42)

    tensor = torch.randn(
        8,
        8,
        8,
    )

    low = reconstruct_tt(
        tt_svd(
            tensor,
            max_rank=2,
        )
    )

    high = reconstruct_tt(
        tt_svd(
            tensor,
            max_rank=4,
        )
    )

    low_error = relative_tt_error(
        tensor,
        low,
    )

    high_error = relative_tt_error(
        tensor,
        high,
    )

    assert high_error <= low_error + 1e-6


def test_model_tt_preserves_type() -> None:
    compressed, _ = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    assert isinstance(
        compressed,
        SharedVLAModel,
    )


def test_model_tt_does_not_modify_original() -> None:
    model = SharedVLAModel()

    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    compress_model_tt(
        model,
        max_rank=4,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            before[name],
        )


def test_model_tt_matches_expected_target_count() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    assert report.total_parameters == 76179

    assert report.compressed_layer_weight_parameters == 34816

    assert len(report.layers) == 7


def test_exact_tt_selected_layer_set() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    selected = {layer.name for layer in report.layers}

    expected = {
        "vision_encoder.projection.1",
        "language_encoder.projection.0",
        "fusion.network.0",
        "fusion.network.2",
        "latent.network.0",
        "latent.network.2",
        "action_head.network.0",
    }

    assert selected == expected


def test_tt_selected_layers_are_profitable() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    for layer in report.layers:
        assert layer.compressed_parameters < layer.original_parameters

        assert layer.compression_ratio > 1.0


def test_tt_rank_two_expected_storage() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=2,
    )

    assert report.effective_stored_parameters == 42307

    assert report.compression_ratio == pytest.approx(1.8006240102110762)


def test_tt_rank_four_expected_storage() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    assert report.effective_stored_parameters == 44147

    assert report.compression_ratio == pytest.approx(1.7255759168233402)


def test_tt_rank_eight_expected_storage() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=8,
    )

    assert report.effective_stored_parameters == 49459

    assert report.compression_ratio == pytest.approx(1.5402454558321033)


def test_rank_two_compresses_more_than_rank_four_and_eight() -> None:
    model = SharedVLAModel()

    _, rank_two = compress_model_tt(
        model,
        max_rank=2,
    )

    _, rank_four = compress_model_tt(
        model,
        max_rank=4,
    )

    _, rank_eight = compress_model_tt(
        model,
        max_rank=8,
    )

    assert (
        rank_two.compressed_size_bytes
        < rank_four.compressed_size_bytes
        < rank_eight.compressed_size_bytes
    )

    assert (
        rank_two.compression_ratio
        > rank_four.compression_ratio
        > rank_eight.compression_ratio
    )


def test_rank_eight_can_use_lower_feasible_internal_rank() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=8,
    )

    projection = next(
        layer
        for layer in report.layers
        if layer.name == "language_encoder.projection.0"
    )

    assert projection.requested_rank == 8

    assert projection.actual_ranks == (
        1,
        4,
        8,
        1,
    )


def test_parameterless_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model must contain parameters",
    ):
        compress_model_tt(
            nn.Identity(),
            max_rank=2,
        )


def test_invalid_rank_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_rank must be greater than zero",
    ):
        compress_model_tt(
            SharedVLAModel(),
            max_rank=0,
        )


def test_small_linear_not_selected() -> None:
    model = nn.Sequential(
        nn.Linear(
            4,
            4,
        )
    )

    _, report = compress_model_tt(
        model,
        max_rank=2,
        minimum_weight_parameters=1024,
    )

    assert not report.layers


def test_report_rank_matches_request() -> None:
    _, report = compress_model_tt(
        SharedVLAModel(),
        max_rank=4,
    )

    assert report.requested_rank == 4
