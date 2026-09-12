"""Tests for truncated SVD model compression."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from q_vla_forge.compression import (
    compress_model_svd,
    profitable_svd_max_rank,
    reconstruct_svd,
    relative_frobenius_error,
    resolve_rank,
    truncated_svd,
)
from q_vla_forge.models import SharedVLAModel


def test_profitable_rank_is_positive_for_large_matrix() -> None:
    rank = profitable_svd_max_rank(
        64,
        64,
    )

    assert rank == 31


def test_profitable_rank_reduces_storage() -> None:
    output_dim = 64
    input_dim = 64

    rank = profitable_svd_max_rank(
        output_dim,
        input_dim,
    )

    original_parameters = output_dim * input_dim

    compressed_parameters = rank * (output_dim + input_dim + 1)

    assert compressed_parameters < original_parameters


def test_profitable_rank_rejects_invalid_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="output_dim must be greater than zero",
    ):
        profitable_svd_max_rank(
            0,
            64,
        )

    with pytest.raises(
        ValueError,
        match="input_dim must be greater than zero",
    ):
        profitable_svd_max_rank(
            64,
            0,
        )


def test_resolve_rank_matches_expected_values() -> None:
    assert (
        resolve_rank(
            64,
            64,
            0.25,
        )
        == 8
    )

    assert (
        resolve_rank(
            64,
            64,
            0.50,
        )
        == 16
    )

    assert (
        resolve_rank(
            64,
            64,
            0.75,
        )
        == 23
    )


def test_resolve_rank_rejects_invalid_fraction() -> None:
    with pytest.raises(
        ValueError,
        match="rank_fraction must be in",
    ):
        resolve_rank(
            64,
            64,
            0.0,
        )

    with pytest.raises(
        ValueError,
        match="rank_fraction must be in",
    ):
        resolve_rank(
            64,
            64,
            1.1,
        )


def test_truncated_svd_shapes() -> None:
    torch.manual_seed(42)

    weight = torch.randn(
        32,
        64,
    )

    u, singular_values, vh = truncated_svd(
        weight,
        rank=8,
    )

    assert u.shape == (
        32,
        8,
    )

    assert singular_values.shape == (8,)

    assert vh.shape == (
        8,
        64,
    )


def test_full_rank_reconstruction_is_close() -> None:
    torch.manual_seed(42)

    weight = torch.randn(
        16,
        8,
    )

    u, singular_values, vh = truncated_svd(
        weight,
        rank=8,
    )

    reconstructed = reconstruct_svd(
        u,
        singular_values,
        vh,
    )

    assert torch.allclose(
        weight,
        reconstructed,
        atol=1e-5,
        rtol=1e-5,
    )


def test_lower_rank_has_finite_nonnegative_error() -> None:
    torch.manual_seed(42)

    weight = torch.randn(
        32,
        32,
    )

    u, singular_values, vh = truncated_svd(
        weight,
        rank=4,
    )

    reconstructed = reconstruct_svd(
        u,
        singular_values,
        vh,
    )

    error = relative_frobenius_error(
        weight,
        reconstructed,
    )

    assert math.isfinite(error)

    assert error >= 0.0


def test_higher_rank_not_worse_reconstruction() -> None:
    torch.manual_seed(42)

    weight = torch.randn(
        32,
        32,
    )

    low_rank = truncated_svd(
        weight,
        rank=2,
    )

    high_rank = truncated_svd(
        weight,
        rank=8,
    )

    low_error = relative_frobenius_error(
        weight,
        reconstruct_svd(*low_rank),
    )

    high_error = relative_frobenius_error(
        weight,
        reconstruct_svd(*high_rank),
    )

    assert high_error <= low_error


def test_model_compression_preserves_type() -> None:
    model = SharedVLAModel()

    compressed, _ = compress_model_svd(
        model,
        rank_fraction=0.5,
    )

    assert isinstance(
        compressed,
        SharedVLAModel,
    )


def test_model_compression_does_not_mutate_original() -> None:
    model = SharedVLAModel()

    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    compress_model_svd(
        model,
        rank_fraction=0.5,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            before[name],
        )


def test_model_report_matches_expected_targets() -> None:
    _, report = compress_model_svd(
        SharedVLAModel(),
        rank_fraction=0.5,
    )

    assert report.total_parameters == 76179

    assert report.compressed_layer_weight_parameters == 34816

    assert len(report.layers) == 7


def test_exact_selected_layer_set() -> None:
    _, report = compress_model_svd(
        SharedVLAModel(),
        rank_fraction=0.5,
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


def test_model_effective_parameter_reduction() -> None:
    _, report = compress_model_svd(
        SharedVLAModel(),
        rank_fraction=0.5,
    )

    assert report.effective_stored_parameters == 58784

    assert report.effective_stored_parameters < report.total_parameters

    assert report.parameter_reduction_percent > 0.0


def test_model_storage_compression() -> None:
    _, report = compress_model_svd(
        SharedVLAModel(),
        rank_fraction=0.5,
    )

    assert report.compressed_size_bytes < report.baseline_size_bytes

    assert report.compression_ratio == pytest.approx(1.295913174)


def test_every_selected_layer_is_locally_profitable() -> None:
    _, report = compress_model_svd(
        SharedVLAModel(),
        rank_fraction=1.0,
    )

    for layer in report.layers:
        assert layer.compressed_parameters < layer.original_parameters

        assert layer.compression_ratio > 1.0


def test_smaller_rank_fraction_compresses_more() -> None:
    model = SharedVLAModel()

    _, small = compress_model_svd(
        model,
        rank_fraction=0.25,
    )

    _, medium = compress_model_svd(
        model,
        rank_fraction=0.50,
    )

    _, large = compress_model_svd(
        model,
        rank_fraction=0.75,
    )

    assert (
        small.compressed_size_bytes
        < medium.compressed_size_bytes
        < large.compressed_size_bytes
    )

    assert small.compression_ratio > medium.compression_ratio > large.compression_ratio


def test_parameterless_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model must contain parameters",
    ):
        compress_model_svd(
            nn.Identity(),
            rank_fraction=0.5,
        )


def test_invalid_minimum_weight_size_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_weight_parameters",
    ):
        compress_model_svd(
            SharedVLAModel(),
            rank_fraction=0.5,
            minimum_weight_parameters=0,
        )


def test_small_linear_is_not_selected() -> None:
    model = nn.Sequential(
        nn.Linear(
            4,
            4,
        )
    )

    _, report = compress_model_svd(
        model,
        rank_fraction=0.5,
        minimum_weight_parameters=1024,
    )

    assert not report.layers
