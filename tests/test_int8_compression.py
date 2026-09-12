"""Tests for symmetric INT8 weight compression."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from q_vla_forge.compression import (
    dequantize_tensor,
    quantize_model_weights_int8,
    quantize_tensor_symmetric,
)
from q_vla_forge.models import SharedVLAModel


def test_tensor_quantization_returns_int8() -> None:
    tensor = torch.tensor(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ],
        dtype=torch.float32,
    )

    quantized, scale = quantize_tensor_symmetric(tensor)

    assert quantized.dtype == torch.int8
    assert scale > 0.0


def test_tensor_quantization_range() -> None:
    tensor = torch.linspace(
        -10.0,
        10.0,
        100,
    )

    quantized, _ = quantize_tensor_symmetric(tensor)

    assert int(quantized.min()) >= -127
    assert int(quantized.max()) <= 127


def test_zero_tensor_quantization() -> None:
    tensor = torch.zeros(
        16,
        dtype=torch.float32,
    )

    quantized, scale = quantize_tensor_symmetric(tensor)

    assert scale == 1.0

    assert torch.count_nonzero(quantized).item() == 0


def test_quantize_dequantize_is_close() -> None:
    tensor = torch.linspace(
        -1.0,
        1.0,
        100,
    )

    quantized, scale = quantize_tensor_symmetric(tensor)

    restored = dequantize_tensor(
        quantized,
        scale,
    )

    max_error = float((tensor - restored).abs().max())

    assert max_error <= (scale / 2.0 + 1e-6)


def test_dequantization_requires_int8() -> None:
    with pytest.raises(
        TypeError,
        match="dtype torch.int8",
    ):
        dequantize_tensor(
            torch.ones(
                3,
                dtype=torch.float32,
            ),
            0.1,
        )


def test_dequantization_rejects_nonpositive_scale() -> None:
    with pytest.raises(
        ValueError,
        match="scale must be greater than zero",
    ):
        dequantize_tensor(
            torch.ones(
                3,
                dtype=torch.int8,
            ),
            0.0,
        )


def test_quantization_requires_float() -> None:
    with pytest.raises(
        TypeError,
        match="floating-point dtype",
    ):
        quantize_tensor_symmetric(
            torch.ones(
                3,
                dtype=torch.int32,
            )
        )


def test_model_quantization_preserves_type() -> None:
    model = SharedVLAModel()

    compressed, _ = quantize_model_weights_int8(model)

    assert isinstance(
        compressed,
        SharedVLAModel,
    )


def test_model_quantization_does_not_modify_original() -> None:
    model = SharedVLAModel()

    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    quantize_model_weights_int8(model)

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            before[name],
        )


def test_int8_report_matches_frozen_model() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    assert report.total_parameters == 76179
    assert report.quantized_weight_parameters == 75184
    assert report.untouched_parameters == 995
    assert report.baseline_size_bytes == 304716
    assert report.compressed_size_bytes == 79224


def test_int8_report_has_expected_compression_ratio() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    assert report.compression_ratio == pytest.approx(3.846255174)

    assert report.storage_reduction_percent == pytest.approx(74.00073511)


def test_model_parameter_count_is_unchanged() -> None:
    compressed, report = quantize_model_weights_int8(SharedVLAModel())

    compressed_parameters = sum(
        parameter.numel() for parameter in compressed.parameters()
    )

    assert compressed_parameters == report.total_parameters


def test_quantization_detects_exact_tensor_count() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    assert len(report.quantized_tensors) == 15


def test_large_tensor_storage_approaches_four_x() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    largest = max(
        report.quantized_tensors,
        key=lambda item: item.elements,
    )

    local_ratio = largest.original_bytes / largest.quantized_bytes

    assert local_ratio > 3.99


def test_each_tensor_storage_includes_scale() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    for item in report.quantized_tensors:
        assert item.quantized_bytes == item.elements + 4


def test_quantized_tensor_storage_is_smaller() -> None:
    _, report = quantize_model_weights_int8(SharedVLAModel())

    for item in report.quantized_tensors:
        assert item.quantized_bytes < item.original_bytes


def test_simple_linear_output_remains_close() -> None:
    torch.manual_seed(42)

    model = nn.Linear(
        32,
        16,
    )

    inputs = torch.randn(
        8,
        32,
    )

    expected = model(inputs)

    compressed, _ = quantize_model_weights_int8(model)

    actual = compressed(inputs)

    mae = (expected - actual).abs().mean()

    mae_value = float(mae.detach())

    assert math.isfinite(mae_value)

    assert mae_value < 0.05


def test_parameterless_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model must contain parameters",
    ):
        quantize_model_weights_int8(nn.Identity())
