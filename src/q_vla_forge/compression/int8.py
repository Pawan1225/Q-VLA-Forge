"""Symmetric INT8 weight quantization for Sprint 2 compression."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QuantizedTensorInfo:
    """Storage information for one INT8-encoded weight tensor."""

    name: str
    shape: tuple[int, ...]
    elements: int
    original_bytes: int
    quantized_bytes: int
    scale: float


@dataclass(frozen=True)
class Int8QuantizationReport:
    """Whole-model report for simulated INT8 weight storage."""

    total_parameters: int
    quantized_weight_parameters: int
    untouched_parameters: int

    baseline_size_bytes: int
    compressed_size_bytes: int

    quantized_tensors: tuple[QuantizedTensorInfo, ...]

    @property
    def compression_ratio(self) -> float:
        """Return whole-model storage compression ratio."""
        return self.baseline_size_bytes / self.compressed_size_bytes

    @property
    def storage_reduction_percent(self) -> float:
        """Return whole-model byte reduction percentage."""
        return (1.0 - (self.compressed_size_bytes / self.baseline_size_bytes)) * 100.0


def quantize_tensor_symmetric(
    tensor: torch.Tensor,
) -> tuple[
    torch.Tensor,
    float,
]:
    """Quantize one floating-point tensor to signed symmetric INT8."""
    if not torch.is_floating_point(tensor):
        raise TypeError("tensor must use a floating-point dtype")

    if tensor.numel() == 0:
        raise ValueError("tensor must not be empty")

    max_abs = float(tensor.detach().abs().max().item())

    if max_abs == 0.0:
        return (
            torch.zeros_like(
                tensor,
                dtype=torch.int8,
            ),
            1.0,
        )

    scale = max_abs / 127.0

    quantized = torch.clamp(
        torch.round(tensor.detach() / scale),
        min=-127,
        max=127,
    ).to(torch.int8)

    return (
        quantized,
        float(scale),
    )


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an INT8 tensor to a floating-point tensor."""
    if quantized.dtype != torch.int8:
        raise TypeError("quantized tensor must have dtype torch.int8")

    if scale <= 0.0:
        raise ValueError("scale must be greater than zero")

    return quantized.to(dtype) * scale


def _eligible_module(
    module: nn.Module,
) -> bool:
    """Return whether a module is eligible for Sprint 2.3 INT8."""
    return isinstance(
        module,
        (
            nn.Linear,
            nn.Conv2d,
            nn.Embedding,
        ),
    )


def quantize_model_weights_int8(
    model: nn.Module,
) -> tuple[
    nn.Module,
    Int8QuantizationReport,
]:
    """
    Quantize eligible weights to INT8 storage and reconstruct them.

    The returned model keeps the original architecture and contains
    FP32-dequantized weights. This isolates quantization error from
    backend-specific native INT8 kernel behavior.
    """
    compressed_model = copy.deepcopy(model)

    total_parameters = sum(
        parameter.numel() for parameter in compressed_model.parameters()
    )

    if total_parameters <= 0:
        raise ValueError("model must contain parameters")

    baseline_size_bytes = total_parameters * 4

    quantized_weight_parameters = 0
    quantized_storage_bytes = 0

    tensor_reports: list[QuantizedTensorInfo] = []

    with torch.no_grad():
        for (
            name,
            module,
        ) in compressed_model.named_modules():
            if not _eligible_module(module):
                continue

            weight = getattr(
                module,
                "weight",
                None,
            )

            if weight is None:
                continue

            quantized, scale = quantize_tensor_symmetric(weight)

            reconstructed = dequantize_tensor(
                quantized,
                scale,
                dtype=weight.dtype,
            )

            weight.copy_(reconstructed)

            elements = int(weight.numel())

            original_bytes = elements * 4

            # INT8 payload plus one FP32 scale.
            compressed_bytes = elements + 4

            quantized_weight_parameters += elements

            quantized_storage_bytes += compressed_bytes

            tensor_reports.append(
                QuantizedTensorInfo(
                    name=name,
                    shape=tuple(int(value) for value in weight.shape),
                    elements=elements,
                    original_bytes=(original_bytes),
                    quantized_bytes=(compressed_bytes),
                    scale=scale,
                )
            )

    untouched_parameters = total_parameters - quantized_weight_parameters

    untouched_storage_bytes = untouched_parameters * 4

    compressed_size_bytes = quantized_storage_bytes + untouched_storage_bytes

    return (
        compressed_model,
        Int8QuantizationReport(
            total_parameters=(total_parameters),
            quantized_weight_parameters=(quantized_weight_parameters),
            untouched_parameters=(untouched_parameters),
            baseline_size_bytes=(baseline_size_bytes),
            compressed_size_bytes=(compressed_size_bytes),
            quantized_tensors=tuple(tensor_reports),
        ),
    )
