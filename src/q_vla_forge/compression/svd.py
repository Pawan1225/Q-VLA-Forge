"""Truncated SVD compression for dense Q-VLA Forge layers."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SVDLayerReport:
    """Compression information for one SVD-compressed Linear layer."""

    name: str

    input_dim: int
    output_dim: int

    original_parameters: int
    compressed_parameters: int

    rank: int
    maximum_rank: int
    profitable_max_rank: int

    original_bytes: int
    compressed_bytes: int

    relative_reconstruction_error: float

    @property
    def compression_ratio(self) -> float:
        """Return local layer compression ratio."""
        return self.original_bytes / self.compressed_bytes


@dataclass(frozen=True)
class SVDCompressionReport:
    """Whole-model SVD compression report."""

    total_parameters: int

    compressed_layer_weight_parameters: int
    effective_stored_parameters: int

    baseline_size_bytes: int
    compressed_size_bytes: int

    layers: tuple[SVDLayerReport, ...]

    @property
    def compression_ratio(self) -> float:
        """Return whole-model effective compression ratio."""
        return self.baseline_size_bytes / self.compressed_size_bytes

    @property
    def storage_reduction_percent(self) -> float:
        """Return whole-model effective storage reduction."""
        return (1.0 - (self.compressed_size_bytes / self.baseline_size_bytes)) * 100.0

    @property
    def parameter_reduction_percent(self) -> float:
        """Return effective stored-coefficient reduction."""
        return (
            1.0 - (self.effective_stored_parameters / self.total_parameters)
        ) * 100.0


def profitable_svd_max_rank(
    output_dim: int,
    input_dim: int,
) -> int:
    """Return the largest rank that still reduces coefficient storage."""
    if output_dim <= 0:
        raise ValueError("output_dim must be greater than zero")

    if input_dim <= 0:
        raise ValueError("input_dim must be greater than zero")

    original_parameters = output_dim * input_dim

    factor_parameters_per_rank = output_dim + input_dim + 1

    profitable_rank = (original_parameters - 1) // factor_parameters_per_rank

    maximum_rank = min(
        output_dim,
        input_dim,
    )

    return min(
        profitable_rank,
        maximum_rank,
    )


def resolve_rank(
    output_dim: int,
    input_dim: int,
    rank_fraction: float,
) -> int:
    """Resolve rank relative to the maximum profitable SVD rank."""
    if not math.isfinite(rank_fraction):
        raise ValueError("rank_fraction must be finite")

    if not (0.0 < rank_fraction <= 1.0):
        raise ValueError("rank_fraction must be in (0, 1]")

    profitable_rank = profitable_svd_max_rank(
        output_dim,
        input_dim,
    )

    if profitable_rank < 1:
        return 0

    rank = max(
        1,
        round(profitable_rank * rank_fraction),
    )

    return min(
        rank,
        profitable_rank,
    )


def truncated_svd(
    weight: torch.Tensor,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return rank-r truncated SVD factors."""
    if weight.ndim != 2:
        raise ValueError("weight must be a two-dimensional matrix")

    if not torch.is_floating_point(weight):
        raise TypeError("weight must use a floating-point dtype")

    maximum_rank = min(
        int(weight.shape[0]),
        int(weight.shape[1]),
    )

    if rank <= 0:
        raise ValueError("rank must be greater than zero")

    if rank > maximum_rank:
        raise ValueError("rank exceeds matrix maximum rank")

    (
        u,
        singular_values,
        vh,
    ) = torch.linalg.svd(
        weight.detach(),
        full_matrices=False,
    )

    return (
        u[:, :rank],
        singular_values[:rank],
        vh[:rank, :],
    )


def reconstruct_svd(
    u: torch.Tensor,
    singular_values: torch.Tensor,
    vh: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a dense matrix from truncated SVD factors."""
    if u.ndim != 2:
        raise ValueError("u must be two-dimensional")

    if singular_values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")

    if vh.ndim != 2:
        raise ValueError("vh must be two-dimensional")

    rank = int(singular_values.shape[0])

    if u.shape[1] != rank or vh.shape[0] != rank:
        raise ValueError("SVD factor ranks must match")

    return (u * singular_values.unsqueeze(0)) @ vh


def relative_frobenius_error(
    original: torch.Tensor,
    approximation: torch.Tensor,
) -> float:
    """Return relative Frobenius reconstruction error."""
    if original.shape != approximation.shape:
        raise ValueError("matrix shapes must match")

    denominator = torch.linalg.norm(original)

    numerator = torch.linalg.norm(original - approximation)

    denominator_value = float(denominator.item())

    if denominator_value == 0.0:
        if float(numerator.item()) == 0.0:
            return 0.0

        return math.inf

    return float((numerator / denominator).item())


def compress_model_svd(
    model: nn.Module,
    *,
    rank_fraction: float,
    minimum_weight_parameters: int = 1024,
    selected_layer_names: set[str] | None = None,
) -> tuple[
    nn.Module,
    SVDCompressionReport,
]:
    """
    Apply truncated SVD to eligible Linear weights.

    The returned model retains ordinary Linear modules containing
    reconstructed FP32 weights. Reported compressed storage represents
    the U, singular-value, and Vh factors that would be stored instead.

    When selected_layer_names is provided, only Linear modules whose
    names are present in that set are considered for compression.
    """
    if minimum_weight_parameters <= 0:
        raise ValueError("minimum_weight_parameters must be greater than zero")

    compressed_model = copy.deepcopy(model)

    total_parameters = sum(
        parameter.numel() for parameter in compressed_model.parameters()
    )

    if total_parameters <= 0:
        raise ValueError("model must contain parameters")

    baseline_size_bytes = total_parameters * 4

    original_target_parameters = 0
    compressed_target_parameters = 0

    layer_reports: list[SVDLayerReport] = []

    with torch.no_grad():
        for (
            name,
            module,
        ) in compressed_model.named_modules():
            if not isinstance(
                module,
                nn.Linear,
            ):
                continue

            if selected_layer_names is not None and name not in selected_layer_names:
                continue

            weight = module.weight

            if weight.numel() < minimum_weight_parameters:
                continue

            output_dim = int(weight.shape[0])

            input_dim = int(weight.shape[1])

            profitable_rank = profitable_svd_max_rank(
                output_dim,
                input_dim,
            )

            if profitable_rank < 1:
                continue

            rank = resolve_rank(
                output_dim,
                input_dim,
                rank_fraction,
            )

            if rank < 1:
                continue

            (
                u,
                singular_values,
                vh,
            ) = truncated_svd(
                weight,
                rank,
            )

            reconstructed = reconstruct_svd(
                u,
                singular_values,
                vh,
            )

            reconstruction_error = relative_frobenius_error(
                weight,
                reconstructed,
            )

            weight.copy_(
                reconstructed.to(
                    dtype=weight.dtype,
                    device=weight.device,
                )
            )

            original_parameters = output_dim * input_dim

            compressed_parameters = rank * (output_dim + input_dim + 1)

            original_target_parameters += original_parameters

            compressed_target_parameters += compressed_parameters

            layer_reports.append(
                SVDLayerReport(
                    name=name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    original_parameters=(original_parameters),
                    compressed_parameters=(compressed_parameters),
                    rank=rank,
                    maximum_rank=min(
                        output_dim,
                        input_dim,
                    ),
                    profitable_max_rank=(profitable_rank),
                    original_bytes=(original_parameters * 4),
                    compressed_bytes=(compressed_parameters * 4),
                    relative_reconstruction_error=(reconstruction_error),
                )
            )

    untouched_parameters = total_parameters - original_target_parameters

    effective_stored_parameters = untouched_parameters + compressed_target_parameters

    compressed_size_bytes = effective_stored_parameters * 4

    return (
        compressed_model,
        SVDCompressionReport(
            total_parameters=(total_parameters),
            compressed_layer_weight_parameters=(original_target_parameters),
            effective_stored_parameters=(effective_stored_parameters),
            baseline_size_bytes=(baseline_size_bytes),
            compressed_size_bytes=(compressed_size_bytes),
            layers=tuple(layer_reports),
        ),
    )
