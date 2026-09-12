"""Tensor Train and TT-SVD compression for Q-VLA Forge."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch
from torch import nn


@dataclass(frozen=True)
class TTDecomposition:
    """Tensor Train representation."""

    original_shape: tuple[int, ...]
    ranks: tuple[int, ...]
    cores: tuple[torch.Tensor, ...]

    @property
    def stored_parameters(self) -> int:
        """Return total number of coefficients stored in TT cores."""
        return sum(int(core.numel()) for core in self.cores)


@dataclass(frozen=True)
class TTLayerReport:
    """Tensor Train compression evidence for one Linear layer."""

    name: str

    input_dim: int
    output_dim: int

    row_factors: tuple[int, ...]
    column_factors: tuple[int, ...]
    tensor_shape: tuple[int, ...]

    requested_rank: int
    actual_ranks: tuple[int, ...]

    original_parameters: int
    compressed_parameters: int

    original_bytes: int
    compressed_bytes: int

    relative_reconstruction_error: float

    @property
    def compression_ratio(self) -> float:
        """Return local TT storage compression ratio."""
        return self.original_bytes / self.compressed_bytes


@dataclass(frozen=True)
class TTCompressionReport:
    """Whole-model Tensor Train compression report."""

    total_parameters: int

    compressed_layer_weight_parameters: int
    effective_stored_parameters: int

    baseline_size_bytes: int
    compressed_size_bytes: int

    requested_rank: int

    layers: tuple[TTLayerReport, ...]

    @property
    def compression_ratio(self) -> float:
        """Return effective whole-model TT compression ratio."""
        return self.baseline_size_bytes / self.compressed_size_bytes

    @property
    def storage_reduction_percent(self) -> float:
        """Return whole-model byte reduction percentage."""
        return (1.0 - (self.compressed_size_bytes / self.baseline_size_bytes)) * 100.0

    @property
    def parameter_reduction_percent(self) -> float:
        """Return effective coefficient reduction percentage."""
        return (
            1.0 - (self.effective_stored_parameters / self.total_parameters)
        ) * 100.0


def _product(
    values: tuple[int, ...],
) -> int:
    """Return the integer product of a tuple."""
    return reduce(
        mul,
        values,
        1,
    )


def factor_dimension(
    dimension: int,
    *,
    order: int = 3,
) -> tuple[int, ...]:
    """Find a deterministic balanced exact factorization."""
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")

    if order <= 0:
        raise ValueError("order must be greater than zero")

    if order == 1:
        return (dimension,)

    best: tuple[int, ...] | None = None
    best_spread: int | None = None

    def search(
        remaining: int,
        depth: int,
        current: tuple[int, ...],
    ) -> None:
        nonlocal best
        nonlocal best_spread

        if depth == order - 1:
            candidate = (
                *current,
                remaining,
            )

            if _product(candidate) != dimension:
                return

            spread = max(candidate) - min(candidate)

            if (
                best is None
                or best_spread is None
                or spread < best_spread
                or (spread == best_spread and candidate < best)
            ):
                best = candidate
                best_spread = spread

            return

        for factor in range(
            1,
            remaining + 1,
        ):
            if remaining % factor != 0:
                continue

            search(
                remaining // factor,
                depth + 1,
                (
                    *current,
                    factor,
                ),
            )

    search(
        dimension,
        0,
        (),
    )

    if best is None:
        raise RuntimeError("unable to factor dimension")

    return best


def interleaved_tensorize_matrix(
    matrix: torch.Tensor,
    row_factors: tuple[int, ...],
    column_factors: tuple[int, ...],
) -> tuple[
    torch.Tensor,
    tuple[int, ...],
]:
    """Tensorize a matrix into paired row/column modes."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if len(row_factors) != len(column_factors):
        raise ValueError("row and column factor counts must match")

    rows = int(matrix.shape[0])

    columns = int(matrix.shape[1])

    if _product(row_factors) != rows:
        raise ValueError("row factors do not match matrix rows")

    if _product(column_factors) != columns:
        raise ValueError("column factors do not match matrix columns")

    order = len(row_factors)

    reshaped = matrix.reshape(
        *row_factors,
        *column_factors,
    )

    permutation: list[int] = []

    for index in range(order):
        permutation.extend(
            [
                index,
                index + order,
            ]
        )

    interleaved = reshaped.permute(*permutation).contiguous()

    tensor_shape = tuple(
        row_factors[index] * column_factors[index] for index in range(order)
    )

    return (
        interleaved.reshape(*tensor_shape),
        tensor_shape,
    )


def detensorize_matrix(
    tensor: torch.Tensor,
    row_factors: tuple[int, ...],
    column_factors: tuple[int, ...],
) -> torch.Tensor:
    """Reverse paired-mode matrix tensorization."""
    if len(row_factors) != len(column_factors):
        raise ValueError("row and column factor counts must match")

    order = len(row_factors)

    tensor_shape = tuple(
        row_factors[index] * column_factors[index] for index in range(order)
    )

    if tuple(tensor.shape) != tensor_shape:
        raise ValueError("tensor shape does not match factorization")

    expanded_shape: list[int] = []

    for index in range(order):
        expanded_shape.extend(
            [
                row_factors[index],
                column_factors[index],
            ]
        )

    expanded = tensor.reshape(*expanded_shape)

    row_positions = [2 * index for index in range(order)]

    column_positions = [2 * index + 1 for index in range(order)]

    permutation = row_positions + column_positions

    matrix_tensor = expanded.permute(*permutation).contiguous()

    return matrix_tensor.reshape(
        _product(row_factors),
        _product(column_factors),
    )


def tt_svd(
    tensor: torch.Tensor,
    max_rank: int,
) -> TTDecomposition:
    """Compute a Tensor Train decomposition using sequential SVD."""
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two modes")

    if not torch.is_floating_point(tensor):
        raise TypeError("tensor must use a floating-point dtype")

    if max_rank <= 0:
        raise ValueError("max_rank must be greater than zero")

    shape = tuple(int(value) for value in tensor.shape)

    order = len(shape)

    working = tensor.detach()

    previous_rank = 1

    ranks: list[int] = [1]

    cores: list[torch.Tensor] = []

    for index in range(order - 1):
        mode_size = shape[index]

        working = working.reshape(
            previous_rank * mode_size,
            -1,
        )

        (
            u,
            singular_values,
            vh,
        ) = torch.linalg.svd(
            working,
            full_matrices=False,
        )

        rank = min(
            max_rank,
            int(singular_values.shape[0]),
        )

        u = u[
            :,
            :rank,
        ]

        singular_values = singular_values[:rank]

        vh = vh[
            :rank,
            :,
        ]

        core = u.reshape(
            previous_rank,
            mode_size,
            rank,
        )

        cores.append(core)

        ranks.append(rank)

        working = singular_values.unsqueeze(1) * vh

        previous_rank = rank

    final_core = working.reshape(
        previous_rank,
        shape[-1],
        1,
    )

    cores.append(final_core)

    ranks.append(1)

    return TTDecomposition(
        original_shape=shape,
        ranks=tuple(ranks),
        cores=tuple(cores),
    )


def reconstruct_tt(
    decomposition: TTDecomposition,
) -> torch.Tensor:
    """Reconstruct a dense tensor from TT cores."""
    cores = decomposition.cores

    if not cores:
        raise ValueError("decomposition must contain cores")

    result = cores[0]

    for core in cores[1:]:
        result = torch.tensordot(
            result,
            core,
            dims=(
                [-1],
                [0],
            ),
        )

    result = result.squeeze(0).squeeze(-1)

    return result.reshape(*decomposition.original_shape)


def relative_tt_error(
    original: torch.Tensor,
    approximation: torch.Tensor,
) -> float:
    """Return relative Frobenius reconstruction error."""
    if original.shape != approximation.shape:
        raise ValueError("tensor shapes must match")

    denominator = torch.linalg.norm(original)

    numerator = torch.linalg.norm(original - approximation)

    denominator_value = float(denominator.item())

    if denominator_value == 0.0:
        if float(numerator.item()) == 0.0:
            return 0.0

        return math.inf

    return float((numerator / denominator).item())


def compress_model_tt(
    model: nn.Module,
    *,
    max_rank: int,
    minimum_weight_parameters: int = 1024,
    tensor_order: int = 3,
) -> tuple[
    nn.Module,
    TTCompressionReport,
]:
    """
    Apply TT-SVD to eligible Linear weights.

    The returned model retains reconstructed FP32 dense matrices.
    Effective storage is calculated from TT cores.
    """
    if max_rank <= 0:
        raise ValueError("max_rank must be greater than zero")

    if minimum_weight_parameters <= 0:
        raise ValueError("minimum_weight_parameters must be greater than zero")

    if tensor_order < 2:
        raise ValueError("tensor_order must be at least two")

    compressed_model = copy.deepcopy(model)

    total_parameters = sum(
        parameter.numel() for parameter in compressed_model.parameters()
    )

    if total_parameters <= 0:
        raise ValueError("model must contain parameters")

    baseline_size_bytes = total_parameters * 4

    original_target_parameters = 0
    compressed_target_parameters = 0

    reports: list[TTLayerReport] = []

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

            weight = module.weight

            if weight.numel() < minimum_weight_parameters:
                continue

            output_dim = int(weight.shape[0])

            input_dim = int(weight.shape[1])

            row_factors = factor_dimension(
                output_dim,
                order=tensor_order,
            )

            column_factors = factor_dimension(
                input_dim,
                order=tensor_order,
            )

            (
                tensorized,
                tensor_shape,
            ) = interleaved_tensorize_matrix(
                weight.detach(),
                row_factors,
                column_factors,
            )

            decomposition = tt_svd(
                tensorized,
                max_rank=max_rank,
            )

            reconstructed_tensor = reconstruct_tt(decomposition)

            reconstructed_weight = detensorize_matrix(
                reconstructed_tensor,
                row_factors,
                column_factors,
            )

            error = relative_tt_error(
                weight,
                reconstructed_weight,
            )

            original_parameters = int(weight.numel())

            tt_parameters = decomposition.stored_parameters

            if tt_parameters >= original_parameters:
                continue

            weight.copy_(
                reconstructed_weight.to(
                    dtype=weight.dtype,
                    device=weight.device,
                )
            )

            original_target_parameters += original_parameters

            compressed_target_parameters += tt_parameters

            reports.append(
                TTLayerReport(
                    name=name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    row_factors=row_factors,
                    column_factors=column_factors,
                    tensor_shape=tensor_shape,
                    requested_rank=max_rank,
                    actual_ranks=decomposition.ranks,
                    original_parameters=(original_parameters),
                    compressed_parameters=(tt_parameters),
                    original_bytes=(original_parameters * 4),
                    compressed_bytes=(tt_parameters * 4),
                    relative_reconstruction_error=(error),
                )
            )

    untouched_parameters = total_parameters - original_target_parameters

    effective_stored_parameters = untouched_parameters + compressed_target_parameters

    compressed_size_bytes = effective_stored_parameters * 4

    return (
        compressed_model,
        TTCompressionReport(
            total_parameters=(total_parameters),
            compressed_layer_weight_parameters=(original_target_parameters),
            effective_stored_parameters=(effective_stored_parameters),
            baseline_size_bytes=(baseline_size_bytes),
            compressed_size_bytes=(compressed_size_bytes),
            requested_rank=max_rank,
            layers=tuple(reports),
        ),
    )
