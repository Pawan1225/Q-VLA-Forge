"""Trainable quantum-inspired TT/MPS layers for Sprint 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from q_vla_forge.compression.tensor_train import (
    TTDecomposition,
    detensorize_matrix,
    factor_dimension,
    interleaved_tensorize_matrix,
    reconstruct_tt,
    relative_tt_error,
    tt_svd,
)


@dataclass(frozen=True)
class TrainableTTLayerReport:
    """Evidence for one dense-to-trainable-TT replacement."""

    name: str

    input_dim: int
    output_dim: int

    row_factors: tuple[int, ...]
    column_factors: tuple[int, ...]
    tensor_shape: tuple[int, ...]

    requested_rank: int
    actual_ranks: tuple[int, ...]

    original_weight_parameters: int
    core_parameters: int
    bias_parameters: int

    original_total_parameters: int
    structured_total_parameters: int

    relative_initialization_error: float

    @property
    def local_parameter_ratio(self) -> float:
        """Dense parameters divided by structured parameters."""
        return self.original_total_parameters / self.structured_total_parameters

    @property
    def parameter_reduction_percent(self) -> float:
        """Return local trainable-parameter reduction."""
        return (
            1.0 - (self.structured_total_parameters / self.original_total_parameters)
        ) * 100.0


@dataclass(frozen=True)
class TrainableTTModelReport:
    """Whole-model trainable TT/MPS conversion report."""

    original_trainable_parameters: int
    structured_trainable_parameters: int

    replaced_weight_parameters: int
    structured_core_parameters: int

    requested_rank: int
    tensor_order: int

    layers: tuple[TrainableTTLayerReport, ...]

    quantum_inspired: bool = True
    quantum_hardware_used: bool = False

    @property
    def parameter_ratio(self) -> float:
        """Dense model parameters divided by structured parameters."""
        return self.original_trainable_parameters / self.structured_trainable_parameters

    @property
    def parameter_reduction_percent(self) -> float:
        """Return whole-model trainable-parameter reduction."""
        return (
            1.0
            - (
                self.structured_trainable_parameters
                / self.original_trainable_parameters
            )
        ) * 100.0


class TrainableTTLinear(nn.Module):
    """
    Linear layer parameterized by trainable TT/open-boundary-MPS cores.

    A dense-equivalent weight is materialized only as a differentiable
    forward-pass intermediate. It is not a trainable model parameter.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        row_factors: tuple[int, ...],
        column_factors: tuple[int, ...],
        tensor_shape: tuple[int, ...],
        ranks: tuple[int, ...],
        cores: tuple[torch.Tensor, ...],
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if len(row_factors) < 2:
            raise ValueError("TT representation must contain at least two modes")

        if len(row_factors) != len(column_factors):
            raise ValueError("row and column factor counts must match")

        if len(cores) != len(row_factors):
            raise ValueError("one TT core is required per tensor mode")

        if len(ranks) != len(cores) + 1:
            raise ValueError("TT rank tuple must have order + 1 entries")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("open-boundary TT/MPS must have boundary rank 1")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.row_factors = row_factors
        self.column_factors = column_factors
        self.tensor_shape = tensor_shape
        self.ranks = ranks

        self.cores = nn.ParameterList(
            [nn.Parameter(core.detach().clone()) for core in cores]
        )

        if bias is None:
            self.register_parameter(
                "bias",
                None,
            )
        else:
            self.bias = nn.Parameter(bias.detach().clone())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        max_rank: int,
        tensor_order: int = 3,
    ) -> TrainableTTLinear:
        """Initialize a trainable TT/MPS layer using TT-SVD."""
        if max_rank <= 0:
            raise ValueError("max_rank must be greater than zero")

        if tensor_order < 2:
            raise ValueError("tensor_order must be at least two")

        weight = linear.weight.detach()

        row_factors = factor_dimension(
            linear.out_features,
            order=tensor_order,
        )

        column_factors = factor_dimension(
            linear.in_features,
            order=tensor_order,
        )

        (
            tensorized,
            tensor_shape,
        ) = interleaved_tensorize_matrix(
            weight,
            row_factors,
            column_factors,
        )

        decomposition = tt_svd(
            tensorized,
            max_rank=max_rank,
        )

        return cls(
            input_dim=linear.in_features,
            output_dim=linear.out_features,
            row_factors=row_factors,
            column_factors=column_factors,
            tensor_shape=tensor_shape,
            ranks=decomposition.ranks,
            cores=decomposition.cores,
            bias=(linear.bias.detach() if linear.bias is not None else None),
        )

    @property
    def core_parameter_count(self) -> int:
        """Return the number of trainable TT-core coefficients."""
        return sum(parameter.numel() for parameter in self.cores)

    def decomposition(
        self,
    ) -> TTDecomposition:
        """Expose current trainable cores as a TT decomposition."""
        return TTDecomposition(
            original_shape=self.tensor_shape,
            ranks=self.ranks,
            cores=tuple(self.cores),
        )

    def reconstructed_tensor(
        self,
    ) -> torch.Tensor:
        """Differentiably reconstruct the tensor represented by the cores."""
        return reconstruct_tt(self.decomposition())

    def reconstructed_weight(
        self,
    ) -> torch.Tensor:
        """Return the differentiable matrix represented by TT/MPS cores."""
        tensor = self.reconstructed_tensor()

        return detensorize_matrix(
            tensor,
            self.row_factors,
            self.column_factors,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the TT/MPS-parameterized linear transformation."""
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "input feature dimension does not " "match TrainableTTLinear"
            )

        weight = self.reconstructed_weight()

        return F.linear(
            inputs,
            weight,
            self.bias,
        )


def _set_module_by_name(
    model: nn.Module,
    name: str,
    replacement: nn.Module,
) -> None:
    """Replace one nested module using its named_modules path."""
    if not name:
        raise ValueError("cannot replace the root module")

    parts = name.split(".")

    parent: nn.Module = model

    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(f"module path not found: {name}")

        child = parent._modules[part]

        if child is None:
            raise KeyError(f"module path not found: {name}")

        parent = child

    leaf = parts[-1]

    if leaf not in parent._modules:
        raise KeyError(f"module path not found: {name}")

    parent._modules[leaf] = replacement


def trainable_parameter_count(
    model: nn.Module,
) -> int:
    """Count gradient-optimized model parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def convert_model_to_trainable_tt(
    model: nn.Module,
    *,
    max_rank: int,
    minimum_weight_parameters: int = 1024,
    tensor_order: int = 3,
    selected_layer_names: set[str] | None = None,
) -> tuple[
    nn.Module,
    TrainableTTModelReport,
]:
    """
    Replace profitable Linear weights with trainable TT/MPS cores.

    The input model is deep-copied and never mutated.
    """
    if max_rank <= 0:
        raise ValueError("max_rank must be greater than zero")

    if minimum_weight_parameters <= 0:
        raise ValueError("minimum_weight_parameters " "must be greater than zero")

    if tensor_order < 2:
        raise ValueError("tensor_order must be at least two")

    structured_model = copy.deepcopy(model)

    original_parameters = trainable_parameter_count(structured_model)

    if original_parameters <= 0:
        raise ValueError("model must contain trainable parameters")

    replacements: list[
        tuple[
            str,
            nn.Linear,
            TrainableTTLinear,
            float,
        ]
    ] = []

    for (
        name,
        module,
    ) in structured_model.named_modules():
        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        if selected_layer_names is not None and name not in selected_layer_names:
            continue

        if module.weight.numel() < minimum_weight_parameters:
            continue

        structured_layer = TrainableTTLinear.from_linear(
            module,
            max_rank=max_rank,
            tensor_order=tensor_order,
        )

        original_weight_parameters = module.weight.numel()

        core_parameters = structured_layer.core_parameter_count

        if core_parameters >= original_weight_parameters:
            continue

        reconstruction_error = relative_tt_error(
            module.weight.detach(),
            structured_layer.reconstructed_weight().detach(),
        )

        replacements.append(
            (
                name,
                module,
                structured_layer,
                reconstruction_error,
            )
        )

    reports: list[TrainableTTLayerReport] = []

    replaced_weight_parameters = 0
    structured_core_parameters = 0

    for (
        name,
        dense_layer,
        structured_layer,
        reconstruction_error,
    ) in replacements:
        original_weight_count = dense_layer.weight.numel()

        core_count = structured_layer.core_parameter_count

        bias_parameters = (
            dense_layer.bias.numel() if dense_layer.bias is not None else 0
        )

        _set_module_by_name(
            structured_model,
            name,
            structured_layer,
        )

        replaced_weight_parameters += original_weight_count

        structured_core_parameters += core_count

        reports.append(
            TrainableTTLayerReport(
                name=name,
                input_dim=dense_layer.in_features,
                output_dim=dense_layer.out_features,
                row_factors=(structured_layer.row_factors),
                column_factors=(structured_layer.column_factors),
                tensor_shape=(structured_layer.tensor_shape),
                requested_rank=max_rank,
                actual_ranks=(structured_layer.ranks),
                original_weight_parameters=(original_weight_count),
                core_parameters=core_count,
                bias_parameters=bias_parameters,
                original_total_parameters=(original_weight_count + bias_parameters),
                structured_total_parameters=(core_count + bias_parameters),
                relative_initialization_error=(reconstruction_error),
            )
        )

    structured_parameters = trainable_parameter_count(structured_model)

    expected_parameters = (
        original_parameters - replaced_weight_parameters + structured_core_parameters
    )

    if structured_parameters != expected_parameters:
        raise RuntimeError("trainable TT parameter " "accounting mismatch")

    return (
        structured_model,
        TrainableTTModelReport(
            original_trainable_parameters=(original_parameters),
            structured_trainable_parameters=(structured_parameters),
            replaced_weight_parameters=(replaced_weight_parameters),
            structured_core_parameters=(structured_core_parameters),
            requested_rank=max_rank,
            tensor_order=tensor_order,
            layers=tuple(reports),
        ),
    )
