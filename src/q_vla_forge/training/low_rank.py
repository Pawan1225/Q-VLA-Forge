"""Trainable classical low-rank layers for Sprint 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from q_vla_forge.compression.svd import (
    relative_frobenius_error,
    resolve_rank,
    truncated_svd,
)


@dataclass(frozen=True)
class TrainableSVDLayerReport:
    """One dense-to-trainable-SVD replacement."""

    name: str
    input_dim: int
    output_dim: int
    rank: int

    original_weight_parameters: int
    factor_parameters: int
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
        """Percentage reduction in trainable parameters."""
        return (
            1.0 - (self.structured_total_parameters / self.original_total_parameters)
        ) * 100.0


@dataclass(frozen=True)
class TrainableSVDModelReport:
    """Whole-model trainable-SVD conversion report."""

    original_trainable_parameters: int
    structured_trainable_parameters: int

    replaced_weight_parameters: int
    structured_factor_parameters: int

    layers: tuple[TrainableSVDLayerReport, ...]

    @property
    def parameter_ratio(self) -> float:
        """Dense model parameters divided by structured parameters."""
        return self.original_trainable_parameters / self.structured_trainable_parameters

    @property
    def parameter_reduction_percent(self) -> float:
        """Whole-model trainable parameter reduction."""
        return (
            1.0
            - (
                self.structured_trainable_parameters
                / self.original_trainable_parameters
            )
        ) * 100.0


class TrainableSVDLinear(nn.Module):
    """Linear layer parameterized directly by trainable SVD factors."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        rank: int,
        bias: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if rank <= 0:
            raise ValueError("rank must be greater than zero")

        if rank > min(
            input_dim,
            output_dim,
        ):
            raise ValueError("rank exceeds matrix maximum rank")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.u = nn.Parameter(
            torch.empty(
                output_dim,
                rank,
                device=device,
                dtype=dtype,
            )
        )

        self.singular_values = nn.Parameter(
            torch.empty(
                rank,
                device=device,
                dtype=dtype,
            )
        )

        self.vh = nn.Parameter(
            torch.empty(
                rank,
                input_dim,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    output_dim,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
    ) -> TrainableSVDLinear:
        """Initialize a trainable low-rank layer from a dense Linear."""
        if rank <= 0:
            raise ValueError("rank must be greater than zero")

        weight = linear.weight.detach()

        (
            u,
            singular_values,
            vh,
        ) = truncated_svd(
            weight,
            rank,
        )

        layer = cls(
            input_dim=linear.in_features,
            output_dim=linear.out_features,
            rank=rank,
            bias=(linear.bias is not None),
            dtype=weight.dtype,
            device=weight.device,
        )

        with torch.no_grad():
            layer.u.copy_(u)
            layer.singular_values.copy_(singular_values)
            layer.vh.copy_(vh)

            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias.detach())

        return layer

    def reconstructed_weight(
        self,
    ) -> torch.Tensor:
        """Return the differentiable dense-equivalent weight."""
        return (self.u * self.singular_values.unsqueeze(0)) @ self.vh

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the factorized linear transformation without
        materializing a dense weight matrix.
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "input feature dimension does not " "match TrainableSVDLinear"
            )

        rank_features = torch.matmul(
            inputs,
            self.vh.transpose(
                0,
                1,
            ),
        )

        scaled = rank_features * self.singular_values

        output = torch.matmul(
            scaled,
            self.u.transpose(
                0,
                1,
            ),
        )

        if self.bias is not None:
            output = output + self.bias

        return output


def _set_module_by_name(
    model: nn.Module,
    name: str,
    replacement: nn.Module,
) -> None:
    """Replace a nested child module using a named_modules path."""
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
    """Count parameters optimized by gradient descent."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def convert_model_to_trainable_svd(
    model: nn.Module,
    *,
    rank_fraction: float,
    minimum_weight_parameters: int = 1024,
    selected_layer_names: set[str] | None = None,
) -> tuple[
    nn.Module,
    TrainableSVDModelReport,
]:
    """
    Replace eligible dense Linear layers with trainable SVD factors.

    The input model is never mutated.
    """
    if minimum_weight_parameters <= 0:
        raise ValueError("minimum_weight_parameters " "must be greater than zero")

    structured_model = copy.deepcopy(model)

    original_parameters = trainable_parameter_count(structured_model)

    if original_parameters <= 0:
        raise ValueError("model must contain trainable parameters")

    replacements: list[
        tuple[
            str,
            nn.Linear,
            int,
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

        weight_parameters = module.weight.numel()

        if weight_parameters < minimum_weight_parameters:
            continue

        rank = resolve_rank(
            module.out_features,
            module.in_features,
            rank_fraction,
        )

        if rank < 1:
            continue

        dense_weight_parameters = module.out_features * module.in_features

        factor_parameters = rank * (module.out_features + module.in_features + 1)

        if factor_parameters >= dense_weight_parameters:
            continue

        replacements.append(
            (
                name,
                module,
                rank,
            )
        )

    reports: list[TrainableSVDLayerReport] = []

    replaced_weight_parameters = 0
    structured_factor_parameters = 0

    for (
        name,
        dense_layer,
        rank,
    ) in replacements:
        structured_layer = TrainableSVDLinear.from_linear(
            dense_layer,
            rank=rank,
        )

        original_weight = dense_layer.weight.detach()

        reconstructed = structured_layer.reconstructed_weight().detach()

        reconstruction_error = relative_frobenius_error(
            original_weight,
            reconstructed,
        )

        bias_parameters = (
            dense_layer.bias.numel() if dense_layer.bias is not None else 0
        )

        original_weight_count = dense_layer.weight.numel()

        factor_count = (
            structured_layer.u.numel()
            + structured_layer.singular_values.numel()
            + structured_layer.vh.numel()
        )

        _set_module_by_name(
            structured_model,
            name,
            structured_layer,
        )

        replaced_weight_parameters += original_weight_count

        structured_factor_parameters += factor_count

        reports.append(
            TrainableSVDLayerReport(
                name=name,
                input_dim=(dense_layer.in_features),
                output_dim=(dense_layer.out_features),
                rank=rank,
                original_weight_parameters=(original_weight_count),
                factor_parameters=(factor_count),
                bias_parameters=(bias_parameters),
                original_total_parameters=(original_weight_count + bias_parameters),
                structured_total_parameters=(factor_count + bias_parameters),
                relative_initialization_error=(reconstruction_error),
            )
        )

    structured_parameters = trainable_parameter_count(structured_model)

    expected_structured_parameters = (
        original_parameters - replaced_weight_parameters + structured_factor_parameters
    )

    if structured_parameters != expected_structured_parameters:
        raise RuntimeError("trainable SVD parameter " "accounting mismatch")

    return (
        structured_model,
        TrainableSVDModelReport(
            original_trainable_parameters=(original_parameters),
            structured_trainable_parameters=(structured_parameters),
            replaced_weight_parameters=(replaced_weight_parameters),
            structured_factor_parameters=(structured_factor_parameters),
            layers=tuple(reports),
        ),
    )
