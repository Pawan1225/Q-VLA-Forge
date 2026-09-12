"""Analyze model layers for Sprint 2 compression suitability."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import nn


@dataclass(frozen=True)
class LayerCompressionTarget:
    """Compression suitability for one parameterized model layer."""

    name: str
    module_type: str

    weight_shape: tuple[int, ...]
    weight_parameters: int
    module_parameters: int

    model_parameter_percent: float

    int8_eligible: bool
    svd_eligible: bool
    tensor_train_eligible: bool
    mps_eligible: bool

    priority: str


@dataclass(frozen=True)
class CompressionTargetReport:
    """Whole-model compression target analysis."""

    total_parameters: int
    trainable_parameters: int

    candidate_weight_parameters: int
    uncategorized_parameters: int

    int8_target_parameters: int
    svd_target_parameters: int
    tensor_train_target_parameters: int
    mps_target_parameters: int

    int8_coverage_percent: float
    svd_coverage_percent: float
    tensor_train_coverage_percent: float
    mps_coverage_percent: float

    layers: tuple[LayerCompressionTarget, ...]


def _parameter_count(
    module: nn.Module,
) -> int:
    """Count parameters owned directly by one module."""
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _weight_parameter_count(
    module: nn.Module,
) -> int:
    """Count elements in a module's primary weight tensor."""
    weight = getattr(
        module,
        "weight",
        None,
    )

    if weight is None:
        return 0

    return int(weight.numel())


def _priority(
    percentage: float,
) -> str:
    """Assign compression priority from whole-model contribution."""
    if percentage >= 5.0:
        return "high"

    if percentage >= 1.0:
        return "medium"

    return "low"


def _is_int8_eligible(
    module: nn.Module,
) -> bool:
    """Return whether a layer is eligible for INT8 experiments."""
    return isinstance(
        module,
        (
            nn.Linear,
            nn.Conv2d,
            nn.Embedding,
        ),
    )


def _is_svd_eligible(
    module: nn.Module,
) -> bool:
    """Return whether a layer is eligible for matrix SVD."""
    return isinstance(
        module,
        nn.Linear,
    )


def _is_tensor_train_eligible(
    module: nn.Module,
) -> bool:
    """Return whether a dense layer is large enough for TT/MPS study."""
    if not isinstance(
        module,
        nn.Linear,
    ):
        return False

    weight = module.weight

    if weight.ndim != 2:
        return False

    output_dim = int(weight.shape[0])
    input_dim = int(weight.shape[1])

    return output_dim >= 16 and input_dim >= 16 and weight.numel() >= 1024


def analyze_compression_targets(
    model: nn.Module,
) -> CompressionTargetReport:
    """Analyze every supported parameterized layer."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if total_parameters <= 0:
        raise ValueError("model must contain parameters")

    targets: list[LayerCompressionTarget] = []

    candidate_weight_parameters = 0

    int8_parameters = 0
    svd_parameters = 0
    tensor_train_parameters = 0
    mps_parameters = 0

    for name, module in model.named_modules():
        if not isinstance(
            module,
            (
                nn.Linear,
                nn.Conv2d,
                nn.Embedding,
            ),
        ):
            continue

        weight_parameters = _weight_parameter_count(module)

        module_parameters = _parameter_count(module)

        candidate_weight_parameters += weight_parameters

        percentage = weight_parameters / total_parameters * 100.0

        int8_eligible = _is_int8_eligible(module)

        svd_eligible = _is_svd_eligible(module)

        tensor_train_eligible = _is_tensor_train_eligible(module)

        mps_eligible = tensor_train_eligible

        if int8_eligible:
            int8_parameters += weight_parameters

        if svd_eligible:
            svd_parameters += weight_parameters

        if tensor_train_eligible:
            tensor_train_parameters += weight_parameters

        if mps_eligible:
            mps_parameters += weight_parameters

        targets.append(
            LayerCompressionTarget(
                name=name,
                module_type=(type(module).__name__),
                weight_shape=tuple(int(value) for value in module.weight.shape),
                weight_parameters=(weight_parameters),
                module_parameters=(module_parameters),
                model_parameter_percent=(float(percentage)),
                int8_eligible=(int8_eligible),
                svd_eligible=(svd_eligible),
                tensor_train_eligible=(tensor_train_eligible),
                mps_eligible=(mps_eligible),
                priority=_priority(percentage),
            )
        )

    targets.sort(
        key=lambda item: (item.weight_parameters),
        reverse=True,
    )

    uncategorized_parameters = total_parameters - candidate_weight_parameters

    return CompressionTargetReport(
        total_parameters=(total_parameters),
        trainable_parameters=(trainable_parameters),
        candidate_weight_parameters=(candidate_weight_parameters),
        uncategorized_parameters=(uncategorized_parameters),
        int8_target_parameters=(int8_parameters),
        svd_target_parameters=(svd_parameters),
        tensor_train_target_parameters=(tensor_train_parameters),
        mps_target_parameters=(mps_parameters),
        int8_coverage_percent=(int8_parameters / total_parameters * 100.0),
        svd_coverage_percent=(svd_parameters / total_parameters * 100.0),
        tensor_train_coverage_percent=(
            tensor_train_parameters / total_parameters * 100.0
        ),
        mps_coverage_percent=(mps_parameters / total_parameters * 100.0),
        layers=tuple(targets),
    )


def save_compression_target_report(
    report: CompressionTargetReport,
    output_path: Path,
) -> None:
    """Save compression target analysis as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        ),
        encoding="utf-8",
    )
