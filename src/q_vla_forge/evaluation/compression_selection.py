"""Compression experiment normalization and candidate selection."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompressionCandidate:
    """Normalized compression experiment used for method selection."""

    experiment_id: str
    domain: str
    method: str
    seed: int

    configuration_label: str

    compression_ratio: float
    storage_reduction_percent: float
    parameter_reduction_percent: float

    baseline_test_mse: float
    compressed_test_mse: float
    mse_change_percent: float

    baseline_test_mae: float
    compressed_test_mae: float
    mae_change_percent: float

    baseline_mean_latency_ms: float
    compressed_mean_latency_ms: float
    latency_change_percent: float

    pilot_feasible: bool
    source_file: str


@dataclass(frozen=True)
class MethodSelection:
    """Selected candidate for one compression family."""

    method: str
    selected: CompressionCandidate
    candidates: tuple[CompressionCandidate, ...]
    feasible_candidates: int
    selection_reason: str


@dataclass(frozen=True)
class DomainCompressionSelection:
    """Compression-method selections for one domain."""

    domain: str
    seed: int

    minimum_compression_ratio: float
    maximum_mse_increase_percent: float

    int8: MethodSelection
    svd: MethodSelection
    tensor_network: MethodSelection

    selected_experiment_ids: tuple[str, ...]


def _percent_change(
    baseline: float,
    compressed: float,
) -> float:
    """Return relative percentage change from baseline."""
    if baseline <= 0.0:
        raise ValueError("baseline metric must be greater than zero")

    return (compressed - baseline) / baseline * 100.0


def _validate_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Reject non-finite derived evidence."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _configuration_label(
    payload: dict[str, Any],
) -> str:
    """Return a compact display label for a compression experiment."""
    method = str(payload["method"])

    configuration = payload.get(
        "configuration",
        {},
    )

    if method == "int8":
        return "INT8"

    if method == "svd":
        fraction = float(configuration["rank_fraction"])

        return f"SVD-{round(fraction * 100)}%"

    if method == "tensor_train":
        rank = int(configuration["requested_max_rank"])

        return f"TT-rank-{rank}"

    raise ValueError(f"unsupported compression method: {method}")


def load_compression_candidate(
    path: Path,
    *,
    minimum_compression_ratio: float = 2.0,
    maximum_mse_increase_percent: float = 5.0,
) -> CompressionCandidate:
    """Load and normalize one compression experiment result."""
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    metrics = payload["metrics"]

    baseline_size = int(metrics["baseline_size_bytes"])

    compressed_size = int(metrics["compressed_size_bytes"])

    if baseline_size <= 0:
        raise ValueError("baseline_size_bytes must be positive")

    if compressed_size <= 0:
        raise ValueError("compressed_size_bytes must be positive")

    compression_ratio = baseline_size / compressed_size

    storage_reduction = (1.0 - (compressed_size / baseline_size)) * 100.0

    baseline_parameters = int(metrics["baseline_parameters"])

    compressed_parameters = int(metrics["compressed_parameters"])

    if baseline_parameters <= 0:
        raise ValueError("baseline_parameters must be positive")

    if compressed_parameters <= 0:
        raise ValueError("compressed_parameters must be positive")

    parameter_reduction = (1.0 - (compressed_parameters / baseline_parameters)) * 100.0

    baseline_mse = float(metrics["baseline_test_mse"])

    compressed_mse = float(metrics["compressed_test_mse"])

    baseline_mae = float(metrics["baseline_test_mae"])

    compressed_mae = float(metrics["compressed_test_mae"])

    baseline_latency = float(metrics["baseline_mean_latency_ms"])

    compressed_latency = float(metrics["compressed_mean_latency_ms"])

    mse_change = _percent_change(
        baseline_mse,
        compressed_mse,
    )

    mae_change = _percent_change(
        baseline_mae,
        compressed_mae,
    )

    latency_change = _percent_change(
        baseline_latency,
        compressed_latency,
    )

    for name, value in (
        (
            "compression_ratio",
            compression_ratio,
        ),
        (
            "storage_reduction_percent",
            storage_reduction,
        ),
        (
            "parameter_reduction_percent",
            parameter_reduction,
        ),
        (
            "mse_change_percent",
            mse_change,
        ),
        (
            "mae_change_percent",
            mae_change,
        ),
        (
            "latency_change_percent",
            latency_change,
        ),
    ):
        _validate_finite(
            float(value),
            name=name,
        )

    pilot_feasible = (
        compression_ratio >= minimum_compression_ratio
        and mse_change <= maximum_mse_increase_percent
    )

    return CompressionCandidate(
        experiment_id=str(payload["experiment_id"]),
        domain=str(payload["domain"]),
        method=str(payload["method"]),
        seed=int(payload["seed"]),
        configuration_label=(_configuration_label(payload)),
        compression_ratio=float(compression_ratio),
        storage_reduction_percent=float(storage_reduction),
        parameter_reduction_percent=float(parameter_reduction),
        baseline_test_mse=(baseline_mse),
        compressed_test_mse=(compressed_mse),
        mse_change_percent=float(mse_change),
        baseline_test_mae=(baseline_mae),
        compressed_test_mae=(compressed_mae),
        mae_change_percent=float(mae_change),
        baseline_mean_latency_ms=(baseline_latency),
        compressed_mean_latency_ms=(compressed_latency),
        latency_change_percent=float(latency_change),
        pilot_feasible=(pilot_feasible),
        source_file=path.name,
    )


def _fallback_score(
    candidate: CompressionCandidate,
) -> float:
    """
    Return compression-error utility when no candidate is feasible.

    Larger is better. Negative MSE change is treated as zero penalty.
    """
    penalty = max(
        candidate.mse_change_percent,
        0.0,
    )

    return candidate.compression_ratio / (1.0 + penalty / 100.0)


def select_method_candidate(
    method: str,
    candidates: Sequence[CompressionCandidate],
) -> MethodSelection:
    """Select the strongest configuration for one method family."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    if any(candidate.method != method for candidate in candidates):
        raise ValueError("all candidates must match the requested method")

    feasible = [candidate for candidate in candidates if candidate.pilot_feasible]

    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                -candidate.compression_ratio,
                candidate.mse_change_percent,
                candidate.mae_change_percent,
                candidate.experiment_id,
            ),
        )

        reason = (
            "Selected from pilot-feasible configurations "
            "by highest whole-model compression ratio, "
            "then lower MSE and MAE change."
        )
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                -_fallback_score(candidate),
                candidate.mse_change_percent,
                -candidate.compression_ratio,
                candidate.experiment_id,
            ),
        )

        reason = (
            "No configuration satisfied the pilot "
            "compression/error criterion; selected the "
            "best compression-error fallback candidate."
        )

    return MethodSelection(
        method=method,
        selected=selected,
        candidates=tuple(candidates),
        feasible_candidates=len(feasible),
        selection_reason=reason,
    )


def build_domain_compression_selection(
    candidates: Sequence[CompressionCandidate],
    *,
    domain: str,
    seed: int,
    minimum_compression_ratio: float = 2.0,
    maximum_mse_increase_percent: float = 5.0,
) -> DomainCompressionSelection:
    """Build INT8, SVD, and TT/MPS selection for a domain and seed."""
    relevant = [
        candidate
        for candidate in candidates
        if (candidate.domain == domain and candidate.seed == seed)
    ]

    if not relevant:
        raise ValueError("no candidates match domain and seed")

    int8_candidates = [
        candidate for candidate in relevant if candidate.method == "int8"
    ]

    svd_candidates = [candidate for candidate in relevant if candidate.method == "svd"]

    tt_candidates = [
        candidate for candidate in relevant if candidate.method == "tensor_train"
    ]

    int8 = select_method_candidate(
        "int8",
        int8_candidates,
    )

    svd = select_method_candidate(
        "svd",
        svd_candidates,
    )

    tensor_network = select_method_candidate(
        "tensor_train",
        tt_candidates,
    )

    selected_ids = (
        int8.selected.experiment_id,
        svd.selected.experiment_id,
        tensor_network.selected.experiment_id,
    )

    return DomainCompressionSelection(
        domain=domain,
        seed=seed,
        minimum_compression_ratio=(minimum_compression_ratio),
        maximum_mse_increase_percent=(maximum_mse_increase_percent),
        int8=int8,
        svd=svd,
        tensor_network=tensor_network,
        selected_experiment_ids=(selected_ids),
    )


def save_domain_compression_selection(
    selection: DomainCompressionSelection,
    output_path: Path,
) -> None:
    """Save selected compression configurations as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(selection),
            indent=2,
        ),
        encoding="utf-8",
    )
