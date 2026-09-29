"""Sprint 7.3 final compression ablation contract and evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_statistics import SeedMetric, summarize_metric

METHODS: tuple[str, ...] = (
    "fp32",
    "int8",
    "svd",
    "tt_mps",
)

DOMAINS: tuple[str, ...] = (
    "autonomous_driving",
    "robotics",
)

MIN_COMPRESSION_RATIO = 2.0
MAX_RELATIVE_MSE_DEGRADATION_PERCENT = 5.0

REQUIRED_SEEDS: tuple[int, ...] = (
    42,
    123,
    456,
)

REFERENCE_METHOD = "fp32"

CANONICAL_COMPRESSION_SOURCE = "results/compression/compression-validation-summary.json"

DOMAIN_SOURCE_KEYS = {
    "autonomous_driving": "driving",
    "robotics": "robotics",
}

METHOD_SOURCE_KEYS = {
    "int8": "int8",
    "svd": "svd",
    "tt_mps": "tensor_network",
}


@dataclass(frozen=True)
class CompressionAblationRecord:
    """One final compression-ablation result."""

    domain: str
    method: str
    compression_ratio: float | None
    mse_mean: float
    mse_sample_std: float
    relative_mse_change_mean_percent: float | None
    relative_mse_change_sample_std_percent: float | None
    criterion_pass: bool | None
    seed_pass_count: int | None
    source_artifacts: tuple[str, ...]
    latency_mean_ms: float | None = None
    latency_sample_std_ms: float | None = None
    classification: str = "evaluated"

    def to_dict(self) -> dict[str, Any]:
        """Convert record to serializable form."""

        return asdict(self)


def validate_domain(
    domain: str,
) -> None:
    """Require one of the locked Phase 1 domains."""

    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")


def validate_method(
    method: str,
) -> None:
    """Require one of the locked compression methods."""

    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")


def validate_required_seeds(
    seeds: tuple[int, ...] | list[int],
) -> None:
    """Require the locked three-seed contract."""

    observed = tuple(sorted(seeds))

    if observed != REQUIRED_SEEDS:
        raise ValueError(f"Expected seeds {REQUIRED_SEEDS}, " f"observed {observed}")


def validate_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Reject NaN and infinite evidence values."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite: {value}")


def compression_quality_pass(
    compression_ratio: float,
    relative_mse_change_percent: float,
) -> bool:
    """Evaluate the frozen Sprint 2 joint criterion."""

    validate_finite(
        compression_ratio,
        name="compression_ratio",
    )

    validate_finite(
        relative_mse_change_percent,
        name="relative_mse_change_percent",
    )

    return (
        compression_ratio >= MIN_COMPRESSION_RATIO
        and relative_mse_change_percent <= MAX_RELATIVE_MSE_DEGRADATION_PERCENT
    )


def classify_method(
    method: str,
    *,
    compression_ratio: float | None = None,
    relative_mse_change_percent: float | None = None,
) -> str:
    """Classify FP32 as reference and compressed methods by criterion."""

    validate_method(method)

    if method == REFERENCE_METHOD:
        return "reference"

    if compression_ratio is None:
        raise ValueError("Compressed method requires compression_ratio.")

    if relative_mse_change_percent is None:
        raise ValueError("Compressed method requires " "relative_mse_change_percent.")

    return (
        "pass"
        if compression_quality_pass(
            compression_ratio,
            relative_mse_change_percent,
        )
        else "fail"
    )


def validate_source_provenance(
    source_artifacts: tuple[str, ...],
) -> None:
    """Require explicit frozen-evidence provenance."""

    if not source_artifacts:
        raise ValueError("At least one source artifact is required.")

    if any(not item.strip() for item in source_artifacts):
        raise ValueError("Source artifact paths must be non-empty.")


def build_reference_record(
    *,
    domain: str,
    mse_mean: float,
    mse_sample_std: float,
    source_artifacts: tuple[str, ...],
    latency_mean_ms: float | None = None,
    latency_sample_std_ms: float | None = None,
) -> CompressionAblationRecord:
    """Build the explicit FP32 reference row."""

    validate_domain(domain)

    validate_source_provenance(source_artifacts)

    validate_finite(
        mse_mean,
        name="mse_mean",
    )

    validate_finite(
        mse_sample_std,
        name="mse_sample_std",
    )

    return CompressionAblationRecord(
        domain=domain,
        method="fp32",
        compression_ratio=1.0,
        mse_mean=mse_mean,
        mse_sample_std=mse_sample_std,
        relative_mse_change_mean_percent=None,
        relative_mse_change_sample_std_percent=None,
        criterion_pass=None,
        seed_pass_count=None,
        source_artifacts=source_artifacts,
        latency_mean_ms=latency_mean_ms,
        latency_sample_std_ms=latency_sample_std_ms,
        classification="reference",
    )


def build_compressed_record(
    *,
    domain: str,
    method: str,
    compression_ratio: float,
    mse_mean: float,
    mse_sample_std: float,
    relative_mse_change_mean_percent: float,
    relative_mse_change_sample_std_percent: float,
    seed_pass_count: int,
    source_artifacts: tuple[str, ...],
    latency_mean_ms: float | None = None,
    latency_sample_std_ms: float | None = None,
) -> CompressionAblationRecord:
    """Build one compressed-method ablation record."""

    validate_domain(domain)

    validate_method(method)

    if method == REFERENCE_METHOD:
        raise ValueError("Use build_reference_record for FP32.")

    validate_source_provenance(source_artifacts)

    for name, value in (
        (
            "compression_ratio",
            compression_ratio,
        ),
        (
            "mse_mean",
            mse_mean,
        ),
        (
            "mse_sample_std",
            mse_sample_std,
        ),
        (
            "relative_mse_change_mean_percent",
            relative_mse_change_mean_percent,
        ),
        (
            "relative_mse_change_sample_std_percent",
            relative_mse_change_sample_std_percent,
        ),
    ):
        validate_finite(
            value,
            name=name,
        )

    if not 0 <= seed_pass_count <= len(REQUIRED_SEEDS):
        raise ValueError(
            "seed_pass_count must be between " f"0 and {len(REQUIRED_SEEDS)}."
        )

    criterion_pass = compression_quality_pass(
        compression_ratio,
        relative_mse_change_mean_percent,
    )

    return CompressionAblationRecord(
        domain=domain,
        method=method,
        compression_ratio=compression_ratio,
        mse_mean=mse_mean,
        mse_sample_std=mse_sample_std,
        relative_mse_change_mean_percent=(relative_mse_change_mean_percent),
        relative_mse_change_sample_std_percent=(relative_mse_change_sample_std_percent),
        criterion_pass=criterion_pass,
        seed_pass_count=seed_pass_count,
        source_artifacts=source_artifacts,
        latency_mean_ms=latency_mean_ms,
        latency_sample_std_ms=latency_sample_std_ms,
        classification=("pass" if criterion_pass else "fail"),
    )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one canonical JSON artifact."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def _summary_from_seed_values(
    *,
    metric: str,
    seeds: list[int],
    values: list[float],
):
    """Recompute mean and sample SD from explicit seed values."""

    validate_required_seeds(seeds)

    return summarize_metric(
        metric,
        [
            SeedMetric(
                seed=int(seed),
                value=float(value),
            )
            for seed, value in zip(
                seeds,
                values,
                strict=True,
            )
        ],
    )


def _require_invariant(
    values: list[float],
    *,
    name: str,
) -> float:
    """Require a deterministic quantity to be invariant by seed."""

    if len(values) != len(REQUIRED_SEEDS):
        raise ValueError(f"{name}: expected three values.")

    first = float(values[0])

    if not all(
        math.isclose(
            float(value),
            first,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value in values
    ):
        raise ValueError(f"{name} is not invariant across seeds.")

    return first


def _validate_reported_summary(
    payload: dict[str, Any],
    *,
    recomputed_mean: float,
    recomputed_std: float,
    metric: str,
) -> None:
    """Cross-check frozen reported statistics against seed values."""

    if not math.isclose(
        float(payload["mean"]),
        recomputed_mean,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{metric}: reported mean does not match seed values.")

    if not math.isclose(
        float(payload["std"]),
        recomputed_std,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{metric}: reported sample SD does not match seed values.")


def _build_fp32_from_domain(
    *,
    domain_name: str,
    domain_payload: dict[str, Any],
) -> CompressionAblationRecord:
    """Build FP32 reference from frozen baseline evidence."""

    int8_payload = domain_payload["int8"]

    seeds = list(int8_payload["seeds"])

    baseline_mse = int8_payload["baseline_test_mse"]

    mse_summary = _summary_from_seed_values(
        metric="baseline_test_mse",
        seeds=seeds,
        values=list(baseline_mse["values"]),
    )

    _validate_reported_summary(
        baseline_mse,
        recomputed_mean=mse_summary.mean,
        recomputed_std=mse_summary.sample_std,
        metric=(f"{domain_name}/fp32/" "baseline_test_mse"),
    )

    latency_mean = None
    latency_std = None

    baseline_latency = int8_payload.get("baseline_mean_latency_ms")

    if baseline_latency is not None:
        latency_summary = _summary_from_seed_values(
            metric="baseline_mean_latency_ms",
            seeds=seeds,
            values=list(baseline_latency["values"]),
        )

        _validate_reported_summary(
            baseline_latency,
            recomputed_mean=latency_summary.mean,
            recomputed_std=latency_summary.sample_std,
            metric=(f"{domain_name}/fp32/" "baseline_mean_latency_ms"),
        )

        latency_mean = latency_summary.mean
        latency_std = latency_summary.sample_std

    return build_reference_record(
        domain=domain_name,
        mse_mean=mse_summary.mean,
        mse_sample_std=mse_summary.sample_std,
        source_artifacts=(CANONICAL_COMPRESSION_SOURCE,),
        latency_mean_ms=latency_mean,
        latency_sample_std_ms=latency_std,
    )


def _build_compressed_from_method(
    *,
    domain_name: str,
    method_name: str,
    method_payload: dict[str, Any],
) -> CompressionAblationRecord:
    """Build compressed-method record from frozen Sprint 2 evidence."""

    seeds = list(method_payload["seeds"])

    validate_required_seeds(seeds)

    ratio_payload = method_payload["compression_ratio"]

    mse_payload = method_payload["compressed_test_mse"]

    change_payload = method_payload["mse_change_percent"]

    ratio_values = [float(value) for value in ratio_payload["values"]]

    compression_ratio = _require_invariant(
        ratio_values,
        name=(f"{domain_name}/{method_name}/" "compression_ratio"),
    )

    mse_summary = _summary_from_seed_values(
        metric="compressed_test_mse",
        seeds=seeds,
        values=list(mse_payload["values"]),
    )

    change_summary = _summary_from_seed_values(
        metric="mse_change_percent",
        seeds=seeds,
        values=list(change_payload["values"]),
    )

    _validate_reported_summary(
        mse_payload,
        recomputed_mean=mse_summary.mean,
        recomputed_std=mse_summary.sample_std,
        metric=(f"{domain_name}/{method_name}/" "compressed_test_mse"),
    )

    _validate_reported_summary(
        change_payload,
        recomputed_mean=change_summary.mean,
        recomputed_std=change_summary.sample_std,
        metric=(f"{domain_name}/{method_name}/" "mse_change_percent"),
    )

    seed_pass_count = sum(
        compression_quality_pass(
            ratio,
            float(mse_change),
        )
        for ratio, mse_change in zip(
            ratio_values,
            change_payload["values"],
            strict=True,
        )
    )

    reported_seed_pass_count = int(method_payload["pilot_feasible_runs"])

    if seed_pass_count != reported_seed_pass_count:
        raise ValueError(
            f"{domain_name}/{method_name}: "
            "derived seed pass count differs "
            "from frozen Sprint 2 evidence."
        )

    reported_all_pass = bool(method_payload["all_runs_pilot_feasible"])

    derived_all_pass = seed_pass_count == len(REQUIRED_SEEDS)

    if reported_all_pass != derived_all_pass:
        raise ValueError(
            f"{domain_name}/{method_name}: " "all-run feasibility mismatch."
        )

    latency_mean = None
    latency_std = None

    latency_payload = method_payload.get("compressed_mean_latency_ms")

    if latency_payload is not None:
        latency_summary = _summary_from_seed_values(
            metric="compressed_mean_latency_ms",
            seeds=seeds,
            values=list(latency_payload["values"]),
        )

        _validate_reported_summary(
            latency_payload,
            recomputed_mean=latency_summary.mean,
            recomputed_std=latency_summary.sample_std,
            metric=(f"{domain_name}/{method_name}/" "compressed_mean_latency_ms"),
        )

        latency_mean = latency_summary.mean
        latency_std = latency_summary.sample_std

    return build_compressed_record(
        domain=domain_name,
        method=method_name,
        compression_ratio=compression_ratio,
        mse_mean=mse_summary.mean,
        mse_sample_std=mse_summary.sample_std,
        relative_mse_change_mean_percent=(change_summary.mean),
        relative_mse_change_sample_std_percent=(change_summary.sample_std),
        seed_pass_count=seed_pass_count,
        source_artifacts=(CANONICAL_COMPRESSION_SOURCE,),
        latency_mean_ms=latency_mean,
        latency_sample_std_ms=latency_std,
    )


def load_canonical_compression_ablation(
    root: Path,
) -> tuple[
    CompressionAblationRecord,
    ...,
]:
    """Load all eight canonical Sprint 7.3 ablation records."""

    source_path = root / CANONICAL_COMPRESSION_SOURCE

    source = _load_json(source_path)

    validate_required_seeds(list(source["seeds"]))

    records: list[CompressionAblationRecord] = []

    for domain_name in DOMAINS:
        source_key = DOMAIN_SOURCE_KEYS[domain_name]

        domain_payload = source[source_key]

        validate_required_seeds(list(domain_payload["seeds"]))

        if not math.isclose(
            float(domain_payload["minimum_compression_ratio"]),
            MIN_COMPRESSION_RATIO,
        ):
            raise ValueError(f"{domain_name}: " "compression threshold mismatch.")

        if not math.isclose(
            float(domain_payload["maximum_mse_increase_percent"]),
            MAX_RELATIVE_MSE_DEGRADATION_PERCENT,
        ):
            raise ValueError(f"{domain_name}: " "MSE threshold mismatch.")

        records.append(
            _build_fp32_from_domain(
                domain_name=domain_name,
                domain_payload=domain_payload,
            )
        )

        for method_name in (
            "int8",
            "svd",
            "tt_mps",
        ):
            source_method = METHOD_SOURCE_KEYS[method_name]

            records.append(
                _build_compressed_from_method(
                    domain_name=domain_name,
                    method_name=method_name,
                    method_payload=domain_payload[source_method],
                )
            )

    expected = {
        (
            domain,
            method,
        )
        for domain in DOMAINS
        for method in METHODS
    }

    observed = {
        (
            record.domain,
            record.method,
        )
        for record in records
    }

    if observed != expected:
        raise ValueError("Canonical compression coverage is incomplete.")

    if len(records) != 8:
        raise ValueError("Expected exactly eight canonical records.")

    return tuple(records)
