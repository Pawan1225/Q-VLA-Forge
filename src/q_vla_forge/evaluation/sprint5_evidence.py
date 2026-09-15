from __future__ import annotations

import hashlib
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PRINCIPAL_SEEDS = (42, 123, 456)

ALLOWED_METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

ALLOWED_DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SCIENTIFIC_SCOPE_FLAGS = {
    "new_training": False,
    "new_principal_execution": False,
    "new_scientific_experiment": False,
    "scientific_scope_frozen": True,
}


@dataclass(frozen=True)
class SeedMetric:
    seed: int
    value: float | None


@dataclass(frozen=True)
class AggregateMetric:
    mean: float | None
    sample_sd: float | None
    seed_values: tuple[SeedMetric, ...]


@dataclass(frozen=True)
class SafetyResult:
    domain: str
    regime: str
    condition: str
    method: str
    metric: str
    aggregate: AggregateMetric
    source_artifact: str


@dataclass(frozen=True)
class ActionRecoveryResult:
    domain: str
    method: str
    unsafe_perturbed_steps: int
    recovered_steps: int
    unresolved_steps: int
    recovery_rate: float | None


@dataclass(frozen=True)
class LyapunovMechanismResult:
    domain: str
    regime: str
    action_bound: int
    domain_constraint: int
    lyapunov_decrease: int
    emergency_fallback: int
    strict_decrease: int
    total_interventions: int


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    claim: str
    status: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    proposal_safe_wording: str
    prohibited_wording: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceArtifact:
    path: str
    role: str
    sha256: str
    size_bytes: int
    required: bool


def aggregate_seed_values(
    values: Iterable[SeedMetric],
) -> AggregateMetric:
    seed_values = tuple(
        sorted(
            values,
            key=lambda item: item.seed,
        )
    )

    if not seed_values:
        return AggregateMetric(
            mean=None,
            sample_sd=None,
            seed_values=(),
        )

    seeds = tuple(item.seed for item in seed_values)

    if seeds != PRINCIPAL_SEEDS:
        raise ValueError("Expected principal seeds " f"{PRINCIPAL_SEEDS}, got {seeds}.")

    numeric_values = [item.value for item in seed_values if item.value is not None]

    if len(numeric_values) != len(seed_values):
        return AggregateMetric(
            mean=None,
            sample_sd=None,
            seed_values=seed_values,
        )

    mean = statistics.mean(numeric_values)

    sample_sd = statistics.stdev(numeric_values)

    return AggregateMetric(
        mean=float(mean),
        sample_sd=float(sample_sd),
        seed_values=seed_values,
    )


def relative_reduction(
    reference: float,
    value: float,
) -> float | None:
    if reference == 0.0:
        return None

    return (reference - value) / reference


def recovery_rate(
    unsafe_perturbed_steps: int,
    recovered_steps: int,
) -> float | None:
    if unsafe_perturbed_steps < 0:
        raise ValueError("unsafe_perturbed_steps cannot be negative")

    if recovered_steps < 0:
        raise ValueError("recovered_steps cannot be negative")

    if recovered_steps > unsafe_perturbed_steps:
        raise ValueError("recovered_steps cannot exceed unsafe steps")

    if unsafe_perturbed_steps == 0:
        return None

    return recovered_steps / unsafe_perturbed_steps


def validate_action_recovery(
    result: ActionRecoveryResult,
) -> None:
    if (
        result.recovered_steps + result.unresolved_steps
        != result.unsafe_perturbed_steps
    ):
        raise ValueError("unsafe steps must equal " "recovered + unresolved")

    expected = recovery_rate(
        result.unsafe_perturbed_steps,
        result.recovered_steps,
    )

    if expected != result.recovery_rate:
        raise ValueError("recovery rate does not match counts")


def validate_mechanism_result(
    result: LyapunovMechanismResult,
) -> None:
    reason_total = (
        result.action_bound
        + result.domain_constraint
        + result.lyapunov_decrease
        + result.emergency_fallback
    )

    if reason_total > result.total_interventions:
        raise ValueError("mechanism reason counts exceed interventions")

    if result.strict_decrease < 0:
        raise ValueError("strict decrease count cannot be negative")


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def artifact_record(
    path: Path,
    *,
    role: str,
    required: bool = True,
) -> EvidenceArtifact:
    if not path.is_file():
        raise FileNotFoundError(path)

    return EvidenceArtifact(
        path=str(path).replace(
            "\\",
            "/",
        ),
        role=role,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        required=required,
    )


def validate_scope_flags(
    payload: dict[str, object],
) -> None:
    for key, expected in SCIENTIFIC_SCOPE_FLAGS.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid scope flag " f"{key}: " f"{payload.get(key)!r}")
