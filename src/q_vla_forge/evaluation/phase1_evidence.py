"""Sprint 7.2 — canonical Phase 1 evidence aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.three_seed_validation import (
    extract_observed_seeds,
    has_result_content,
    is_generated_validation_path,
    path_seed,
)

CONFIGURATION_KEYS: tuple[str, ...] = (
    "epochs",
    "batch_size",
    "optimizer",
    "scheduler",
    "learning_rate",
    "lr",
    "parameters",
    "fp32_model_size_bytes",
    "rank",
    "qubits",
    "layers",
    "episodes",
    "total_timesteps",
)

METRIC_HINTS: tuple[str, ...] = (
    "mse",
    "mae",
    "loss",
    "reward",
    "success",
    "latency",
    "compression",
    "runtime",
    "violation",
    "intervention",
    "efficiency",
    "recovery",
    "parameter",
)


@dataclass(frozen=True)
class EvidenceArtifact:
    """Canonical description of one frozen Phase 1 artifact."""

    experiment: str
    method: str
    domain: str
    seeds: tuple[int, ...]
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    hardware_resource_metadata: dict[str, Any]
    artifact_path: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_string(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    default: str,
) -> str:
    for key in keys:
        value = payload.get(key)

        if isinstance(value, str) and value:
            return value

    return default


def collect_configuration(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Collect explicit top-level configuration fields."""

    return {key: payload[key] for key in CONFIGURATION_KEYS if key in payload}


def collect_metrics(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
) -> dict[str, Any]:
    """Collect compact result metrics without copying histories."""

    if depth > 4:
        return {}

    metrics: dict[str, Any] = {}

    if not isinstance(value, dict):
        return metrics

    for key, item in value.items():
        normalized = str(key).lower()

        if normalized in {
            "training_history",
            "history",
            "epochs_history",
        }:
            continue

        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(item, (int, float)):
            if any(hint in normalized for hint in METRIC_HINTS):
                metrics[name] = item

        elif isinstance(item, dict):
            if (
                "mean" in item
                and "std" in item
                and isinstance(
                    item["mean"],
                    (int, float),
                )
            ):
                metrics[name] = {
                    field: item[field]
                    for field in (
                        "mean",
                        "std",
                        "values",
                    )
                    if field in item
                }
            else:
                metrics.update(
                    collect_metrics(
                        item,
                        prefix=name,
                        depth=depth + 1,
                    )
                )

    return metrics


def find_metadata(
    value: Any,
) -> dict[str, Any]:
    """Find explicit hardware/resource metadata."""

    found: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()

            if normalized in {
                "hardware",
                "hardware_info",
                "resource_metadata",
                "resources",
                "device",
            }:
                found[str(key)] = item

    return found


def artifact_seeds(
    path: Path,
    payload: dict[str, Any],
) -> tuple[int, ...]:
    seeds = path_seed(path)

    seeds.update(
        extract_observed_seeds(
            payload,
            allow_aggregate_seeds=(
                has_result_content(payload)
                and "manifest" not in path.as_posix().lower()
            ),
        )
    )

    return tuple(sorted(seeds))


def build_phase1_evidence(
    root: Path,
) -> dict[str, Any]:
    """Build canonical inventory from frozen JSON evidence."""

    results = root / "results"

    if not results.exists():
        raise FileNotFoundError(results)

    artifacts: list[EvidenceArtifact] = []

    for path in sorted(results.rglob("*.json")):
        if is_generated_validation_path(root, path):
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(payload, dict):
            continue

        relative = path.relative_to(root)

        experiment = first_string(
            payload,
            (
                "experiment_id",
                "experiment",
                "protocol",
                "name",
            ),
            relative.stem,
        )

        method = first_string(
            payload,
            (
                "method",
                "baseline_name",
                "algorithm",
                "model",
            ),
            "not_explicitly_recorded",
        )

        domain = first_string(
            payload,
            (
                "domain",
                "environment",
                "task_domain",
            ),
            "cross_domain_or_unspecified",
        )

        artifacts.append(
            EvidenceArtifact(
                experiment=experiment,
                method=method,
                domain=domain,
                seeds=artifact_seeds(
                    relative,
                    payload,
                ),
                configuration=collect_configuration(payload),
                metrics=collect_metrics(payload),
                hardware_resource_metadata=(find_metadata(payload)),
                artifact_path=relative.as_posix(),
                provenance=("frozen_phase1_repository_artifact"),
            )
        )

    return {
        "sprint": "7.2",
        "protocol": "cross_sprint_evidence_aggregation",
        "scope": "frozen_phase1_evidence_only",
        "artifact_count": len(artifacts),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
