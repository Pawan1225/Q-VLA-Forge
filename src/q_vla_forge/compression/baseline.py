"""Frozen Sprint 1 baseline references for compression experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from q_vla_forge.data import Domain
from q_vla_forge.utils.reproducibility import DEFAULT_SEEDS


@dataclass(frozen=True)
class FrozenBaselineReference:
    """Aggregate frozen Sprint 1 FP32 reference."""

    baseline_name: str
    baseline_version: str
    seeds: tuple[int, ...]

    parameters: int
    fp32_model_size_bytes: int

    driving_test_mse_mean: float
    driving_test_mse_std: float
    driving_test_mae_mean: float
    driving_test_mae_std: float
    driving_mean_latency_ms: float
    driving_p95_latency_ms: float

    robotics_test_mse_mean: float
    robotics_test_mse_std: float
    robotics_test_mae_mean: float
    robotics_test_mae_std: float
    robotics_mean_latency_ms: float
    robotics_p95_latency_ms: float


@dataclass(frozen=True)
class SeedBaselineReference:
    """Exact per-seed FP32 baseline for one domain."""

    domain: Domain
    seed: int

    parameters: int
    fp32_model_size_bytes: int

    test_mse: float
    test_mae: float

    mean_latency_ms: float
    p95_latency_ms: float


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"required baseline file not found: {path}")

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in baseline file: {path}")

    return payload


def load_frozen_baseline(
    results_dir: Path,
) -> FrozenBaselineReference:
    """Load and validate the frozen Sprint 1 aggregate reference."""
    manifest = _load_json(results_dir / "sprint1-baseline-manifest.json")

    summary = _load_json(results_dir / "baseline-validation-summary.json")

    if manifest["baseline_name"] != "shared_vla_fp32":
        raise ValueError("unexpected Sprint 1 baseline name")

    if manifest["baseline_version"] != "1.0":
        raise ValueError("unexpected Sprint 1 baseline version")

    manifest_seeds = tuple(int(seed) for seed in manifest["seeds"])

    summary_seeds = tuple(int(seed) for seed in summary["seeds"])

    if manifest_seeds != DEFAULT_SEEDS:
        raise ValueError("manifest seeds do not match DEFAULT_SEEDS")

    if summary_seeds != DEFAULT_SEEDS:
        raise ValueError("summary seeds do not match DEFAULT_SEEDS")

    if summary["method"] != "shared_vla_fp32":
        raise ValueError("unexpected baseline summary method")

    driving = summary["driving"]
    robotics = summary["robotics"]

    if driving["domain"] != Domain.AUTONOMOUS_DRIVING.value:
        raise ValueError("unexpected driving domain")

    if robotics["domain"] != Domain.ROBOTICS.value:
        raise ValueError("unexpected robotics domain")

    if int(driving["parameters"]) != int(robotics["parameters"]):
        raise ValueError("cross-domain parameter counts must match")

    if int(driving["fp32_model_size_bytes"]) != int(robotics["fp32_model_size_bytes"]):
        raise ValueError("cross-domain FP32 model sizes must match")

    if int(manifest["parameters"]) != int(driving["parameters"]):
        raise ValueError("manifest and summary parameter counts differ")

    if int(manifest["fp32_model_size_bytes"]) != int(driving["fp32_model_size_bytes"]):
        raise ValueError("manifest and summary model sizes differ")

    return FrozenBaselineReference(
        baseline_name=str(manifest["baseline_name"]),
        baseline_version=str(manifest["baseline_version"]),
        seeds=manifest_seeds,
        parameters=int(manifest["parameters"]),
        fp32_model_size_bytes=int(manifest["fp32_model_size_bytes"]),
        driving_test_mse_mean=float(driving["test_mse"]["mean"]),
        driving_test_mse_std=float(driving["test_mse"]["std"]),
        driving_test_mae_mean=float(driving["test_mae"]["mean"]),
        driving_test_mae_std=float(driving["test_mae"]["std"]),
        driving_mean_latency_ms=float(driving["mean_latency_ms"]["mean"]),
        driving_p95_latency_ms=float(driving["p95_latency_ms"]["mean"]),
        robotics_test_mse_mean=float(robotics["test_mse"]["mean"]),
        robotics_test_mse_std=float(robotics["test_mse"]["std"]),
        robotics_test_mae_mean=float(robotics["test_mae"]["mean"]),
        robotics_test_mae_std=float(robotics["test_mae"]["std"]),
        robotics_mean_latency_ms=float(robotics["mean_latency_ms"]["mean"]),
        robotics_p95_latency_ms=float(robotics["p95_latency_ms"]["mean"]),
    )


def load_seed_baseline(
    results_dir: Path,
    domain: Domain,
    seed: int,
) -> SeedBaselineReference:
    """Load the exact Sprint 1 FP32 result for one seed."""
    if seed not in DEFAULT_SEEDS:
        raise ValueError(f"unsupported baseline seed: {seed}")

    if domain == Domain.AUTONOMOUS_DRIVING:
        prefix = "driving"
    elif domain == Domain.ROBOTICS:
        prefix = "robotics"
    else:
        raise ValueError(f"unsupported domain: {domain}")

    payload = _load_json(results_dir / f"{prefix}-baseline-seed-{seed}.json")

    if payload["domain"] != domain.value:
        raise ValueError("baseline payload domain mismatch")

    if int(payload["seed"]) != seed:
        raise ValueError("baseline payload seed mismatch")

    if payload["method"] != "shared_vla_fp32":
        raise ValueError("unexpected baseline method")

    return SeedBaselineReference(
        domain=domain,
        seed=seed,
        parameters=int(payload["parameters"]),
        fp32_model_size_bytes=int(payload["fp32_model_size_bytes"]),
        test_mse=float(payload["test_metrics"]["test_mse"]),
        test_mae=float(payload["test_metrics"]["test_mae"]),
        mean_latency_ms=float(payload["latency"]["mean_ms"]),
        p95_latency_ms=float(payload["latency"]["p95_ms"]),
    )
