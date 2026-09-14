"""Sprint 5.13 safety consolidation foundation.

Analysis only.

This module must not:
- train policies
- execute PPO
- execute environments
- modify safety filters
- modify thresholds
- create principal trajectories
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class ThreeSeedMetric:
    seed42: float
    seed123: float
    seed456: float
    mean: float
    sample_sd: float


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sprint: str
    role: str
    group: str
    sha256: str
    size_bytes: int
    required: bool
    frozen: bool


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sample_sd(
    values: Iterable[float],
) -> float:
    materialized = [float(value) for value in values]

    if not materialized:
        raise ValueError("values must be non-empty")

    if len(materialized) == 1:
        return 0.0

    return float(stdev(materialized))


def three_seed_metric(
    *,
    seed42: float,
    seed123: float,
    seed456: float,
) -> ThreeSeedMetric:
    values = [
        float(seed42),
        float(seed123),
        float(seed456),
    ]

    return ThreeSeedMetric(
        seed42=values[0],
        seed123=values[1],
        seed456=values[2],
        mean=float(mean(values)),
        sample_sd=sample_sd(values),
    )


def paired_seed_deltas(
    *,
    baseline: dict[int, float],
    comparison: dict[int, float],
) -> dict[int, float]:
    expected = set(PRINCIPAL_SEEDS)

    if set(baseline) != expected:
        raise ValueError("baseline must contain exactly seeds 42, 123, and 456")

    if set(comparison) != expected:
        raise ValueError("comparison must contain exactly seeds 42, 123, and 456")

    return {
        seed: (float(comparison[seed]) - float(baseline[seed]))
        for seed in PRINCIPAL_SEEDS
    }


def recovery_rate(
    *,
    unsafe_steps: int,
    recovered_steps: int,
) -> float | None:
    if unsafe_steps < 0:
        raise ValueError("unsafe_steps must be nonnegative")

    if recovered_steps < 0:
        raise ValueError("recovered_steps must be nonnegative")

    if recovered_steps > unsafe_steps:
        raise ValueError("recovered_steps cannot exceed unsafe_steps")

    if unsafe_steps == 0:
        return None

    return recovered_steps / unsafe_steps


def make_manifest_entry(
    *,
    root: Path,
    path: Path,
    sprint: str,
    role: str,
    group: str,
    required: bool = True,
    frozen: bool = True,
) -> ManifestEntry:
    resolved_root = root.resolve()
    resolved_path = path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)

    relative = resolved_path.relative_to(resolved_root)

    return ManifestEntry(
        path=str(relative).replace(
            "\\",
            "/",
        ),
        sprint=sprint,
        role=role,
        group=group,
        sha256=sha256_file(resolved_path),
        size_bytes=resolved_path.stat().st_size,
        required=required,
        frozen=frozen,
    )
