"""Pilot-readiness audit utilities for Q-VLA Forge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

LOCKED_SEEDS = (42, 123, 456)

LOCKED_DOMAINS = (
    "autonomous_driving",
    "robotics",
)


@dataclass(frozen=True)
class SplitAudit:
    """Integrity evidence for one dataset split."""

    domain: str
    seed: int
    train_size: int
    validation_size: int
    test_size: int
    train_unique: int
    validation_unique: int
    test_unique: int
    train_validation_overlap: int
    train_test_overlap: int
    validation_test_overlap: int

    @property
    def passed(self) -> bool:
        return (
            self.train_unique == self.train_size
            and self.validation_unique == self.validation_size
            and self.test_unique == self.test_size
            and self.train_validation_overlap == 0
            and self.train_test_overlap == 0
            and self.validation_test_overlap == 0
        )


@dataclass(frozen=True)
class DistributionSummary:
    """Compact numerical distribution summary."""

    minimum: list[float]
    maximum: list[float]
    mean: list[float]
    standard_deviation: list[float]


def array_fingerprint(array: np.ndarray) -> bytes:
    """Return a stable byte representation for one array."""
    contiguous = np.ascontiguousarray(
        array,
        dtype=np.float32,
    )

    shape = ",".join(str(value) for value in contiguous.shape).encode("utf-8")

    return shape + b"|" + contiguous.tobytes()


def sample_fingerprint(
    *,
    visual: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    language_goal: str,
) -> str:
    """Build a stable SHA-256 fingerprint for one supervised sample."""
    digest = hashlib.sha256()

    digest.update(array_fingerprint(visual))

    digest.update(array_fingerprint(state))

    digest.update(array_fingerprint(action))

    digest.update(language_goal.encode("utf-8"))

    return digest.hexdigest()


def overlap_count(
    left: set[str],
    right: set[str],
) -> int:
    """Count exact sample fingerprints shared by two splits."""
    return len(left.intersection(right))


def summarize_distribution(
    values: np.ndarray,
) -> DistributionSummary:
    """Summarize a two-dimensional sample-by-feature matrix."""
    if values.ndim != 2:
        raise ValueError("values must have shape [samples, features]")

    return DistributionSummary(
        minimum=np.min(
            values,
            axis=0,
        ).tolist(),
        maximum=np.max(
            values,
            axis=0,
        ).tolist(),
        mean=np.mean(
            values,
            axis=0,
        ).tolist(),
        standard_deviation=np.std(
            values,
            axis=0,
            ddof=0,
        ).tolist(),
    )


def require_locked_seeds(
    seeds: set[int],
) -> None:
    """Require the frozen three-seed protocol."""
    if seeds != set(LOCKED_SEEDS):
        raise ValueError("pilot readiness requires seeds " "42, 123, and 456")


def require_locked_domains(
    domains: set[str],
) -> None:
    """Require both frozen proxy domains."""
    if domains != set(LOCKED_DOMAINS):
        raise ValueError("pilot readiness requires autonomous_driving " "and robotics")


def json_safe(
    value: Any,
) -> Any:
    """Convert common NumPy values into JSON-safe Python values."""
    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    return value
