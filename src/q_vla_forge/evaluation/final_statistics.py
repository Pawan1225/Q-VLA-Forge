"""Sprint 7.2 final multi-seed statistical aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean, stdev

LOCKED_SEEDS: tuple[int, ...] = (42, 123, 456)


@dataclass(frozen=True)
class SeedMetric:
    """One metric value associated with one experimental seed."""

    seed: int
    value: float


@dataclass(frozen=True)
class MetricStatistics:
    """Final three-seed statistics for one metric."""

    metric: str
    values: tuple[SeedMetric, ...]
    mean: float
    sample_std: float
    minimum: float
    maximum: float
    n: int

    @property
    def formatted(self) -> str:
        """Return proposal-ready mean ± sample standard deviation."""

        return f"{self.mean:.6f} " f"± {self.sample_std:.6f}"

    def to_dict(self) -> dict:
        """Convert statistics to a serializable dictionary."""

        payload = asdict(self)
        payload["formatted"] = self.formatted

        return payload


def validate_seed_metrics(
    values: Sequence[SeedMetric],
) -> None:
    """Require exactly one finite value for every locked seed."""

    observed = [item.seed for item in values]

    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate seed values detected: " f"{observed}")

    if set(observed) != set(LOCKED_SEEDS):
        raise ValueError(
            "Expected exactly seeds "
            f"{LOCKED_SEEDS}, observed "
            f"{tuple(sorted(observed))}"
        )

    for item in values:
        if not math.isfinite(item.value):
            raise ValueError("Non-finite value for seed " f"{item.seed}: {item.value}")


def summarize_metric(
    metric: str,
    values: Sequence[SeedMetric],
) -> MetricStatistics:
    """Calculate mean and sample standard deviation."""

    validate_seed_metrics(values)

    ordered = tuple(
        sorted(
            values,
            key=lambda item: item.seed,
        )
    )

    numeric = [item.value for item in ordered]

    return MetricStatistics(
        metric=metric,
        values=ordered,
        mean=mean(numeric),
        sample_std=stdev(numeric),
        minimum=min(numeric),
        maximum=max(numeric),
        n=len(numeric),
    )
