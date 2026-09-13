"""Shared environment audit utilities for Sprint 4.4."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

AUDIT_EPISODE_COUNT = 100
AUDIT_SEED_START = 10_000

AUDIT_SEEDS = tuple(
    range(
        AUDIT_SEED_START,
        AUDIT_SEED_START + AUDIT_EPISODE_COUNT,
    )
)


@dataclass(frozen=True)
class EpisodeAuditRecord:
    """Result from one deterministic audit episode."""

    domain: str
    policy: str
    seed: int
    total_reward: float
    episode_length: int
    success: bool
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_reward):
            raise ValueError("episode reward must be finite")

        if self.episode_length <= 0:
            raise ValueError("episode length must be positive")

        if self.terminated and self.truncated:
            raise ValueError("episode cannot be both terminated " "and truncated")


@dataclass(frozen=True)
class PolicyAuditSummary:
    """Aggregate statistics for one policy/domain pair."""

    domain: str
    policy: str
    episodes: int
    mean_reward: float
    reward_standard_deviation: float
    minimum_reward: float
    maximum_reward: float
    success_rate: float
    mean_episode_length: float
    terminated_rate: float
    truncated_rate: float

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")

        for value in (
            self.success_rate,
            self.terminated_rate,
            self.truncated_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rates must lie in [0, 1]")


def summarize_episode_records(
    records: list[EpisodeAuditRecord],
) -> PolicyAuditSummary:
    """Aggregate one homogeneous episode collection."""
    if not records:
        raise ValueError("records cannot be empty")

    domains = {record.domain for record in records}

    policies = {record.policy for record in records}

    if len(domains) != 1:
        raise ValueError("records must contain one domain")

    if len(policies) != 1:
        raise ValueError("records must contain one policy")

    rewards = [record.total_reward for record in records]

    lengths = [record.episode_length for record in records]

    successes = [float(record.success) for record in records]

    terminated = [float(record.terminated) for record in records]

    truncated = [float(record.truncated) for record in records]

    reward_sd = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    return PolicyAuditSummary(
        domain=records[0].domain,
        policy=records[0].policy,
        episodes=len(records),
        mean_reward=statistics.mean(rewards),
        reward_standard_deviation=reward_sd,
        minimum_reward=min(rewards),
        maximum_reward=max(rewards),
        success_rate=statistics.mean(successes),
        mean_episode_length=statistics.mean(lengths),
        terminated_rate=statistics.mean(terminated),
        truncated_rate=statistics.mean(truncated),
    )


def heuristic_beats_random(
    *,
    heuristic: PolicyAuditSummary,
    random: PolicyAuditSummary,
) -> bool:
    """Check the minimum task-sanity relationship."""
    if heuristic.domain != random.domain:
        raise ValueError("domain mismatch")

    return bool(heuristic.mean_reward > random.mean_reward)


def summary_to_dict(
    summary: PolicyAuditSummary,
) -> dict[str, Any]:
    """Convert summary to JSON-safe data."""
    return asdict(summary)


def episode_to_dict(
    record: EpisodeAuditRecord,
) -> dict[str, Any]:
    """Convert episode record to JSON-safe data."""
    return asdict(record)
