"""Proposal-facing Sprint 4 RL evidence utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProposalMethodSummary:
    domain: str
    method: str

    actor_parameters: int
    target_reach_count: int
    target_total: int

    normalized_auc_mean: float
    normalized_auc_sd: float

    best_progress_mean: float
    best_progress_sd: float

    final_progress_mean: float
    final_progress_sd: float


@dataclass(frozen=True)
class ProposalCompactnessSummary:
    domain: str

    full_actor_parameters: int
    compact_actor_parameters: int
    parameter_reduction_percent: float


@dataclass(frozen=True)
class ProposalRepresentationSummary:
    domain: str

    matched_actor_parameters: int
    qml_actor_parameters: int

    matched_auc_mean: float
    qml_auc_mean: float

    matched_minus_qml_auc: float
    direction: str


def method_summary_to_dict(
    summary: ProposalMethodSummary,
) -> dict[str, Any]:
    return asdict(summary)


def compactness_summary_to_dict(
    summary: ProposalCompactnessSummary,
) -> dict[str, Any]:
    return asdict(summary)


def representation_summary_to_dict(
    summary: ProposalRepresentationSummary,
) -> dict[str, Any]:
    return asdict(summary)


def proposal_status(
    *,
    supported: bool,
    limitation: bool = False,
) -> str:
    if supported and limitation:
        return "supported_with_limitation"

    if supported:
        return "supported"

    return "not_supported"
