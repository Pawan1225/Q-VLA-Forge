"""Sprint 5.13G safety claim-matrix utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    SUPPORTED_WITH_LIMITATIONS = "supported_with_limitations"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SafetyClaim:
    claim_id: str
    statement: str
    status: ClaimStatus
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


def count_by_status(
    claims: list[SafetyClaim],
) -> dict[str, int]:
    counts = {status.value: 0 for status in ClaimStatus}

    for claim in claims:
        counts[claim.status.value] += 1

    return counts


def unsupported_claim_ids(
    claims: list[SafetyClaim],
) -> list[str]:
    return [
        claim.claim_id for claim in claims if claim.status is ClaimStatus.UNSUPPORTED
    ]
