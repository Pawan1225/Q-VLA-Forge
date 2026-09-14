"""Tests for Sprint 5.13G safety claim matrix."""

from q_vla_forge.evaluation.safety_claim_matrix import (
    ClaimStatus,
    SafetyClaim,
    count_by_status,
    unsupported_claim_ids,
)


def test_status_counts() -> None:
    claims = [
        SafetyClaim(
            claim_id="a",
            statement="A",
            status=ClaimStatus.SUPPORTED,
            evidence=("x",),
            limitations=(),
        ),
        SafetyClaim(
            claim_id="b",
            statement="B",
            status=ClaimStatus.SUPPORTED_WITH_LIMITATIONS,
            evidence=("y",),
            limitations=("limited",),
        ),
        SafetyClaim(
            claim_id="c",
            statement="C",
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=("unsupported",),
        ),
    ]

    counts = count_by_status(claims)

    assert counts["supported"] == 1

    assert counts["supported_with_limitations"] == 1

    assert counts["unsupported"] == 1


def test_unsupported_ids() -> None:
    claims = [
        SafetyClaim(
            claim_id="a",
            statement="A",
            status=ClaimStatus.SUPPORTED,
            evidence=("x",),
            limitations=(),
        ),
        SafetyClaim(
            claim_id="b",
            statement="B",
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=("no evidence",),
        ),
    ]

    assert unsupported_claim_ids(claims) == ["b"]
