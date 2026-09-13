from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.training_evidence import (
    build_cross_domain_claim,
    build_method_claim,
    domain_label,
    method_label,
    prohibited_claims,
)

EVIDENCE_PATH = (
    Path("results") / "training" / "evidence" / "sprint3-training-evidence.json"
)

MARKDOWN_PATH = (
    Path("results") / "training" / "evidence" / "sprint3-training-evidence.md"
)

CSV_PATH = Path("results") / "training" / "evidence" / "sprint3-training-evidence.csv"


def test_method_labels() -> None:
    assert method_label("trainable_svd") == "Trainable SVD"

    assert method_label("trainable_tt_mps") == "Trainable TT/MPS"


def test_domain_labels() -> None:
    assert domain_label("autonomous_driving") == "autonomous driving"

    assert domain_label("robotics") == "robotics"


def test_robust_method_claim() -> None:
    claim = build_method_claim(
        domain="autonomous_driving",
        method="trainable_tt_mps",
        reached=3,
        total=3,
        step_reduction_mean=15.0,
        step_reduction_std=2.0,
        robust_ten_percent=True,
    )

    assert claim.supported
    assert "15.00 ± 2.00%" in claim.statement


def test_partial_reach_claim_is_cautious() -> None:
    claim = build_method_claim(
        domain="robotics",
        method="trainable_tt_mps",
        reached=2,
        total=3,
        step_reduction_mean=20.0,
        step_reduction_std=3.0,
        robust_ten_percent=False,
    )

    assert claim.supported

    assert "does not support a robust" in claim.statement


def test_inconsistent_robust_claim_rejected() -> None:
    with pytest.raises(ValueError):
        build_method_claim(
            domain="robotics",
            method="trainable_tt_mps",
            reached=2,
            total=3,
            step_reduction_mean=20.0,
            step_reduction_std=3.0,
            robust_ten_percent=True,
        )


def test_cross_domain_positive_claim() -> None:
    claim = build_cross_domain_claim(
        method="trainable_tt_mps",
        robust_cross_domain=True,
        target_consistency="strong",
        efficiency_consistency="strong",
    )

    assert claim.supported

    assert "both autonomous-driving and robotics" in claim.statement


def test_cross_domain_mixed_claim() -> None:
    claim = build_cross_domain_claim(
        method="trainable_tt_mps",
        robust_cross_domain=False,
        target_consistency="strong",
        efficiency_consistency="mixed",
    )

    assert claim.supported

    assert "did not demonstrate robust" in claim.statement


def test_prohibited_claims_are_unsupported() -> None:
    claims = prohibited_claims()

    assert len(claims) >= 4

    assert all(not claim.supported for claim in claims)


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError):
        method_label("unknown")


def test_generated_evidence_shape() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]

    assert len(payload["table"]) == 4


def test_generated_evidence_has_both_domains() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert set(payload["domains"]) == {
        "autonomous_driving",
        "robotics",
    }


def test_evidence_disables_quantum_advantage() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    statements = {item["claim_id"]: item for item in payload["unsupported_claims"]}

    assert statements["quantum-advantage"]["supported"] is False

    assert statements["quantum-speedup"]["supported"] is False


def test_evidence_markdown_exists() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    assert MARKDOWN_PATH.exists()


def test_evidence_csv_exists() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    assert CSV_PATH.exists()


def test_evidence_uses_lightweight_text_encoder_wording() -> None:
    if not EVIDENCE_PATH.exists():
        pytest.skip("Sprint 3 evidence not generated")

    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    text = " ".join(payload["limitations"])

    assert "lightweight" in text.lower()

    assert "not a large Transformer" in text
