from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.cross_domain_training import (
    DomainMethodEvidence,
    build_cross_domain_comparison,
    efficiency_consistency_label,
    target_consistency_label,
)

ARTIFACT = Path(
    "results/training/cross-domain/" "cross-domain-training-comparison.json"
)


def _evidence(
    *,
    domain: str,
    method: str = "trainable_tt_mps",
    reached: int = 3,
    robust: bool = True,
    parameter_reduction: float = 40.0,
) -> DomainMethodEvidence:
    return DomainMethodEvidence(
        domain=domain,
        method=method,
        target_reached=reached,
        target_total=3,
        parameter_reduction_percent=parameter_reduction,
        step_reduction_percent_mean=15.0,
        step_reduction_percent_std=2.0,
        test_mse_mean=0.01,
        test_mse_std=0.001,
        robust_ten_percent_efficiency=robust,
    )


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_target_consistency_strong() -> None:
    driving = _evidence(
        domain="autonomous_driving",
    )

    robotics = _evidence(
        domain="robotics",
    )

    assert (
        target_consistency_label(
            driving,
            robotics,
        )
        == "strong"
    )


def test_target_consistency_mixed() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        reached=3,
    )

    robotics = _evidence(
        domain="robotics",
        reached=2,
    )

    assert (
        target_consistency_label(
            driving,
            robotics,
        )
        == "mixed"
    )


def test_target_consistency_weak() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        reached=2,
    )

    robotics = _evidence(
        domain="robotics",
        reached=1,
    )

    assert (
        target_consistency_label(
            driving,
            robotics,
        )
        == "weak"
    )


def test_efficiency_consistency_strong() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        robust=True,
    )

    robotics = _evidence(
        domain="robotics",
        robust=True,
    )

    assert (
        efficiency_consistency_label(
            driving,
            robotics,
        )
        == "strong"
    )


def test_efficiency_consistency_mixed() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        robust=True,
    )

    robotics = _evidence(
        domain="robotics",
        robust=False,
    )

    assert (
        efficiency_consistency_label(
            driving,
            robotics,
        )
        == "mixed"
    )


def test_build_cross_domain_comparison() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        parameter_reduction=40.0,
    )

    robotics = _evidence(
        domain="robotics",
        parameter_reduction=35.0,
    )

    result = build_cross_domain_comparison(
        driving,
        robotics,
    )

    assert result.robust_cross_domain_efficiency is True

    assert result.parameter_reduction_difference_percent == pytest.approx(5.0)


def test_method_mismatch_rejected() -> None:
    driving = _evidence(
        domain="autonomous_driving",
        method="trainable_svd",
    )

    robotics = _evidence(
        domain="robotics",
        method="trainable_tt_mps",
    )

    with pytest.raises(ValueError):
        build_cross_domain_comparison(
            driving,
            robotics,
        )


def test_generated_artifact_has_two_methods() -> None:
    payload = _artifact()

    assert [item["method"] for item in payload["methods"]] == [
        "trainable_svd",
        "trainable_tt_mps",
    ]


def test_generated_artifact_has_both_domains() -> None:
    payload = _artifact()

    assert payload["domains"] == [
        "autonomous_driving",
        "robotics",
    ]


def test_generated_artifact_architecture_claims() -> None:
    payload = _artifact()

    interpretation = payload["architecture_interpretation"]

    assert interpretation["shared_architecture_claim"] is True

    assert interpretation["universal_trained_model_claim"] is False


def test_generated_artifact_rejects_quantum_advantage_claims() -> None:
    payload = _artifact()

    controls = payload["claim_control"]

    assert controls["cross_domain_quantum_advantage"] is False

    assert controls["quantum_speedup"] is False

    assert controls["quantum_hardware_used"] is False

    assert controls["native_tt_runtime_acceleration"] is False


def test_generated_artifact_figures_exist() -> None:
    payload = _artifact()

    figures = [Path(path) for path in payload["figures"]]

    assert len(figures) == 2

    assert all(path.exists() for path in figures)
