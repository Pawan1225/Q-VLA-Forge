from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.training_analysis import (
    EfficiencyQualityPoint,
    mean_curve,
    parameter_reduction_percent,
    percentage_improvement,
)

ANALYSIS = (
    Path("results") / "training" / "analysis" / "training-efficiency-analysis.json"
)


def test_parameter_reduction() -> None:
    result = parameter_reduction_percent(
        100.0,
        75.0,
    )

    assert result == pytest.approx(25.0)


def test_parameter_reduction_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError):
        parameter_reduction_percent(
            0.0,
            10.0,
        )


def test_percentage_improvement() -> None:
    assert percentage_improvement(
        100.0,
        80.0,
    ) == pytest.approx(20.0)


def test_percentage_improvement_can_be_negative() -> None:
    assert percentage_improvement(
        100.0,
        120.0,
    ) == pytest.approx(-20.0)


def test_mean_curve() -> None:
    result = mean_curve(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]
    )

    assert result == pytest.approx(
        (
            2.0,
            3.0,
            4.0,
        )
    )


def test_mean_curve_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        mean_curve(
            [
                [1.0, 2.0],
                [1.0],
            ]
        )


def test_efficiency_quality_point_reach_rate() -> None:
    point = EfficiencyQualityPoint(
        domain="autonomous_driving",
        method="trainable_tt_mps",
        target_reach_count=2,
        target_reach_total=3,
        trainable_parameters=100.0,
        parameter_reduction_percent=25.0,
        step_reduction_percent_mean=10.0,
        step_reduction_percent_std=2.0,
        test_mse_mean=0.1,
        test_mse_std=0.01,
        test_mae_mean=0.05,
        test_mae_std=0.01,
        robust_ten_percent_efficiency=False,
    )

    assert point.target_reach_rate_percent == pytest.approx(66.6666666667)


def test_generated_analysis_has_four_points() -> None:
    if not ANALYSIS.exists():
        pytest.skip("analysis artifact not generated")

    payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))

    assert len(payload["points"]) == 4


def test_generated_analysis_uses_three_seeds() -> None:
    if not ANALYSIS.exists():
        pytest.skip("analysis artifact not generated")

    payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]


def test_quantum_advantage_claim_disabled() -> None:
    if not ANALYSIS.exists():
        pytest.skip("analysis artifact not generated")

    payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))

    assert payload["claim_rules"]["quantum_advantage_claim"] is False
