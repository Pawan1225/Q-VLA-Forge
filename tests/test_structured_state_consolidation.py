"""Tests for Sprint 5.13D structured-state consolidation."""

import pytest

from q_vla_forge.evaluation.structured_state_consolidation import (
    StructuredStateSeedResult,
    summarize_rows,
    summarize_seed_values,
    worst_condition,
)


def _row(
    seed: int,
    violation: float,
) -> StructuredStateSeedResult:
    return StructuredStateSeedResult(
        domain="robotics",
        method="none",
        perturbation_family="object_position",
        perturbation_name="object_x_minus_0p05",
        principal_seed=seed,
        violation_step_rate=violation,
        critical_violation_step_rate=0.0,
        constraint_violation_rate=violation,
        reward=1.0,
        success_rate=0.0,
        intervention_rate=0.0,
        mean_correction_l2=0.0,
        violation_delta_from_clean=violation,
        critical_delta_from_clean=0.0,
        constraint_delta_from_clean=violation,
        reward_delta_from_clean=0.0,
        success_delta_from_clean=0.0,
        strict_lyapunov_decrease_rate=0.0,
        lyapunov_nonincrease_rate=0.0,
        steps_object_grasped=0,
    )


def test_seed_summary() -> None:
    summary = summarize_seed_values(
        {
            42: 1.0,
            123: 2.0,
            456: 3.0,
        }
    )

    assert summary.mean == pytest.approx(2.0)

    assert summary.sample_sd == pytest.approx(1.0)


def test_structured_summary() -> None:
    summary = summarize_rows(
        [
            _row(
                42,
                0.01,
            ),
            _row(
                123,
                0.02,
            ),
            _row(
                456,
                0.03,
            ),
        ]
    )

    assert summary["violation_step_rate"].mean == pytest.approx(0.02)


def test_missing_seed_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="three seeds",
    ):
        summarize_rows(
            [
                _row(
                    42,
                    0.01,
                ),
                _row(
                    123,
                    0.02,
                ),
            ]
        )


def test_worst_condition_max() -> None:
    rows = [
        {
            "perturbation_name": "a",
            "violation_delta_from_clean": {
                "mean": 0.01,
            },
        },
        {
            "perturbation_name": "b",
            "violation_delta_from_clean": {
                "mean": 0.03,
            },
        },
    ]

    result = worst_condition(
        rows,
        metric="violation_delta_from_clean",
        mode="max",
    )

    assert result["perturbation_name"] == "b"


def test_worst_condition_min() -> None:
    rows = [
        {
            "perturbation_name": "a",
            "reward_delta_from_clean": {
                "mean": -0.02,
            },
        },
        {
            "perturbation_name": "b",
            "reward_delta_from_clean": {
                "mean": -0.05,
            },
        },
    ]

    result = worst_condition(
        rows,
        metric="reward_delta_from_clean",
        mode="min",
    )

    assert result["perturbation_name"] == "b"
