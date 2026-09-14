"""Tests for Sprint 5.13C Gaussian consolidation."""

import pytest

from q_vla_forge.evaluation.gaussian_safety_consolidation import (
    GaussianSeedResult,
    summarize_gaussian_rows,
    summarize_three_seeds,
    worst_condition,
)


def _row(
    seed: int,
    violation: float,
) -> GaussianSeedResult:
    return GaussianSeedResult(
        domain="robotics",
        method="none",
        sigma=0.1,
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


def test_three_seed_summary() -> None:
    result = summarize_three_seeds(
        {
            42: 1.0,
            123: 2.0,
            456: 3.0,
        }
    )

    assert result.mean == pytest.approx(2.0)

    assert result.sample_sd == pytest.approx(1.0)


def test_gaussian_row_summary() -> None:
    result = summarize_gaussian_rows(
        [
            _row(
                42,
                0.03,
            ),
            _row(
                123,
                0.01,
            ),
            _row(
                456,
                0.02,
            ),
        ]
    )

    assert result["violation_step_rate"].mean == pytest.approx(0.02)


def test_missing_seed_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="three principal seeds",
    ):
        summarize_gaussian_rows(
            [
                _row(
                    42,
                    0.1,
                ),
                _row(
                    123,
                    0.2,
                ),
            ]
        )


def test_worst_condition_max() -> None:
    rows = [
        {
            "sigma": 0.01,
            "violation_delta_from_clean": {"mean": 0.1},
        },
        {
            "sigma": 0.10,
            "violation_delta_from_clean": {"mean": 0.4},
        },
    ]

    result = worst_condition(
        rows,
        metric="violation_delta_from_clean",
        mode="max",
    )

    assert result["sigma"] == pytest.approx(0.10)


def test_worst_condition_min() -> None:
    rows = [
        {
            "sigma": 0.01,
            "reward_delta_from_clean": {"mean": -0.1},
        },
        {
            "sigma": 0.10,
            "reward_delta_from_clean": {"mean": -0.5},
        },
    ]

    result = worst_condition(
        rows,
        metric="reward_delta_from_clean",
        mode="min",
    )

    assert result["sigma"] == pytest.approx(0.10)
