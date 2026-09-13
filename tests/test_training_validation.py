from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from q_vla_forge.evaluation.training_validation import (
    MeanStd,
    build_reach_summary,
    mean_std,
    optional_mean_std,
    robust_ten_percent_efficiency,
)

SUMMARY_PATH = Path("results/training/training-validation-summary.json")


def test_mean_std_three_values() -> None:
    result = mean_std(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert result.mean == 20.0
    assert result.n == 3

    assert result.std == pytest.approx(10.0)


def test_mean_std_single_value() -> None:
    result = mean_std([12.0])

    assert result.mean == 12.0
    assert result.std == 0.0
    assert result.n == 1


def test_empty_values_rejected() -> None:
    with pytest.raises(ValueError):
        mean_std([])


def test_non_finite_values_rejected() -> None:
    with pytest.raises(ValueError):
        mean_std(
            [
                1.0,
                math.inf,
            ]
        )


def test_optional_mean_std_ignores_none() -> None:
    result = optional_mean_std(
        [
            10.0,
            None,
            20.0,
        ]
    )

    assert result is not None

    assert result.mean == 15.0
    assert result.n == 2


def test_optional_mean_std_all_none() -> None:
    assert (
        optional_mean_std(
            [
                None,
                None,
            ]
        )
        is None
    )


def test_reach_summary_all_reached() -> None:
    result = build_reach_summary(
        [
            True,
            True,
            True,
        ]
    )

    assert result.reached == 3
    assert result.total == 3
    assert result.rate == 1.0
    assert result.rate_percent == 100.0
    assert result.all_reached


def test_reach_summary_partial() -> None:
    result = build_reach_summary(
        [
            True,
            False,
            True,
        ]
    )

    assert result.reached == 2
    assert result.total == 3
    assert result.rate_percent == pytest.approx(66.6666666667)

    assert not result.all_reached


def test_robust_efficiency_passes() -> None:
    reach = build_reach_summary(
        [
            True,
            True,
            True,
        ]
    )

    reduction = MeanStd(
        mean=12.0,
        std=2.0,
        n=3,
    )

    assert robust_ten_percent_efficiency(
        reach_summary=reach,
        step_reduction=reduction,
    )


def test_robust_efficiency_fails_if_one_seed_misses() -> None:
    reach = build_reach_summary(
        [
            True,
            False,
            True,
        ]
    )

    reduction = MeanStd(
        mean=20.0,
        std=2.0,
        n=2,
    )

    assert not robust_ten_percent_efficiency(
        reach_summary=reach,
        step_reduction=reduction,
    )


def test_robust_efficiency_fails_below_threshold() -> None:
    reach = build_reach_summary(
        [
            True,
            True,
            True,
        ]
    )

    reduction = MeanStd(
        mean=9.99,
        std=1.0,
        n=3,
    )

    assert not robust_ten_percent_efficiency(
        reach_summary=reach,
        step_reduction=reduction,
    )


def test_generated_validation_summary_shape() -> None:
    """Generated three-seed summary should have the expected structure."""
    if not SUMMARY_PATH.exists():
        return

    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]

    assert len(payload["methods"]) == 4


def test_generated_validation_summary_has_three_seed_reach_totals() -> None:
    """Every structured method summary should cover all three seeds."""
    if not SUMMARY_PATH.exists():
        return

    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    for method in payload["methods"]:
        assert method["target_reach"]["total"] == 3
