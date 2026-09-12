from __future__ import annotations

import pytest

from q_vla_forge.evaluation.pareto import (
    ParetoPoint,
    build_domain_pareto_analysis,
    dominates,
    pareto_frontier,
)


def _point(
    name: str,
    *,
    ratio: float,
    mse: float,
    reference: bool = False,
) -> ParetoPoint:
    return ParetoPoint(
        method=name,
        configuration=name,
        family="test",
        compression_ratio_mean=ratio,
        compression_ratio_std=0.0,
        mse_change_mean=mse,
        mse_change_std=0.0,
        mae_change_mean=0.0,
        mae_change_std=0.0,
        pilot_feasible_runs=3,
        all_runs_pilot_feasible=True,
        is_reference=reference,
    )


def test_clear_dominance() -> None:
    better = _point(
        "better",
        ratio=3.0,
        mse=2.0,
    )

    worse = _point(
        "worse",
        ratio=2.0,
        mse=4.0,
    )

    assert dominates(
        better,
        worse,
    )


def test_tradeoff_does_not_dominate() -> None:
    high_compression = _point(
        "compression",
        ratio=4.0,
        mse=5.0,
    )

    low_error = _point(
        "error",
        ratio=2.0,
        mse=1.0,
    )

    assert not dominates(
        high_compression,
        low_error,
    )

    assert not dominates(
        low_error,
        high_compression,
    )


def test_equal_point_does_not_dominate() -> None:
    first = _point(
        "first",
        ratio=2.0,
        mse=2.0,
    )

    second = _point(
        "second",
        ratio=2.0,
        mse=2.0,
    )

    assert not dominates(
        first,
        second,
    )


def test_negative_mse_is_supported() -> None:
    better = _point(
        "better",
        ratio=2.0,
        mse=-1.0,
    )

    worse = _point(
        "worse",
        ratio=2.0,
        mse=1.0,
    )

    assert dominates(
        better,
        worse,
    )


def test_frontier_removes_dominated_point() -> None:
    points = [
        _point(
            "a",
            ratio=2.0,
            mse=4.0,
        ),
        _point(
            "b",
            ratio=3.0,
            mse=2.0,
        ),
        _point(
            "c",
            ratio=4.0,
            mse=5.0,
        ),
    ]

    frontier = pareto_frontier(points)

    names = {point.configuration for point in frontier}

    assert "a" not in names

    assert "b" in names
    assert "c" in names


def test_reference_excluded_by_default() -> None:
    points = [
        _point(
            "FP32",
            ratio=1.0,
            mse=0.0,
            reference=True,
        ),
        _point(
            "INT8",
            ratio=4.0,
            mse=2.0,
        ),
    ]

    frontier = pareto_frontier(points)

    assert len(frontier) == 1

    assert frontier[0].configuration == "INT8"


def test_reference_can_be_included() -> None:
    points = [
        _point(
            "FP32",
            ratio=1.0,
            mse=0.0,
            reference=True,
        ),
        _point(
            "INT8",
            ratio=4.0,
            mse=2.0,
        ),
    ]

    frontier = pareto_frontier(
        points,
        include_reference=True,
    )

    assert len(frontier) == 2


def test_build_domain_analysis() -> None:
    points = [
        _point(
            "FP32",
            ratio=1.0,
            mse=0.0,
            reference=True,
        ),
        _point(
            "INT8",
            ratio=4.0,
            mse=2.0,
        ),
        _point(
            "SVD",
            ratio=2.0,
            mse=4.0,
        ),
    ]

    result = build_domain_pareto_analysis(
        domain="autonomous_driving",
        points=points,
    )

    assert result.domain == "autonomous_driving"

    assert "INT8" in result.pareto_frontier

    assert "SVD" in result.dominated


def test_empty_points_rejected() -> None:
    with pytest.raises(ValueError):
        pareto_frontier([])


def test_invalid_compression_ratio_rejected() -> None:
    point = _point(
        "bad",
        ratio=0.0,
        mse=1.0,
    )

    with pytest.raises(ValueError):
        pareto_frontier([point])


def test_frontier_sorted_by_compression_ratio() -> None:
    points = [
        _point(
            "high",
            ratio=4.0,
            mse=5.0,
        ),
        _point(
            "low",
            ratio=2.0,
            mse=1.0,
        ),
    ]

    frontier = pareto_frontier(points)

    assert [point.configuration for point in frontier] == [
        "low",
        "high",
    ]
