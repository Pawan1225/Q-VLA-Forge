import math

import pytest

from q_vla_forge.evaluation.final_statistics import (
    LOCKED_SEEDS,
    SeedMetric,
    summarize_metric,
)


def test_locked_seeds() -> None:
    assert LOCKED_SEEDS == (
        42,
        123,
        456,
    )


def test_mean_and_sample_std() -> None:
    result = summarize_metric(
        "example",
        (
            SeedMetric(
                42,
                1.0,
            ),
            SeedMetric(
                123,
                2.0,
            ),
            SeedMetric(
                456,
                3.0,
            ),
        ),
    )

    assert result.n == 3
    assert result.mean == pytest.approx(2.0)

    # Sample SD of [1, 2, 3] = 1.
    assert result.sample_std == pytest.approx(1.0)


def test_zero_variance() -> None:
    result = summarize_metric(
        "example",
        (
            SeedMetric(
                42,
                5.0,
            ),
            SeedMetric(
                123,
                5.0,
            ),
            SeedMetric(
                456,
                5.0,
            ),
        ),
    )

    assert result.mean == pytest.approx(5.0)

    assert result.sample_std == pytest.approx(0.0)


def test_order_does_not_change_result() -> None:
    result = summarize_metric(
        "example",
        (
            SeedMetric(
                456,
                3.0,
            ),
            SeedMetric(
                42,
                1.0,
            ),
            SeedMetric(
                123,
                2.0,
            ),
        ),
    )

    assert [item.seed for item in result.values] == [
        42,
        123,
        456,
    ]


def test_missing_seed_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_metric(
            "example",
            (
                SeedMetric(
                    42,
                    1.0,
                ),
                SeedMetric(
                    123,
                    2.0,
                ),
            ),
        )


def test_duplicate_seed_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_metric(
            "example",
            (
                SeedMetric(
                    42,
                    1.0,
                ),
                SeedMetric(
                    42,
                    2.0,
                ),
                SeedMetric(
                    456,
                    3.0,
                ),
            ),
        )


def test_non_finite_value_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_metric(
            "example",
            (
                SeedMetric(
                    42,
                    1.0,
                ),
                SeedMetric(
                    123,
                    math.nan,
                ),
                SeedMetric(
                    456,
                    3.0,
                ),
            ),
        )


def test_formatted_statistics() -> None:
    result = summarize_metric(
        "example",
        (
            SeedMetric(
                42,
                1.0,
            ),
            SeedMetric(
                123,
                2.0,
            ),
            SeedMetric(
                456,
                3.0,
            ),
        ),
    )

    assert result.formatted == "2.000000 ± 1.000000"


def test_to_dict_contains_formatted_value() -> None:
    result = summarize_metric(
        "example",
        (
            SeedMetric(
                42,
                1.0,
            ),
            SeedMetric(
                123,
                2.0,
            ),
            SeedMetric(
                456,
                3.0,
            ),
        ),
    )

    payload = result.to_dict()

    assert payload["metric"] == "example"

    assert payload["n"] == 3

    assert payload["formatted"] == "2.000000 ± 1.000000"
