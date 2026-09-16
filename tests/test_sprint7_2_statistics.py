from pathlib import Path

import pytest

from q_vla_forge.evaluation.final_statistics_aggregation import (
    aggregate_baseline,
    build_final_statistics,
)

ROOT = Path(__file__).resolve().parents[1]


def find_record(
    payload: dict,
    *,
    domain: str,
    experiment: str,
    metric: str,
) -> dict:
    matches = [
        record
        for record in payload["records"]
        if record["domain"] == domain
        and record["experiment"] == experiment
        and record["metric"] == metric
    ]

    assert matches

    return matches[0]


def test_baseline_statistics_reproduce_driving() -> None:
    records, fixed = aggregate_baseline(ROOT)

    assert records
    assert fixed

    payload = {
        "records": records,
    }

    mse = find_record(
        payload,
        domain="autonomous_driving",
        experiment="baseline",
        metric="test_mse",
    )

    mae = find_record(
        payload,
        domain="autonomous_driving",
        experiment="baseline",
        metric="test_mae",
    )

    latency = find_record(
        payload,
        domain="autonomous_driving",
        experiment="baseline",
        metric="mean_latency_ms",
    )

    assert mse["mean"] == pytest.approx(0.012067932014664015)

    assert mse["sample_std"] == pytest.approx(0.0015086088391748967)

    assert mae["mean"] == pytest.approx(0.06861458470424016)

    assert latency["mean"] == pytest.approx(1.736594332808939)


def test_baseline_statistics_reproduce_robotics() -> None:
    records, _ = aggregate_baseline(ROOT)

    payload = {
        "records": records,
    }

    mse = find_record(
        payload,
        domain="robotics",
        experiment="baseline",
        metric="test_mse",
    )

    latency = find_record(
        payload,
        domain="robotics",
        experiment="baseline",
        metric="mean_latency_ms",
    )

    assert mse["mean"] == pytest.approx(0.008104669706275066)

    assert mse["sample_std"] == pytest.approx(0.002484056535872967)

    assert latency["mean"] == pytest.approx(1.9888236651119466)


def test_final_statistics_use_locked_protocol() -> None:
    payload = build_final_statistics(ROOT)

    definition = payload["statistics_definition"]

    assert definition["locked_seeds"] == [
        42,
        123,
        456,
    ]

    assert definition["ddof"] == 1

    assert definition["spread"] == ("sample_standard_deviation")

    assert not payload["new_training"]

    assert not payload["new_experiments"]


def test_deterministic_values_are_separate() -> None:
    payload = build_final_statistics(ROOT)

    parameter_records = [
        item for item in payload["fixed_values"] if item["quantity"] == "parameters"
    ]

    assert parameter_records

    assert all(item["reporting_mode"] == "fixed_value" for item in parameter_records)

    assert not any(record["metric"] == "parameters" for record in payload["records"])


def test_statistics_cover_all_major_areas() -> None:
    payload = build_final_statistics(ROOT)

    experiments = {record["experiment"] for record in payload["records"]}

    assert "baseline" in experiments
    assert "compression" in experiments
    assert "training_efficiency" in experiments
    assert "rl_qml" in experiments
    assert "safety_clean" in experiments

    assert any(name.startswith("robustness_") for name in experiments)
