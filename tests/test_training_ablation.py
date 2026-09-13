from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.training_ablation import (
    ParameterBudget,
    relative_parameter_difference_percent,
    select_closest_parameter_pair,
)

SUMMARY = (
    Path("results") / "training" / "ablation" / "classical-vs-qi-ablation-summary.json"
)


def test_relative_parameter_difference() -> None:
    result = relative_parameter_difference_percent(
        80,
        100,
    )

    assert result == pytest.approx(20.0)


def test_relative_parameter_difference_symmetric() -> None:
    left = relative_parameter_difference_percent(
        80,
        100,
    )

    right = relative_parameter_difference_percent(
        100,
        80,
    )

    assert left == right


def test_invalid_parameter_count_rejected() -> None:
    with pytest.raises(ValueError):
        relative_parameter_difference_percent(
            0,
            100,
        )


def test_select_closest_pair() -> None:
    svd = [
        ParameterBudget(
            method="trainable_svd",
            configuration="SVD-25%",
            trainable_parameters=70,
        ),
        ParameterBudget(
            method="trainable_svd",
            configuration="SVD-50%",
            trainable_parameters=80,
        ),
    ]

    tt = [
        ParameterBudget(
            method="trainable_tt_mps",
            configuration="TT-rank-2",
            trainable_parameters=79,
        ),
        ParameterBudget(
            method="trainable_tt_mps",
            configuration="TT-rank-4",
            trainable_parameters=95,
        ),
    ]

    pair = select_closest_parameter_pair(
        svd,
        tt,
    )

    assert pair.svd.configuration == "SVD-50%"

    assert pair.tt_mps.configuration == "TT-rank-2"

    assert pair.absolute_parameter_difference == 1


def test_pair_selection_does_not_need_performance_metrics() -> None:
    svd = [
        ParameterBudget(
            method="trainable_svd",
            configuration="SVD-X",
            trainable_parameters=100,
        )
    ]

    tt = [
        ParameterBudget(
            method="trainable_tt_mps",
            configuration="TT-X",
            trainable_parameters=100,
        )
    ]

    pair = select_closest_parameter_pair(
        svd,
        tt,
    )

    assert pair.relative_difference_percent == 0.0


def test_generated_ablation_has_two_domains() -> None:
    if not SUMMARY.exists():
        pytest.skip("training ablation not generated")

    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert len(payload["domains"]) == 2


def test_generated_ablation_uses_seed_42() -> None:
    if not SUMMARY.exists():
        pytest.skip("training ablation not generated")

    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert payload["seed"] == 42


def test_generated_ablation_does_not_claim_quantum_advantage() -> None:
    if not SUMMARY.exists():
        pytest.skip("training ablation not generated")

    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert payload["claim_control"]["quantum_advantage"] is False


def test_generated_ablation_uses_same_layers() -> None:
    if not SUMMARY.exists():
        pytest.skip("training ablation not generated")

    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    for domain in payload["domains"]:
        control = domain["scientific_control"]

        assert control["same_selected_layers"] is True

        assert control["approximately_matched_parameters"] is True


def test_parameter_pair_selection_uses_no_performance_data() -> None:
    if not SUMMARY.exists():
        pytest.skip("training ablation not generated")

    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    for domain in payload["domains"]:
        assert domain["selection_uses_performance_data"] is False
