from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULT = (
    Path("results")
    / "training"
    / "robotics"
    / "robotics-training-efficiency-seed-42.json"
)


@pytest.fixture
def payload() -> dict:
    if not RESULT.exists():
        pytest.skip(
            "robotics training-efficiency " "artifact has not been generated yet"
        )

    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_domain_and_seed(
    payload: dict,
) -> None:
    assert payload["domain"] == "robotics"

    assert payload["seed"] == 42


def test_experiment_is_exploratory(
    payload: dict,
) -> None:
    assert payload["exploratory"] is True


def test_svd_configuration_is_frozen(
    payload: dict,
) -> None:
    assert payload["configurations"]["svd"]["rank_fraction"] == 0.50


def test_tt_configuration_is_frozen(
    payload: dict,
) -> None:
    assert payload["configurations"]["tt_mps"]["max_rank"] == 2


def test_tt_is_quantum_inspired(
    payload: dict,
) -> None:
    config = payload["configurations"]["tt_mps"]

    assert config["quantum_inspired"] is True

    assert config["quantum_hardware_used"] is False


def test_structured_models_reduce_parameters(
    payload: dict,
) -> None:
    baseline = payload["fp32_reference"]["trainable_parameters"]

    svd = payload["svd"]["summary"]["trainable_parameters"]

    tt = payload["tt_mps"]["summary"]["trainable_parameters"]

    assert svd < baseline
    assert tt < baseline


def test_same_fp32_target_used(
    payload: dict,
) -> None:
    target = payload["target"]["target_validation_loss"]

    assert payload["svd"]["summary"]["target"]["target_validation_loss"] == target

    assert payload["tt_mps"]["summary"]["target"]["target_validation_loss"] == target


def test_histories_have_twenty_epochs(
    payload: dict,
) -> None:
    assert len(payload["svd"]["summary"]["history"]) == 20

    assert len(payload["tt_mps"]["summary"]["history"]) == 20


def test_final_step_count(
    payload: dict,
) -> None:
    assert payload["svd"]["summary"]["history"][-1]["optimizer_steps"] == 320

    assert payload["tt_mps"]["summary"]["history"][-1]["optimizer_steps"] == 320


def test_final_sample_count(
    payload: dict,
) -> None:
    assert payload["svd"]["summary"]["history"][-1]["samples_processed"] == 10240

    assert payload["tt_mps"]["summary"]["history"][-1]["samples_processed"] == 10240
