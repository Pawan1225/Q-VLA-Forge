from __future__ import annotations

import json
from pathlib import Path

from dashboard.data_loader import (
    experiments_dataframe,
    flatten_experiment_record,
    load_result_files,
    load_sprint4_rl_evidence,
)


def test_load_result_files(tmp_path: Path) -> None:
    record = {
        "experiment_id": "test-001",
        "domain": "autonomous_driving",
        "method": "baseline",
        "seed": 42,
        "metrics": {
            "score": 1.0,
        },
    }

    result_file = tmp_path / "test-001.json"
    result_file.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    records = load_result_files(tmp_path)

    assert len(records) == 1
    assert records[0]["experiment_id"] == "test-001"


def test_missing_results_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    records = load_result_files(missing_dir)

    assert records == []


def test_invalid_json_is_ignored(tmp_path: Path) -> None:
    invalid_file = tmp_path / "broken.json"
    invalid_file.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    records = load_result_files(tmp_path)

    assert records == []


def test_flatten_experiment_record() -> None:
    record = {
        "experiment_id": "test-002",
        "domain": "robotics",
        "method": "ppo",
        "seed": 123,
        "metrics": {
            "reward": 25.0,
        },
        "hardware": {
            "python_version": "3.13.11",
            "torch_version": "2.14.0+cpu",
            "cuda_available": False,
        },
    }

    flattened = flatten_experiment_record(record)

    assert flattened["experiment_id"] == "test-002"
    assert flattened["domain"] == "robotics"
    assert flattened["metric_reward"] == 25.0
    assert flattened["cuda_available"] is False


def test_experiments_dataframe() -> None:
    records = [
        {
            "experiment_id": "test-003",
            "domain": "autonomous_driving",
            "method": "baseline",
            "seed": 42,
            "metrics": {
                "accuracy": 0.9,
            },
        }
    ]

    dataframe = experiments_dataframe(records)

    assert len(dataframe) == 1
    assert "experiment_id" in dataframe.columns
    assert "metric_accuracy" in dataframe.columns


def test_empty_dataframe() -> None:
    dataframe = experiments_dataframe([])

    assert dataframe.empty


def test_load_sprint4_rl_evidence(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "sprint4-rl-evidence.json"

    payload = {
        "sprint": "4.14",
        "target_reach": {
            "full_ppo": {
                "reached": 6,
                "total": 6,
            },
            "matched_classical": {
                "reached": 1,
                "total": 6,
            },
            "hybrid_qml": {
                "reached": 0,
                "total": 6,
            },
        },
        "domains": {
            "autonomous_driving": {
                "methods": {
                    "full_ppo": {},
                    "matched_classical": {},
                    "hybrid_qml": {},
                },
                "compactness": {
                    "parameter_reduction_percent": 95.9,
                },
                "matched_budget_representation": {
                    "direction": "hybrid_qml",
                },
            },
            "robotics": {
                "methods": {
                    "full_ppo": {},
                    "matched_classical": {},
                    "hybrid_qml": {},
                },
                "compactness": {
                    "parameter_reduction_percent": 95.5,
                },
                "matched_budget_representation": {
                    "direction": "matched_classical",
                },
            },
        },
        "claims": [],
        "figure_index": [],
    }

    evidence_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    data = load_sprint4_rl_evidence(evidence_path)

    assert data["sprint"] == "4.14"

    assert data["target_reach"]["full_ppo"]["reached"] == 6

    assert data["target_reach"]["matched_classical"]["reached"] == 1

    assert data["target_reach"]["hybrid_qml"]["reached"] == 0

    assert set(data["domains"]) == {
        "autonomous_driving",
        "robotics",
    }

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        methods = data["domains"][domain]["methods"]

        assert set(methods) == {
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        }

        assert "parameter_reduction_percent" in data["domains"][domain]["compactness"]

        assert "direction" in data["domains"][domain]["matched_budget_representation"]


def test_missing_sprint4_rl_evidence(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-sprint4.json"

    data = load_sprint4_rl_evidence(missing)

    assert data == {}


def test_invalid_sprint4_rl_evidence(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "broken-sprint4.json"

    invalid.write_text(
        "{broken-json",
        encoding="utf-8",
    )

    data = load_sprint4_rl_evidence(invalid)

    assert data == {}
