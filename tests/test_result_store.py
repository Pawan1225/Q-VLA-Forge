import json

from q_vla_forge.evaluation.experiment_record import ExperimentRecord
from q_vla_forge.evaluation.result_store import save_experiment_record


def test_save_experiment_record(tmp_path) -> None:
    record = ExperimentRecord(
        experiment_id="exp-test-001",
        domain="autonomous_driving",
        method="baseline",
        seed=42,
        metrics={"accuracy": 0.95},
        parameters={"learning_rate": 0.001},
        hardware={"cuda_available": False},
    )

    file_path = save_experiment_record(
        record=record,
        output_dir=tmp_path,
    )

    assert file_path.exists()
    assert file_path.name == "exp-test-001.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["experiment_id"] == "exp-test-001"
    assert data["domain"] == "autonomous_driving"
    assert data["method"] == "baseline"
    assert data["seed"] == 42
    assert data["metrics"]["accuracy"] == 0.95
    assert data["parameters"]["learning_rate"] == 0.001
    assert data["hardware"]["cuda_available"] is False
    assert "created_at" in data
