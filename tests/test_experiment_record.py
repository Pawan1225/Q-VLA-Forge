from q_vla_forge.evaluation.experiment_record import ExperimentRecord


def test_experiment_record_defaults() -> None:
    record = ExperimentRecord(
        experiment_id="exp-001",
        domain="autonomous_driving",
        method="baseline",
        seed=42,
    )

    assert record.experiment_id == "exp-001"
    assert record.domain == "autonomous_driving"
    assert record.method == "baseline"
    assert record.seed == 42
    assert record.metrics == {}
    assert record.parameters == {}
    assert record.hardware == {}
    assert record.notes is None
    assert record.created_at


def test_experiment_record_to_dict() -> None:
    record = ExperimentRecord(
        experiment_id="exp-002",
        domain="robotics",
        method="pqc_policy",
        seed=123,
        metrics={"reward": 12.5},
        parameters={"layers": 2},
        hardware={"cuda_available": False},
        notes="test run",
    )

    data = record.to_dict()

    assert data["experiment_id"] == "exp-002"
    assert data["domain"] == "robotics"
    assert data["method"] == "pqc_policy"
    assert data["seed"] == 123
    assert data["metrics"]["reward"] == 12.5
    assert data["parameters"]["layers"] == 2
    assert data["hardware"]["cuda_available"] is False
    assert data["notes"] == "test run"
    assert "created_at" in data
