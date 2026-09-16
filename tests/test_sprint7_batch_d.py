import json
from pathlib import Path

from q_vla_forge.evaluation.repository_evidence import (
    build_evidence_index,
)


def test_evidence_index_contains_batch_d() -> None:
    result = build_evidence_index(Path("."))

    assert "7.11 Final Claim Registry Freeze" in result

    assert "7.12 Final Figures & Tables" in result

    assert "7.13 README / Repository Evidence Layer" in result


def test_batch_c_acceptance_exists() -> None:
    path = Path("results/final-validation/" "batch-c/batch-c-acceptance.json")

    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"]

    assert not payload["new_training"]

    assert not payload["new_experiments"]
