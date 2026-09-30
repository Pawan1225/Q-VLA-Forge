from __future__ import annotations

import csv
import json
from pathlib import Path

from q_vla_forge.evaluation.phase_separation import (
    COMPONENT_ONLY_CELLS,
    DIRECT_FULL_SYSTEM_RUNS,
    PHASE1_STATUS,
    PHASE2_STATUS,
    PRINCIPAL_SEEDS,
    build_phase_separation_matrix,
    phase_separation_as_dicts,
    validate_phase_separation_matrix,
)

ROOT = Path(".")

FULL_SYSTEM_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "full-system-ablation"
    / "full-system-ablation.json"
)

PROPOSAL_EVIDENCE_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "proposal-evidence"
    / "proposal-evidence-matrix.json"
)

CROSS_DOMAIN_PATH = (
    ROOT / "results" / "final-validation" / "tables" / "cross-domain-table.csv"
)


def test_phase_constants() -> None:
    assert PHASE1_STATUS == "evaluated_component_evidence"
    assert PHASE2_STATUS == "candidate_integrated_validation"
    assert DIRECT_FULL_SYSTEM_RUNS == 0
    assert COMPONENT_ONLY_CELLS == 16
    assert PRINCIPAL_SEEDS == (42, 123, 456)


def test_phase_separation_has_eight_rows() -> None:
    rows = build_phase_separation_matrix()

    assert len(rows) == 8


def test_phase_separation_areas_are_frozen() -> None:
    rows = build_phase_separation_matrix()

    assert [row.area for row in rows] == [
        "shared_architecture",
        "compression",
        "training_efficiency",
        "rl",
        "qml",
        "safety",
        "robustness",
        "full_system",
    ]


def test_phase_separation_validation_passes() -> None:
    rows = build_phase_separation_matrix()

    validate_phase_separation_matrix(rows)


def test_phase_separation_dict_projection() -> None:
    rows = phase_separation_as_dicts()

    assert len(rows) == 8

    for row in rows:
        assert row["area"]
        assert row["phase1_executed"]
        assert row["phase1_evidence"]
        assert row["phase1_boundary"]
        assert row["phase2_objective"]
        assert row["phase2_status"] == PHASE2_STATUS
        assert row["evidence_artifact"]


def test_full_system_artifact_exists() -> None:
    assert FULL_SYSTEM_PATH.is_file()


def test_proposal_evidence_artifact_exists() -> None:
    assert PROPOSAL_EVIDENCE_PATH.is_file()


def test_cross_domain_artifact_exists() -> None:
    assert CROSS_DOMAIN_PATH.is_file()


def test_full_system_counts_match_frozen_evidence() -> None:
    payload = json.loads(FULL_SYSTEM_PATH.read_text(encoding="utf-8"))

    assert payload["phase1"]["direct_count"] == DIRECT_FULL_SYSTEM_RUNS
    assert payload["phase1"]["component_only_count"] == COMPONENT_ONLY_CELLS
    assert payload["phase1"]["status"] == PHASE1_STATUS
    assert payload["phase2"]["status"] == PHASE2_STATUS


def test_full_system_remains_component_only() -> None:
    payload = json.loads(FULL_SYSTEM_PATH.read_text(encoding="utf-8"))

    matrix = payload["phase1"]["matrix"]

    assert len(matrix) == 16

    assert all(row["evidence_status"] == "component_only" for row in matrix)

    assert all(row["can_report_end_to_end_metrics"] is False for row in matrix)


def test_full_system_scientific_controls_remain_blocked() -> None:
    payload = json.loads(FULL_SYSTEM_PATH.read_text(encoding="utf-8"))

    controls = payload["scientific_controls"]

    assert controls["full_system_end_to_end_metrics_reported"] is False

    assert controls["interaction_effects_estimable_with_component_only"] is False

    assert controls["predicted_phase2_performance_reported"] is False

    assert controls["synthetic_metric_composition_allowed"] is False


def test_phase2_integrated_factorial_remains_future_work() -> None:
    payload = json.loads(FULL_SYSTEM_PATH.read_text(encoding="utf-8"))

    assert payload["phase2"]["matched_integrated_factorial_validation_required"] is True


def test_training_negative_result_preserved() -> None:
    rows = {row.area: row for row in build_phase_separation_matrix()}

    training = rows["training_efficiency"]

    assert "did not demonstrate" in training.phase1_boundary
    assert "10%" in training.phase1_boundary


def test_qml_boundary_preserved() -> None:
    rows = {row.area: row for row in build_phase_separation_matrix()}

    qml = rows["qml"]

    assert "0 of 6" in qml.phase1_boundary
    assert "No QML sample-efficiency" in qml.phase1_boundary
    assert "quantum-speedup" in qml.phase1_boundary
    assert "quantum-hardware advantage" in qml.phase1_boundary


def test_safety_boundary_preserved() -> None:
    rows = {row.area: row for row in build_phase_separation_matrix()}

    safety = rows["safety"]

    assert "empirical pilot evidence only" in safety.phase1_boundary
    assert "formal safety" in safety.phase1_boundary
    assert "certification" in safety.phase1_boundary
    assert "production readiness" in safety.phase1_boundary


def test_cross_domain_boundary_preserved() -> None:
    rows = {row.area: row for row in build_phase_separation_matrix()}

    shared = rows["shared_architecture"]

    assert "No universal trained policy" in shared.phase1_boundary
    assert "zero-shot transfer" in shared.phase1_boundary
    assert "universal safety controller" in shared.phase1_boundary


def test_cross_domain_table_agrees_with_boundary() -> None:
    with CROSS_DOMAIN_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    architecture = next(
        row for row in rows if row["Component / Method"] == "Shared AI/DL architecture"
    )

    assert "same trained policy weights: NO" in architecture["Shared Across Domains?"]


def test_proposal_evidence_phase1_flags_are_frozen() -> None:
    payload = json.loads(PROPOSAL_EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["phase"] == "Phase 1"
    assert payload["evidence_frozen"] is True
    assert payload["new_experiments"] is False
    assert payload["new_training"] is False
    assert payload["new_scientific_results"] is False
    assert payload["direct_full_system_runs"] == 0
    assert payload["component_only_cells"] == 16


def test_full_system_boundary_text_is_explicit() -> None:
    rows = {row.area: row for row in build_phase_separation_matrix()}

    full_system = rows["full_system"]

    assert "DIRECT = 0" in full_system.phase1_boundary
    assert "COMPONENT_ONLY = 16" in full_system.phase1_boundary
    assert "No synthetic full-system metric" in full_system.phase1_boundary
    assert "interaction effect" in full_system.phase1_boundary
    assert "full-system superiority" in full_system.phase1_boundary


def test_every_phase2_item_is_future_validation() -> None:
    rows = build_phase_separation_matrix()

    assert all(row.phase2_status == PHASE2_STATUS for row in rows)
