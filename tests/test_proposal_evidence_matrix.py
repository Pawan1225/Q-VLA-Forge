from __future__ import annotations

import csv
from pathlib import Path

from q_vla_forge.evaluation.proposal_evidence_matrix import (
    COMPONENT_ONLY_CELLS,
    DIRECT_FULL_SYSTEM_RUNS,
    PHASE1_SEEDS,
    build_proposal_evidence_matrix,
    proposal_evidence_as_dicts,
    validate_proposal_evidence_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

CLAIM_REGISTRY = (
    ROOT / "results" / "final-validation" / "tables" / "final-claim-registry.csv"
)

SCORECARD = (
    ROOT
    / "results"
    / "final-validation"
    / "tables"
    / "challenge-bottleneck-scorecard.csv"
)


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def test_phase1_constants() -> None:
    assert PHASE1_SEEDS == (
        42,
        123,
        456,
    )

    assert DIRECT_FULL_SYSTEM_RUNS == 0
    assert COMPONENT_ONLY_CELLS == 16


def test_matrix_has_nine_rows() -> None:
    rows = build_proposal_evidence_matrix()

    assert len(rows) == 9


def test_matrix_ids_are_frozen_claim_ids() -> None:
    rows = build_proposal_evidence_matrix()

    assert [row.evidence_id for row in rows] == [
        "S7-C01",
        "S7-C02",
        "S7-C03",
        "S7-C04",
        "S7-C05",
        "S7-C06",
        "S7-C07",
        "S7-C08",
        "S7-C09",
    ]


def test_matrix_validation_passes() -> None:
    rows = build_proposal_evidence_matrix()

    validate_proposal_evidence_matrix(rows)


def test_dict_export_has_all_fields() -> None:
    rows = proposal_evidence_as_dicts()

    expected_fields = {
        "evidence_id",
        "challenge_bottleneck",
        "research_question",
        "method",
        "metric_or_criterion",
        "observed_result",
        "phase1_status",
        "proposal_safe_claim",
        "limitation",
        "evidence_artifact",
        "phase2_follow_up",
    }

    assert len(rows) == 9

    for row in rows:
        assert set(row) == expected_fields


def test_claim_registry_exists() -> None:
    assert CLAIM_REGISTRY.is_file()


def test_scorecard_exists() -> None:
    assert SCORECARD.is_file()


def test_claim_statements_match_frozen_registry() -> None:
    matrix = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    registry = load_csv(CLAIM_REGISTRY)

    assert len(registry) == 9

    for claim in registry:
        evidence_id = claim["claim_id"]

        assert evidence_id in matrix

        assert matrix[evidence_id].proposal_safe_claim == claim["statement"]


def test_claim_limitations_preserve_registry_boundaries() -> None:
    matrix = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    registry = load_csv(CLAIM_REGISTRY)

    for claim in registry:
        limitation = claim["limitation"].strip()

        if not limitation:
            continue

        evidence_id = claim["claim_id"]

        assert limitation in (matrix[evidence_id].limitation)


def test_all_rows_have_provenance() -> None:
    rows = build_proposal_evidence_matrix()

    for row in rows:
        assert row.evidence_artifact
        assert row.evidence_artifact.endswith(".json")


def test_negative_training_result_preserved() -> None:
    rows = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    row = rows["S7-C03"]

    assert row.phase1_status == "not_demonstrated"

    assert "did not demonstrate" in row.proposal_safe_claim


def test_qml_boundary_preserved() -> None:
    rows = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    row = rows["S7-C05"]

    assert "parameter-compact" in row.proposal_safe_claim

    assert "No QML sample-efficiency" in row.limitation

    assert "quantum-speedup" in row.limitation

    assert "quantum-hardware advantage" in row.limitation


def test_safety_boundary_preserved() -> None:
    rows = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    row = rows["S7-C06"]

    assert "zero" in row.proposal_safe_claim

    assert "Empirical pilot evidence only" in row.limitation

    assert "no formal safety" in row.limitation


def test_cross_domain_boundary_preserved() -> None:
    rows = {row.evidence_id: row for row in build_proposal_evidence_matrix()}

    row = rows["S7-C09"]

    assert "framework" in row.proposal_safe_claim

    assert "zero-shot policy transfer" in row.limitation


def test_scorecard_bottlenecks_are_represented() -> None:
    scorecard = load_csv(SCORECARD)

    matrix_bottlenecks = {
        row.challenge_bottleneck for row in build_proposal_evidence_matrix()
    }

    scorecard_bottlenecks = {row["bottleneck"] for row in scorecard}

    assert scorecard_bottlenecks.issubset(matrix_bottlenecks)
