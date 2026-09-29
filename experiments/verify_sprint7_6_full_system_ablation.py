"""Independent verification for Sprint 7.6 full-system ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.full_system_ablation import (
    ALLOW_SYNTHETIC_METRIC_COMPOSITION,
    CLAIM_CONTROLS,
    CONFIGURATIONS,
    DOMAINS,
    END_TO_END_METRIC_NAMES,
    FULL_SYSTEM_DIRECT_EXECUTION_FOUND,
    INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY,
    build_phase1_evidence_matrix,
)

OUTPUT_DIR = Path("results/final-validation/full-system-ablation")

JSON_PATH = OUTPUT_DIR / "full-system-ablation.json"

CSV_PATH = OUTPUT_DIR / "full-system-ablation.csv"

MARKDOWN_PATH = OUTPUT_DIR / "full-system-ablation.md"

PHASE2_PLAN_PATH = OUTPUT_DIR / "phase2-factorial-plan.csv"


def _load_json() -> dict[
    str,
    Any,
]:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _check(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_artifact_presence() -> None:
    for path in (
        JSON_PATH,
        CSV_PATH,
        MARKDOWN_PATH,
        PHASE2_PLAN_PATH,
    ):
        _check(
            path.is_file(),
            f"Missing artifact: {path}",
        )


def _verify_factorial_design(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    design = payload["factorial_design"]

    _check(
        design["factors"]
        == [
            "compression",
            "qml",
            "safety",
        ],
        "Unexpected factorial factors.",
    )

    _check(
        len(design["configurations"]) == 8,
        "Expected exactly eight configurations.",
    )

    _check(
        set(design["configurations"]) == set(CONFIGURATIONS),
        "Configuration names do not match contract.",
    )

    _check(
        design["domains"] == list(DOMAINS),
        "Domain set does not match contract.",
    )

    _check(
        design["expected_classifications"] == 16,
        "Expected sixteen classifications.",
    )


def _verify_matrix_shape(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    matrix = payload["phase1"]["matrix"]

    _check(
        len(matrix) == 16,
        "Phase 1 matrix must contain sixteen records.",
    )

    observed = {
        (
            row["domain"],
            row["configuration"],
        )
        for row in matrix
    }

    expected = {
        (
            domain,
            configuration,
        )
        for domain in DOMAINS
        for configuration in CONFIGURATIONS
    }

    _check(
        observed == expected,
        "Phase 1 matrix does not cover all domain/configuration pairs.",
    )


def _verify_evidence_classification(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    phase1 = payload["phase1"]

    matrix = phase1["matrix"]

    _check(
        FULL_SYSTEM_DIRECT_EXECUTION_FOUND is False,
        "Contract unexpectedly reports direct full-system execution.",
    )

    _check(
        phase1["direct_full_system_execution_found"] is False,
        "Artifact unexpectedly reports direct execution.",
    )

    _check(
        phase1["direct_count"] == 0,
        "Direct evidence count must be zero.",
    )

    _check(
        phase1["component_only_count"] == 16,
        "Component-only count must be sixteen.",
    )

    _check(
        phase1["not_evaluated_count"] == 0,
        "Not-evaluated count must be zero.",
    )

    for row in matrix:
        _check(
            row["evidence_status"] == "component_only",
            "Every Phase 1 factorial cell must be COMPONENT_ONLY.",
        )

        _check(
            row["can_report_end_to_end_metrics"] is False,
            "COMPONENT_ONLY evidence cannot report end-to-end metrics.",
        )


def _verify_contract_matrix_consistency(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    artifact_matrix = {
        (
            row["domain"],
            row["configuration"],
        ): row
        for row in payload["phase1"]["matrix"]
    }

    contract_matrix = build_phase1_evidence_matrix()

    for record in contract_matrix:
        row = artifact_matrix[
            (
                record.domain,
                record.name,
            )
        ]

        _check(
            row["compression"] == record.compression,
            "Compression factor mismatch.",
        )

        _check(
            row["qml"] == record.qml,
            "QML factor mismatch.",
        )

        _check(
            row["safety"] == record.safety,
            "Safety factor mismatch.",
        )

        _check(
            row["evidence_status"] == record.evidence_status,
            "Evidence-status mismatch.",
        )


def _verify_no_synthetic_end_to_end_metrics(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    matrix = payload["phase1"]["matrix"]

    forbidden_metric_names = set(END_TO_END_METRIC_NAMES)

    for row in matrix:
        if row["evidence_status"] != "direct":
            present = forbidden_metric_names & set(row)

            _check(
                not present,
                (
                    "Non-direct record contains "
                    f"forbidden end-to-end metrics: {sorted(present)}"
                ),
            )

    controls = payload["scientific_controls"]

    _check(
        controls["synthetic_metric_composition_allowed"] is False,
        "Synthetic metric composition must be disabled.",
    )

    _check(
        controls["full_system_end_to_end_metrics_reported"] is False,
        "Artifact must not report full-system end-to-end metrics.",
    )


def _verify_interaction_controls(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    controls = payload["scientific_controls"]

    _check(
        ALLOW_SYNTHETIC_METRIC_COMPOSITION is False,
        "Contract permits synthetic metric composition.",
    )

    _check(
        INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY is False,
        "Component-only interaction inference must remain disabled.",
    )

    _check(
        controls["interaction_effects_estimable_with_component_only"] is False,
        "Artifact permits component-only interaction estimation.",
    )


def _verify_phase_separation(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    _check(
        payload["phase1"]["status"] == "evaluated_component_evidence",
        "Unexpected Phase 1 status.",
    )

    _check(
        payload["phase2"]["status"] == "candidate_integrated_validation",
        "Unexpected Phase 2 status.",
    )

    _check(
        payload["phase2"]["matched_integrated_factorial_validation_required"] is True,
        "Phase 2 integrated validation must be required.",
    )


def _verify_provenance(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    for row in payload["phase1"]["matrix"]:
        sources = row["source_artifacts"]

        _check(
            isinstance(
                sources,
                list,
            ),
            "Source provenance must be serialized as a list.",
        )

        _check(
            len(sources) >= 2,
            "Component-only records must retain provenance.",
        )

        for source in sources:
            _check(
                bool(source.strip()),
                "Blank source provenance found.",
            )


def _verify_phase2_plan() -> None:
    rows = _load_csv(PHASE2_PLAN_PATH)

    _check(
        len(rows) == 16,
        "Phase 2 factorial plan must contain sixteen rows.",
    )

    observed = {
        (
            row["domain"],
            row["configuration"],
        )
        for row in rows
    }

    expected = {
        (
            domain,
            configuration,
        )
        for domain in DOMAINS
        for configuration in CONFIGURATIONS
    }

    _check(
        observed == expected,
        "Phase 2 plan does not cover all factorial cells.",
    )

    for row in rows:
        _check(
            row["phase1_evidence_status"] == "component_only",
            "Phase 2 plan contains unexpected Phase 1 status.",
        )

        _check(
            row["phase2_run_required"].lower() == "true",
            "Every factorial cell must require a Phase 2 matched run.",
        )

        _check(
            row["priority"]
            in {
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
            },
            "Invalid Phase 2 priority.",
        )

        forbidden = {
            "predicted_performance",
            "expected_reward",
            "expected_mse",
            "expected_latency",
            "expected_success",
            "expected_violation_rate",
            "expected_winner",
        }

        _check(
            not (forbidden & set(row)),
            "Phase 2 plan contains fabricated prediction fields.",
        )


def _verify_csv_consistency(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    csv_rows = _load_csv(CSV_PATH)

    _check(
        len(csv_rows) == 16,
        "Full-system CSV must contain sixteen rows.",
    )

    json_pairs = {
        (
            row["domain"],
            row["configuration"],
        )
        for row in payload["phase1"]["matrix"]
    }

    csv_pairs = {
        (
            row["domain"],
            row["configuration"],
        )
        for row in csv_rows
    }

    _check(
        csv_pairs == json_pairs,
        "JSON and CSV matrix coverage differs.",
    )

    for row in csv_rows:
        _check(
            row["evidence_status"] == "component_only",
            "CSV contains non-component-only evidence.",
        )

        _check(
            row["can_report_end_to_end_metrics"].lower() == "false",
            "CSV incorrectly permits end-to-end metrics.",
        )


def _verify_markdown() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")

    required_fragments = (
        "Phase 1 Component Evidence",
        "Full-System Evidence Matrix",
        "COMPONENT_ONLY",
        "DIRECT: 0",
        "COMPONENT_ONLY: 16",
        "NOT_EVALUATED: 0",
        "No exact Compression x QML x Safety factorial configuration",
        "Interaction Effects",
        "Phase 2 Integrated Factorial Plan",
        "no predicted performance",
        "Scientific Conclusion",
        "Quantum advantage claim: blocked",
        "Full-system superiority claim: blocked",
        "Production-readiness claim: blocked",
    )

    for fragment in required_fragments:
        _check(
            fragment in text,
            ("Markdown missing required " f"fragment: {fragment!r}"),
        )


def _verify_claim_controls(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    controls = payload["claim_controls"]

    _check(
        controls == CLAIM_CONTROLS,
        "Artifact claim controls differ from contract.",
    )

    _check(
        all(value is False for value in controls.values()),
        "All Sprint 7.6 claim controls must remain blocked.",
    )


def verify() -> None:
    """Run independent Sprint 7.6 verification."""

    _verify_artifact_presence()

    payload = _load_json()

    _verify_factorial_design(payload)
    _verify_matrix_shape(payload)
    _verify_evidence_classification(payload)
    _verify_contract_matrix_consistency(payload)
    _verify_no_synthetic_end_to_end_metrics(payload)
    _verify_interaction_controls(payload)
    _verify_phase_separation(payload)
    _verify_provenance(payload)
    _verify_phase2_plan()
    _verify_csv_consistency(payload)
    _verify_markdown()
    _verify_claim_controls(payload)

    print("=" * 64)
    print(" SPRINT 7.6 FULL-SYSTEM ABLATION VERIFICATION")
    print("=" * 64)
    print()
    print("Factorial design                  PASS")
    print("8 configurations                  PASS")
    print("2 domains                         PASS")
    print("16 evidence classifications       PASS")
    print("Direct/component distinction      PASS")
    print("No synthetic metric composition   PASS")
    print("No fabricated full-system metrics PASS")
    print("Interaction controls              PASS")
    print("Phase 1/Phase 2 separation        PASS")
    print("Source provenance                 PASS")
    print("Phase 2 factorial plan            PASS")
    print("JSON/CSV consistency              PASS")
    print("Reviewer Markdown                 PASS")
    print("Claim controls                    PASS")
    print()
    print("SPRINT 7.6 INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    verify()
