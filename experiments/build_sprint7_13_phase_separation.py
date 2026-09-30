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
    validate_phase_separation_matrix,
)

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "final-validation" / "phase-separation"

CSV_PATH = OUTPUT_DIR / "phase1-vs-phase2.csv"

JSON_PATH = OUTPUT_DIR / "phase1-vs-phase2.json"

MARKDOWN_PATH = OUTPUT_DIR / "phase1-vs-phase2.md"

MANIFEST_PATH = OUTPUT_DIR / "phase-separation-manifest.json"


FIELDNAMES = (
    "area",
    "phase1_executed",
    "phase1_evidence",
    "phase1_boundary",
    "phase2_objective",
    "phase2_status",
    "evidence_artifact",
)


def write_csv(
    rows: list[dict[str, str]],
) -> None:
    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    rows: list[dict[str, str]],
) -> None:
    payload = {
        "artifact": ("Q-VLA Forge Sprint 7.13 " "Phase 1 vs Phase 2 Separation"),
        "phase1_status": PHASE1_STATUS,
        "phase2_status": PHASE2_STATUS,
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "direct_full_system_runs": (DIRECT_FULL_SYSTEM_RUNS),
        "component_only_cells": (COMPONENT_ONLY_CELLS),
        "row_count": len(rows),
        "new_experiments": False,
        "new_training": False,
        "new_scientific_results": False,
        "phase2_predictions_reported": False,
        "rows": rows,
        "claim_controls": {
            "quantum_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "qml_sample_efficiency_advantage_claimed": False,
            "tt_mps_superiority_claimed": False,
            "formal_safety_guarantee_claimed": False,
            "production_readiness_claimed": False,
            "certification_claimed": False,
            "zero_shot_cross_domain_transfer_claimed": False,
            "full_system_superiority_claimed": False,
            "interaction_effects_claimed": False,
            "synthetic_metric_composition": False,
            "predicted_phase2_performance_claimed": False,
        },
    }

    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def markdown_escape(
    value: str,
) -> str:
    return (
        value.replace(
            "|",
            "\\|",
        )
        .replace(
            "\r",
            " ",
        )
        .replace(
            "\n",
            " ",
        )
    )


def write_markdown(
    rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Q-VLA Forge — Phase 1 vs Phase 2 Separation",
        "",
        (
            "Sprint 7.13 explicitly separates executed and frozen "
            "Phase 1 evidence from future Phase 2 validation objectives."
        ),
        "",
        (
            "No Phase 2 objective in this artifact is reported as an "
            "executed result, demonstrated improvement, or predicted outcome."
        ),
        "",
        "## Frozen Phase 1 State",
        "",
        "- Principal seeds: 42, 123, 456",
        "- Phase 1 status: evaluated component evidence",
        "- Direct integrated full-system runs: 0",
        "- Component-only factorial cells: 16",
        "- New experiments in Sprint 7.13: no",
        "- New training in Sprint 7.13: no",
        "- New scientific results in Sprint 7.13: no",
        "",
        "## Phase Boundary Matrix",
        "",
        (
            "| Area | Phase 1 Executed | Phase 1 Evidence | "
            "Phase 1 Boundary | Phase 2 Objective | "
            "Phase 2 Status | Evidence |"
        ),
        ("|---|---|---|---|---|---|---|"),
    ]

    for row in rows:
        values = [
            markdown_escape(row["area"]),
            markdown_escape(row["phase1_executed"]),
            markdown_escape(row["phase1_evidence"]),
            markdown_escape(row["phase1_boundary"]),
            markdown_escape(row["phase2_objective"]),
            markdown_escape(row["phase2_status"]),
            markdown_escape(row["evidence_artifact"]),
        ]

        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Phase 1 Claim Boundaries",
            "",
            "- No quantum advantage claim.",
            "- No quantum speedup claim.",
            "- No QML sample-efficiency superiority claim.",
            "- No TT/MPS superiority claim.",
            "- No formal safety or Lyapunov stability guarantee.",
            "- No production-readiness claim.",
            "- No certification claim.",
            "- No zero-shot cross-domain transfer claim.",
            "- No full-system superiority claim.",
            "- No interaction-effect claim from component-only evidence.",
            "- No synthetic end-to-end metric composition.",
            "- No predicted Phase 2 performance claim.",
            "",
            "## Phase 2 Integrated Validation",
            "",
            (
                "The primary carry-over is direct matched execution "
                "of the 16 Compression × QML × Safety "
                "domain/configuration cells."
            ),
            "",
            (
                "Those runs are required before main effects, "
                "interaction effects, or integrated full-system "
                "performance can be reported."
            ),
            "",
            (
                "Phase 2 does not assume that quantum or "
                "quantum-inspired methods will outperform "
                "matched classical baselines."
            ),
            "",
        ]
    )

    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_manifest(
    rows: list[dict[str, str]],
) -> None:
    manifest = {
        "sprint": "7.13",
        "artifact_group": ("phase1_vs_phase2_separation"),
        "status": "frozen",
        "row_count": len(rows),
        "areas": [row["area"] for row in rows],
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "phase1_status": PHASE1_STATUS,
        "phase2_status": PHASE2_STATUS,
        "direct_full_system_runs": (DIRECT_FULL_SYSTEM_RUNS),
        "component_only_cells": (COMPONENT_ONLY_CELLS),
        "generated_from_frozen_evidence": True,
        "new_experiments": False,
        "new_training": False,
        "new_scientific_results": False,
        "phase2_predictions_reported": False,
        "outputs": [
            str(CSV_PATH),
            str(JSON_PATH),
            str(MARKDOWN_PATH),
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = build_phase_separation_matrix()

    validate_phase_separation_matrix(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionaries = [row.to_dict() for row in rows]

    write_csv(dictionaries)

    write_json(dictionaries)

    write_markdown(dictionaries)

    write_manifest(dictionaries)

    print("=" * 68)

    print(" SPRINT 7.13 — PHASE 1 VS PHASE 2 SEPARATION")

    print("=" * 68)

    print()

    print(f"Rows: {len(dictionaries)}")

    print(f"Phase 1 status: {PHASE1_STATUS}")

    print(f"Phase 2 status: {PHASE2_STATUS}")

    print("Direct full-system runs: " f"{DIRECT_FULL_SYSTEM_RUNS}")

    print("Component-only cells: " f"{COMPONENT_ONLY_CELLS}")

    print()

    print(f"[PASS] {CSV_PATH}")

    print(f"[PASS] {JSON_PATH}")

    print(f"[PASS] {MARKDOWN_PATH}")

    print(f"[PASS] {MANIFEST_PATH}")

    print()

    print("SPRINT 7.13 BUILD: PASS")


if __name__ == "__main__":
    main()
