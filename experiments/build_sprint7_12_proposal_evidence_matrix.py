from __future__ import annotations

import csv
import json
from pathlib import Path

from q_vla_forge.evaluation.proposal_evidence_matrix import (
    COMPONENT_ONLY_CELLS,
    DIRECT_FULL_SYSTEM_RUNS,
    PHASE1_SEEDS,
    build_proposal_evidence_matrix,
    validate_proposal_evidence_matrix,
)

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "final-validation" / "proposal-evidence"

CSV_PATH = OUTPUT_DIR / "proposal-evidence-matrix.csv"

JSON_PATH = OUTPUT_DIR / "proposal-evidence-matrix.json"

MARKDOWN_PATH = OUTPUT_DIR / "proposal-evidence-matrix.md"

MANIFEST_PATH = OUTPUT_DIR / "proposal-evidence-manifest.json"


FIELDNAMES = (
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
        "artifact": ("Q-VLA Forge Sprint 7.12 " "Proposal Evidence Matrix"),
        "phase": "Phase 1",
        "evidence_frozen": True,
        "new_experiments": False,
        "new_training": False,
        "new_scientific_results": False,
        "principal_seeds": list(PHASE1_SEEDS),
        "direct_full_system_runs": (DIRECT_FULL_SYSTEM_RUNS),
        "component_only_cells": (COMPONENT_ONLY_CELLS),
        "row_count": len(rows),
        "rows": rows,
        "claim_controls": {
            "quantum_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "qml_sample_efficiency_advantage_claimed": False,
            "tt_mps_superiority_claimed": False,
            "formal_safety_guarantee_claimed": False,
            "production_readiness_claimed": False,
            "zero_shot_cross_domain_transfer_claimed": False,
            "full_system_superiority_claimed": False,
            "synthetic_end_to_end_metric_composition": False,
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
        "# Q-VLA Forge — Phase 1 Proposal Evidence Matrix",
        "",
        (
            "Sprint 7.12 consolidates the frozen Phase 1 evidence into "
            "a reviewer-facing traceability matrix."
        ),
        "",
        (
            "This artifact introduces no new experiments, training, "
            "scientific results, or synthetic end-to-end metrics."
        ),
        "",
        "## Evidence Protocol",
        "",
        "- Domains: autonomous driving and robotics proxy domains",
        "- Principal seeds: 42, 123, 456",
        "- Frozen Phase 1 evidence: yes",
        "- Direct integrated full-system runs: 0",
        "- Component-only factorial cells: 16",
        "",
        "## Proposal Evidence Matrix",
        "",
        (
            "| ID | Bottleneck | Research Question | Method | "
            "Metric / Criterion | Observed Result | Status | "
            "Proposal-Safe Claim | Limitation | Evidence | "
            "Phase 2 Follow-Up |"
        ),
        ("|---|---|---|---|---|---|---|---|---|---|---|"),
    ]

    for row in rows:
        values = [
            markdown_escape(row["evidence_id"]),
            markdown_escape(row["challenge_bottleneck"]),
            markdown_escape(row["research_question"]),
            markdown_escape(row["method"]),
            markdown_escape(row["metric_or_criterion"]),
            markdown_escape(row["observed_result"]),
            markdown_escape(row["phase1_status"]),
            markdown_escape(row["proposal_safe_claim"]),
            markdown_escape(row["limitation"]),
            markdown_escape(row["evidence_artifact"]),
            markdown_escape(row["phase2_follow_up"]),
        ]

        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Phase 1 Boundaries",
            "",
            "- No quantum advantage claim.",
            "- No quantum speedup claim.",
            "- No QML sample-efficiency superiority claim.",
            "- No TT/MPS superiority claim.",
            "- No formal safety or Lyapunov stability guarantee.",
            "- No production or certification claim.",
            "- No zero-shot cross-domain transfer claim.",
            "- No full-system superiority claim.",
            "- No synthetic end-to-end metric composition.",
            "",
            "## Phase 2 Carry-Over",
            "",
            (
                "The principal integrated follow-up is matched "
                "execution of the 16 Compression × QML × Safety "
                "domain/configuration cells required to estimate "
                "full-system main and interaction effects."
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
        "sprint": "7.12",
        "artifact_group": ("proposal_evidence_matrix"),
        "status": "frozen",
        "row_count": len(rows),
        "claim_ids": [row["evidence_id"] for row in rows],
        "principal_seeds": list(PHASE1_SEEDS),
        "direct_full_system_runs": (DIRECT_FULL_SYSTEM_RUNS),
        "component_only_cells": (COMPONENT_ONLY_CELLS),
        "generated_from_frozen_evidence": True,
        "new_experiments": False,
        "new_training": False,
        "new_scientific_results": False,
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
    rows = build_proposal_evidence_matrix()

    validate_proposal_evidence_matrix(rows)

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

    print(" SPRINT 7.12 — PROPOSAL EVIDENCE MATRIX BUILD")

    print("=" * 68)

    print()

    print(f"Rows: {len(dictionaries)}")

    print("Claims: S7-C01 through S7-C09")

    print(f"Seeds: {PHASE1_SEEDS}")

    print("Direct full-system runs: " f"{DIRECT_FULL_SYSTEM_RUNS}")

    print("Component-only cells: " f"{COMPONENT_ONLY_CELLS}")

    print()

    print(f"[PASS] {CSV_PATH}")

    print(f"[PASS] {JSON_PATH}")

    print(f"[PASS] {MARKDOWN_PATH}")

    print(f"[PASS] {MANIFEST_PATH}")

    print()

    print("SPRINT 7.12 BUILD: PASS")


if __name__ == "__main__":
    main()
