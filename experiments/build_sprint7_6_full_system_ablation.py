"""Build Sprint 7.6 full-system ablation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.full_system_ablation import (
    ALLOW_SYNTHETIC_METRIC_COMPOSITION,
    CLAIM_CONTROLS,
    COMPONENT_EVIDENCE,
    CONFIGURATIONS,
    DOMAINS,
    FULL_SYSTEM_DIRECT_EXECUTION_FOUND,
    FULL_SYSTEM_PHASE1_NOTE,
    INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY,
    PHASE1_STATUS,
    PHASE2_STATUS,
    build_phase1_evidence_matrix,
    component_only_configuration_count,
    direct_configuration_count,
    not_evaluated_configuration_count,
)

OUTPUT_DIR = Path("results/final-validation/full-system-ablation")

JSON_PATH = OUTPUT_DIR / "full-system-ablation.json"

CSV_PATH = OUTPUT_DIR / "full-system-ablation.csv"

MARKDOWN_PATH = OUTPUT_DIR / "full-system-ablation.md"

PHASE2_PLAN_PATH = OUTPUT_DIR / "phase2-factorial-plan.csv"


SCIENTIFIC_CONCLUSION = (
    "Phase 1 independently evaluated compression, QML policy, "
    "and safety-filter components across both proxy domains. "
    "Exact end-to-end factorial combinations were not directly "
    "executed, so independently measured component results are "
    "not combined into synthetic full-system performance estimates. "
    "Matched integrated factorial execution remains a Phase 2 "
    "validation objective."
)

INTERACTION_CONCLUSION = (
    "Compression, QML, and safety main effects or interaction "
    "effects cannot be estimated from the Phase 1 full-system "
    "factorial matrix because no matched integrated factorial "
    "configuration was directly executed."
)


def _bool_text(
    value: bool,
) -> str:
    return "yes" if value else "no"


def _phase2_priority(
    configuration: str,
) -> int:
    priorities = {
        "baseline": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 5,
        "e": 6,
        "f": 7,
        "full": 8,
    }

    return priorities[configuration]


def _matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in build_phase1_evidence_matrix():
        rows.append(
            {
                "domain": record.domain,
                "configuration": record.name,
                "compression": (record.compression),
                "qml": record.qml,
                "safety": record.safety,
                "evidence_status": (record.evidence_status),
                "can_report_end_to_end_metrics": (record.can_report_end_to_end_metrics),
                "phase_status": (record.phase_status),
                "source_artifacts": list(record.source_artifacts),
                "notes": record.notes,
            }
        )

    return rows


def _phase2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in build_phase1_evidence_matrix():
        rows.append(
            {
                "domain": record.domain,
                "configuration": record.name,
                "compression": (record.compression),
                "qml": record.qml,
                "safety": record.safety,
                "phase1_evidence_status": (record.evidence_status),
                "phase2_run_required": (record.evidence_status != "direct"),
                "priority": (_phase2_priority(record.name)),
            }
        )

    return rows


def _build_json_payload() -> dict[
    str,
    Any,
]:
    matrix = _matrix_rows()

    return {
        "protocol": ("sprint7_6_full_system_ablation"),
        "sprint": "7.6",
        "status": "FROZEN",
        "new_experiments": False,
        "new_training": False,
        "factorial_design": {
            "factors": [
                "compression",
                "qml",
                "safety",
            ],
            "configurations": {
                name: {
                    "compression": factors[0],
                    "qml": factors[1],
                    "safety": factors[2],
                }
                for name, factors in CONFIGURATIONS.items()
            },
            "domains": list(DOMAINS),
            "expected_classifications": 16,
        },
        "phase1": {
            "status": PHASE1_STATUS,
            "direct_full_system_execution_found": (FULL_SYSTEM_DIRECT_EXECUTION_FOUND),
            "direct_count": (direct_configuration_count()),
            "component_only_count": (component_only_configuration_count()),
            "not_evaluated_count": (not_evaluated_configuration_count()),
            "component_evidence": (COMPONENT_EVIDENCE),
            "matrix": matrix,
        },
        "phase2": {
            "status": PHASE2_STATUS,
            "matched_integrated_factorial_validation_required": True,
            "interaction_effects_to_measure": [
                "compression",
                "qml",
                "safety",
                "compression_x_qml",
                "compression_x_safety",
                "qml_x_safety",
                "compression_x_qml_x_safety",
            ],
        },
        "scientific_controls": {
            "synthetic_metric_composition_allowed": (
                ALLOW_SYNTHETIC_METRIC_COMPOSITION
            ),
            "interaction_effects_estimable_with_component_only": (
                INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY
            ),
            "full_system_end_to_end_metrics_reported": False,
            "predicted_phase2_performance_reported": False,
        },
        "claim_controls": (dict(CLAIM_CONTROLS)),
        "phase1_note": (FULL_SYSTEM_PHASE1_NOTE),
        "scientific_conclusion": (SCIENTIFIC_CONCLUSION),
        "interaction_conclusion": (INTERACTION_CONCLUSION),
    }


def _write_json(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv() -> None:
    rows = _matrix_rows()

    fieldnames = [
        "domain",
        "configuration",
        "compression",
        "qml",
        "safety",
        "evidence_status",
        "can_report_end_to_end_metrics",
        "phase_status",
        "source_artifacts",
        "notes",
    ]

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "source_artifacts": (";".join(row["source_artifacts"])),
                }
            )


def _write_phase2_plan() -> None:
    rows = _phase2_rows()

    fieldnames = [
        "domain",
        "configuration",
        "compression",
        "qml",
        "safety",
        "phase1_evidence_status",
        "phase2_run_required",
        "priority",
    ]

    with PHASE2_PLAN_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def _markdown_table() -> str:
    lines = [
        "| Domain | Config | Compression | QML | Safety | Evidence |",
        "|---|---|---:|---:|---:|---|",
    ]

    for record in build_phase1_evidence_matrix():
        lines.append(
            "| "
            f"{record.domain} | "
            f"{record.name} | "
            f"{_bool_text(record.compression)} | "
            f"{_bool_text(record.qml)} | "
            f"{_bool_text(record.safety)} | "
            f"{record.evidence_status.upper()} |"
        )

    return "\n".join(lines)


def _write_markdown() -> None:
    markdown = f"""# Q-VLA Forge - Sprint 7.6 Full-System Ablation

Phase 1 component evidence only. No new training, PPO execution,
QML execution, safety episodes, robustness runs, or matched integrated
factorial experiments were performed in Sprint 7.6.

## Phase 1 Component Evidence

The following independently frozen evidence exists:

- Compression: Sprint 7.3
- QML / RL policy comparison: Sprint 7.4
- Safety filtering: Sprint 7.5

These component results are not combined into synthetic full-system
performance values.

## Full-System Evidence Matrix

{_markdown_table()}

### Evidence Summary

- DIRECT: {direct_configuration_count()}
- COMPONENT_ONLY: {component_only_configuration_count()}
- NOT_EVALUATED: {not_evaluated_configuration_count()}

All sixteen Phase 1 factorial cells are classified as COMPONENT_ONLY.

No configuration is permitted to report synthetic end-to-end reward,
MSE, latency, success, violation rate, or other integrated metrics.

## Directly Evaluated vs Not Directly Evaluated

No exact Compression x QML x Safety factorial configuration was directly
executed as one matched end-to-end pipeline.

Sprint 4 RL/QML policies operated on compact environment state rather than
the Sprint 1 shared VLA latent representation. Therefore the independently
evaluated compression, policy, and safety pathways cannot be treated as one
measured integrated system.

## Interaction Effects

The following interaction effects are not established in Phase 1:

- Compression x QML
- Compression x Safety
- QML x Safety
- Compression x QML x Safety

Component-only evidence cannot be used to infer these effects.

## Phase 2 Integrated Factorial Plan

All sixteen domain/configuration cells require matched integrated execution
before they can be promoted from COMPONENT_ONLY to DIRECT evidence.

The Phase 2 plan contains no predicted performance, expected winner, or
fabricated target values.

## Scientific Conclusion

{SCIENTIFIC_CONCLUSION}

## Interaction Conclusion

{INTERACTION_CONCLUSION}

## Claim Controls

- Quantum advantage claim: blocked
- Full-system superiority claim: blocked
- Integrated factorial validation claim: blocked
- Cross-domain zero-shot transfer claim: blocked
- Production-readiness claim: blocked
- Interaction-effect claims without direct evidence: blocked
"""

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )


def build() -> None:
    """Build all Sprint 7.6 full-system ablation artifacts."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = _build_json_payload()

    _write_json(payload)
    _write_csv()
    _write_markdown()
    _write_phase2_plan()

    expected_paths = (
        JSON_PATH,
        CSV_PATH,
        MARKDOWN_PATH,
        PHASE2_PLAN_PATH,
    )

    missing = [path for path in expected_paths if not path.is_file()]

    if missing:
        raise RuntimeError(
            "Missing Sprint 7.6 artifact(s): "
            + ", ".join(str(path) for path in missing)
        )

    print("=" * 64)
    print(" SPRINT 7.6 FULL-SYSTEM ABLATION BUILD")
    print("=" * 64)
    print()
    print("Artifacts:")
    print("  full-system-ablation.json          PASS")
    print("  full-system-ablation.csv           PASS")
    print("  full-system-ablation.md            PASS")
    print("  phase2-factorial-plan.csv          PASS")
    print()
    print("Factorial configurations:            8")
    print("Domains:                             2")
    print("Evidence classifications:           16")
    print(f"Direct evidence:                    " f"{direct_configuration_count()}")
    print(
        f"Component-only evidence:            "
        f"{component_only_configuration_count()}"
    )
    print(
        f"Not evaluated:                      " f"{not_evaluated_configuration_count()}"
    )
    print("Synthetic metric composition:        BLOCKED")
    print("Interaction inference:               BLOCKED")
    print("Phase 2 factorial plan:              PASS")
    print("Claim controls:                      PASS")
    print()
    print("SPRINT 7.6 FULL-SYSTEM ABLATION BUILD: PASS")


if __name__ == "__main__":
    build()
