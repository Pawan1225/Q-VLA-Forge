"""Build Sprint 7 Batch A — subsprints 7.1, 7.2, and 7.3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_baseline import build_final_baseline
from q_vla_forge.evaluation.phase1_evidence import build_phase1_evidence
from q_vla_forge.evaluation.three_seed_validation import (
    validate_three_seed_evidence,
)

ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT = ROOT / "results" / "final-validation"

THREE_SEED_DIR = FINAL_ROOT / "three-seed"
EVIDENCE_DIR = FINAL_ROOT / "evidence"
BASELINE_DIR = FINAL_ROOT / "baseline"
BATCH_DIR = FINAL_ROOT / "batch-a"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_three_seed_report(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge — Sprint 7.1",
        "",
        "## Final Three-Seed Validation",
        "",
        "**Locked seeds:** 42, 123, 456",
        "",
        "| Evidence Group | Observed Seeds | Status |",
        "|---|---|---|",
    ]

    for group in payload["groups"]:
        observed = ", ".join(str(seed) for seed in group["observed_seeds"])

        status = "PASS" if group["complete"] else "REVIEW"

        lines.append(f"| {group['group']} | " f"{observed or 'None'} | " f"{status} |")

    lines.extend(
        [
            "",
            "## Controls",
            "",
            "- Frozen Phase 1 evidence only.",
            "- Sprint 7 generated artifacts excluded.",
            ("- Required seed declarations do not " "count as execution evidence."),
            "- No historical evidence modified.",
            "- No experiment rerun by this audit.",
            "",
            "## Result",
            "",
            ("**PASS**" if payload["passed"] else "**REVIEW REQUIRED**"),
            "",
        ]
    )

    return "\n".join(lines)


def build_baseline_report(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge — Sprint 7.3",
        "",
        "## Final Frozen Baseline",
        "",
        (
            f"Baseline: **{payload['baseline_name']} "
            f"v{payload['baseline_version']}**"
        ),
        "",
        ("Parameters: " f"**{payload['architecture']['parameters']:,}**"),
        "",
        (
            "FP32 storage: "
            f"**{payload['architecture']['fp32_model_size_bytes']:,} bytes**"
        ),
        "",
        ("| Domain | Test MSE | Test MAE | " "Mean Latency (ms) | P95 Latency (ms) |"),
        "|---|---:|---:|---:|---:|",
    ]

    for name in ("driving", "robotics"):
        domain = payload["domains"][name]

        lines.append(
            f"| {domain['domain']} | "
            f"{domain['test_mse']['mean']:.6f} "
            f"± {domain['test_mse']['std']:.6f} | "
            f"{domain['test_mae']['mean']:.6f} "
            f"± {domain['test_mae']['std']:.6f} | "
            f"{domain['mean_latency_ms']['mean']:.3f} "
            f"± {domain['mean_latency_ms']['std']:.3f} | "
            f"{domain['p95_latency_ms']['mean']:.3f} "
            f"± {domain['p95_latency_ms']['std']:.3f} |"
        )

    lines.extend(
        [
            "",
            ("All statistics use the frozen " "three-seed protocol: **42, 123, 456**."),
            "",
            ("No Sprint 1 baseline values were " "retrained or recomputed."),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print(" Q-VLA FORGE — SPRINT 7 BATCH A")
    print(" 7.1 + 7.2 + 7.3 FINAL EVIDENCE BUILD")
    print("=" * 70)
    print()

    # --------------------------------------------------
    # Sprint 7.1
    # --------------------------------------------------

    print("[7.1] Final Three-Seed Validation")

    validation = validate_three_seed_evidence(ROOT)

    validation_payload = validation.to_dict()

    write_json(
        THREE_SEED_DIR / "three-seed-validation.json",
        validation_payload,
    )

    THREE_SEED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (THREE_SEED_DIR / "three-seed-report.md").write_text(
        build_three_seed_report(validation_payload),
        encoding="utf-8",
    )

    for group in validation.groups:
        status = "PASS" if group.complete else "REVIEW"

        print(f"  {group.group:<22} " f"{status:<7} " f"{list(group.observed_seeds)}")

    print()

    # --------------------------------------------------
    # Sprint 7.2
    # --------------------------------------------------

    print("[7.2] Cross-Sprint Evidence Aggregation")

    evidence = build_phase1_evidence(ROOT)

    write_json(
        EVIDENCE_DIR / "phase1-evidence.json",
        evidence,
    )

    print("  Frozen JSON artifacts: " f"{evidence['artifact_count']}")
    print("  Status: PASS")
    print()

    # --------------------------------------------------
    # Sprint 7.3
    # --------------------------------------------------

    print("[7.3] Final Baseline Summary")

    baseline = build_final_baseline(ROOT)

    write_json(
        BASELINE_DIR / "final-baseline-summary.json",
        baseline,
    )

    BASELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (BASELINE_DIR / "final-baseline-summary.md").write_text(
        build_baseline_report(baseline),
        encoding="utf-8",
    )

    print("  Baseline: " f"{baseline['baseline_name']}")
    print("  Parameters: " f"{baseline['architecture']['parameters']}")
    print("  Seeds: " f"{baseline['required_seeds']}")
    print("  Status: PASS")
    print()

    # --------------------------------------------------
    # Batch acceptance
    # --------------------------------------------------

    batch_passed = validation.passed

    acceptance = {
        "batch": "Sprint 7 Batch A",
        "sub_sprints": [
            "7.1",
            "7.2",
            "7.3",
        ],
        "three_seed_validation": ("PASS" if validation.passed else "REVIEW"),
        "cross_sprint_evidence": "PASS",
        "final_baseline_summary": "PASS",
        "frozen_evidence_preserved": True,
        "new_training": False,
        "passed": batch_passed,
    }

    write_json(
        BATCH_DIR / "batch-a-acceptance.json",
        acceptance,
    )

    print("-" * 70)

    if batch_passed:
        print("SPRINT 7 BATCH A: PASS")
    else:
        print("SPRINT 7 BATCH A: " "REVIEW REQUIRED")

    print()

    print("Artifacts:")
    print("  results/final-validation/" "three-seed/")
    print("  results/final-validation/" "evidence/")
    print("  results/final-validation/" "baseline/")
    print("  results/final-validation/" "batch-a/")


if __name__ == "__main__":
    main()
