"""Build Sprint 7 Batch E — final reproducibility and acceptance."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_acceptance import (
    build_final_acceptance,
)
from q_vla_forge.evaluation.final_reproducibility import (
    build_final_reproducibility,
)

ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT = ROOT / "results" / "final-validation"

REPRO_DIR = FINAL_ROOT / "reproducibility"

FINAL_DIR = FINAL_ROOT / "final-acceptance"

BATCH_DIR = FINAL_ROOT / "batch-e"


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


def run_command(
    command: list[str],
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output = (result.stdout + result.stderr).strip()

    return {
        "command": command,
        "return_code": (result.returncode),
        "passed": (result.returncode == 0),
        "output": output,
    }


def extract_pytest_count(
    output: str,
) -> int | None:
    matches = re.findall(
        r"(\d+)\s+passed",
        output,
    )

    if not matches:
        return None

    return int(matches[-1])


def run_quality_gate() -> dict[str, Any]:
    ruff = shutil.which("ruff")

    black = shutil.which("black")

    if ruff is None:
        raise RuntimeError("ruff executable not found.")

    if black is None:
        raise RuntimeError("black executable not found.")

    pytest_result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
        ]
    )

    pytest_result["passed_tests"] = extract_pytest_count(pytest_result["output"])

    ruff_result = run_command(
        [
            ruff,
            "check",
            "src",
            "tests",
            "dashboard",
        ]
    )

    black_result = run_command(
        [
            black,
            "--check",
            "src",
            "tests",
            "dashboard",
        ]
    )

    return {
        "pytest": pytest_result,
        "ruff": ruff_result,
        "black": black_result,
    }


def main() -> None:
    print("=" * 70)
    print(" Q-VLA FORGE — SPRINT 7 BATCH E")
    print(" 7.14 + 7.15 FINAL ACCEPTANCE")
    print("=" * 70)
    print()

    print("[7.14] Final Reproducibility Verification")

    reproducibility = build_final_reproducibility(ROOT)

    write_json(
        REPRO_DIR / "final-reproducibility-verification.json",
        reproducibility,
    )

    print(
        "  Batches A-D: " f"{'PASS' if reproducibility['batches_passed'] else 'FAIL'}"
    )

    print(
        "  Required artifacts: "
        f"{reproducibility['required_artifacts']['present_count']}/"
        f"{reproducibility['required_artifacts']['required_count']}"
    )

    print(
        "  Locked seed contract: "
        f"{'PASS' if reproducibility['seed_contract']['passed'] else 'FAIL'}"
    )

    print("  Claim registry frozen: " f"{reproducibility['claim_freeze']['frozen']}")

    print("  Status: " f"{reproducibility['status']}")
    print()

    if not reproducibility["passed"]:
        print("SPRINT 7 BATCH E: FAIL")
        raise SystemExit(1)

    print("[7.15] Repository Quality Gate")

    quality_gate = run_quality_gate()

    pytest_result = quality_gate["pytest"]

    print("  Pytest: " f"{'PASS' if pytest_result['passed'] else 'FAIL'}")

    if pytest_result["passed_tests"] is not None:
        print("  Tests passed: " f"{pytest_result['passed_tests']}")

    print("  Ruff: " f"{'PASS' if quality_gate['ruff']['passed'] else 'FAIL'}")

    print("  Black: " f"{'PASS' if quality_gate['black']['passed'] else 'FAIL'}")
    print()

    acceptance = build_final_acceptance(
        ROOT,
        quality_gate,
    )

    write_json(
        FINAL_DIR / "sprint7-final-acceptance.json",
        acceptance,
    )

    batch_acceptance = {
        "batch": ("Sprint 7 Batch E"),
        "sub_sprints": [
            "7.14",
            "7.15",
        ],
        "reproducibility_verification": (reproducibility["status"]),
        "repository_quality_gate": (
            "PASS"
            if all(
                item["passed"]
                for item in (
                    quality_gate["pytest"],
                    quality_gate["ruff"],
                    quality_gate["black"],
                )
            )
            else "FAIL"
        ),
        "final_acceptance": (acceptance["status"]),
        "frozen_evidence_preserved": True,
        "new_training": False,
        "new_experiments": False,
        "passed": acceptance["passed"],
    }

    write_json(
        BATCH_DIR / "batch-e-acceptance.json",
        batch_acceptance,
    )

    print("[7.15] Sprint 7 Final Acceptance")

    print("  Phase 1 evidence frozen: " f"{acceptance['phase1_evidence_frozen']}")

    print("  Sprint 7 complete: " f"{acceptance['sprint7_complete']}")

    print(
        "  Ready for submission packaging: "
        f"{acceptance['ready_for_submission_packaging']}"
    )

    print("  Status: " f"{acceptance['status']}")
    print()

    print("-" * 70)

    if acceptance["passed"]:
        print("SPRINT 7 BATCH E: PASS")
        print("SPRINT 7 FINAL ACCEPTANCE: PASS")
        print("Q-VLA FORGE PHASE 1 EVIDENCE: FROZEN")
    else:
        print("SPRINT 7 BATCH E: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
