from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(".")

EVIDENCE_DIR = ROOT / "results" / "safety" / "evidence"

REQUIRED_EVIDENCE = (
    EVIDENCE_DIR / "sprint5-safety-evidence.json",
    EVIDENCE_DIR / "sprint5-safety-evidence.csv",
    EVIDENCE_DIR / "sprint5-safety-evidence.md",
    EVIDENCE_DIR / "sprint5-safety-claim-matrix.json",
    EVIDENCE_DIR / "sprint5-safety-manifest.json",
    EVIDENCE_DIR / "sprint5-safety-figure-index.json",
)

REQUIRED_FIGURES = (
    ROOT / "figures" / "safety" / "sprint5-clean-safety.png",
    ROOT / "figures" / "safety" / "sprint5-robustness.png",
    ROOT / "figures" / "safety" / "sprint5-action-recovery.png",
    ROOT / "figures" / "safety" / "sprint5-lyapunov-mechanisms.png",
    ROOT / "figures" / "safety" / "sprint5-cross-domain.png",
)

REQUIRED_SCRIPTS = (
    ROOT / "experiments" / "verify_sprint5_safety_evidence_readiness.py",
    ROOT / "experiments" / "build_sprint5_safety_evidence.py",
    ROOT / "experiments" / "build_sprint5_safety_assets.py",
    ROOT / "experiments" / "verify_sprint5_safety_evidence.py",
    ROOT / "experiments" / "verify_sprint5_final_gate.py",
)

REQUIRED_DASHBOARD = ROOT / "dashboard" / "pages" / "5_Safety_Evidence.py"

HISTORICAL_ARTIFACTS = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-safety-evidence-package.json",
    ROOT / "results" / "safety" / "cross-domain" / "sprint5-cross-domain-package.json",
    ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json",
    ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json",
    ROOT / "results" / "safety" / "final" / "sprint5-final-limitations.json",
    ROOT / "results" / "safety" / "final" / "sprint5-freeze-record.json",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty artifact: {path}")


def run(
    *args: str,
) -> None:
    subprocess.run(
        list(args),
        check=True,
    )


def verify_artifacts() -> None:
    for path in (
        *REQUIRED_EVIDENCE,
        *REQUIRED_FIGURES,
        *REQUIRED_SCRIPTS,
        REQUIRED_DASHBOARD,
    ):
        require_file(path)


def verify_historical_integrity() -> None:
    for path in HISTORICAL_ARTIFACTS:
        require_file(path)

    freeze_record = load_json(
        ROOT / "results" / "safety" / "final" / "sprint5-freeze-record.json"
    )

    required_true = {
        "analysis_only_closeout",
        "artifact_manifest_verified",
        "claim_boundary_frozen",
        "limitations_frozen",
        "scientific_invariants_frozen",
        "scientific_work_frozen",
        "phase2_bridge_present",
        "sprint7_handoff_ready",
    }

    for key in required_true:
        if freeze_record.get(key) is not True:
            raise RuntimeError("Historical Sprint 5 freeze " f"invariant failed: {key}")

    required_false = {
        "new_metrics",
        "new_principal_runs",
        "new_robustness_runs",
        "new_safety_runs",
        "new_training",
        "sprint6_execution_required",
    }

    for key in required_false:
        if freeze_record.get(key) is not False:
            raise RuntimeError("Historical Sprint 5 freeze " f"invariant failed: {key}")

    if freeze_record.get("status") != "COMPLETE":
        raise RuntimeError("Historical Sprint 5 freeze " "status is not COMPLETE")

    if freeze_record.get("claim_count") != 25:
        raise RuntimeError("Historical Sprint 5 claim count changed")

    if freeze_record.get("limitation_count") != 15:
        raise RuntimeError("Historical Sprint 5 limitation " "count changed")

    if freeze_record.get("chain_stage_count") != 16:
        raise RuntimeError("Historical Sprint 5 chain-stage " "count changed")


def verify_evidence_scope() -> None:
    evidence = load_json(EVIDENCE_DIR / "sprint5-safety-evidence.json")

    metadata = evidence.get("metadata")

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError("Evidence metadata missing")

    expected = {
        "new_training": False,
        "new_principal_execution": False,
        "new_scientific_experiment": False,
        "scientific_scope_frozen": True,
    }

    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Evidence scope mismatch: {key}")


def verify_claims() -> None:
    matrix = load_json(EVIDENCE_DIR / "sprint5-safety-claim-matrix.json")

    proposal = matrix.get("proposal_claims")

    if not isinstance(
        proposal,
        list,
    ):
        raise TypeError("Proposal claims missing")

    ids = {
        str(row["claim_id"])
        for row in proposal
        if isinstance(
            row,
            dict,
        )
    }

    if ids != {
        "S5-E01",
        "S5-E02",
        "S5-E03",
        "S5-E04",
        "S5-E05",
        "S5-E06",
    }:
        raise RuntimeError("Proposal claim set mismatch")


def verify_manifest() -> None:
    manifest = load_json(EVIDENCE_DIR / "sprint5-safety-manifest.json")

    if manifest.get("scientific_scope_frozen") is not True:
        raise RuntimeError("Manifest freeze mismatch")

    if manifest.get("new_training") is not False:
        raise RuntimeError("Manifest training flag mismatch")

    if manifest.get("new_principal_execution") is not False:
        raise RuntimeError("Manifest execution flag mismatch")

    if manifest.get("new_scientific_experiment") is not False:
        raise RuntimeError("Manifest science flag mismatch")


def main() -> None:
    print("=" * 64)
    print(" Q-VLA FORGE - SPRINT 5.15 FINAL ACCEPTANCE GATE")
    print("=" * 64)
    print()

    verify_artifacts()

    print("Required evidence artifacts           PASS")

    verify_historical_integrity()

    print("Historical Sprint 5 integrity          PASS")

    verify_evidence_scope()

    print("Scientific freeze controls            PASS")

    verify_claims()

    print("Proposal claim controls               PASS")

    verify_manifest()

    print("Evidence manifest controls            PASS")

    run(
        sys.executable,
        "experiments/" "verify_sprint5_safety_evidence_readiness.py",
    )

    print("Packaging readiness verifier          PASS")

    run(
        sys.executable,
        "experiments/" "verify_sprint5_safety_evidence.py",
    )

    print("Independent evidence verifier         PASS")

    print()
    print("No new training                       PASS")
    print("No principal execution                PASS")
    print("No new scientific experiment          PASS")
    print()
    print("SPRINT 5.15 FINAL ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
