"""Sprint 7.15 — final Sprint 7 acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_acceptance(
    root: Path,
    quality_gate: dict[str, Any],
) -> dict[str, Any]:
    """Build final Sprint 7 acceptance record."""

    reproducibility = load_json(
        root
        / "results"
        / "final-validation"
        / "reproducibility"
        / "final-reproducibility-verification.json"
    )

    batches = {}

    for name in (
        "a",
        "b",
        "c",
        "d",
    ):
        path = (
            root
            / "results"
            / "final-validation"
            / f"batch-{name}"
            / f"batch-{name}-acceptance.json"
        )

        payload = load_json(path)

        batches[f"batch-{name}"] = bool(
            payload.get(
                "passed",
                False,
            )
        )

    quality_passed = all(
        (
            quality_gate["pytest"]["passed"],
            quality_gate["ruff"]["passed"],
            quality_gate["black"]["passed"],
        )
    )

    passed = (
        all(batches.values()) and bool(reproducibility["passed"]) and quality_passed
    )

    return {
        "sprint": "7.15",
        "project": "Q-VLA Forge",
        "protocol": ("sprint7_final_acceptance"),
        "phase": "Phase 1",
        "batch_acceptance": batches,
        "reproducibility_verification": (reproducibility["status"]),
        "quality_gate": quality_gate,
        "frozen_evidence_preserved": True,
        "new_principal_training": False,
        "new_scientific_experiments": False,
        "phase1_evidence_frozen": passed,
        "sprint7_complete": passed,
        "ready_for_submission_packaging": (passed),
        "status": ("PASS" if passed else "FAIL"),
        "passed": passed,
    }
