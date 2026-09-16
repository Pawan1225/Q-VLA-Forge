"""Sprint 7.10 — final challenge-bottleneck scorecard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_challenge_scorecard(
    root: Path,
) -> dict[str, Any]:
    """Build the frozen four-bottleneck Phase 1 scorecard."""

    compression = load_json(
        root / "results/final-validation/compression/" "final-compression-summary.json"
    )

    training = load_json(
        root / "results/final-validation/training-efficiency/"
        "final-training-efficiency-summary.json"
    )

    rl_qml = load_json(
        root / "results/final-validation/rl-qml/" "final-rl-qml-summary.json"
    )

    safety = load_json(
        root / "results/final-validation/safety-robustness/"
        "final-safety-robustness-summary.json"
    )

    bottlenecks = [
        {
            "bottleneck": "model_footprint",
            "phase1_status": "demonstrated",
            "headline": compression["claim"],
            "strongest_evidence": (
                "INT8 ~3.85x effective compression with " "cross-domain Pareto support."
            ),
            "boundary": ("No TT/MPS superiority demonstrated."),
        },
        {
            "bottleneck": "training_efficiency",
            "phase1_status": "not_demonstrated",
            "headline": training["claim"],
            "strongest_evidence": (
                "All methods evaluated under the locked "
                "three-seed target-reaching protocol."
            ),
            "boundary": ("No robust >=10% optimizer-step reduction."),
        },
        {
            "bottleneck": "rl_alignment_sample_efficiency",
            "phase1_status": "mixed_evidence",
            "headline": rl_qml["claim"],
            "strongest_evidence": (
                "PPO reached 6/6 targets; QML actor achieved "
                "~95.5-95.9% parameter reduction."
            ),
            "boundary": (
                "No QML sample-efficiency or computational " "advantage demonstrated."
            ),
        },
        {
            "bottleneck": "safety",
            "phase1_status": "empirically_supported",
            "headline": safety["claim"],
            "strongest_evidence": (
                "Clean filtered violation-step rate reached zero "
                "in both proxy domains; direct action recovery "
                "was quantified."
            ),
            "boundary": (
                "Empirical proxy evidence only; no formal "
                "safety or Lyapunov stability guarantee."
            ),
        },
    ]

    return {
        "sprint": "7.10",
        "protocol": "challenge_bottleneck_scorecard",
        "status": "FROZEN",
        "bottlenecks": bottlenecks,
        "all_four_bottlenecks_experimentally_addressed": True,
        "all_four_bottlenecks_solved": False,
        "new_training": False,
        "new_experiments": False,
    }
