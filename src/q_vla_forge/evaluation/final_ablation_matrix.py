"""Sprint 7.9 — final Phase 1 ablation matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_ablation_matrix(
    root: Path,
) -> dict[str, Any]:
    """Combine frozen Sprint 7 summaries into one ablation matrix."""

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

    cross_domain = load_json(
        root / "results/final-validation/cross-domain/"
        "final-cross-domain-summary.json"
    )

    rows = [
        {
            "area": "compression",
            "comparison": "FP32 vs INT8",
            "finding": ("INT8 is on the Pareto frontier in both domains."),
            "status": "supported",
        },
        {
            "area": "compression",
            "comparison": "INT8 vs SVD",
            "finding": (
                "Evaluated SVD settings did not satisfy the "
                "joint frozen pilot criterion."
            ),
            "status": "supported",
        },
        {
            "area": "compression",
            "comparison": "INT8 vs TT/MPS",
            "finding": (
                "TT/MPS was implemented and evaluated but did "
                "not demonstrate superiority."
            ),
            "status": "supported",
        },
        {
            "area": "training_efficiency",
            "comparison": "trainable SVD / TT-MPS vs FP32 target",
            "finding": training["claim"],
            "status": "not_demonstrated",
        },
        {
            "area": "rl_qml",
            "comparison": "Full PPO vs matched classical",
            "finding": (
                "Full PPO reached 6/6 frozen targets; matched " "classical reached 1/6."
            ),
            "status": "supported",
        },
        {
            "area": "rl_qml",
            "comparison": "Full PPO vs QML",
            "finding": (
                "Full PPO reached 6/6 frozen targets while QML "
                "reached 0/6; QML actor size was substantially lower."
            ),
            "status": "supported_with_limitation",
        },
        {
            "area": "safety",
            "comparison": "No filter vs clipping",
            "finding": (
                "Clipping reduced clean observed violation-step "
                "rate to zero in both proxy domains."
            ),
            "status": "supported",
        },
        {
            "area": "safety",
            "comparison": "No filter vs Lyapunov filter",
            "finding": (
                "Lyapunov filtering reduced clean observed "
                "violation-step rate to zero in both proxy domains."
            ),
            "status": "supported_with_limitation",
        },
        {
            "area": "robustness",
            "comparison": "Action perturbation recovery",
            "finding": (
                f"Recovered {safety['action_recovery']['recovered_unsafe_steps']} "
                f"of {safety['action_recovery']['unsafe_perturbed_steps']} "
                "unsafe perturbed steps."
            ),
            "status": "supported",
        },
        {
            "area": "cross_domain",
            "comparison": "Framework reuse",
            "finding": cross_domain["claim"],
            "status": "supported_with_limitation",
        },
    ]

    return {
        "sprint": "7.9",
        "protocol": "final_ablation_matrix",
        "status": "FROZEN",
        "rows": rows,
        "source_summaries": {
            "compression": compression["source"],
            "training_efficiency": training["source"],
            "rl_qml": rl_qml["source"],
            "safety": safety["source"],
            "cross_domain": cross_domain["source"],
        },
        "new_training": False,
        "new_experiments": False,
    }
