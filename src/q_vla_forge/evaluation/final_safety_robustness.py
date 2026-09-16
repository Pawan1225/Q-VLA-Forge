"""Sprint 7.7 — final frozen safety and robustness summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SAFETY_PACKAGE = "results/safety/consolidated/" "sprint5-safety-evidence-package.json"

EXPECTED_SEEDS = (42, 123, 456)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_safety_robustness(
    root: Path,
) -> dict[str, Any]:
    """Build Sprint 7.7 from frozen Sprint 5 safety evidence."""

    source_path = root / SAFETY_PACKAGE

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source = load_json(source_path)

    seeds = tuple(source.get("principal_seeds", []))

    if seeds != EXPECTED_SEEDS:
        raise ValueError("Safety package does not use locked seeds " "42, 123, 456.")

    clean = source["clean_evidence"]
    robustness = source["robustness_corpus"]
    mechanism = source["lyapunov_mechanism"]

    driving_none = clean["autonomous_driving"]["none"]["violation_step_rate"]

    driving_clipping = clean["autonomous_driving"]["clipping"]["violation_step_rate"]

    driving_lyapunov = clean["autonomous_driving"]["lyapunov"]["violation_step_rate"]

    robotics_none = clean["robotics"]["none"]["violation_step_rate"]

    robotics_clipping = clean["robotics"]["clipping"]["violation_step_rate"]

    robotics_lyapunov = clean["robotics"]["lyapunov"]["violation_step_rate"]

    clean_zero_both = (
        driving_clipping == 0.0
        and driving_lyapunov == 0.0
        and robotics_clipping == 0.0
        and robotics_lyapunov == 0.0
    )

    action = robustness["action"]

    return {
        "sprint": "7.7",
        "protocol": "final_safety_robustness_summary",
        "status": "FROZEN",
        "source": SAFETY_PACKAGE,
        "required_seeds": list(EXPECTED_SEEDS),
        "clean_evidence": {
            "autonomous_driving": {
                "none_violation_step_rate": driving_none,
                "clipping_violation_step_rate": driving_clipping,
                "lyapunov_violation_step_rate": driving_lyapunov,
            },
            "robotics": {
                "none_violation_step_rate": robotics_none,
                "clipping_violation_step_rate": robotics_clipping,
                "lyapunov_violation_step_rate": robotics_lyapunov,
            },
        },
        "clean_explicit_filtering_zero_violation_rate_both_domains": (clean_zero_both),
        "robustness": {
            "gaussian": robustness["gaussian"],
            "structured_state": robustness["structured_state"],
            "action": action,
        },
        "action_recovery": {
            "unsafe_perturbed_steps": action["unsafe_perturbed_steps"],
            "recovered_unsafe_steps": action["recovered_unsafe_steps"],
            "unresolved_unsafe_steps": action["unresolved_unsafe_steps"],
            "recovery_fraction": action["recovery_fraction"],
        },
        "lyapunov_mechanism": mechanism,
        "formal_safety_guarantee": False,
        "formal_lyapunov_stability_proof": False,
        "claim": (
            "Explicit safety filtering produced strong empirical "
            "pilot effects across both proxy domains, including "
            "zero observed clean violation-step rate for clipping "
            "and Lyapunov filtering and quantified recovery under "
            "direct action perturbations."
        ),
        "limitations": source.get(
            "global_limitations",
            [],
        ),
        "new_training": False,
        "new_safety_episodes": False,
    }
