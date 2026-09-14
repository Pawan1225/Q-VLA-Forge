"""Build frozen Sprint 5.7 Lyapunov-filter mechanism evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyMethod,
)
from q_vla_forge.safety.lyapunov_candidates import (
    ACTION_DEDUPLICATION_TOLERANCE,
    MAX_DRIVING_CANDIDATES,
    MAX_ROBOTICS_CANDIDATES,
)
from q_vla_forge.safety.lyapunov_filter import (
    INTERVENTION_TOLERANCE_L2,
    LYAPUNOV_IMPROVEMENT_TOLERANCE,
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.predictors import (
    RoboticsPredictionContext,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "safety" / "lyapunov-filter"

JSON_PATH = OUTPUT_DIR / "sprint5-lyapunov-filter-mechanism.json"

MARKDOWN_PATH = OUTPUT_DIR / "sprint5-lyapunov-filter-mechanism.md"

SOURCE_FILES = {
    "sprint5_protocol": ROOT / "src/q_vla_forge/safety/protocol.py",
    "driving_constraints": (ROOT / "src/q_vla_forge/safety/driving_constraints.py"),
    "robotics_constraints": (ROOT / "src/q_vla_forge/safety/robotics_constraints.py"),
    "lyapunov_foundation": (ROOT / "src/q_vla_forge/safety/lyapunov.py"),
    "predictors": (ROOT / "src/q_vla_forge/safety/predictors.py"),
    "candidates": (ROOT / "src/q_vla_forge/safety/lyapunov_candidates.py"),
    "filter": (ROOT / "src/q_vla_forge/safety/lyapunov_filter.py"),
}


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _decision_summary(
    decision: Any,
) -> dict[str, Any]:
    return {
        "method": decision.method.value,
        "proposed_action": (decision.proposed_action.tolist()),
        "executed_action": (decision.executed_action.tolist()),
        "intervened": decision.intervened,
        "intervention_reason": (decision.intervention_reason.value),
        "correction_l2": (decision.correction_l2),
        "candidate_count": (decision.metadata["candidate_count"]),
        "eligible_candidate_count": (decision.metadata["eligible_candidate_count"]),
        "selected_candidate_index": (decision.metadata["selected_candidate_index"]),
        "selected_candidate_source": (decision.metadata["selected_candidate_source"]),
        "current_v": (decision.metadata["current_v"]),
        "selected_next_v": (decision.metadata["selected_next_v"]),
        "selected_delta_v": (decision.metadata["selected_delta_v"]),
        "emergency_fallback": (decision.metadata["emergency_fallback"]),
    }


def _build_probes() -> dict[str, Any]:
    driving_safe = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.30,
                0.00,
                0.00,
                2.00,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                0.10,
                0.20,
                0.00,
            ],
            dtype=np.float64,
        ),
    )

    driving_guard = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.40,
                0.70,
                0.60,
                1.00,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                0.90,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
    )

    driving_decrease = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.40,
                0.80,
                0.00,
                1.00,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
    )

    robotics_safe = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.00,
                0.00,
                0.20,
                0.20,
                0.50,
                0.50,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                0.20,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=False)),
    )

    robotics_guard = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.89,
                0.00,
                0.00,
                0.00,
                0.50,
                0.50,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                1.00,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=False)),
    )

    robotics_decrease = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.95,
                0.95,
                0.95,
                0.95,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=True)),
    )

    return {
        "driving_safe_preservation": (_decision_summary(driving_safe)),
        "driving_hard_guard": (_decision_summary(driving_guard)),
        "driving_lyapunov_decrease": (_decision_summary(driving_decrease)),
        "robotics_safe_preservation": (_decision_summary(robotics_safe)),
        "robotics_hard_guard": (_decision_summary(robotics_guard)),
        "robotics_lyapunov_decrease": (_decision_summary(robotics_decrease)),
    }


def _property_checks(
    probes: dict[str, Any],
) -> dict[str, bool]:
    return {
        "method_is_lyapunov": all(
            probe["method"] == SafetyMethod.LYAPUNOV.value for probe in probes.values()
        ),
        "driving_safe_action_preserved": (
            not probes["driving_safe_preservation"]["intervened"]
        ),
        "robotics_safe_action_preserved": (
            not probes["robotics_safe_preservation"]["intervened"]
        ),
        "driving_hard_guard_intervenes": (
            probes["driving_hard_guard"]["intervention_reason"]
            == InterventionReason.DOMAIN_CONSTRAINT.value
        ),
        "robotics_hard_guard_intervenes": (
            probes["robotics_hard_guard"]["intervention_reason"]
            == InterventionReason.DOMAIN_CONSTRAINT.value
        ),
        "driving_strict_lyapunov_decrease": (
            probes["driving_lyapunov_decrease"]["selected_delta_v"] < 0.0
        ),
        "robotics_strict_lyapunov_decrease": (
            probes["robotics_lyapunov_decrease"]["selected_delta_v"] < 0.0
        ),
        "driving_candidate_ceiling_respected": all(
            probe["candidate_count"] <= MAX_DRIVING_CANDIDATES
            for name, probe in probes.items()
            if name.startswith("driving_")
        ),
        "robotics_candidate_ceiling_respected": all(
            probe["candidate_count"] <= MAX_ROBOTICS_CANDIDATES
            for name, probe in probes.items()
            if name.startswith("robotics_")
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance = {
        name: {
            "path": str(path.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            "sha256": _sha256(path),
        }
        for name, path in SOURCE_FILES.items()
    }

    probes = _build_probes()

    property_checks = _property_checks(probes)

    artifact: dict[str, Any] = {
        "sprint": "5.7",
        "artifact": ("lyapunov_safety_filter_mechanism"),
        "method": ("deterministic_one_step_lyapunov_filter"),
        "scope": {
            "policy_loading": False,
            "checkpoint_loading": False,
            "ppo_training": False,
            "policy_fine_tuning": False,
            "principal_policy_evaluation": False,
            "robustness_evaluation": False,
            "empirical_parameter_tuning": False,
            "formal_stability_proof": False,
            "formal_safety_guarantee": False,
        },
        "constants": {
            "action_deduplication_tolerance": (ACTION_DEDUPLICATION_TOLERANCE),
            "intervention_tolerance_l2": (INTERVENTION_TOLERANCE_L2),
            "lyapunov_improvement_tolerance": (LYAPUNOV_IMPROVEMENT_TOLERANCE),
            "max_driving_candidates": (MAX_DRIVING_CANDIDATES),
            "max_robotics_candidates": (MAX_ROBOTICS_CANDIDATES),
        },
        "selection_order": [
            "hard_guard_pass",
            "lower_predicted_next_v",
            "lower_delta_v",
            "lower_action_correction_l2",
            "lower_candidate_index",
        ],
        "provenance": provenance,
        "synthetic_probes": probes,
        "property_checks": (property_checks),
        "claim_controls": {
            "filter_implementation_verified": True,
            "principal_violation_reduction_tested": False,
            "principal_reward_effect_tested": False,
            "principal_success_effect_tested": False,
            "lyapunov_vs_clipping_tested": False,
            "robustness_tested": False,
            "formal_stability_established": False,
            "formal_safety_guarantee_established": False,
        },
    }

    JSON_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# Sprint 5.7 Lyapunov Safety Filter Mechanism",
        "",
        "## Scope",
        "",
        "- Deterministic one-step candidate filter",
        "- No PPO training",
        "- No policy loading",
        "- No principal policy evaluation",
        "- No robustness evaluation",
        "- No formal stability or safety guarantee",
        "",
        "## Synthetic Property Checks",
        "",
    ]

    for name, passed in property_checks.items():
        markdown_lines.append(f"- {name}: " f"{'PASS' if passed else 'FAIL'}")

    markdown_lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            (
                "Sprint 5.7 verifies implementation and "
                "synthetic mechanism behavior only."
            ),
            (
                "Empirical violation reduction, reward impact, "
                "success impact, robustness, and comparison "
                "against clipping remain untested."
            ),
            "",
        ]
    )

    MARKDOWN_PATH.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print(
        "Driving safe preservation:",
        "PASS" if property_checks["driving_safe_action_preserved"] else "FAIL",
    )

    print(
        "Driving hard guard:",
        "PASS" if property_checks["driving_hard_guard_intervenes"] else "FAIL",
    )

    print(
        "Driving Lyapunov decrease:",
        "PASS" if property_checks["driving_strict_lyapunov_decrease"] else "FAIL",
    )

    print(
        "Robotics safe preservation:",
        "PASS" if property_checks["robotics_safe_action_preserved"] else "FAIL",
    )

    print(
        "Robotics hard guard:",
        "PASS" if property_checks["robotics_hard_guard_intervenes"] else "FAIL",
    )

    print(
        "Robotics Lyapunov decrease:",
        "PASS" if property_checks["robotics_strict_lyapunov_decrease"] else "FAIL",
    )

    print(
        "Policy evaluation:",
        "NO",
    )

    print(
        "Empirical tuning:",
        "NO",
    )

    print(
        "Formal safety guarantee:",
        "NO",
    )

    if not all(property_checks.values()):
        raise SystemExit("SPRINT 5.7 LYAPUNOV FILTER MECHANISM: FAIL")

    print("SPRINT 5.7 LYAPUNOV FILTER MECHANISM BUILT")


if __name__ == "__main__":
    main()
