"""Build the Sprint 4 to Sprint 5 handoff artifact."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path("results/rl/final/sprint4-handoff.json")


def build_handoff() -> dict[str, object]:
    """Build the locked Sprint 5 safety handoff."""

    return {
        "from_sprint": "4",
        "to_sprint": "5",
        "handoff_status": "ready",
        "frozen_inputs": {
            "environments": [
                "autonomous_driving",
                "robotics",
            ],
            "environment_files": {
                "autonomous_driving": ("src/q_vla_forge/rl/driving_env.py"),
                "robotics": ("src/q_vla_forge/rl/robotics_env.py"),
            },
            "policies": [
                "classical_ppo",
                "hybrid_qml",
                "matched_classical",
            ],
            "principal_seeds": [
                42,
                123,
                456,
            ],
            "evaluation_seeds": list(
                range(
                    20000,
                    20020,
                )
            ),
            "principal_interaction_budget": 20000,
            "observation_interface": ("raw_environment_state"),
            "action_dimension": 3,
        },
        "locked_interfaces": {
            "policy_to_safety_filter": ("policy_proposed_action"),
            "safety_filter_to_environment": ("filtered_safe_action"),
            "environment_action_dimension": 3,
            "existing_action_bounds_are_not_a_safety_guarantee": True,
        },
        "safety_scope": {
            "methods": [
                "none",
                "heuristic_constraint_clipping",
                "lyapunov_safety_filter",
            ],
            "robustness_conditions": [
                "clean",
                "gaussian_state_perturbation",
                "state_perturbation",
                "action_perturbation",
            ],
            "evaluation_metrics": [
                "constraint_violations",
                "intervention_rate",
                "reward_degradation",
                "success_degradation",
                "safety_recovery",
                "robustness_curves",
            ],
        },
        "core_scientific_question": (
            "Can heuristic clipping and Lyapunov-based safety "
            "filtering reduce constraint violations and improve "
            "robustness under state/action perturbations while "
            "preserving acceptable task performance?"
        ),
        "safety_layer_principle": {
            "flow": [
                "policy_proposed_action",
                "safety_filter",
                "safe_action",
                "environment",
            ],
            "interpretation": (
                "The AI/RL policy proposes an action; the safety "
                "layer verifies or corrects it before the "
                "environment receives the action."
            ),
            "compatible_policies": [
                "classical_ppo",
                "hybrid_qml",
                "matched_classical",
            ],
        },
        "scientific_boundary": {
            "existing_action_scaling": ("valid_action_range_enforcement"),
            "existing_action_scaling_is_formal_safety": False,
            "sprint4_established_safety_guarantee": False,
            "sprint5_adds_explicit_safety_logic": True,
        },
        "planned_comparison": {
            "safety_methods": [
                "none",
                "clipping",
                "lyapunov",
            ],
            "conditions": [
                "clean",
                "gaussian_noise",
                "state_perturbation",
                "action_perturbation",
            ],
            "domains": [
                "autonomous_driving",
                "robotics",
            ],
            "principal_seeds": [
                42,
                123,
                456,
            ],
            "status": "planned_not_executed",
        },
        "open_questions": [],
    }


def main() -> None:
    handoff = build_handoff()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            handoff,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Sprint 5 handoff:",
        OUTPUT_PATH.as_posix(),
    )


if __name__ == "__main__":
    main()
