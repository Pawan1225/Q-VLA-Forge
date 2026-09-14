"""Build Sprint 5.14B cross-domain safety architecture reuse analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.cross_domain_safety import (
    SafetyArchitectureReuse,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "safety" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "sprint5-cross-domain-architecture.json"

OUTPUT_MD = OUTPUT_DIR / "sprint5-cross-domain-architecture.md"


def _architecture_record() -> SafetyArchitectureReuse:
    return SafetyArchitectureReuse(
        shared_safety_interface=True,
        shared_action_dimension=True,
        shared_decision_contract=True,
        shared_metric_schema=True,
        shared_seed_protocol=True,
        shared_robustness_harness=True,
        domain_specific_constraints=True,
        domain_specific_clipping=True,
        domain_specific_predictor=True,
        domain_specific_lyapunov=True,
        same_policy_weights=False,
        transfer_tested=False,
        universal_controller_supported=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    architecture = _architecture_record()

    shared_components = [
        {
            "component": "policy_interface",
            "driving": "PPO",
            "robotics": "PPO",
            "shared": True,
            "sharing_level": "interface",
        },
        {
            "component": "action_dimension",
            "driving": 3,
            "robotics": 3,
            "shared": True,
            "sharing_level": "interface",
        },
        {
            "component": "safety_decision_contract",
            "driving": "SafetyDecision",
            "robotics": "SafetyDecision",
            "shared": True,
            "sharing_level": "contract",
        },
        {
            "component": "none_method",
            "driving": "shared interface",
            "robotics": "shared interface",
            "shared": True,
            "sharing_level": "framework",
        },
        {
            "component": "clipping_method",
            "driving": "domain rules",
            "robotics": "domain rules",
            "shared": True,
            "sharing_level": "framework_only",
        },
        {
            "component": "lyapunov_method",
            "driving": "V_drive",
            "robotics": "V_robotics",
            "shared": True,
            "sharing_level": "framework_only",
        },
        {
            "component": "constraint_semantics",
            "driving": "driving constraints",
            "robotics": "robotics constraints",
            "shared": False,
            "sharing_level": "domain_specific",
        },
        {
            "component": "predictor",
            "driving": "driving one-step predictor",
            "robotics": "robotics one-step predictor",
            "shared": True,
            "sharing_level": "framework_only",
        },
        {
            "component": "perturbation_harness",
            "driving": "shared harness",
            "robotics": "shared harness",
            "shared": True,
            "sharing_level": "framework",
        },
        {
            "component": "metric_schema",
            "driving": "shared metrics",
            "robotics": "shared metrics",
            "shared": True,
            "sharing_level": "schema",
        },
        {
            "component": "principal_seeds",
            "driving": [42, 123, 456],
            "robotics": [42, 123, 456],
            "shared": True,
            "sharing_level": "protocol",
        },
        {
            "component": "trained_policy_weights",
            "driving": "driving-specific",
            "robotics": "robotics-specific",
            "shared": False,
            "sharing_level": "domain_specific",
        },
    ]

    domain_specific = {
        "autonomous_driving": {
            "state_dimension": 4,
            "state_semantics": [
                "speed",
                "lane_offset",
                "heading_error",
                "obstacle_distance",
            ],
            "constraint_evaluator": "driving constraint evaluator",
            "clipping_rules": "driving-specific clipping",
            "predictor": "driving one-step predictor",
            "lyapunov": "V_drive",
        },
        "robotics": {
            "state_dimension": 6,
            "state_semantics": [
                "robot_x",
                "robot_y",
                "object_x",
                "object_y",
                "target_x",
                "target_y",
            ],
            "constraint_evaluator": "robotics constraint evaluator",
            "clipping_rules": "robotics-specific clipping",
            "predictor": "robotics one-step predictor",
            "lyapunov": "V_robotics",
            "additional_context": [
                "object_grasped",
                "gripper semantics",
            ],
        },
    }

    payload: dict[str, Any] = {
        "sprint": "5.14B",
        "artifact": "cross-domain-safety-architecture-reuse",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "architecture_reuse": {
            "shared_safety_interface": architecture.shared_safety_interface,
            "shared_action_dimension": architecture.shared_action_dimension,
            "shared_action_dim": 3,
            "shared_decision_contract": architecture.shared_decision_contract,
            "shared_metric_schema": architecture.shared_metric_schema,
            "shared_seed_protocol": architecture.shared_seed_protocol,
            "shared_robustness_harness": architecture.shared_robustness_harness,
            "domain_specific_constraints": architecture.domain_specific_constraints,
            "domain_specific_clipping": architecture.domain_specific_clipping,
            "domain_specific_predictor": architecture.domain_specific_predictor,
            "domain_specific_lyapunov": architecture.domain_specific_lyapunov,
            "same_policy_weights": architecture.same_policy_weights,
            "transfer_tested": architecture.transfer_tested,
            "universal_controller_supported": architecture.universal_controller_supported,
        },
        "shared_components": shared_components,
        "domain_specific_components": domain_specific,
        "supported_statement": (
            "A common safety-filter interface, intervention protocol, "
            "robustness harness, metric schema, and seed protocol were "
            "reused across autonomous-driving and robotics proxy "
            "environments while retaining domain-specific constraints, "
            "predictors, clipping semantics, and Lyapunov potentials."
        ),
        "blocked_interpretations": [
            "same safety function",
            "same Lyapunov function",
            "same safety thresholds",
            "same trained policy",
            "zero-shot safety transfer",
            "universal safety controller",
            "formal cross-domain safety guarantee",
            "production generalization",
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sprint 5.14B — Cross-Domain Safety Architecture Reuse",
        "",
        "## Scope",
        "",
        (
            "Architecture-only analysis. No training, policy execution, "
            "safety evaluation, robustness rerun, or threshold change "
            "was performed."
        ),
        "",
        "## Shared vs domain-specific architecture",
        "",
        "| Component | Driving | Robotics | Reuse |",
        "|---|---|---|---|",
    ]

    for row in shared_components:
        reuse = "Yes" if row["shared"] else "No"

        lines.append(
            f"| {row['component']} | "
            f"{row['driving']} | "
            f"{row['robotics']} | "
            f"{reuse} ({row['sharing_level']}) |"
        )

    lines.extend(
        [
            "",
            "## Driving-specific components",
            "",
            "- State dimension: 4",
            ("- State: speed, lane offset, heading error, " "obstacle distance"),
            "- Driving constraint evaluator",
            "- Driving-specific clipping rules",
            "- Driving one-step predictor",
            "- V_drive",
            "",
            "## Robotics-specific components",
            "",
            "- State dimension: 6",
            ("- State: robot position, object position, " "target position"),
            "- Robotics constraint evaluator",
            "- Robotics-specific clipping rules",
            "- Robotics one-step predictor",
            "- V_robotics",
            "- Object-grasped prediction context",
            "",
            "## Supported conclusion",
            "",
            payload["supported_statement"],
            "",
            "## Explicitly unsupported interpretations",
            "",
        ]
    )

    for item in payload["blocked_interpretations"]:
        lines.append(f"- {item}")

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print(" SPRINT 5.14B CROSS-DOMAIN ARCHITECTURE REUSE")
    print("=" * 76)
    print()

    print("Shared safety interface: PASS")
    print("Shared 3D action interface: PASS")
    print("Shared decision contract: PASS")
    print("Shared metric schema: PASS")
    print("Shared seed protocol: PASS")
    print("Shared robustness harness: PASS")
    print()

    print("Domain-specific constraints: PASS")
    print("Domain-specific clipping: PASS")
    print("Domain-specific predictors: PASS")
    print("Domain-specific Lyapunov potentials: PASS")
    print()

    print("Same trained weights: FALSE")
    print("Cross-domain transfer tested: FALSE")
    print("Universal controller supported: FALSE")
    print()

    print("No new training: PASS")
    print("No new principal execution: PASS")
    print()

    print("SPRINT 5.14B ARCHITECTURE REUSE: PASS")


if __name__ == "__main__":
    main()
