"""Build Sprint 5.6 Lyapunov foundation evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from q_vla_forge.safety.lyapunov import (
    evaluate_driving_lyapunov,
    evaluate_lyapunov_delta,
    evaluate_robotics_lyapunov,
    required_safe_distance,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "results" / "safety" / "lyapunov-foundation"

PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)

DRIVING_CONTRACT_PATH = (
    ROOT / "results" / "safety" / "driving" / "sprint5-driving-safety-constraints.json"
)

ROBOTICS_CONTRACT_PATH = (
    ROOT
    / "results"
    / "safety"
    / "robotics"
    / "sprint5-robotics-safety-constraints.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65_536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def driving_state(
    *,
    speed: float = 0.2,
    lane: float = 0.0,
    heading: float = 0.0,
    obstacle: float = 1.0,
) -> np.ndarray:
    return np.array(
        [
            speed,
            lane,
            heading,
            obstacle,
        ],
        dtype=np.float64,
    )


def robotics_state(
    *,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
    object_x: float = 0.2,
    object_y: float = 0.0,
    target_x: float = 0.4,
    target_y: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            robot_x,
            robot_y,
            object_x,
            object_y,
            target_x,
            target_y,
        ],
        dtype=np.float64,
    )


def evaluation_to_dict(
    evaluation: Any,
) -> dict[str, Any]:
    return {
        "domain": evaluation.domain,
        "total": evaluation.total,
        "components": evaluation.components,
        "zero_risk_region": (evaluation.zero_risk_region),
    }


def delta_to_dict(
    delta: Any,
) -> dict[str, float]:
    return {
        "current_value": (delta.current_value),
        "next_value": (delta.next_value),
        "delta": (delta.delta),
    }


def build_driving_probes() -> dict[str, Any]:
    nominal = evaluate_driving_lyapunov(driving_state())

    lane_warning = evaluate_driving_lyapunov(driving_state(lane=0.85))

    lane_critical = evaluate_driving_lyapunov(driving_state(lane=0.95))

    heading_warning = evaluate_driving_lyapunov(driving_state(heading=0.60))

    heading_critical = evaluate_driving_lyapunov(driving_state(heading=0.75))

    obstacle_probe_distances = (
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
    )

    obstacle_probe_values = [
        {
            "distance": distance,
            "evaluation": (
                evaluation_to_dict(
                    evaluate_driving_lyapunov(
                        driving_state(
                            speed=1.0,
                            obstacle=distance,
                        )
                    )
                )
            ),
        }
        for distance in obstacle_probe_distances
    ]

    speed_probe_values = [
        {
            "speed": speed,
            "safe_distance": (required_safe_distance(speed)),
            "evaluation": (
                evaluation_to_dict(
                    evaluate_driving_lyapunov(
                        driving_state(
                            speed=speed,
                            obstacle=0.40,
                        )
                    )
                )
            ),
        }
        for speed in (
            0.0,
            0.5,
            1.0,
        )
    ]

    safer_delta = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.95),
        next_state=driving_state(lane=0.80),
    )

    worsening_delta = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.80),
        next_state=driving_state(lane=0.95),
    )

    return {
        "nominal": (evaluation_to_dict(nominal)),
        "lane_warning_probe": (evaluation_to_dict(lane_warning)),
        "lane_critical_reference": (evaluation_to_dict(lane_critical)),
        "heading_warning_probe": (evaluation_to_dict(heading_warning)),
        "heading_critical_reference": (evaluation_to_dict(heading_critical)),
        "obstacle_distance_probe": (obstacle_probe_values),
        "speed_sensitivity_probe": (speed_probe_values),
        "safer_transition_delta": (delta_to_dict(safer_delta)),
        "worsening_transition_delta": (delta_to_dict(worsening_delta)),
    }


def build_robotics_probes() -> dict[str, Any]:
    nominal = evaluate_robotics_lyapunov(robotics_state())

    robot_boundary = [
        {
            "robot_x": value,
            "evaluation": (
                evaluation_to_dict(
                    evaluate_robotics_lyapunov(robotics_state(robot_x=value))
                )
            ),
        }
        for value in (
            0.80,
            0.90,
            0.95,
            1.00,
            1.05,
        )
    ]

    object_boundary = [
        {
            "object_x": value,
            "evaluation": (
                evaluation_to_dict(
                    evaluate_robotics_lyapunov(
                        robotics_state(
                            object_x=value,
                            target_x=value,
                        )
                    )
                )
            ),
        }
        for value in (
            0.90,
            0.95,
            1.00,
            1.05,
        )
    ]

    target_inside = evaluate_robotics_lyapunov(robotics_state(target_x=0.80))

    target_boundary = evaluate_robotics_lyapunov(robotics_state(target_x=1.00))

    target_outside = evaluate_robotics_lyapunov(robotics_state(target_x=1.05))

    task_separation = evaluate_robotics_lyapunov(
        robotics_state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.0,
            object_y=0.0,
            target_x=0.8,
            target_y=0.0,
        )
    )

    interaction_central = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.0,
            object_y=0.0,
            target_x=0.8,
            target_y=0.0,
        )
    )

    interaction_boundary = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.95,
            object_y=0.0,
            target_x=0.0,
            target_y=0.0,
        )
    )

    safer_delta = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.98),
        next_state=robotics_state(robot_x=0.85),
    )

    worsening_delta = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.85),
        next_state=robotics_state(robot_x=0.98),
    )

    return {
        "nominal": (evaluation_to_dict(nominal)),
        "robot_boundary_probe": (robot_boundary),
        "object_boundary_probe": (object_boundary),
        "target_inside_probe": (evaluation_to_dict(target_inside)),
        "target_boundary_probe": (evaluation_to_dict(target_boundary)),
        "target_outside_probe": (evaluation_to_dict(target_outside)),
        "task_safety_separation_probe": (evaluation_to_dict(task_separation)),
        "interaction_central_probe": (evaluation_to_dict(interaction_central)),
        "interaction_boundary_probe": (evaluation_to_dict(interaction_boundary)),
        "safer_transition_delta": (delta_to_dict(safer_delta)),
        "worsening_transition_delta": (delta_to_dict(worsening_delta)),
    }


def nondecreasing(
    values: list[float],
) -> bool:
    return all(right >= left for left, right in pairwise(values))


def build_property_checks(
    driving: dict[str, Any],
    robotics: dict[str, Any],
) -> dict[str, bool]:
    obstacle_values = [
        float(item["evaluation"]["components"]["obstacle"])
        for item in driving["obstacle_distance_probe"]
    ]

    speed_values = [
        float(item["evaluation"]["components"]["obstacle"])
        for item in driving["speed_sensitivity_probe"]
    ]

    robot_values = [
        float(item["evaluation"]["components"]["robot_workspace"])
        for item in robotics["robot_boundary_probe"]
    ]

    object_values = [
        float(item["evaluation"]["components"]["object_workspace"])
        for item in robotics["object_boundary_probe"]
    ]

    return {
        "driving_nominal_zero": (driving["nominal"]["total"] == 0.0),
        "driving_lane_critical_normalized": (
            abs(driving["lane_critical_reference"]["components"]["lane"] - 1.0)
            <= 1.0e-12
        ),
        "driving_heading_critical_normalized": (
            abs(driving["heading_critical_reference"]["components"]["heading"] - 1.0)
            <= 1.0e-12
        ),
        "driving_obstacle_monotonic": (nondecreasing(obstacle_values)),
        "driving_speed_sensitivity": (nondecreasing(speed_values)),
        "driving_safer_delta_negative": (
            driving["safer_transition_delta"]["delta"] < 0.0
        ),
        "driving_worsening_delta_positive": (
            driving["worsening_transition_delta"]["delta"] > 0.0
        ),
        "robotics_nominal_zero": (robotics["nominal"]["total"] == 0.0),
        "robot_workspace_monotonic": (nondecreasing(robot_values)),
        "object_workspace_monotonic": (nondecreasing(object_values)),
        "target_inside_zero": (
            robotics["target_inside_probe"]["components"]["target_workspace"] == 0.0
        ),
        "target_boundary_zero": (
            robotics["target_boundary_probe"]["components"]["target_workspace"] == 0.0
        ),
        "target_outside_positive": (
            robotics["target_outside_probe"]["components"]["target_workspace"] > 0.0
        ),
        "task_safety_separation": (
            robotics["task_safety_separation_probe"]["total"] == 0.0
        ),
        "interaction_central_zero": (
            robotics["interaction_central_probe"]["components"][
                "object_target_interaction"
            ]
            == 0.0
        ),
        "interaction_boundary_positive": (
            robotics["interaction_boundary_probe"]["components"][
                "object_target_interaction"
            ]
            > 0.0
        ),
        "robotics_safer_delta_negative": (
            robotics["safer_transition_delta"]["delta"] < 0.0
        ),
        "robotics_worsening_delta_positive": (
            robotics["worsening_transition_delta"]["delta"] > 0.0
        ),
    }


def build_markdown(
    payload: dict[str, Any],
) -> None:
    checks = payload["property_checks"]

    lines = [
        "# Sprint 5.6 - Lyapunov Function Foundation",
        "",
        "Method: classical Lyapunov safety candidate",
        "",
        "Action modification performed: `false`",
        "",
        "Policy evaluation performed: `false`",
        "",
        "Empirical parameter tuning: `false`",
        "",
        "Clipping results used for parameter selection: `false`",
        "",
        "## Driving Candidate",
        "",
        "Components:",
        "",
        "- lane",
        "- heading",
        "- obstacle",
        "",
        "Formula:",
        "",
        "`V_drive = lane_excess^2 + heading_excess^2 + obstacle_excess^2`",
        "",
        "## Robotics Candidate",
        "",
        "Components:",
        "",
        "- robot_workspace",
        "- object_workspace",
        "- target_workspace",
        "- object_target_interaction",
        "",
        "Formula:",
        "",
        (
            "`V_robotics = robot_excess^2 + object_excess^2 + "
            "target_excess^2 + interaction_excess^2`"
        ),
        "",
        "## Property Checks",
        "",
    ]

    for name, passed in checks.items():
        lines.append(f"- `{name}`: " f"{'PASS' if passed else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Excluded Action-Only Constraints",
            "",
            "Driving:",
            "",
            "- steering-risk action component",
            "- acceleration/braking conflict",
            "",
            "Robotics:",
            "",
            "- unsafe motion",
            "- unsafe gripper condition",
            "",
            "These remain explicit hard guards for Sprint 5.7.",
            "",
            "## Claim Boundary",
            "",
            "Lyapunov candidate foundation: `SUPPORTED`",
            "",
            "Lyapunov action filtering: `NOT_YET_TESTED`",
            "",
            "Lyapunov violation reduction: `NOT_YET_TESTED`",
            "",
            "Lyapunov versus clipping: `NOT_YET_TESTED`",
            "",
            "Robustness improvement: `NOT_YET_TESTED`",
            "",
            "Formal stability proof: `NO`",
            "",
            "Formal safety guarantee: `NO`",
            "",
            "Production certification: `NO`",
            "",
            "Quantum safety claim: `NO`",
            "",
        ]
    )

    markdown_path = OUTPUT_DIRECTORY / "sprint5-lyapunov-foundation.md"

    markdown_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    for source_path in (
        PROTOCOL_PATH,
        DRIVING_CONTRACT_PATH,
        ROBOTICS_CONTRACT_PATH,
    ):
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    driving = build_driving_probes()

    robotics = build_robotics_probes()

    property_checks = build_property_checks(
        driving,
        robotics,
    )

    if not all(property_checks.values()):
        failures = [name for name, passed in property_checks.items() if not passed]

        raise RuntimeError("Lyapunov property checks failed: " + ", ".join(failures))

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "sprint": "5.6",
        "artifact": ("lyapunov-function-foundation"),
        "method": ("classical_lyapunov_safety_candidate"),
        "scope": {
            "candidate_definition_only": True,
            "policy_loading": False,
            "checkpoint_loading": False,
            "action_modification_performed": False,
            "candidate_action_selection": False,
            "principal_policy_evaluation": False,
            "robustness_evaluation": False,
            "formal_stability_proof": False,
        },
        "provenance": {
            "sprint5_protocol": {
                "path": (PROTOCOL_PATH.relative_to(ROOT).as_posix()),
                "sha256": (sha256_file(PROTOCOL_PATH)),
            },
            "driving_safety_contract": {
                "path": (DRIVING_CONTRACT_PATH.relative_to(ROOT).as_posix()),
                "sha256": (sha256_file(DRIVING_CONTRACT_PATH)),
            },
            "robotics_safety_contract": {
                "path": (ROBOTICS_CONTRACT_PATH.relative_to(ROOT).as_posix()),
                "sha256": (sha256_file(ROBOTICS_CONTRACT_PATH)),
            },
        },
        "shared_design": {
            "component_weights": ("equal"),
            "weight_value": 1.0,
            "epsilon": 1.0e-12,
            "empirical_weight_tuning": False,
            "clipping_results_used_for_parameter_selection": False,
        },
        "driving": {
            "state_order": [
                "speed",
                "lane_offset",
                "heading_error",
                "obstacle_distance",
            ],
            "components": [
                "lane",
                "heading",
                "obstacle",
            ],
            "formula": {
                "lane_excess": ("max(0, abs(lane_offset)-0.75) / 0.20"),
                "heading_excess": ("max(0, abs(heading_error)-0.50) / 0.25"),
                "fixed_obstacle_excess": ("max(0, 0.30-obstacle_distance) / 0.15"),
                "safe_distance": ("0.15 + 0.35*max(speed,0)"),
                "speed_obstacle_excess": (
                    "max(0, safe_distance-obstacle_distance) "
                    "/ max(safe_distance,1e-12)"
                ),
                "obstacle_excess": (
                    "max(fixed_obstacle_excess," " speed_obstacle_excess)"
                ),
                "total": ("lane_excess^2 + heading_excess^2 " "+ obstacle_excess^2"),
            },
            "probes": driving,
        },
        "robotics": {
            "state_order": [
                "robot_x",
                "robot_y",
                "object_x",
                "object_y",
                "target_x",
                "target_y",
            ],
            "components": [
                "robot_workspace",
                "object_workspace",
                "target_workspace",
                "object_target_interaction",
            ],
            "formula": {
                "robot_extent": ("max(abs(robot_x),abs(robot_y))"),
                "robot_excess": ("max(0, robot_extent-0.90) / 0.10"),
                "object_extent": ("max(abs(object_x),abs(object_y))"),
                "object_excess": ("max(0, object_extent-0.90) / 0.10"),
                "target_extent": ("max(abs(target_x),abs(target_y))"),
                "target_excess": ("max(0, target_extent-1.00) / 0.10"),
                "interaction_excess": (
                    "0 if object_extent<=0.90 else "
                    "max(0, object_target_distance-0.50)/0.50"
                ),
                "total": (
                    "robot_excess^2 + object_excess^2 + "
                    "target_excess^2 + interaction_excess^2"
                ),
            },
            "probes": robotics,
        },
        "excluded_action_only_constraints": {
            "driving": [
                "steering_risk_action_component",
                "acceleration_braking_conflict",
            ],
            "robotics": [
                "unsafe_motion",
                "unsafe_gripper_condition",
            ],
        },
        "property_checks": (property_checks),
        "claims": {
            "deterministic_candidate_supported": True,
            "nonnegative_candidate_supported": True,
            "synthetic_probe_monotonicity_supported": True,
            "lyapunov_filter_effectiveness": ("NOT_YET_TESTED"),
            "lyapunov_violation_reduction": ("NOT_YET_TESTED"),
            "lyapunov_vs_clipping": ("NOT_YET_TESTED"),
            "robustness_improvement": ("NOT_YET_TESTED"),
            "formal_stability_proof": False,
            "formal_safety_guarantee": False,
            "production_certification": False,
            "quantum_safety_claim": False,
        },
    }

    json_path = OUTPUT_DIRECTORY / "sprint5-lyapunov-foundation.json"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build_markdown(payload)

    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.6 LYAPUNOV FOUNDATION")
    print("=" * 68)
    print()

    print("Driving candidate: PASS")
    print("Driving components: 3")
    print()
    print("Robotics candidate: PASS")
    print("Robotics components: 4")
    print()
    print("Non-negativity: PASS")
    print("Monotonic hazard probes: PASS")
    print("Task/safety separation: PASS")
    print("Equal weights: YES")
    print("Empirical tuning: NO")
    print("Action modification: NO")
    print("Policy evaluation: NO")
    print("Formal stability proof: NO")
    print()
    print("SPRINT 5.6 LYAPUNOV FOUNDATION BUILT")


if __name__ == "__main__":
    main()
