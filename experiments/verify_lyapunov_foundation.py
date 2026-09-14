"""Independent verification for Sprint 5.6 Lyapunov foundation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from q_vla_forge.safety.lyapunov import (
    DRIVING_HEADING_CRITICAL_ABS,
    DRIVING_HEADING_WARNING_ABS,
    DRIVING_LANE_CRITICAL_ABS,
    DRIVING_LANE_WARNING_ABS,
    DRIVING_OBSTACLE_CRITICAL_DISTANCE,
    DRIVING_OBSTACLE_WARNING_DISTANCE,
    DRIVING_SAFE_DISTANCE_BASE,
    DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT,
    ROBOTICS_OBJECT_CRITICAL_ABS,
    ROBOTICS_OBJECT_TARGET_CRITICAL_DISTANCE,
    ROBOTICS_OBJECT_TARGET_WARNING_DISTANCE,
    ROBOTICS_OBJECT_WARNING_ABS,
    ROBOTICS_TARGET_CRITICAL_ABS,
    ROBOTICS_WORKSPACE_CRITICAL_ABS,
    ROBOTICS_WORKSPACE_WARNING_ABS,
    evaluate_driving_lyapunov,
    evaluate_lyapunov_delta,
    evaluate_robotics_lyapunov,
)

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-foundation"
    / "sprint5-lyapunov-foundation.json"
)

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

LYAPUNOV_SOURCE_PATH = ROOT / "src" / "q_vla_forge" / "safety" / "lyapunov.py"

FLOAT_TOLERANCE = 1.0e-12


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


def require(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def require_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        label,
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    require(
        path.exists(),
        f"artifact exists: {path.name}",
    )

    return json.loads(path.read_text(encoding="utf-8"))


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


def verify_provenance(
    artifact: dict[str, Any],
) -> None:
    provenance = artifact["provenance"]

    expected = {
        "sprint5_protocol": (PROTOCOL_PATH),
        "driving_safety_contract": (DRIVING_CONTRACT_PATH),
        "robotics_safety_contract": (ROBOTICS_CONTRACT_PATH),
    }

    for key, path in expected.items():
        require(
            path.exists(),
            f"{key} source exists",
        )

        require(
            provenance[key]["path"] == path.relative_to(ROOT).as_posix(),
            f"{key} path matches",
        )

        require(
            provenance[key]["sha256"] == sha256_file(path),
            f"{key} SHA256 matches",
        )


def verify_scope(
    artifact: dict[str, Any],
) -> None:
    scope = artifact["scope"]

    require(
        scope["candidate_definition_only"] is True,
        "candidate definition only",
    )

    require(
        scope["policy_loading"] is False,
        "no policy loading",
    )

    require(
        scope["checkpoint_loading"] is False,
        "no checkpoint loading",
    )

    require(
        scope["action_modification_performed"] is False,
        "no action modification",
    )

    require(
        scope["candidate_action_selection"] is False,
        "no candidate action selection",
    )

    require(
        scope["principal_policy_evaluation"] is False,
        "no principal policy evaluation",
    )

    require(
        scope["robustness_evaluation"] is False,
        "no robustness evaluation",
    )

    require(
        scope["formal_stability_proof"] is False,
        "no formal stability proof",
    )


def verify_weights(
    artifact: dict[str, Any],
) -> None:
    shared = artifact["shared_design"]

    require(
        shared["component_weights"] == "equal",
        "equal component weights",
    )

    require_close(
        shared["weight_value"],
        1.0,
        label="weight value equals one",
    )

    require(
        shared["empirical_weight_tuning"] is False,
        "no empirical weight tuning",
    )

    require(
        shared["clipping_results_used_for_parameter_selection"] is False,
        "clipping results not used for parameter selection",
    )


def verify_constants() -> None:
    require_close(
        DRIVING_LANE_WARNING_ABS,
        0.75,
        label="driving lane warning threshold",
    )

    require_close(
        DRIVING_LANE_CRITICAL_ABS,
        0.95,
        label="driving lane critical reference",
    )

    require_close(
        DRIVING_HEADING_WARNING_ABS,
        0.50,
        label="driving heading warning threshold",
    )

    require_close(
        DRIVING_HEADING_CRITICAL_ABS,
        0.75,
        label="driving heading critical reference",
    )

    require_close(
        DRIVING_OBSTACLE_WARNING_DISTANCE,
        0.30,
        label="driving obstacle warning distance",
    )

    require_close(
        DRIVING_OBSTACLE_CRITICAL_DISTANCE,
        0.15,
        label="driving obstacle critical reference",
    )

    require_close(
        DRIVING_SAFE_DISTANCE_BASE,
        0.15,
        label="driving safe-distance base",
    )

    require_close(
        DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT,
        0.35,
        label="driving safe-distance speed coefficient",
    )

    require_close(
        ROBOTICS_WORKSPACE_WARNING_ABS,
        0.90,
        label="robot workspace warning threshold",
    )

    require_close(
        ROBOTICS_WORKSPACE_CRITICAL_ABS,
        1.00,
        label="robot workspace critical reference",
    )

    require_close(
        ROBOTICS_OBJECT_WARNING_ABS,
        0.90,
        label="object workspace warning threshold",
    )

    require_close(
        ROBOTICS_OBJECT_CRITICAL_ABS,
        1.00,
        label="object workspace critical reference",
    )

    require_close(
        ROBOTICS_TARGET_CRITICAL_ABS,
        1.00,
        label="target workspace boundary",
    )

    require_close(
        ROBOTICS_OBJECT_TARGET_WARNING_DISTANCE,
        0.50,
        label="object-target warning distance",
    )

    require_close(
        ROBOTICS_OBJECT_TARGET_CRITICAL_DISTANCE,
        1.00,
        label="object-target critical reference",
    )


def verify_driving_properties() -> None:
    nominal = evaluate_driving_lyapunov(driving_state())

    require_close(
        nominal.total,
        0.0,
        label="driving nominal zero",
    )

    require(
        nominal.zero_risk_region,
        "driving nominal zero-risk flag",
    )

    lane_critical = evaluate_driving_lyapunov(driving_state(lane=0.95))

    require_close(
        lane_critical.components["lane"],
        1.0,
        label="driving lane critical normalization",
    )

    heading_critical = evaluate_driving_lyapunov(driving_state(heading=0.75))

    require_close(
        heading_critical.components["heading"],
        1.0,
        label="driving heading critical normalization",
    )

    lane_values = [
        evaluate_driving_lyapunov(driving_state(lane=value)).components["lane"]
        for value in (
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        )
    ]

    require(
        lane_values == sorted(lane_values),
        "driving lane monotonicity",
    )

    positive_lane = evaluate_driving_lyapunov(driving_state(lane=0.90))

    negative_lane = evaluate_driving_lyapunov(driving_state(lane=-0.90))

    require_close(
        positive_lane.components["lane"],
        negative_lane.components["lane"],
        label="driving lane symmetry",
    )

    obstacle_values = [
        evaluate_driving_lyapunov(
            driving_state(
                speed=1.0,
                obstacle=distance,
            )
        ).components["obstacle"]
        for distance in (
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
        )
    ]

    require(
        obstacle_values == sorted(obstacle_values),
        "driving obstacle hazard monotonicity",
    )

    speed_values = [
        evaluate_driving_lyapunov(
            driving_state(
                speed=speed,
                obstacle=0.40,
            )
        ).components["obstacle"]
        for speed in (
            0.0,
            0.5,
            1.0,
        )
    ]

    require(
        speed_values == sorted(speed_values),
        "driving speed sensitivity",
    )

    safer = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.95),
        next_state=driving_state(lane=0.80),
    )

    require(
        safer.delta < 0.0,
        "driving safer transition has negative delta",
    )

    worsening = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.80),
        next_state=driving_state(lane=0.95),
    )

    require(
        worsening.delta > 0.0,
        "driving worsening transition has positive delta",
    )


def verify_robotics_properties() -> None:
    nominal = evaluate_robotics_lyapunov(robotics_state())

    require_close(
        nominal.total,
        0.0,
        label="robotics nominal zero",
    )

    require(
        nominal.zero_risk_region,
        "robotics nominal zero-risk flag",
    )

    robot_critical = evaluate_robotics_lyapunov(robotics_state(robot_x=1.0))

    require_close(
        robot_critical.components["robot_workspace"],
        1.0,
        label="robot workspace critical normalization",
    )

    positive_robot = evaluate_robotics_lyapunov(robotics_state(robot_x=0.95))

    negative_robot = evaluate_robotics_lyapunov(robotics_state(robot_x=-0.95))

    require_close(
        positive_robot.components["robot_workspace"],
        negative_robot.components["robot_workspace"],
        label="robot workspace symmetry",
    )

    target_boundary = evaluate_robotics_lyapunov(robotics_state(target_x=1.0))

    require_close(
        target_boundary.components["target_workspace"],
        0.0,
        label="target boundary has zero target potential",
    )

    target_outside = evaluate_robotics_lyapunov(robotics_state(target_x=1.05))

    require(
        target_outside.components["target_workspace"] > 0.0,
        "target outside has positive target potential",
    )

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

    require_close(
        task_separation.total,
        0.0,
        label="task distance alone is not safety risk",
    )

    interaction_central = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.0,
            object_y=0.0,
            target_x=0.8,
            target_y=0.0,
        )
    )

    require_close(
        interaction_central.components["object_target_interaction"],
        0.0,
        label="central object interaction inactive",
    )

    interaction_boundary = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.95,
            object_y=0.0,
            target_x=0.0,
            target_y=0.0,
        )
    )

    require(
        interaction_boundary.components["object_target_interaction"] > 0.0,
        "boundary object interaction active",
    )

    safer = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.98),
        next_state=robotics_state(robot_x=0.85),
    )

    require(
        safer.delta < 0.0,
        "robotics recovery transition has negative delta",
    )

    worsening = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.85),
        next_state=robotics_state(robot_x=0.98),
    )

    require(
        worsening.delta > 0.0,
        "robotics worsening transition has positive delta",
    )


def verify_nonnegativity() -> None:
    driving_count = 0

    for speed in (
        0.0,
        0.5,
        1.0,
    ):
        for lane in (
            -1.0,
            0.0,
            1.0,
        ):
            for heading in (
                -0.8,
                0.0,
                0.8,
            ):
                for obstacle in (
                    0.1,
                    0.5,
                    1.0,
                ):
                    result = evaluate_driving_lyapunov(
                        driving_state(
                            speed=speed,
                            lane=lane,
                            heading=heading,
                            obstacle=obstacle,
                        )
                    )

                    require(
                        result.total >= 0.0,
                        ("driving nonnegative probe " f"{driving_count}"),
                    )

                    driving_count += 1

    robotics_count = 0

    values = (
        -1.05,
        0.0,
        1.05,
    )

    for robot_x in values:
        for object_x in values:
            for target_x in values:
                result = evaluate_robotics_lyapunov(
                    robotics_state(
                        robot_x=robot_x,
                        object_x=object_x,
                        target_x=target_x,
                    )
                )

                require(
                    result.total >= 0.0,
                    ("robotics nonnegative probe " f"{robotics_count}"),
                )

                robotics_count += 1


def verify_source_boundaries() -> None:
    source = LYAPUNOV_SOURCE_PATH.read_text(encoding="utf-8")

    forbidden_policy_terms = (
        "load_frozen_ppo_policy",
        "GaussianActorCritic",
        "deterministic_policy_action",
        "checkpoint_path",
        "hybrid_qml",
    )

    for term in forbidden_policy_terms:
        require(
            term not in source,
            f"foundation source excludes {term}",
        )

    forbidden_filter_functions = (
        "def filter_action",
        "def select_candidate",
        "def safe_action",
        "def correct_action",
        "def execute_action",
    )

    for term in forbidden_filter_functions:
        require(
            term not in source,
            f"foundation source excludes {term}",
        )


def verify_claims(
    artifact: dict[str, Any],
) -> None:
    claims = artifact["claims"]

    require(
        claims["deterministic_candidate_supported"] is True,
        "deterministic candidate supported",
    )

    require(
        claims["nonnegative_candidate_supported"] is True,
        "nonnegative candidate supported",
    )

    require(
        claims["synthetic_probe_monotonicity_supported"] is True,
        "synthetic monotonicity supported",
    )

    require(
        claims["lyapunov_filter_effectiveness"] == "NOT_YET_TESTED",
        "filter effectiveness remains NOT_YET_TESTED",
    )

    require(
        claims["lyapunov_violation_reduction"] == "NOT_YET_TESTED",
        "violation reduction remains NOT_YET_TESTED",
    )

    require(
        claims["lyapunov_vs_clipping"] == "NOT_YET_TESTED",
        "Lyapunov vs clipping remains NOT_YET_TESTED",
    )

    require(
        claims["robustness_improvement"] == "NOT_YET_TESTED",
        "robustness remains NOT_YET_TESTED",
    )

    require(
        claims["formal_stability_proof"] is False,
        "no formal stability proof",
    )

    require(
        claims["formal_safety_guarantee"] is False,
        "no formal safety guarantee",
    )

    require(
        claims["production_certification"] is False,
        "no production certification",
    )

    require(
        claims["quantum_safety_claim"] is False,
        "no quantum safety claim",
    )


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.6 LYAPUNOV FOUNDATION VERIFICATION")
    print("=" * 68)
    print()

    artifact = load_json(ARTIFACT_PATH)

    require(
        artifact["sprint"] == "5.6",
        "artifact sprint equals 5.6",
    )

    require(
        artifact["method"] == "classical_lyapunov_safety_candidate",
        "classical Lyapunov candidate method",
    )

    verify_provenance(artifact)

    verify_scope(artifact)

    verify_weights(artifact)

    verify_constants()

    verify_driving_properties()

    verify_robotics_properties()

    verify_nonnegativity()

    verify_source_boundaries()

    verify_claims(artifact)

    require(
        all(artifact["property_checks"].values()),
        "all builder property checks pass",
    )

    print()
    print("Source provenance: PASS")
    print("Driving candidate: PASS")
    print("Robotics candidate: PASS")
    print("Non-negativity: PASS")
    print("Monotonicity: PASS")
    print("Delta behavior: PASS")
    print("Policy independence: PASS")
    print("Action-filter absence: PASS")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.6 LYAPUNOV FOUNDATION: PASS")


if __name__ == "__main__":
    main()
