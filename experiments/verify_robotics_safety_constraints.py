"""Independent verification for Sprint 5.3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from q_vla_forge.safety.contracts import (
    ViolationSeverity,
)
from q_vla_forge.safety.robotics_constraints import (
    GRASP_SAFE_DISTANCE,
    GRIPPER_CLOSE_THRESHOLD,
    GRIPPER_CRITICAL_DISTANCE,
    MOVEMENT_SCALE,
    OBJECT_BOUNDARY_CRITICAL_ABS,
    OBJECT_BOUNDARY_WARNING_ABS,
    OBJECT_TARGET_CRITICAL_DISTANCE,
    OBJECT_TARGET_WARNING_DISTANCE,
    VIOLATION_CATEGORIES,
    WORKSPACE_CRITICAL_ABS,
    WORKSPACE_WARNING_ABS,
    count_robotics_violations,
    evaluate_robotics_violations,
    predicted_robot_position,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "robotics_safety.yaml"

ARTIFACT_PATH = (
    ROOT
    / "results"
    / "safety"
    / "robotics"
    / "sprint5-robotics-safety-constraints.json"
)

PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def by_name(
    state: np.ndarray,
    action: np.ndarray,
):
    return {
        record.name: record
        for record in evaluate_robotics_violations(
            state,
            action,
        )
    }


def main() -> None:
    require(
        PROTOCOL_PATH.exists(),
        "Sprint 5.1 protocol exists",
    )

    require(
        CONFIG_PATH.exists(),
        "Robotics safety config exists",
    )

    require(
        ARTIFACT_PATH.exists(),
        "Robotics safety artifact exists",
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    require(
        artifact["state_contract"]["dimension"] == 6,
        "Robotics state dimension is 6",
    )

    require(
        artifact["action_contract"]["dimension"] == 3,
        "Robotics action dimension is 3",
    )

    require(
        tuple(artifact["violation_categories"]) == VIOLATION_CATEGORIES,
        "Exactly five robotics categories are frozen",
    )

    require(
        WORKSPACE_WARNING_ABS == 0.90,
        "Workspace warning is frozen",
    )

    require(
        WORKSPACE_CRITICAL_ABS == 1.00,
        "Workspace critical is frozen",
    )

    require(
        MOVEMENT_SCALE == 0.08,
        "Movement scale matches frozen environment",
    )

    require(
        GRIPPER_CLOSE_THRESHOLD == 0.50,
        "Gripper close threshold is frozen",
    )

    require(
        GRASP_SAFE_DISTANCE == 0.12,
        "Grasp distance matches frozen environment",
    )

    require(
        GRIPPER_CRITICAL_DISTANCE == 0.30,
        "Gripper critical distance is frozen",
    )

    require(
        OBJECT_BOUNDARY_WARNING_ABS == 0.90,
        "Object warning boundary is frozen",
    )

    require(
        OBJECT_BOUNDARY_CRITICAL_ABS == 1.00,
        "Object critical boundary is frozen",
    )

    require(
        OBJECT_TARGET_WARNING_DISTANCE == 0.50,
        "Object-target warning distance is frozen",
    )

    require(
        OBJECT_TARGET_CRITICAL_DISTANCE == 1.00,
        "Object-target critical distance is frozen",
    )

    nominal_state = np.array(
        [
            0.0,
            0.0,
            0.2,
            0.0,
            0.4,
            0.0,
        ],
        dtype=np.float32,
    )

    neutral_action = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    require(
        count_robotics_violations(
            nominal_state,
            neutral_action,
        )
        == 0,
        "Nominal center case is safe",
    )

    inward_state = np.array(
        [
            0.95,
            0.0,
            0.2,
            0.0,
            0.4,
            0.0,
        ],
        dtype=np.float32,
    )

    inward_action = np.array(
        [
            -1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    inward = by_name(
        inward_state,
        inward_action,
    )

    require(
        inward["workspace_boundary"].violated,
        "Boundary state is detected",
    )

    require(
        not inward["unsafe_motion"].violated,
        "Inward recovery motion remains allowed",
    )

    outward_state = np.array(
        [
            0.88,
            0.0,
            0.2,
            0.0,
            0.4,
            0.0,
        ],
        dtype=np.float32,
    )

    outward_action = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    predicted_x, predicted_y = predicted_robot_position(
        outward_state,
        outward_action,
    )

    require(
        np.isclose(
            predicted_x,
            0.96,
        )
        and np.isclose(
            predicted_y,
            0.0,
        ),
        "Predicted robot position is correct",
    )

    require(
        by_name(
            outward_state,
            outward_action,
        )["unsafe_motion"].violated,
        "Outward boundary motion is detected",
    )

    workspace_critical = by_name(
        np.array(
            [
                1.05,
                0.0,
                0.2,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["workspace_boundary"]

    require(
        workspace_critical.violated
        and workspace_critical.severity == ViolationSeverity.CRITICAL,
        "Known workspace-critical case verified",
    )

    far_gripper = by_name(
        np.array(
            [
                0.0,
                0.0,
                0.5,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )["unsafe_gripper_condition"]

    require(
        far_gripper.violated,
        "Far gripper close is unsafe",
    )

    near_gripper = by_name(
        np.array(
            [
                0.0,
                0.0,
                0.10,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )["unsafe_gripper_condition"]

    require(
        not near_gripper.violated,
        "Near gripper close remains safe",
    )

    object_boundary = by_name(
        np.array(
            [
                0.0,
                0.0,
                0.95,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["object_boundary"]

    require(
        object_boundary.violated,
        "Object boundary warning is detected",
    )

    target_outside = by_name(
        np.array(
            [
                0.0,
                0.0,
                0.2,
                0.0,
                1.10,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["unsafe_object_target_interaction"]

    require(
        target_outside.violated
        and target_outside.severity == ViolationSeverity.CRITICAL,
        "Target outside workspace is critical",
    )

    interior_far_target = by_name(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["unsafe_object_target_interaction"]

    require(
        not interior_far_target.violated,
        "Task distance is not automatically unsafe",
    )

    positive = by_name(
        np.array(
            [
                0.95,
                0.0,
                0.2,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["workspace_boundary"]

    negative = by_name(
        np.array(
            [
                -0.95,
                0.0,
                0.2,
                0.0,
                0.4,
                0.0,
            ],
            dtype=np.float32,
        ),
        neutral_action,
    )["workspace_boundary"]

    require(
        positive.violated == negative.violated,
        "Workspace constraint is sign symmetric",
    )

    state = outward_state.copy()
    action = outward_action.copy()

    before_state = state.copy()
    before_action = action.copy()

    evaluate_robotics_violations(
        state,
        action,
    )

    require(
        np.array_equal(
            state,
            before_state,
        )
        and np.array_equal(
            action,
            before_action,
        ),
        "Constraint evaluator does not mutate inputs",
    )

    require(
        artifact["action_contract"]["constraint_evaluator_clips_actions"] is False,
        "Constraint evaluator does not clip actions",
    )

    require(
        config["claims"]["safety_improvement_demonstrated"] is False,
        "No robotics safety-improvement claim exists",
    )

    require(
        config["claims"]["iso_10218_certification"] is False,
        "No ISO 10218 certification claim exists",
    )

    require(
        config["claims"]["iec_62061_certification"] is False,
        "No IEC 62061 certification claim exists",
    )

    require(
        config["claims"]["formal_safety_guarantee"] is False,
        "No formal safety guarantee exists",
    )

    results = artifact["results"]

    require(
        results["safety_results_present"] is False,
        "No robotics safety results exist",
    )

    require(
        results["safety_filter_evaluated"] is False,
        "No safety filter has been evaluated",
    )

    serialized = json.dumps(artifact).lower()

    require(
        "best_method" not in serialized,
        "No winner field exists",
    )

    print()
    print("=" * 62)
    print(" SPRINT 5.3 — ROBOTICS SAFETY CONSTRAINTS")
    print("=" * 62)
    print()
    print("Constraint taxonomy: PASS")
    print("Environment alignment: PASS")
    print("Workspace safety: PASS")
    print("Outward-motion semantics: PASS")
    print("Gripper safety: PASS")
    print("Object boundary: PASS")
    print("Object/target separation: PASS")
    print("Symmetry: PASS")
    print("Purity: PASS")
    print("Claim controls: PASS")
    print("Premature-result protection: PASS")
    print()
    print("SPRINT 5.3 ROBOTICS SAFETY CONSTRAINTS: PASS")


if __name__ == "__main__":
    main()
