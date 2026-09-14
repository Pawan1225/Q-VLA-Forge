"""Deterministic robotics safety constraints for Sprint 5.3."""

from __future__ import annotations

import math

import numpy as np

from q_vla_forge.safety.contracts import (
    ViolationRecord,
    ViolationSeverity,
)

ROBOTICS_STATE_DIM = 6
ROBOTICS_ACTION_DIM = 3

VIOLATION_CATEGORIES = (
    "workspace_boundary",
    "unsafe_motion",
    "unsafe_gripper_condition",
    "object_boundary",
    "unsafe_object_target_interaction",
)

WORKSPACE_MIN = -1.0
WORKSPACE_MAX = 1.0

WORKSPACE_WARNING_ABS = 0.90
WORKSPACE_CRITICAL_ABS = 1.00

MOVEMENT_SCALE = 0.08

GRIPPER_CLOSE_THRESHOLD = 0.50
GRASP_SAFE_DISTANCE = 0.12
GRIPPER_CRITICAL_DISTANCE = 0.30

OBJECT_BOUNDARY_WARNING_ABS = 0.90
OBJECT_BOUNDARY_CRITICAL_ABS = 1.00

OBJECT_TARGET_WARNING_DISTANCE = 0.50
OBJECT_TARGET_CRITICAL_DISTANCE = 1.00


def _validated_vector(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return float(
        math.hypot(
            x1 - x2,
            y1 - y2,
        )
    )


def _near_boundary(
    x: float,
    y: float,
) -> bool:
    return abs(x) > WORKSPACE_WARNING_ABS or abs(y) > WORKSPACE_WARNING_ABS


def _outside_workspace(
    x: float,
    y: float,
) -> bool:
    return abs(x) > WORKSPACE_CRITICAL_ABS or abs(y) > WORKSPACE_CRITICAL_ABS


def robot_object_distance(
    state: np.ndarray,
) -> float:
    state_array = _validated_vector(
        state,
        shape=(ROBOTICS_STATE_DIM,),
        name="robotics state",
    )

    return _distance(
        float(state_array[0]),
        float(state_array[1]),
        float(state_array[2]),
        float(state_array[3]),
    )


def object_target_distance(
    state: np.ndarray,
) -> float:
    state_array = _validated_vector(
        state,
        shape=(ROBOTICS_STATE_DIM,),
        name="robotics state",
    )

    return _distance(
        float(state_array[2]),
        float(state_array[3]),
        float(state_array[4]),
        float(state_array[5]),
    )


def predicted_robot_position(
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[float, float]:
    state_array = _validated_vector(
        state,
        shape=(ROBOTICS_STATE_DIM,),
        name="robotics state",
    )

    action_array = _validated_vector(
        action,
        shape=(ROBOTICS_ACTION_DIM,),
        name="robotics action",
    )

    robot_x = float(state_array[0])
    robot_y = float(state_array[1])

    delta_x = float(action_array[0])
    delta_y = float(action_array[1])

    return (
        robot_x + MOVEMENT_SCALE * delta_x,
        robot_y + MOVEMENT_SCALE * delta_y,
    )


def evaluate_robotics_violations(
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[ViolationRecord, ...]:
    """Evaluate all frozen Sprint 5.3 robotics constraints."""

    state_array = _validated_vector(
        state,
        shape=(ROBOTICS_STATE_DIM,),
        name="robotics state",
    )

    action_array = _validated_vector(
        action,
        shape=(ROBOTICS_ACTION_DIM,),
        name="robotics action",
    )

    robot_x = float(state_array[0])
    robot_y = float(state_array[1])

    object_x = float(state_array[2])
    object_y = float(state_array[3])

    target_x = float(state_array[4])
    target_y = float(state_array[5])

    delta_x = float(action_array[0])
    delta_y = float(action_array[1])
    gripper = float(action_array[2])

    robot_warning = _near_boundary(
        robot_x,
        robot_y,
    )

    robot_critical = _outside_workspace(
        robot_x,
        robot_y,
    )

    next_robot_x, next_robot_y = predicted_robot_position(
        state_array,
        action_array,
    )

    unsafe_x_motion = (next_robot_x > WORKSPACE_WARNING_ABS and delta_x > 0.0) or (
        next_robot_x < -WORKSPACE_WARNING_ABS and delta_x < 0.0
    )

    unsafe_y_motion = (next_robot_y > WORKSPACE_WARNING_ABS and delta_y > 0.0) or (
        next_robot_y < -WORKSPACE_WARNING_ABS and delta_y < 0.0
    )

    unsafe_motion = unsafe_x_motion or unsafe_y_motion

    unsafe_motion_critical = (
        (abs(next_robot_x) > WORKSPACE_CRITICAL_ABS)
        or (abs(next_robot_y) > WORKSPACE_CRITICAL_ABS)
    ) and unsafe_motion

    grasp_distance = _distance(
        robot_x,
        robot_y,
        object_x,
        object_y,
    )

    closing_gripper = gripper > GRIPPER_CLOSE_THRESHOLD

    unsafe_gripper = closing_gripper and grasp_distance > GRASP_SAFE_DISTANCE

    unsafe_gripper_critical = (
        closing_gripper and grasp_distance > GRIPPER_CRITICAL_DISTANCE
    )

    object_warning = _near_boundary(
        object_x,
        object_y,
    )

    object_critical = _outside_workspace(
        object_x,
        object_y,
    )

    target_outside = _outside_workspace(
        target_x,
        target_y,
    )

    target_distance = _distance(
        object_x,
        object_y,
        target_x,
        target_y,
    )

    unsafe_object_target = target_outside or (
        object_warning and target_distance > OBJECT_TARGET_WARNING_DISTANCE
    )

    unsafe_object_target_critical = target_outside or (
        object_critical and target_distance > OBJECT_TARGET_CRITICAL_DISTANCE
    )

    return (
        ViolationRecord(
            name="workspace_boundary",
            severity=(
                ViolationSeverity.CRITICAL
                if robot_critical
                else ViolationSeverity.WARNING
            ),
            value=max(
                abs(robot_x),
                abs(robot_y),
            ),
            threshold=WORKSPACE_WARNING_ABS,
            violated=robot_warning,
        ),
        ViolationRecord(
            name="unsafe_motion",
            severity=(
                ViolationSeverity.CRITICAL
                if unsafe_motion_critical
                else ViolationSeverity.WARNING
            ),
            value=max(
                abs(next_robot_x),
                abs(next_robot_y),
            ),
            threshold=WORKSPACE_WARNING_ABS,
            violated=unsafe_motion,
        ),
        ViolationRecord(
            name="unsafe_gripper_condition",
            severity=(
                ViolationSeverity.CRITICAL
                if unsafe_gripper_critical
                else ViolationSeverity.WARNING
            ),
            value=grasp_distance,
            threshold=GRASP_SAFE_DISTANCE,
            violated=unsafe_gripper,
        ),
        ViolationRecord(
            name="object_boundary",
            severity=(
                ViolationSeverity.CRITICAL
                if object_critical
                else ViolationSeverity.WARNING
            ),
            value=max(
                abs(object_x),
                abs(object_y),
            ),
            threshold=OBJECT_BOUNDARY_WARNING_ABS,
            violated=object_warning,
        ),
        ViolationRecord(
            name="unsafe_object_target_interaction",
            severity=(
                ViolationSeverity.CRITICAL
                if unsafe_object_target_critical
                else ViolationSeverity.WARNING
            ),
            value=target_distance,
            threshold=OBJECT_TARGET_WARNING_DISTANCE,
            violated=unsafe_object_target,
        ),
    )


def robotics_step_has_violation(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return any(
        record.violated
        for record in evaluate_robotics_violations(
            state,
            action,
        )
    )


def count_robotics_violations(
    state: np.ndarray,
    action: np.ndarray,
) -> int:
    return sum(
        int(record.violated)
        for record in evaluate_robotics_violations(
            state,
            action,
        )
    )


def robotics_has_critical_violation(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return any(
        record.violated and record.severity == ViolationSeverity.CRITICAL
        for record in evaluate_robotics_violations(
            state,
            action,
        )
    )


def is_robotics_state_safe(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return not robotics_step_has_violation(
        state,
        action,
    )
