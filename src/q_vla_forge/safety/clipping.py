"""Deterministic heuristic constraint clipping for Sprint 5.5."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyDecision,
    SafetyMethod,
)
from q_vla_forge.safety.driving_constraints import (
    evaluate_driving_violations,
)
from q_vla_forge.safety.robotics_constraints import (
    evaluate_robotics_violations,
)

# Frozen Sprint 5.2 driving thresholds.
DRIVING_LANE_WARNING_ABS = 0.75
DRIVING_RISKY_LANE_ABS = 0.60
DRIVING_RISKY_HEADING_ABS = 0.50
DRIVING_STEERING_WARNING_ABS = 0.80

DRIVING_SAFE_DISTANCE_BASE = 0.15
DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT = 0.35
DRIVING_CRITICAL_DISTANCE_FRACTION = 0.50

DRIVING_ACCEL_BRAKE_CONFLICT_THRESHOLD = 0.25

# Sprint 5.5 clipping-only intervention constants.
DRIVING_WARNING_MINIMUM_BRAKING = 0.50
DRIVING_CRITICAL_MINIMUM_BRAKING = 1.00

# Frozen Sprint 5.3 robotics thresholds.
ROBOTICS_WORKSPACE_WARNING_ABS = 0.90
ROBOTICS_MOVEMENT_SCALE = 0.08
ROBOTICS_GRIPPER_CLOSE_THRESHOLD = 0.50
ROBOTICS_SAFE_GRASP_DISTANCE = 0.12

INTERVENTION_TOLERANCE_L2 = 1e-8


def _validated_vector(
    value: np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.size != expected_size:
        raise ValueError(f"{name} must contain exactly {expected_size} values")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array.copy()


def required_safe_distance(
    speed: float,
) -> float:
    """Frozen Sprint 5.2 heuristic speed-distance envelope."""

    if not math.isfinite(float(speed)):
        raise ValueError("speed must be finite")

    return float(
        DRIVING_SAFE_DISTANCE_BASE
        + DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT
        * max(
            float(speed),
            0.0,
        )
    )


def _append_rule(
    rules: list[str],
    rule: str,
) -> None:
    if rule not in rules:
        rules.append(rule)


def clip_driving_action(
    *,
    state: np.ndarray,
    proposed_action: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    dict[str, Any],
]:
    """Apply frozen Sprint 5.5 driving clipping rules."""

    state_array = _validated_vector(
        state,
        expected_size=4,
        name="driving state",
    )

    original_action = _validated_vector(
        proposed_action,
        expected_size=3,
        name="driving action",
    )

    action = original_action.copy()

    triggered_rules: list[str] = []

    speed = float(state_array[0])
    lane_offset = float(state_array[1])
    heading_error = float(state_array[2])
    obstacle_distance = float(state_array[3])

    # 1. Valid action bounds.
    bounded = np.array(
        [
            np.clip(
                action[0],
                -1.0,
                1.0,
            ),
            np.clip(
                action[1],
                -1.0,
                1.0,
            ),
            np.clip(
                action[2],
                0.0,
                1.0,
            ),
        ],
        dtype=np.float64,
    )

    if not np.array_equal(
        bounded,
        action,
    ):
        _append_rule(
            triggered_rules,
            "action_bounds",
        )

    action = bounded

    # 2. Suppress outward steering while preserving inward recovery.
    if (
        lane_offset > DRIVING_LANE_WARNING_ABS
        and action[0] > 0.0
        or lane_offset < -DRIVING_LANE_WARNING_ABS
        and action[0] < 0.0
    ):
        action[0] = 0.0
        _append_rule(
            triggered_rules,
            "outward_lane_steering",
        )

    # 3. Contextual steering cap.
    if (
        abs(lane_offset) > DRIVING_RISKY_LANE_ABS
        or abs(heading_error) > DRIVING_RISKY_HEADING_ABS
    ):
        clipped_steering = float(
            np.clip(
                action[0],
                -DRIVING_STEERING_WARNING_ABS,
                DRIVING_STEERING_WARNING_ABS,
            )
        )

        if clipped_steering != action[0]:
            action[0] = clipped_steering

            _append_rule(
                triggered_rules,
                "contextual_steering_cap",
            )

    # 4-5. Speed-distance acceleration suppression and braking.
    safe_distance = required_safe_distance(speed)

    critical_distance = DRIVING_CRITICAL_DISTANCE_FRACTION * safe_distance

    obstacle_risk = obstacle_distance < safe_distance

    critical_obstacle_risk = obstacle_distance < critical_distance

    if obstacle_risk:
        if action[1] > 0.0:
            action[1] = 0.0

            _append_rule(
                triggered_rules,
                "speed_distance_acceleration",
            )

        minimum_braking = (
            DRIVING_CRITICAL_MINIMUM_BRAKING
            if critical_obstacle_risk
            else DRIVING_WARNING_MINIMUM_BRAKING
        )

        if action[2] < minimum_braking:
            action[2] = minimum_braking

            _append_rule(
                triggered_rules,
                "speed_distance_braking",
            )

    # 6. Acceleration/braking conflict resolution.
    if (
        action[1] > DRIVING_ACCEL_BRAKE_CONFLICT_THRESHOLD
        and action[2] > DRIVING_ACCEL_BRAKE_CONFLICT_THRESHOLD
    ):
        action[1] = 0.0

        _append_rule(
            triggered_rules,
            "accel_brake_conflict",
        )

    # 7. Final bounds.
    final_action = np.array(
        [
            np.clip(
                action[0],
                -1.0,
                1.0,
            ),
            np.clip(
                action[1],
                -1.0,
                1.0,
            ),
            np.clip(
                action[2],
                0.0,
                1.0,
            ),
        ],
        dtype=np.float64,
    )

    if not np.array_equal(
        final_action,
        action,
    ):
        _append_rule(
            triggered_rules,
            "final_action_bounds",
        )

    metadata: dict[
        str,
        Any,
    ] = {
        "required_safe_distance": (safe_distance),
        "critical_safe_distance": (critical_distance),
        "obstacle_risk": (obstacle_risk),
        "critical_obstacle_risk": (critical_obstacle_risk),
    }

    return (
        final_action,
        tuple(triggered_rules),
        metadata,
    )


def _clip_workspace_axis(
    *,
    position: float,
    command: float,
    movement_scale: float = ROBOTICS_MOVEMENT_SCALE,
    warning_abs: float = ROBOTICS_WORKSPACE_WARNING_ABS,
) -> tuple[
    float,
    bool,
]:
    """Clip only outward workspace motion; preserve inward recovery."""

    if not all(
        math.isfinite(float(value))
        for value in (
            position,
            command,
            movement_scale,
            warning_abs,
        )
    ):
        raise ValueError("workspace clipping inputs must be finite")

    if movement_scale <= 0.0:
        raise ValueError("movement scale must be positive")

    command = float(
        np.clip(
            command,
            -1.0,
            1.0,
        )
    )

    # Already outside positive warning region.
    if position > warning_abs and command > 0.0:
        return (
            0.0,
            True,
        )

    # Already outside negative warning region.
    if position < -warning_abs and command < 0.0:
        return (
            0.0,
            True,
        )

    predicted = position + movement_scale * command

    if command > 0.0 and predicted > warning_abs:
        allowed = (warning_abs - position) / movement_scale

        allowed = float(
            np.clip(
                allowed,
                -1.0,
                1.0,
            )
        )

        return (
            allowed,
            True,
        )

    if command < 0.0 and predicted < -warning_abs:
        allowed = (-warning_abs - position) / movement_scale

        allowed = float(
            np.clip(
                allowed,
                -1.0,
                1.0,
            )
        )

        return (
            allowed,
            True,
        )

    return (
        command,
        False,
    )


def clip_robotics_action(
    *,
    state: np.ndarray,
    proposed_action: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    dict[str, Any],
]:
    """Apply frozen Sprint 5.5 robotics clipping rules."""

    state_array = _validated_vector(
        state,
        expected_size=6,
        name="robotics state",
    )

    original_action = _validated_vector(
        proposed_action,
        expected_size=3,
        name="robotics action",
    )

    action = np.clip(
        original_action,
        -1.0,
        1.0,
    )

    triggered_rules: list[str] = []

    if not np.array_equal(
        action,
        original_action,
    ):
        _append_rule(
            triggered_rules,
            "action_bounds",
        )

    robot_x = float(state_array[0])
    robot_y = float(state_array[1])
    object_x = float(state_array[2])
    object_y = float(state_array[3])

    before_predicted = [
        (robot_x + ROBOTICS_MOVEMENT_SCALE * float(action[0])),
        (robot_y + ROBOTICS_MOVEMENT_SCALE * float(action[1])),
    ]

    clipped_x, changed_x = _clip_workspace_axis(
        position=robot_x,
        command=float(action[0]),
    )

    if changed_x:
        action[0] = clipped_x

        _append_rule(
            triggered_rules,
            "workspace_x_clip",
        )

    clipped_y, changed_y = _clip_workspace_axis(
        position=robot_y,
        command=float(action[1]),
    )

    if changed_y:
        action[1] = clipped_y

        _append_rule(
            triggered_rules,
            "workspace_y_clip",
        )

    robot_object_distance = math.hypot(
        robot_x - object_x,
        robot_y - object_y,
    )

    if (
        action[2] > ROBOTICS_GRIPPER_CLOSE_THRESHOLD
        and robot_object_distance > ROBOTICS_SAFE_GRASP_DISTANCE
    ):
        action[2] = 0.0

        _append_rule(
            triggered_rules,
            "unsafe_gripper_close",
        )

    final_action = np.clip(
        action,
        -1.0,
        1.0,
    )

    if not np.array_equal(
        final_action,
        action,
    ):
        _append_rule(
            triggered_rules,
            "final_action_bounds",
        )

    after_predicted = [
        (robot_x + ROBOTICS_MOVEMENT_SCALE * float(final_action[0])),
        (robot_y + ROBOTICS_MOVEMENT_SCALE * float(final_action[1])),
    ]

    metadata: dict[
        str,
        Any,
    ] = {
        "robot_object_distance": (robot_object_distance),
        "predicted_position_before": (before_predicted),
        "predicted_position_after": (after_predicted),
    }

    return (
        np.asarray(
            final_action,
            dtype=np.float64,
        ),
        tuple(triggered_rules),
        metadata,
    )


def apply_clipping_safety_filter(
    *,
    domain: str,
    true_state: np.ndarray,
    proposed_action: np.ndarray,
) -> SafetyDecision:
    """Apply the deterministic Sprint 5.5 clipping safety filter."""

    proposed = np.asarray(
        proposed_action,
        dtype=np.float64,
    )

    if domain == "autonomous_driving":
        proposed = _validated_vector(
            proposed,
            expected_size=3,
            name="driving action",
        )

        state = _validated_vector(
            true_state,
            expected_size=4,
            name="driving state",
        )

        violations_before = evaluate_driving_violations(
            state,
            proposed,
        )

        (
            executed,
            triggered_rules,
            domain_metadata,
        ) = clip_driving_action(
            state=state,
            proposed_action=proposed,
        )

        violations_after = evaluate_driving_violations(
            state,
            executed,
        )

    elif domain == "robotics":
        proposed = _validated_vector(
            proposed,
            expected_size=3,
            name="robotics action",
        )

        state = _validated_vector(
            true_state,
            expected_size=6,
            name="robotics state",
        )

        violations_before = evaluate_robotics_violations(
            state,
            proposed,
        )

        (
            executed,
            triggered_rules,
            domain_metadata,
        ) = clip_robotics_action(
            state=state,
            proposed_action=proposed,
        )

        violations_after = evaluate_robotics_violations(
            state,
            executed,
        )

    else:
        raise ValueError(f"unsupported domain: {domain}")

    correction_l2 = float(np.linalg.norm(executed - proposed))

    intervened = bool(correction_l2 > INTERVENTION_TOLERANCE_L2)

    domain_rules = tuple(
        rule
        for rule in triggered_rules
        if rule
        not in {
            "action_bounds",
            "final_action_bounds",
        }
    )

    if not intervened:
        reason = InterventionReason.NONE

    elif domain_rules:
        reason = InterventionReason.DOMAIN_CONSTRAINT

    else:
        reason = InterventionReason.ACTION_BOUND

    before_count = sum(int(record.violated) for record in violations_before)

    after_count = sum(int(record.violated) for record in violations_after)

    metadata = {
        **domain_metadata,
        "triggered_rules": list(triggered_rules),
        "rule_count": len(triggered_rules),
        "pre_filter_violation_count": (before_count),
        "post_filter_violation_count": (after_count),
        "immediate_violation_reduction": (before_count - after_count),
        "intervention_tolerance_l2": (INTERVENTION_TOLERANCE_L2),
    }

    return SafetyDecision(
        method=(SafetyMethod.CLIPPING),
        proposed_action=(proposed.copy()),
        executed_action=(executed.copy()),
        intervened=intervened,
        intervention_reason=(reason),
        correction_l2=(correction_l2),
        violations_before=(violations_before),
        violations_after=(violations_after),
        metadata=metadata,
    )
