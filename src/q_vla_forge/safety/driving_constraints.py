"""Deterministic driving safety constraints for Sprint 5.2."""

from __future__ import annotations

import numpy as np

from q_vla_forge.safety.contracts import (
    ViolationRecord,
    ViolationSeverity,
)

DRIVING_STATE_DIM = 4
DRIVING_ACTION_DIM = 3

VIOLATION_CATEGORIES = (
    "lane_boundary",
    "unsafe_obstacle_distance",
    "unsafe_speed_condition",
    "steering_risk",
    "acceleration_braking_conflict",
)

LANE_WARNING_ABS = 0.75
LANE_CRITICAL_ABS = 0.95

OBSTACLE_WARNING_DISTANCE = 0.30
OBSTACLE_CRITICAL_DISTANCE = 0.15

SAFE_DISTANCE_BASE = 0.15
SAFE_DISTANCE_SPEED_SCALE = 0.35
SAFE_DISTANCE_CRITICAL_FRACTION = 0.50

STEERING_WARNING_ABS = 0.80
STEERING_CRITICAL_ABS = 0.95

LANE_STEERING_RISK_ABS = 0.60
HEADING_STEERING_RISK_ABS = 0.50

LANE_STEERING_CRITICAL_ABS = 0.80
HEADING_STEERING_CRITICAL_ABS = 0.75

ACCEL_BRAKE_CONFLICT_THRESHOLD = 0.25
ACCEL_BRAKE_CRITICAL_THRESHOLD = 0.60


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


def required_safe_distance(speed: float) -> float:
    """Return the pilot speed-distance safety envelope."""

    speed_value = float(speed)

    if not np.isfinite(speed_value):
        raise ValueError("speed must be finite")

    return float(SAFE_DISTANCE_BASE + SAFE_DISTANCE_SPEED_SCALE * max(speed_value, 0.0))


def evaluate_driving_violations(
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[ViolationRecord, ...]:
    """Evaluate all frozen Sprint 5.2 driving constraints."""

    state_array = _validated_vector(
        state,
        shape=(DRIVING_STATE_DIM,),
        name="driving state",
    )
    action_array = _validated_vector(
        action,
        shape=(DRIVING_ACTION_DIM,),
        name="driving action",
    )

    speed = float(state_array[0])
    lane_offset = float(state_array[1])
    heading_error = float(state_array[2])
    obstacle_distance = float(state_array[3])

    steering = float(action_array[0])
    acceleration = float(action_array[1])
    braking = float(action_array[2])

    abs_lane = abs(lane_offset)
    abs_heading = abs(heading_error)
    abs_steering = abs(steering)

    lane_violated = abs_lane > LANE_WARNING_ABS
    lane_critical = abs_lane > LANE_CRITICAL_ABS

    obstacle_violated = obstacle_distance < OBSTACLE_WARNING_DISTANCE
    obstacle_critical = obstacle_distance < OBSTACLE_CRITICAL_DISTANCE

    safe_distance = required_safe_distance(speed)

    speed_condition_violated = obstacle_distance < safe_distance
    speed_condition_critical = (
        obstacle_distance < SAFE_DISTANCE_CRITICAL_FRACTION * safe_distance
    )

    steering_violated = abs_steering > STEERING_WARNING_ABS and (
        abs_lane > LANE_STEERING_RISK_ABS or abs_heading > HEADING_STEERING_RISK_ABS
    )

    steering_critical = abs_steering > STEERING_CRITICAL_ABS and (
        abs_lane > LANE_STEERING_CRITICAL_ABS
        or abs_heading > HEADING_STEERING_CRITICAL_ABS
    )

    conflict_violated = (
        acceleration > ACCEL_BRAKE_CONFLICT_THRESHOLD
        and braking > ACCEL_BRAKE_CONFLICT_THRESHOLD
    )

    conflict_critical = (
        acceleration > ACCEL_BRAKE_CRITICAL_THRESHOLD
        and braking > ACCEL_BRAKE_CRITICAL_THRESHOLD
    )

    return (
        ViolationRecord(
            name="lane_boundary",
            severity=(
                ViolationSeverity.CRITICAL
                if lane_critical
                else ViolationSeverity.WARNING
            ),
            value=abs_lane,
            threshold=LANE_WARNING_ABS,
            violated=lane_violated,
        ),
        ViolationRecord(
            name="unsafe_obstacle_distance",
            severity=(
                ViolationSeverity.CRITICAL
                if obstacle_critical
                else ViolationSeverity.WARNING
            ),
            value=obstacle_distance,
            threshold=OBSTACLE_WARNING_DISTANCE,
            violated=obstacle_violated,
        ),
        ViolationRecord(
            name="unsafe_speed_condition",
            severity=(
                ViolationSeverity.CRITICAL
                if speed_condition_critical
                else ViolationSeverity.WARNING
            ),
            value=obstacle_distance,
            threshold=safe_distance,
            violated=speed_condition_violated,
        ),
        ViolationRecord(
            name="steering_risk",
            severity=(
                ViolationSeverity.CRITICAL
                if steering_critical
                else ViolationSeverity.WARNING
            ),
            value=abs_steering,
            threshold=STEERING_WARNING_ABS,
            violated=steering_violated,
        ),
        ViolationRecord(
            name="acceleration_braking_conflict",
            severity=(
                ViolationSeverity.CRITICAL
                if conflict_critical
                else ViolationSeverity.WARNING
            ),
            value=min(acceleration, braking),
            threshold=ACCEL_BRAKE_CONFLICT_THRESHOLD,
            violated=conflict_violated,
        ),
    )


def driving_step_has_violation(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return any(
        record.violated
        for record in evaluate_driving_violations(
            state,
            action,
        )
    )


def count_driving_violations(
    state: np.ndarray,
    action: np.ndarray,
) -> int:
    return sum(
        int(record.violated)
        for record in evaluate_driving_violations(
            state,
            action,
        )
    )


def driving_has_critical_violation(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return any(
        record.violated and record.severity == ViolationSeverity.CRITICAL
        for record in evaluate_driving_violations(
            state,
            action,
        )
    )


def is_driving_state_safe(
    state: np.ndarray,
    action: np.ndarray,
) -> bool:
    return not driving_step_has_violation(
        state,
        action,
    )
