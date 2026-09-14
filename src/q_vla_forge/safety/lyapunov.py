"""Lyapunov safety candidate functions for Sprint 5.6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1.0e-12

DRIVING_LANE_WARNING_ABS = 0.75
DRIVING_LANE_CRITICAL_ABS = 0.95

DRIVING_HEADING_WARNING_ABS = 0.50
DRIVING_HEADING_CRITICAL_ABS = 0.75

DRIVING_OBSTACLE_WARNING_DISTANCE = 0.30
DRIVING_OBSTACLE_CRITICAL_DISTANCE = 0.15

DRIVING_SAFE_DISTANCE_BASE = 0.15
DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT = 0.35

ROBOTICS_WORKSPACE_WARNING_ABS = 0.90
ROBOTICS_WORKSPACE_CRITICAL_ABS = 1.00

ROBOTICS_OBJECT_WARNING_ABS = 0.90
ROBOTICS_OBJECT_CRITICAL_ABS = 1.00

ROBOTICS_TARGET_CRITICAL_ABS = 1.00

ROBOTICS_OBJECT_TARGET_WARNING_DISTANCE = 0.50
ROBOTICS_OBJECT_TARGET_CRITICAL_DISTANCE = 1.00


@dataclass(frozen=True)
class LyapunovEvaluation:
    """Evaluation of a domain-specific Lyapunov safety candidate."""

    domain: str
    total: float
    components: dict[str, float]
    zero_risk_region: bool


@dataclass(frozen=True)
class LyapunovDelta:
    """Change in Lyapunov safety potential between two states."""

    current_value: float
    next_value: float
    delta: float


def _validated_state(
    state: np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        state,
        dtype=np.float64,
    )

    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array.copy()


def positive_excess(
    value: float,
    threshold: float,
    scale: float,
) -> float:
    """Return normalized positive threshold excess."""

    value = float(value)
    threshold = float(threshold)
    scale = float(scale)

    if not np.isfinite(value):
        raise ValueError("value must be finite")

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if not np.isfinite(scale):
        raise ValueError("scale must be finite")

    if scale <= 0.0:
        raise ValueError("scale must be positive")

    return float(
        max(
            0.0,
            value - threshold,
        )
        / scale
    )


def required_safe_distance(
    speed: float,
) -> float:
    """Frozen Sprint 5.2 heuristic speed-distance envelope."""

    speed = float(speed)

    if not np.isfinite(speed):
        raise ValueError("speed must be finite")

    return float(
        DRIVING_SAFE_DISTANCE_BASE
        + DRIVING_SAFE_DISTANCE_SPEED_COEFFICIENT
        * max(
            speed,
            0.0,
        )
    )


def evaluate_driving_lyapunov(
    state: np.ndarray,
) -> LyapunovEvaluation:
    """Evaluate the autonomous-driving Lyapunov safety candidate."""

    array = _validated_state(
        state,
        expected_size=4,
        name="driving state",
    )

    speed = float(array[0])
    lane_offset = float(array[1])
    heading_error = float(array[2])
    obstacle_distance = float(array[3])

    lane_scale = DRIVING_LANE_CRITICAL_ABS - DRIVING_LANE_WARNING_ABS

    lane_excess = positive_excess(
        abs(lane_offset),
        DRIVING_LANE_WARNING_ABS,
        lane_scale,
    )

    heading_scale = DRIVING_HEADING_CRITICAL_ABS - DRIVING_HEADING_WARNING_ABS

    heading_excess = positive_excess(
        abs(heading_error),
        DRIVING_HEADING_WARNING_ABS,
        heading_scale,
    )

    fixed_obstacle_scale = (
        DRIVING_OBSTACLE_WARNING_DISTANCE - DRIVING_OBSTACLE_CRITICAL_DISTANCE
    )

    fixed_obstacle_excess = float(
        max(
            0.0,
            DRIVING_OBSTACLE_WARNING_DISTANCE - obstacle_distance,
        )
        / fixed_obstacle_scale
    )

    safe_distance = required_safe_distance(speed)

    speed_obstacle_excess = float(
        max(
            0.0,
            safe_distance - obstacle_distance,
        )
        / max(
            safe_distance,
            EPSILON,
        )
    )

    obstacle_excess = max(
        fixed_obstacle_excess,
        speed_obstacle_excess,
    )

    components = {
        "lane": float(lane_excess**2),
        "heading": float(heading_excess**2),
        "obstacle": float(obstacle_excess**2),
    }

    total = float(sum(components.values()))

    return LyapunovEvaluation(
        domain="autonomous_driving",
        total=total,
        components=components,
        zero_risk_region=(total == 0.0),
    )


def evaluate_robotics_lyapunov(
    state: np.ndarray,
) -> LyapunovEvaluation:
    """Evaluate the robotics Lyapunov safety candidate."""

    array = _validated_state(
        state,
        expected_size=6,
        name="robotics state",
    )

    robot_x = float(array[0])
    robot_y = float(array[1])

    object_x = float(array[2])
    object_y = float(array[3])

    target_x = float(array[4])
    target_y = float(array[5])

    robot_extent = max(
        abs(robot_x),
        abs(robot_y),
    )

    workspace_scale = ROBOTICS_WORKSPACE_CRITICAL_ABS - ROBOTICS_WORKSPACE_WARNING_ABS

    robot_excess = positive_excess(
        robot_extent,
        ROBOTICS_WORKSPACE_WARNING_ABS,
        workspace_scale,
    )

    object_extent = max(
        abs(object_x),
        abs(object_y),
    )

    object_scale = ROBOTICS_OBJECT_CRITICAL_ABS - ROBOTICS_OBJECT_WARNING_ABS

    object_excess = positive_excess(
        object_extent,
        ROBOTICS_OBJECT_WARNING_ABS,
        object_scale,
    )

    target_extent = max(
        abs(target_x),
        abs(target_y),
    )

    target_excess = positive_excess(
        target_extent,
        ROBOTICS_TARGET_CRITICAL_ABS,
        workspace_scale,
    )

    object_target_distance = float(
        np.hypot(
            object_x - target_x,
            object_y - target_y,
        )
    )

    if object_extent > ROBOTICS_OBJECT_WARNING_ABS:
        interaction_scale = (
            ROBOTICS_OBJECT_TARGET_CRITICAL_DISTANCE
            - ROBOTICS_OBJECT_TARGET_WARNING_DISTANCE
        )

        interaction_excess = positive_excess(
            object_target_distance,
            ROBOTICS_OBJECT_TARGET_WARNING_DISTANCE,
            interaction_scale,
        )
    else:
        interaction_excess = 0.0

    components = {
        "robot_workspace": float(robot_excess**2),
        "object_workspace": float(object_excess**2),
        "target_workspace": float(target_excess**2),
        "object_target_interaction": float(interaction_excess**2),
    }

    total = float(sum(components.values()))

    return LyapunovEvaluation(
        domain="robotics",
        total=total,
        components=components,
        zero_risk_region=(total == 0.0),
    )


def evaluate_lyapunov(
    *,
    domain: str,
    state: np.ndarray,
) -> LyapunovEvaluation:
    """Route Lyapunov evaluation to the requested domain."""

    if domain == "autonomous_driving":
        return evaluate_driving_lyapunov(state)

    if domain == "robotics":
        return evaluate_robotics_lyapunov(state)

    raise ValueError(f"unsupported domain: {domain}")


def evaluate_lyapunov_delta(
    *,
    domain: str,
    current_state: np.ndarray,
    next_state: np.ndarray,
) -> LyapunovDelta:
    """Compute delta V without selecting or modifying an action."""

    current = evaluate_lyapunov(
        domain=domain,
        state=current_state,
    )

    next_evaluation = evaluate_lyapunov(
        domain=domain,
        state=next_state,
    )

    return LyapunovDelta(
        current_value=current.total,
        next_value=next_evaluation.total,
        delta=float(next_evaluation.total - current.total),
    )
