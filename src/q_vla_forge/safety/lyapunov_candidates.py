"""Deterministic candidate generation and hard guards for Sprint 5.7."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q_vla_forge.safety.contracts import ViolationRecord
from q_vla_forge.safety.driving_constraints import (
    evaluate_driving_violations,
)
from q_vla_forge.safety.robotics_constraints import (
    evaluate_robotics_violations,
)

ACTION_DEDUPLICATION_TOLERANCE = 1.0e-12

MAX_DRIVING_CANDIDATES = 10
MAX_ROBOTICS_CANDIDATES = 9

DRIVING_HARD_GUARD_NAMES = frozenset(
    {
        "steering_risk",
        "acceleration_braking_conflict",
    }
)

ROBOTICS_HARD_GUARD_NAMES = frozenset(
    {
        "unsafe_motion",
        "unsafe_gripper_condition",
    }
)


@dataclass(frozen=True)
class LyapunovActionCandidate:
    """One deterministic action candidate for later Lyapunov scoring."""

    index: int
    source: str
    action: np.ndarray


@dataclass(frozen=True)
class HardGuardResult:
    """Result of screening one action against action-dependent constraints."""

    passed: bool
    violated_guard_names: tuple[str, ...]
    violations: tuple[ViolationRecord, ...]


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


def _actions_equal(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    return bool(
        np.all(
            np.abs(
                np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            )
            <= ACTION_DEDUPLICATION_TOLERANCE
        )
    )


def _deduplicate_candidates(
    candidates: list[tuple[str, np.ndarray]],
) -> tuple[LyapunovActionCandidate, ...]:
    retained: list[tuple[str, np.ndarray]] = []

    for source, action in candidates:
        action_array = np.asarray(
            action,
            dtype=np.float64,
        ).copy()

        if any(
            _actions_equal(
                action_array,
                retained_action,
            )
            for _, retained_action in retained
        ):
            continue

        retained.append(
            (
                source,
                action_array,
            )
        )

    return tuple(
        LyapunovActionCandidate(
            index=index,
            source=source,
            action=action.copy(),
        )
        for index, (source, action) in enumerate(retained)
    )


def _driving_inward_steering(
    state: np.ndarray,
) -> float:
    """Return deterministic steering toward lower lane/heading risk."""

    lane_offset = float(state[1])
    heading_error = float(state[2])

    if abs(heading_error) > 1.0e-12:
        return -0.80 if heading_error > 0.0 else 0.80

    if abs(lane_offset) > 1.0e-12:
        return -0.80 if lane_offset > 0.0 else 0.80

    return 0.0


def generate_driving_candidates(
    *,
    state: np.ndarray,
    proposed_action: np.ndarray,
) -> tuple[LyapunovActionCandidate, ...]:
    """Generate the finite Sprint 5.7 driving candidate set."""

    state_array = _validated_vector(
        state,
        expected_size=4,
        name="driving state",
    )

    proposed = _validated_vector(
        proposed_action,
        expected_size=3,
        name="driving proposed action",
    )

    bounded = np.array(
        [
            np.clip(proposed[0], -1.0, 1.0),
            np.clip(proposed[1], -1.0, 1.0),
            np.clip(proposed[2], 0.0, 1.0),
        ],
        dtype=np.float64,
    )

    steering = float(bounded[0])
    acceleration = float(bounded[1])
    braking = float(bounded[2])

    inward_steering = _driving_inward_steering(state_array)

    moderate_braking = max(
        braking,
        0.50,
    )

    raw_candidates: list[tuple[str, np.ndarray]] = [
        (
            "bounded_proposed",
            bounded,
        ),
        (
            "neutral_steering",
            np.array(
                [
                    0.0,
                    acceleration,
                    braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "coast",
            np.array(
                [
                    steering,
                    0.0,
                    braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "neutral_steering_coast",
            np.array(
                [
                    0.0,
                    0.0,
                    braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "moderate_brake",
            np.array(
                [
                    steering,
                    0.0,
                    moderate_braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "full_brake",
            np.array(
                [
                    steering,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "neutral_full_brake",
            np.array(
                [
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_steer",
            np.array(
                [
                    inward_steering,
                    0.0,
                    braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_moderate_brake",
            np.array(
                [
                    inward_steering,
                    0.0,
                    moderate_braking,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_full_brake",
            np.array(
                [
                    inward_steering,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
        ),
    ]

    candidates = _deduplicate_candidates(raw_candidates)

    if len(candidates) > MAX_DRIVING_CANDIDATES:
        raise RuntimeError("driving candidate ceiling exceeded")

    return candidates


def _robotics_inward_command(
    position: float,
) -> float:
    if position > 0.0:
        return -1.0

    if position < 0.0:
        return 1.0

    return 0.0


def generate_robotics_candidates(
    *,
    state: np.ndarray,
    proposed_action: np.ndarray,
) -> tuple[LyapunovActionCandidate, ...]:
    """Generate the finite Sprint 5.7 robotics candidate set."""

    state_array = _validated_vector(
        state,
        expected_size=6,
        name="robotics state",
    )

    proposed = _validated_vector(
        proposed_action,
        expected_size=3,
        name="robotics proposed action",
    )

    bounded = np.clip(
        proposed,
        -1.0,
        1.0,
    ).astype(np.float64)

    delta_x = float(bounded[0])
    delta_y = float(bounded[1])
    gripper = float(bounded[2])

    inward_x = _robotics_inward_command(float(state_array[0]))

    inward_y = _robotics_inward_command(float(state_array[1]))

    raw_candidates: list[tuple[str, np.ndarray]] = [
        (
            "bounded_proposed",
            bounded,
        ),
        (
            "hold_gripper",
            np.array(
                [
                    delta_x,
                    delta_y,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "x_only",
            np.array(
                [
                    delta_x,
                    0.0,
                    gripper,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "y_only",
            np.array(
                [
                    0.0,
                    delta_y,
                    gripper,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "stop_preserve_gripper",
            np.array(
                [
                    0.0,
                    0.0,
                    gripper,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "full_stop",
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_motion",
            np.array(
                [
                    inward_x,
                    inward_y,
                    gripper,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_motion_hold_gripper",
            np.array(
                [
                    inward_x,
                    inward_y,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "inward_x_hold_gripper",
            np.array(
                [
                    inward_x,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        ),
    ]

    candidates = _deduplicate_candidates(raw_candidates)

    if len(candidates) > MAX_ROBOTICS_CANDIDATES:
        raise RuntimeError("robotics candidate ceiling exceeded")

    return candidates


def _hard_guard_result(
    *,
    violations: tuple[ViolationRecord, ...],
    guard_names: frozenset[str],
) -> HardGuardResult:
    violated_names = tuple(
        record.name
        for record in violations
        if (record.name in guard_names and record.violated)
    )

    return HardGuardResult(
        passed=(len(violated_names) == 0),
        violated_guard_names=violated_names,
        violations=violations,
    )


def evaluate_driving_hard_guards(
    *,
    state: np.ndarray,
    action: np.ndarray,
) -> HardGuardResult:
    """Screen only action-dependent driving constraints."""

    state_array = _validated_vector(
        state,
        expected_size=4,
        name="driving state",
    )

    action_array = _validated_vector(
        action,
        expected_size=3,
        name="driving action",
    )

    violations = evaluate_driving_violations(
        state_array,
        action_array,
    )

    return _hard_guard_result(
        violations=violations,
        guard_names=DRIVING_HARD_GUARD_NAMES,
    )


def evaluate_robotics_hard_guards(
    *,
    state: np.ndarray,
    action: np.ndarray,
) -> HardGuardResult:
    """Screen only action-dependent robotics constraints."""

    state_array = _validated_vector(
        state,
        expected_size=6,
        name="robotics state",
    )

    action_array = _validated_vector(
        action,
        expected_size=3,
        name="robotics action",
    )

    violations = evaluate_robotics_violations(
        state_array,
        action_array,
    )

    return _hard_guard_result(
        violations=violations,
        guard_names=ROBOTICS_HARD_GUARD_NAMES,
    )


def evaluate_hard_guards(
    *,
    domain: str,
    state: np.ndarray,
    action: np.ndarray,
) -> HardGuardResult:
    """Route hard-guard evaluation to the frozen domain contract."""

    if domain == "autonomous_driving":
        return evaluate_driving_hard_guards(
            state=state,
            action=action,
        )

    if domain == "robotics":
        return evaluate_robotics_hard_guards(
            state=state,
            action=action,
        )

    raise ValueError(f"unsupported domain: {domain}")
