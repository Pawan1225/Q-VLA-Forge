from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.contracts import ViolationSeverity
from q_vla_forge.safety.driving_constraints import (
    VIOLATION_CATEGORIES,
    count_driving_violations,
    driving_has_critical_violation,
    driving_step_has_violation,
    evaluate_driving_violations,
    is_driving_state_safe,
    required_safe_distance,
)


def _state(
    *,
    speed: float = 0.2,
    lane_offset: float = 0.0,
    heading_error: float = 0.0,
    obstacle_distance: float = 1.0,
) -> np.ndarray:
    return np.array(
        [
            speed,
            lane_offset,
            heading_error,
            obstacle_distance,
        ],
        dtype=np.float32,
    )


def _action(
    *,
    steering: float = 0.0,
    acceleration: float = 0.0,
    braking: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            steering,
            acceleration,
            braking,
        ],
        dtype=np.float32,
    )


def _by_name(
    state: np.ndarray,
    action: np.ndarray,
):
    return {
        record.name: record
        for record in evaluate_driving_violations(
            state,
            action,
        )
    }


def test_exactly_five_categories() -> None:
    records = evaluate_driving_violations(
        _state(),
        _action(),
    )

    assert tuple(record.name for record in records) == VIOLATION_CATEGORIES


def test_safe_nominal_state() -> None:
    state = _state()
    action = _action()

    assert is_driving_state_safe(state, action)
    assert count_driving_violations(state, action) == 0


def test_lane_warning_violation() -> None:
    record = _by_name(
        _state(lane_offset=0.80),
        _action(),
    )["lane_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.WARNING


def test_lane_critical_violation() -> None:
    record = _by_name(
        _state(lane_offset=0.98),
        _action(),
    )["lane_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_lane_warning_boundary_is_not_violated() -> None:
    record = _by_name(
        _state(lane_offset=0.75),
        _action(),
    )["lane_boundary"]

    assert not record.violated


def test_lane_critical_boundary_remains_warning() -> None:
    record = _by_name(
        _state(lane_offset=0.95),
        _action(),
    )["lane_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.WARNING


@pytest.mark.parametrize("offset", [0.80, -0.80])
def test_lane_symmetry(offset: float) -> None:
    record = _by_name(
        _state(lane_offset=offset),
        _action(),
    )["lane_boundary"]

    assert record.violated


def test_obstacle_warning_violation() -> None:
    record = _by_name(
        _state(obstacle_distance=0.20),
        _action(),
    )["unsafe_obstacle_distance"]

    assert record.violated


def test_obstacle_critical_violation() -> None:
    record = _by_name(
        _state(obstacle_distance=0.10),
        _action(),
    )["unsafe_obstacle_distance"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_obstacle_warning_boundary_is_safe() -> None:
    record = _by_name(
        _state(obstacle_distance=0.30),
        _action(),
    )["unsafe_obstacle_distance"]

    assert not record.violated


def test_required_safe_distance_values() -> None:
    assert required_safe_distance(0.0) == pytest.approx(0.15)
    assert required_safe_distance(0.5) == pytest.approx(0.325)
    assert required_safe_distance(1.0) == pytest.approx(0.50)


def test_safe_distance_monotonicity() -> None:
    values = [
        required_safe_distance(speed)
        for speed in (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        )
    ]

    assert values == sorted(values)


def test_speed_distance_envelope() -> None:
    record = _by_name(
        _state(
            speed=1.0,
            obstacle_distance=0.40,
        ),
        _action(),
    )["unsafe_speed_condition"]

    assert record.violated


def test_speed_distance_critical() -> None:
    record = _by_name(
        _state(
            speed=1.0,
            obstacle_distance=0.20,
        ),
        _action(),
    )["unsafe_speed_condition"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_steering_risk_requires_context() -> None:
    record = _by_name(
        _state(),
        _action(steering=0.90),
    )["steering_risk"]

    assert not record.violated


@pytest.mark.parametrize("steering", [0.90, -0.90])
def test_steering_risk_symmetry(steering: float) -> None:
    record = _by_name(
        _state(lane_offset=0.70),
        _action(steering=steering),
    )["steering_risk"]

    assert record.violated


def test_steering_risk_from_heading_error() -> None:
    record = _by_name(
        _state(heading_error=0.60),
        _action(steering=0.90),
    )["steering_risk"]

    assert record.violated


def test_acceleration_braking_conflict() -> None:
    record = _by_name(
        _state(),
        _action(
            acceleration=0.50,
            braking=0.50,
        ),
    )["acceleration_braking_conflict"]

    assert record.violated


def test_acceleration_braking_critical_conflict() -> None:
    record = _by_name(
        _state(),
        _action(
            acceleration=0.80,
            braking=0.80,
        ),
    )["acceleration_braking_conflict"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_conflict_boundary_not_violated() -> None:
    record = _by_name(
        _state(),
        _action(
            acceleration=0.25,
            braking=0.25,
        ),
    )["acceleration_braking_conflict"]

    assert not record.violated


def test_multiple_violations_same_step() -> None:
    state = _state(
        speed=1.0,
        lane_offset=0.90,
        obstacle_distance=0.10,
    )

    action = _action(
        steering=0.99,
        acceleration=0.80,
        braking=0.80,
    )

    assert driving_step_has_violation(state, action)
    assert count_driving_violations(state, action) >= 4


def test_critical_helper() -> None:
    assert driving_has_critical_violation(
        _state(lane_offset=1.0),
        _action(),
    )


def test_invalid_state_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_driving_violations(
            np.zeros(3, dtype=np.float32),
            _action(),
        )


def test_invalid_action_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_driving_violations(
            _state(),
            np.zeros(2, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_state_rejected(
    bad_value: float,
) -> None:
    state = _state()
    state[0] = bad_value

    with pytest.raises(ValueError):
        evaluate_driving_violations(
            state,
            _action(),
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_action_rejected(
    bad_value: float,
) -> None:
    action = _action()
    action[0] = bad_value

    with pytest.raises(ValueError):
        evaluate_driving_violations(
            _state(),
            action,
        )


def test_evaluator_does_not_mutate_inputs() -> None:
    state = _state(
        speed=0.8,
        lane_offset=0.7,
    )

    action = _action(
        steering=0.9,
        acceleration=0.4,
        braking=0.4,
    )

    state_before = state.copy()
    action_before = action.copy()

    evaluate_driving_violations(
        state,
        action,
    )

    np.testing.assert_array_equal(
        state,
        state_before,
    )

    np.testing.assert_array_equal(
        action,
        action_before,
    )


def test_out_of_range_action_is_not_clipped() -> None:
    action = _action(
        steering=2.0,
        acceleration=0.5,
        braking=0.5,
    )

    record = _by_name(
        _state(lane_offset=0.70),
        action,
    )["steering_risk"]

    assert record.value == pytest.approx(2.0)
    assert record.violated
