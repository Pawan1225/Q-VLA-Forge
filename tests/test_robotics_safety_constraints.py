from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.contracts import ViolationSeverity
from q_vla_forge.safety.robotics_constraints import (
    VIOLATION_CATEGORIES,
    count_robotics_violations,
    evaluate_robotics_violations,
    is_robotics_state_safe,
    object_target_distance,
    predicted_robot_position,
    robot_object_distance,
    robotics_has_critical_violation,
    robotics_step_has_violation,
)


def _state(
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
        dtype=np.float32,
    )


def _action(
    *,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    gripper: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            delta_x,
            delta_y,
            gripper,
        ],
        dtype=np.float32,
    )


def _by_name(
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


def test_exactly_five_categories() -> None:
    records = evaluate_robotics_violations(
        _state(),
        _action(),
    )

    assert tuple(record.name for record in records) == VIOLATION_CATEGORIES


def test_nominal_state_is_safe() -> None:
    assert is_robotics_state_safe(
        _state(),
        _action(),
    )


def test_workspace_warning() -> None:
    record = _by_name(
        _state(robot_x=0.95),
        _action(),
    )["workspace_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.WARNING


def test_workspace_critical() -> None:
    record = _by_name(
        _state(robot_x=1.05),
        _action(),
    )["workspace_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_workspace_warning_boundary_safe() -> None:
    record = _by_name(
        _state(robot_x=0.90),
        _action(),
    )["workspace_boundary"]

    assert not record.violated


def test_workspace_critical_boundary_warning() -> None:
    record = _by_name(
        _state(robot_x=1.00),
        _action(),
    )["workspace_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.WARNING


@pytest.mark.parametrize(
    "value",
    [0.95, -0.95],
)
def test_workspace_symmetry(
    value: float,
) -> None:
    record = _by_name(
        _state(robot_x=value),
        _action(),
    )["workspace_boundary"]

    assert record.violated


def test_predicted_robot_position() -> None:
    x, y = predicted_robot_position(
        _state(),
        _action(
            delta_x=1.0,
            delta_y=-1.0,
        ),
    )

    assert x == pytest.approx(0.08)
    assert y == pytest.approx(-0.08)


def test_inward_recovery_motion_not_unsafe() -> None:
    records = _by_name(
        _state(robot_x=0.95),
        _action(delta_x=-1.0),
    )

    assert records["workspace_boundary"].violated

    assert not records["unsafe_motion"].violated


def test_outward_motion_near_boundary_is_unsafe() -> None:
    records = _by_name(
        _state(robot_x=0.88),
        _action(delta_x=1.0),
    )

    assert records["unsafe_motion"].violated


def test_outward_motion_can_be_critical() -> None:
    record = _by_name(
        _state(robot_x=0.99),
        _action(delta_x=1.0),
    )["unsafe_motion"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_robot_object_distance() -> None:
    distance = robot_object_distance(
        _state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.12,
            object_y=0.0,
        )
    )

    assert distance == pytest.approx(0.12)


def test_object_target_distance() -> None:
    distance = object_target_distance(
        _state(
            object_x=0.0,
            object_y=0.0,
            target_x=0.5,
            target_y=0.0,
        )
    )

    assert distance == pytest.approx(0.5)


def test_gripper_close_far_from_object_is_unsafe() -> None:
    record = _by_name(
        _state(
            robot_x=0.0,
            object_x=0.5,
        ),
        _action(gripper=1.0),
    )["unsafe_gripper_condition"]

    assert record.violated


def test_gripper_close_near_object_is_safe() -> None:
    record = _by_name(
        _state(
            robot_x=0.0,
            object_x=0.10,
        ),
        _action(gripper=1.0),
    )["unsafe_gripper_condition"]

    assert not record.violated


def test_gripper_threshold_exact_not_closing() -> None:
    record = _by_name(
        _state(
            robot_x=0.0,
            object_x=0.5,
        ),
        _action(gripper=0.50),
    )["unsafe_gripper_condition"]

    assert not record.violated


def test_grasp_distance_boundary_safe() -> None:
    record = _by_name(
        _state(
            robot_x=0.0,
            object_x=0.12,
        ),
        _action(gripper=1.0),
    )["unsafe_gripper_condition"]

    assert not record.violated


def test_gripper_critical_distance() -> None:
    record = _by_name(
        _state(
            robot_x=0.0,
            object_x=0.50,
        ),
        _action(gripper=1.0),
    )["unsafe_gripper_condition"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_object_boundary_warning() -> None:
    record = _by_name(
        _state(object_x=0.95),
        _action(),
    )["object_boundary"]

    assert record.violated


def test_object_boundary_critical() -> None:
    record = _by_name(
        _state(object_x=1.05),
        _action(),
    )["object_boundary"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_far_target_in_interior_is_not_automatically_unsafe() -> None:
    record = _by_name(
        _state(
            object_x=0.0,
            target_x=0.8,
        ),
        _action(),
    )["unsafe_object_target_interaction"]

    assert not record.violated


def test_target_outside_workspace_is_critical() -> None:
    record = _by_name(
        _state(target_x=1.1),
        _action(),
    )["unsafe_object_target_interaction"]

    assert record.violated
    assert record.severity == ViolationSeverity.CRITICAL


def test_multiple_violations() -> None:
    state = _state(
        robot_x=0.95,
        object_x=0.95,
        target_x=1.10,
    )

    action = _action(
        delta_x=1.0,
        gripper=1.0,
    )

    assert robotics_step_has_violation(
        state,
        action,
    )

    assert (
        count_robotics_violations(
            state,
            action,
        )
        >= 3
    )


def test_critical_helper() -> None:
    assert robotics_has_critical_violation(
        _state(robot_x=1.10),
        _action(),
    )


def test_invalid_state_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_robotics_violations(
            np.zeros(5, dtype=np.float32),
            _action(),
        )


def test_invalid_action_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_robotics_violations(
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
        evaluate_robotics_violations(
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
        evaluate_robotics_violations(
            _state(),
            action,
        )


def test_evaluator_does_not_mutate_inputs() -> None:
    state = _state(
        robot_x=0.95,
        object_x=0.95,
    )

    action = _action(
        delta_x=1.0,
        gripper=1.0,
    )

    state_before = state.copy()
    action_before = action.copy()

    evaluate_robotics_violations(
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
    record = _by_name(
        _state(robot_x=0.80),
        _action(delta_x=2.0),
    )["unsafe_motion"]

    assert record.violated
