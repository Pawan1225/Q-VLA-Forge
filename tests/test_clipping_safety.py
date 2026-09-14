from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.clipping import (
    apply_clipping_safety_filter,
    clip_driving_action,
    clip_robotics_action,
)
from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyMethod,
)


def driving_state(
    *,
    speed: float = 0.25,
    lane: float = 0.0,
    heading: float = 0.0,
    obstacle: float = 2.0,
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
    object_x: float = 0.1,
    object_y: float = 0.0,
    target_x: float = 0.5,
    target_y: float = 0.5,
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


def test_driving_safe_action_unchanged() -> None:
    action = np.array(
        [
            0.2,
            0.1,
            0.0,
        ]
    )

    executed, rules, _ = clip_driving_action(
        state=driving_state(),
        proposed_action=action,
    )

    np.testing.assert_array_equal(
        executed,
        action,
    )

    assert rules == ()


def test_positive_lane_outward_steering_suppressed() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(lane=0.90),
        proposed_action=np.array(
            [
                0.60,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == 0.0
    assert "outward_lane_steering" in rules


def test_positive_lane_inward_recovery_preserved() -> None:
    executed, _, _ = clip_driving_action(
        state=driving_state(lane=0.90),
        proposed_action=np.array(
            [
                -0.60,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == -0.60


def test_negative_lane_outward_steering_suppressed() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(lane=-0.90),
        proposed_action=np.array(
            [
                -0.60,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == 0.0
    assert "outward_lane_steering" in rules


def test_negative_lane_inward_recovery_preserved() -> None:
    executed, _, _ = clip_driving_action(
        state=driving_state(lane=-0.90),
        proposed_action=np.array(
            [
                0.60,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == 0.60


def test_contextual_steering_capped() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(heading=0.60),
        proposed_action=np.array(
            [
                0.95,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == pytest.approx(0.80)

    assert "contextual_steering_cap" in rules


def test_safe_context_steering_untouched() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(),
        proposed_action=np.array(
            [
                0.95,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == pytest.approx(0.95)

    assert "contextual_steering_cap" not in rules


def test_speed_distance_suppresses_positive_acceleration() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(
            speed=1.0,
            obstacle=0.40,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.8,
                0.0,
            ]
        ),
    )

    assert executed[1] == 0.0
    assert "speed_distance_acceleration" in rules


def test_warning_obstacle_imposes_half_braking() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(
            speed=1.0,
            obstacle=0.40,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                0.1,
            ]
        ),
    )

    assert executed[2] == pytest.approx(0.50)

    assert "speed_distance_braking" in rules


def test_critical_obstacle_imposes_full_braking() -> None:
    executed, _, metadata = clip_driving_action(
        state=driving_state(
            speed=1.0,
            obstacle=0.20,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[2] == pytest.approx(1.0)

    assert metadata["critical_obstacle_risk"] is True


def test_accel_brake_conflict_braking_wins() -> None:
    executed, rules, _ = clip_driving_action(
        state=driving_state(),
        proposed_action=np.array(
            [
                0.0,
                0.8,
                0.8,
            ]
        ),
    )

    assert executed[1] == 0.0
    assert executed[2] == pytest.approx(0.8)

    assert "accel_brake_conflict" in rules


def test_driving_output_bounded() -> None:
    executed, _, _ = clip_driving_action(
        state=driving_state(),
        proposed_action=np.array(
            [
                5.0,
                -5.0,
                5.0,
            ]
        ),
    )

    assert -1.0 <= executed[0] <= 1.0

    assert -1.0 <= executed[1] <= 1.0

    assert 0.0 <= executed[2] <= 1.0


def test_driving_inputs_not_mutated() -> None:
    state = driving_state(lane=0.9)

    action = np.array(
        [
            0.8,
            0.0,
            0.0,
        ]
    )

    state_before = state.copy()
    action_before = action.copy()

    clip_driving_action(
        state=state,
        proposed_action=action,
    )

    np.testing.assert_array_equal(
        state,
        state_before,
    )

    np.testing.assert_array_equal(
        action,
        action_before,
    )


def test_robotics_center_motion_unchanged() -> None:
    action = np.array(
        [
            0.5,
            -0.5,
            0.0,
        ]
    )

    executed, rules, _ = clip_robotics_action(
        state=robotics_state(),
        proposed_action=action,
    )

    np.testing.assert_array_equal(
        executed,
        action,
    )

    assert rules == ()


def test_robotics_positive_boundary_clipped() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=0.88,
        ),
        proposed_action=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == pytest.approx(0.25)

    assert "workspace_x_clip" in rules


def test_robotics_negative_boundary_clipped() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=-0.88,
        ),
        proposed_action=np.array(
            [
                -1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == pytest.approx(-0.25)

    assert "workspace_x_clip" in rules


def test_robotics_outside_warning_inward_motion_preserved() -> None:
    executed, _, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=0.95,
        ),
        proposed_action=np.array(
            [
                -0.5,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == pytest.approx(-0.5)


def test_robotics_outside_warning_outward_motion_zeroed() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=0.95,
        ),
        proposed_action=np.array(
            [
                0.5,
                0.0,
                0.0,
            ]
        ),
    )

    assert executed[0] == 0.0
    assert "workspace_x_clip" in rules


def test_robotics_y_boundary_clipped() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_y=0.88,
        ),
        proposed_action=np.array(
            [
                0.0,
                1.0,
                0.0,
            ]
        ),
    )

    assert executed[1] == pytest.approx(0.25)

    assert "workspace_y_clip" in rules


def test_robotics_far_close_becomes_hold() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.5,
            object_y=0.0,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                0.9,
            ]
        ),
    )

    assert executed[2] == 0.0

    assert "unsafe_gripper_close" in rules


def test_robotics_close_near_object_preserved() -> None:
    executed, rules, _ = clip_robotics_action(
        state=robotics_state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.10,
            object_y=0.0,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                0.9,
            ]
        ),
    )

    assert executed[2] == pytest.approx(0.9)

    assert "unsafe_gripper_close" not in rules


def test_robotics_open_preserved() -> None:
    executed, _, _ = clip_robotics_action(
        state=robotics_state(
            object_x=0.8,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                -0.9,
            ]
        ),
    )

    assert executed[2] == pytest.approx(-0.9)


def test_robotics_output_bounded() -> None:
    executed, _, _ = clip_robotics_action(
        state=robotics_state(),
        proposed_action=np.array(
            [
                5.0,
                -5.0,
                5.0,
            ]
        ),
    )

    assert np.all(executed >= -1.0)

    assert np.all(executed <= 1.0)


def test_robotics_inputs_not_mutated() -> None:
    state = robotics_state(robot_x=0.88)

    action = np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    state_before = state.copy()
    action_before = action.copy()

    clip_robotics_action(
        state=state,
        proposed_action=action,
    )

    np.testing.assert_array_equal(
        state,
        state_before,
    )

    np.testing.assert_array_equal(
        action,
        action_before,
    )


def test_safe_filter_active_without_intervention() -> None:
    decision = apply_clipping_safety_filter(
        domain="autonomous_driving",
        true_state=driving_state(),
        proposed_action=np.array(
            [
                0.1,
                0.0,
                0.0,
            ]
        ),
    )

    assert decision.method == SafetyMethod.CLIPPING

    assert not decision.intervened

    assert decision.intervention_reason == InterventionReason.NONE

    assert decision.correction_l2 == 0.0


def test_domain_intervention_reason() -> None:
    decision = apply_clipping_safety_filter(
        domain="autonomous_driving",
        true_state=driving_state(lane=0.90),
        proposed_action=np.array(
            [
                0.9,
                0.0,
                0.0,
            ]
        ),
    )

    assert decision.intervened

    assert decision.intervention_reason == InterventionReason.DOMAIN_CONSTRAINT


def test_action_bound_only_reason() -> None:
    decision = apply_clipping_safety_filter(
        domain="autonomous_driving",
        true_state=driving_state(),
        proposed_action=np.array(
            [
                1.5,
                0.0,
                0.0,
            ]
        ),
    )

    assert decision.intervened

    assert decision.intervention_reason == InterventionReason.ACTION_BOUND


def test_driving_conflict_immediate_violation_reduced() -> None:
    decision = apply_clipping_safety_filter(
        domain="autonomous_driving",
        true_state=driving_state(),
        proposed_action=np.array(
            [
                0.0,
                0.8,
                0.8,
            ]
        ),
    )

    assert (
        decision.metadata["post_filter_violation_count"]
        < decision.metadata["pre_filter_violation_count"]
    )


def test_robotics_gripper_violation_reduced() -> None:
    decision = apply_clipping_safety_filter(
        domain="robotics",
        true_state=robotics_state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.5,
            object_y=0.0,
        ),
        proposed_action=np.array(
            [
                0.0,
                0.0,
                0.9,
            ]
        ),
    )

    assert (
        decision.metadata["post_filter_violation_count"]
        < decision.metadata["pre_filter_violation_count"]
    )


def test_unknown_domain_rejected() -> None:
    with pytest.raises(ValueError):
        apply_clipping_safety_filter(
            domain="unknown",
            true_state=np.zeros(1),
            proposed_action=np.zeros(1),
        )
