from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.lyapunov import (
    evaluate_driving_lyapunov,
    evaluate_lyapunov,
    evaluate_lyapunov_delta,
    evaluate_robotics_lyapunov,
    positive_excess,
    required_safe_distance,
)


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


def test_positive_excess_zero_below_threshold() -> None:
    assert (
        positive_excess(
            0.5,
            0.75,
            0.2,
        )
        == 0.0
    )


def test_positive_excess_positive_above_threshold() -> None:
    assert positive_excess(
        0.95,
        0.75,
        0.20,
    ) == pytest.approx(1.0)


def test_positive_excess_invalid_scale() -> None:
    with pytest.raises(ValueError):
        positive_excess(
            1.0,
            0.5,
            0.0,
        )


def test_required_safe_distance() -> None:
    assert required_safe_distance(1.0) == pytest.approx(0.50)


def test_driving_nominal_zero() -> None:
    result = evaluate_driving_lyapunov(driving_state())

    assert result.total == 0.0
    assert result.zero_risk_region


def test_driving_lane_warning_positive() -> None:
    result = evaluate_driving_lyapunov(driving_state(lane=0.85))

    assert result.components["lane"] > 0.0


def test_driving_lane_critical_reference_is_one() -> None:
    result = evaluate_driving_lyapunov(driving_state(lane=0.95))

    assert result.components["lane"] == pytest.approx(1.0)


def test_driving_lane_monotonicity() -> None:
    values = [
        evaluate_driving_lyapunov(driving_state(lane=value)).components["lane"]
        for value in (
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        )
    ]

    assert values == sorted(values)


def test_driving_lane_symmetry() -> None:
    positive = evaluate_driving_lyapunov(driving_state(lane=0.90))

    negative = evaluate_driving_lyapunov(driving_state(lane=-0.90))

    assert positive.components["lane"] == pytest.approx(negative.components["lane"])


def test_heading_threshold_zero() -> None:
    result = evaluate_driving_lyapunov(driving_state(heading=0.50))

    assert result.components["heading"] == 0.0


def test_heading_critical_reference_is_one() -> None:
    result = evaluate_driving_lyapunov(driving_state(heading=0.75))

    assert result.components["heading"] == pytest.approx(1.0)


def test_heading_symmetry() -> None:
    positive = evaluate_driving_lyapunov(driving_state(heading=0.60))

    negative = evaluate_driving_lyapunov(driving_state(heading=-0.60))

    assert positive.components["heading"] == pytest.approx(
        negative.components["heading"]
    )


def test_fixed_obstacle_risk_positive() -> None:
    result = evaluate_driving_lyapunov(
        driving_state(
            speed=0.0,
            obstacle=0.20,
        )
    )

    assert result.components["obstacle"] > 0.0


def test_obstacle_risk_monotonic_as_distance_decreases() -> None:
    values = [
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

    assert values == sorted(values)


def test_obstacle_risk_non_decreasing_with_speed() -> None:
    values = [
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

    assert values == sorted(values)


def test_driving_component_sum_identity() -> None:
    result = evaluate_driving_lyapunov(
        driving_state(
            speed=1.0,
            lane=0.90,
            heading=0.60,
            obstacle=0.20,
        )
    )

    assert result.total == pytest.approx(
        sum(result.components.values()),
        abs=1e-12,
    )


def test_robotics_nominal_zero() -> None:
    result = evaluate_robotics_lyapunov(robotics_state())

    assert result.total == 0.0
    assert result.zero_risk_region


def test_robot_workspace_warning_zero() -> None:
    result = evaluate_robotics_lyapunov(robotics_state(robot_x=0.90))

    assert result.components["robot_workspace"] == 0.0


def test_robot_workspace_critical_reference_is_one() -> None:
    result = evaluate_robotics_lyapunov(robotics_state(robot_x=1.00))

    assert result.components["robot_workspace"] == pytest.approx(1.0)


def test_robot_workspace_monotonicity() -> None:
    values = [
        evaluate_robotics_lyapunov(robotics_state(robot_x=value)).components[
            "robot_workspace"
        ]
        for value in (
            0.80,
            0.90,
            0.95,
            1.00,
            1.05,
        )
    ]

    assert values == sorted(values)


@pytest.mark.parametrize(
    "robot_x,robot_y",
    [
        (0.95, 0.0),
        (-0.95, 0.0),
        (0.0, 0.95),
        (0.0, -0.95),
    ],
)
def test_robot_workspace_symmetry(
    robot_x: float,
    robot_y: float,
) -> None:
    result = evaluate_robotics_lyapunov(
        robotics_state(
            robot_x=robot_x,
            robot_y=robot_y,
        )
    )

    assert result.components["robot_workspace"] == pytest.approx(0.25)


def test_object_boundary_monotonicity() -> None:
    values = [
        evaluate_robotics_lyapunov(
            robotics_state(
                object_x=value,
                target_x=value,
            )
        ).components["object_workspace"]
        for value in (
            0.90,
            0.95,
            1.00,
            1.05,
        )
    ]

    assert values == sorted(values)


def test_target_inside_workspace_zero() -> None:
    result = evaluate_robotics_lyapunov(robotics_state(target_x=0.80))

    assert result.components["target_workspace"] == 0.0


def test_target_at_boundary_zero() -> None:
    result = evaluate_robotics_lyapunov(robotics_state(target_x=1.00))

    assert result.components["target_workspace"] == 0.0


def test_target_outside_workspace_positive() -> None:
    result = evaluate_robotics_lyapunov(robotics_state(target_x=1.05))

    assert result.components["target_workspace"] > 0.0


def test_task_distance_alone_is_not_safety_risk() -> None:
    result = evaluate_robotics_lyapunov(
        robotics_state(
            robot_x=0.0,
            robot_y=0.0,
            object_x=0.0,
            object_y=0.0,
            target_x=0.8,
            target_y=0.0,
        )
    )

    assert result.total == 0.0


def test_interaction_inactive_when_object_central() -> None:
    result = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.0,
            object_y=0.0,
            target_x=0.8,
            target_y=0.0,
        )
    )

    assert result.components["object_target_interaction"] == 0.0


def test_interaction_active_near_object_boundary() -> None:
    result = evaluate_robotics_lyapunov(
        robotics_state(
            object_x=0.95,
            object_y=0.0,
            target_x=0.0,
            target_y=0.0,
        )
    )

    assert result.components["object_target_interaction"] > 0.0


def test_robotics_component_sum_identity() -> None:
    result = evaluate_robotics_lyapunov(
        robotics_state(
            robot_x=0.95,
            object_x=0.95,
            target_x=-0.2,
        )
    )

    assert result.total == pytest.approx(
        sum(result.components.values()),
        abs=1e-12,
    )


def test_driving_nonnegative_grid() -> None:
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

                    assert result.total >= 0.0


def test_robotics_nonnegative_grid() -> None:
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

                assert result.total >= 0.0


def test_driving_deterministic() -> None:
    state = driving_state(
        speed=1.0,
        lane=0.9,
        heading=0.6,
        obstacle=0.2,
    )

    first = evaluate_driving_lyapunov(state)

    second = evaluate_driving_lyapunov(state)

    assert first == second


def test_robotics_deterministic() -> None:
    state = robotics_state(
        robot_x=0.95,
        object_x=0.95,
        target_x=-0.2,
    )

    first = evaluate_robotics_lyapunov(state)

    second = evaluate_robotics_lyapunov(state)

    assert first == second


def test_driving_input_not_mutated() -> None:
    state = driving_state(lane=0.9)

    before = state.copy()

    evaluate_driving_lyapunov(state)

    np.testing.assert_array_equal(
        state,
        before,
    )


def test_robotics_input_not_mutated() -> None:
    state = robotics_state(robot_x=0.95)

    before = state.copy()

    evaluate_robotics_lyapunov(state)

    np.testing.assert_array_equal(
        state,
        before,
    )


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(3),
        np.zeros(5),
    ],
)
def test_driving_invalid_shape(
    state: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        evaluate_driving_lyapunov(state)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_driving_rejects_nonfinite(
    bad_value: float,
) -> None:
    state = driving_state()
    state[0] = bad_value

    with pytest.raises(ValueError):
        evaluate_driving_lyapunov(state)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_robotics_rejects_nonfinite(
    bad_value: float,
) -> None:
    state = robotics_state()
    state[0] = bad_value

    with pytest.raises(ValueError):
        evaluate_robotics_lyapunov(state)


def test_driving_delta_decreases_for_safer_lane_state() -> None:
    delta = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.95),
        next_state=driving_state(lane=0.80),
    )

    assert delta.delta < 0.0


def test_driving_delta_increases_for_worsening_lane_state() -> None:
    delta = evaluate_lyapunov_delta(
        domain="autonomous_driving",
        current_state=driving_state(lane=0.80),
        next_state=driving_state(lane=0.95),
    )

    assert delta.delta > 0.0


def test_robotics_delta_decreases_for_workspace_recovery() -> None:
    delta = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.98),
        next_state=robotics_state(robot_x=0.85),
    )

    assert delta.delta < 0.0


def test_robotics_delta_increases_for_workspace_worsening() -> None:
    delta = evaluate_lyapunov_delta(
        domain="robotics",
        current_state=robotics_state(robot_x=0.85),
        next_state=robotics_state(robot_x=0.98),
    )

    assert delta.delta > 0.0


def test_domain_router_driving() -> None:
    result = evaluate_lyapunov(
        domain="autonomous_driving",
        state=driving_state(),
    )

    assert result.domain == "autonomous_driving"


def test_domain_router_robotics() -> None:
    result = evaluate_lyapunov(
        domain="robotics",
        state=robotics_state(),
    )

    assert result.domain == "robotics"


def test_unknown_domain_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_lyapunov(
            domain="unknown",
            state=np.zeros(1),
        )
