from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.rl.driving_env import (
    DrivingRLEnv,
)
from q_vla_forge.rl.robotics_env import (
    RoboticsRLEnv,
)
from q_vla_forge.safety.predictors import (
    RoboticsPredictionContext,
    predict_driving_next_state,
    predict_robotics_next_state,
)


def _set_driving_state(
    env: DrivingRLEnv,
    state: np.ndarray,
) -> None:
    env._state = np.asarray(
        state,
        dtype=np.float32,
    ).copy()


def _set_robotics_state(
    env: RoboticsRLEnv,
    state: np.ndarray,
    *,
    object_grasped: bool,
) -> None:
    env._state = np.asarray(
        state,
        dtype=np.float32,
    ).copy()

    env._object_grasped = object_grasped


@pytest.mark.parametrize(
    "state,action",
    [
        (
            np.array(
                [
                    0.20,
                    0.00,
                    0.00,
                    3.00,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.00,
                    0.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    0.30,
                    0.10,
                    -0.05,
                    2.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.75,
                    0.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    0.30,
                    -0.10,
                    0.05,
                    2.00,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.00,
                    1.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    0.60,
                    0.20,
                    0.10,
                    1.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    -0.40,
                    0.00,
                    1.00,
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    0.80,
                    0.90,
                    0.60,
                    0.40,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    -0.80,
                    -0.40,
                    0.70,
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    0.95,
                    -1.20,
                    -0.70,
                    0.20,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    2.00,
                    2.00,
                    2.00,
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_driving_predictor_matches_environment(
    state: np.ndarray,
    action: np.ndarray,
) -> None:
    env = DrivingRLEnv()

    env.reset(seed=42)

    _set_driving_state(
        env,
        state,
    )

    predicted = predict_driving_next_state(
        state,
        action,
        config=env.config,
    )

    actual, _, _, _, _ = env.step(action)

    np.testing.assert_allclose(
        predicted,
        actual,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    (
        "state",
        "action",
        "object_grasped",
    ),
    [
        (
            np.array(
                [
                    0.00,
                    0.00,
                    0.50,
                    0.50,
                    -0.50,
                    -0.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.50,
                    0.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
            False,
        ),
        (
            np.array(
                [
                    0.00,
                    0.00,
                    0.50,
                    0.50,
                    -0.50,
                    -0.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.00,
                    -1.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
            False,
        ),
        (
            np.array(
                [
                    0.98,
                    0.98,
                    0.20,
                    0.20,
                    -0.50,
                    -0.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    1.00,
                    1.00,
                    0.00,
                ],
                dtype=np.float32,
            ),
            False,
        ),
        (
            np.array(
                [
                    0.00,
                    0.00,
                    0.50,
                    0.50,
                    -0.50,
                    -0.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.00,
                    0.00,
                    1.00,
                ],
                dtype=np.float32,
            ),
            False,
        ),
        (
            np.array(
                [
                    0.00,
                    0.00,
                    0.05,
                    0.00,
                    -0.50,
                    -0.50,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.00,
                    0.00,
                    1.00,
                ],
                dtype=np.float32,
            ),
            False,
        ),
        (
            np.array(
                [
                    0.20,
                    -0.10,
                    0.20,
                    -0.10,
                    0.60,
                    0.60,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.50,
                    -0.50,
                    0.00,
                ],
                dtype=np.float32,
            ),
            True,
        ),
        (
            np.array(
                [
                    0.20,
                    -0.10,
                    0.20,
                    -0.10,
                    0.60,
                    0.60,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.50,
                    -0.50,
                    -1.00,
                ],
                dtype=np.float32,
            ),
            True,
        ),
        (
            np.array(
                [
                    -0.20,
                    0.30,
                    -0.20,
                    0.30,
                    0.60,
                    -0.60,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    -2.00,
                    2.00,
                    2.00,
                ],
                dtype=np.float32,
            ),
            True,
        ),
    ],
)
def test_robotics_predictor_matches_environment(
    state: np.ndarray,
    action: np.ndarray,
    object_grasped: bool,
) -> None:
    env = RoboticsRLEnv()

    env.reset(seed=42)

    _set_robotics_state(
        env,
        state,
        object_grasped=(object_grasped),
    )

    prediction = predict_robotics_next_state(
        state,
        action,
        context=(RoboticsPredictionContext(object_grasped=(object_grasped))),
        config=env.config,
    )

    actual, _, _, _, info = env.step(action)

    np.testing.assert_allclose(
        prediction.next_state,
        actual,
        rtol=0.0,
        atol=0.0,
    )

    assert prediction.next_context.object_grasped == env.object_grasped

    assert prediction.grasped_this_step == bool(info["grasped_this_step"])

    assert prediction.released_this_step == bool(info["released_this_step"])

    assert prediction.invalid_grasp == bool(info["invalid_grasp"])


def test_robotics_predictor_close_after_motion_can_grasp() -> None:
    env = RoboticsRLEnv()

    state = np.array(
        [
            0.00,
            0.00,
            0.12,
            0.00,
            -0.50,
            -0.50,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            0.50,
            0.00,
            1.00,
        ],
        dtype=np.float32,
    )

    env.reset(seed=42)

    _set_robotics_state(
        env,
        state,
        object_grasped=False,
    )

    prediction = predict_robotics_next_state(
        state,
        action,
        context=(RoboticsPredictionContext(object_grasped=False)),
        config=env.config,
    )

    actual, _, _, _, _ = env.step(action)

    np.testing.assert_allclose(
        prediction.next_state,
        actual,
        rtol=0.0,
        atol=0.0,
    )

    assert prediction.next_context.object_grasped is True

    assert prediction.grasped_this_step is True


def test_robotics_grasped_object_follows_robot_exactly() -> None:
    state = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.10,
            0.80,
            0.80,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            0.75,
            -0.50,
            0.00,
        ],
        dtype=np.float32,
    )

    prediction = predict_robotics_next_state(
        state,
        action,
        context=(RoboticsPredictionContext(object_grasped=True)),
    )

    np.testing.assert_array_equal(
        prediction.next_state[0:2],
        prediction.next_state[2:4],
    )


def test_driving_predictor_does_not_mutate_inputs() -> None:
    state = np.array(
        [
            0.30,
            0.10,
            0.05,
            2.00,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            0.30,
            0.40,
            0.20,
        ],
        dtype=np.float32,
    )

    state_before = state.copy()

    action_before = action.copy()

    predict_driving_next_state(
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


def test_robotics_predictor_does_not_mutate_inputs() -> None:
    state = np.array(
        [
            0.00,
            0.00,
            0.50,
            0.50,
            -0.50,
            -0.50,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            0.40,
            -0.20,
            0.00,
        ],
        dtype=np.float32,
    )

    state_before = state.copy()

    action_before = action.copy()

    predict_robotics_next_state(
        state,
        action,
        context=(RoboticsPredictionContext(object_grasped=False)),
    )

    np.testing.assert_array_equal(
        state,
        state_before,
    )

    np.testing.assert_array_equal(
        action,
        action_before,
    )


@pytest.mark.parametrize(
    "bad_state",
    [
        np.zeros(
            3,
            dtype=np.float32,
        ),
        np.array(
            [
                0.0,
                np.nan,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    ],
)
def test_driving_predictor_rejects_invalid_state(
    bad_state: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        predict_driving_next_state(
            bad_state,
            np.zeros(
                3,
                dtype=np.float32,
            ),
        )


@pytest.mark.parametrize(
    "bad_state",
    [
        np.zeros(
            5,
            dtype=np.float32,
        ),
        np.array(
            [
                0.0,
                0.0,
                np.inf,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    ],
)
def test_robotics_predictor_rejects_invalid_state(
    bad_state: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        predict_robotics_next_state(
            bad_state,
            np.zeros(
                3,
                dtype=np.float32,
            ),
            context=(RoboticsPredictionContext(object_grasped=False)),
        )


def test_driving_predictor_is_deterministic() -> None:
    state = np.array(
        [
            0.60,
            0.80,
            0.40,
            0.50,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            -0.50,
            -0.25,
            0.60,
        ],
        dtype=np.float32,
    )

    first = predict_driving_next_state(
        state,
        action,
    )

    second = predict_driving_next_state(
        state,
        action,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_robotics_predictor_is_deterministic() -> None:
    state = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.10,
            0.80,
            0.80,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            0.60,
            -0.30,
            0.00,
        ],
        dtype=np.float32,
    )

    context = RoboticsPredictionContext(object_grasped=True)

    first = predict_robotics_next_state(
        state,
        action,
        context=context,
    )

    second = predict_robotics_next_state(
        state,
        action,
        context=context,
    )

    np.testing.assert_array_equal(
        first.next_state,
        second.next_state,
    )

    assert first.next_context == second.next_context
