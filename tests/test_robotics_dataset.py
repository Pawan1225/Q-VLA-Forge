from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.data import (
    ROBOTICS_ACTION_DIM,
    ROBOTICS_GOALS,
    ROBOTICS_STATE_DIM,
    Domain,
    SyntheticRoboticsDataset,
    compute_robotics_target_action,
    render_robotics_visual,
    robotics_safety_constraints,
)


def test_robotics_dataset_length() -> None:
    dataset = SyntheticRoboticsDataset(
        size=10,
        seed=42,
    )

    assert len(dataset) == 10


def test_robotics_sample_domain() -> None:
    dataset = SyntheticRoboticsDataset(
        size=2,
        seed=42,
    )

    sample = dataset[0]

    assert sample.domain == Domain.ROBOTICS
    assert sample.sample_id == "robotics-000000"


def test_robotics_observation_shapes() -> None:
    dataset = SyntheticRoboticsDataset(
        size=1,
        seed=42,
        image_size=32,
    )

    observation = dataset[0].observation

    assert observation.visual.shape == (3, 32, 32)
    assert observation.state.shape == (ROBOTICS_STATE_DIM,)
    assert observation.visual.dtype == np.float32
    assert observation.state.dtype == np.float32


def test_robotics_action_shape() -> None:
    dataset = SyntheticRoboticsDataset(
        size=1,
        seed=42,
    )

    action = dataset[0].target_action

    assert action.values.shape == (ROBOTICS_ACTION_DIM,)
    assert action.values.dtype == np.float32


def test_robotics_goal_is_supported() -> None:
    dataset = SyntheticRoboticsDataset(
        size=20,
        seed=42,
    )

    for sample in dataset:
        assert sample.observation.language_goal in ROBOTICS_GOALS


def test_robotics_dataset_is_reproducible() -> None:
    first = SyntheticRoboticsDataset(
        size=5,
        seed=123,
    )

    second = SyntheticRoboticsDataset(
        size=5,
        seed=123,
    )

    for first_sample, second_sample in zip(
        first,
        second,
        strict=True,
    ):
        assert np.array_equal(
            first_sample.observation.visual,
            second_sample.observation.visual,
        )

        assert np.array_equal(
            first_sample.observation.state,
            second_sample.observation.state,
        )

        assert (
            first_sample.observation.language_goal
            == second_sample.observation.language_goal
        )

        assert np.array_equal(
            first_sample.target_action.values,
            second_sample.target_action.values,
        )


def test_different_seeds_generate_different_data() -> None:
    first = SyntheticRoboticsDataset(
        size=1,
        seed=42,
    )

    second = SyntheticRoboticsDataset(
        size=1,
        seed=123,
    )

    assert not np.array_equal(
        first[0].observation.state,
        second[0].observation.state,
    )


def test_robotics_safety_constraints() -> None:
    constraints = robotics_safety_constraints()

    assert constraints.lower_bounds.shape == (ROBOTICS_ACTION_DIM,)

    assert constraints.upper_bounds.shape == (ROBOTICS_ACTION_DIM,)

    assert np.all(constraints.lower_bounds <= constraints.upper_bounds)


def test_target_action_respects_bounds() -> None:
    dataset = SyntheticRoboticsDataset(
        size=100,
        seed=42,
    )

    constraints = robotics_safety_constraints()

    for sample in dataset:
        action = sample.target_action.values

        assert np.all(action >= constraints.lower_bounds)

        assert np.all(action <= constraints.upper_bounds)


def test_move_left_goal() -> None:
    state = np.zeros(
        ROBOTICS_STATE_DIM,
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="move left",
    )

    assert action.values[0] < 0.0


def test_move_right_goal() -> None:
    state = np.zeros(
        ROBOTICS_STATE_DIM,
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="move right",
    )

    assert action.values[0] > 0.0


def test_close_gripper_goal() -> None:
    state = np.zeros(
        ROBOTICS_STATE_DIM,
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="close gripper",
    )

    assert action.values[2] == pytest.approx(1.0)


def test_open_gripper_goal() -> None:
    state = np.zeros(
        ROBOTICS_STATE_DIM,
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="open gripper",
    )

    assert action.values[2] == pytest.approx(-1.0)


def test_move_object_to_target_moves_toward_object_when_far() -> None:
    state = np.array(
        [
            -0.8,
            -0.8,
            0.5,
            0.5,
            0.9,
            0.9,
        ],
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="move object to target",
    )

    assert action.values[0] > 0.0
    assert action.values[1] > 0.0
    assert action.values[2] < 0.0


def test_move_object_to_target_moves_toward_target_when_close() -> None:
    state = np.array(
        [
            0.10,
            0.10,
            0.12,
            0.12,
            0.80,
            0.80,
        ],
        dtype=np.float32,
    )

    action = compute_robotics_target_action(
        state=state,
        language_goal="move object to target",
    )

    assert action.values[0] > 0.0
    assert action.values[1] > 0.0
    assert action.values[2] > 0.0


def test_render_robotics_visual() -> None:
    visual = render_robotics_visual(
        robot_position=(0.0, 0.0),
        object_position=(0.5, 0.5),
        target_position=(-0.5, -0.5),
        image_size=32,
    )

    assert visual.shape == (3, 32, 32)
    assert visual.dtype == np.float32
    assert visual.min() >= 0.0
    assert visual.max() <= 1.0

    assert visual[0].max() == pytest.approx(1.0)
    assert visual[1].max() == pytest.approx(1.0)
    assert visual[2].max() == pytest.approx(1.0)


def test_invalid_dataset_size() -> None:
    with pytest.raises(ValueError):
        SyntheticRoboticsDataset(
            size=0,
            seed=42,
        )


def test_invalid_image_size() -> None:
    with pytest.raises(ValueError):
        SyntheticRoboticsDataset(
            size=1,
            seed=42,
            image_size=4,
        )
