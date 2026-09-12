from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.data import (
    DRIVING_ACTION_DIM,
    DRIVING_GOALS,
    DRIVING_STATE_DIM,
    Domain,
    SyntheticDrivingDataset,
    compute_target_action,
    driving_safety_constraints,
    render_driving_visual,
)


def test_driving_dataset_length() -> None:
    dataset = SyntheticDrivingDataset(
        size=10,
        seed=42,
    )

    assert len(dataset) == 10


def test_driving_sample_domain() -> None:
    dataset = SyntheticDrivingDataset(
        size=2,
        seed=42,
    )

    sample = dataset[0]

    assert sample.domain == Domain.AUTONOMOUS_DRIVING
    assert sample.sample_id == "driving-000000"


def test_driving_observation_shapes() -> None:
    dataset = SyntheticDrivingDataset(
        size=1,
        seed=42,
        image_size=32,
    )

    observation = dataset[0].observation

    assert observation.visual.shape == (3, 32, 32)
    assert observation.state.shape == (DRIVING_STATE_DIM,)
    assert observation.visual.dtype == np.float32
    assert observation.state.dtype == np.float32


def test_driving_action_shape() -> None:
    dataset = SyntheticDrivingDataset(
        size=1,
        seed=42,
    )

    action = dataset[0].target_action

    assert action.values.shape == (DRIVING_ACTION_DIM,)
    assert action.values.dtype == np.float32


def test_driving_goal_is_supported() -> None:
    dataset = SyntheticDrivingDataset(
        size=5,
        seed=42,
    )

    for sample in dataset:
        assert sample.observation.language_goal in DRIVING_GOALS


def test_driving_dataset_is_reproducible() -> None:
    first = SyntheticDrivingDataset(
        size=5,
        seed=123,
    )

    second = SyntheticDrivingDataset(
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
    first = SyntheticDrivingDataset(
        size=1,
        seed=42,
    )

    second = SyntheticDrivingDataset(
        size=1,
        seed=123,
    )

    assert not np.array_equal(
        first[0].observation.state,
        second[0].observation.state,
    )


def test_driving_safety_constraints() -> None:
    constraints = driving_safety_constraints()

    assert constraints.lower_bounds.shape == (DRIVING_ACTION_DIM,)
    assert constraints.upper_bounds.shape == (DRIVING_ACTION_DIM,)

    assert np.all(constraints.lower_bounds <= constraints.upper_bounds)


def test_target_action_respects_bounds() -> None:
    dataset = SyntheticDrivingDataset(
        size=100,
        seed=42,
    )

    constraints = driving_safety_constraints()

    for sample in dataset:
        action = sample.target_action.values

        assert np.all(action >= constraints.lower_bounds)

        assert np.all(action <= constraints.upper_bounds)


def test_close_obstacle_causes_braking() -> None:
    state = np.array(
        [
            0.8,
            0.0,
            0.0,
            0.05,
        ],
        dtype=np.float32,
    )

    action = compute_target_action(
        state=state,
        language_goal="keep lane",
    )

    assert action.values[1] == pytest.approx(0.0)
    assert action.values[2] > 0.0


def test_render_driving_visual() -> None:
    visual = render_driving_visual(
        lane_offset=0.0,
        obstacle_distance=0.5,
        image_size=32,
    )

    assert visual.shape == (3, 32, 32)
    assert visual.dtype == np.float32
    assert visual.max() <= 1.0
    assert visual.min() >= 0.0


def test_invalid_dataset_size() -> None:
    with pytest.raises(ValueError):
        SyntheticDrivingDataset(
            size=0,
            seed=42,
        )


def test_invalid_image_size() -> None:
    with pytest.raises(ValueError):
        SyntheticDrivingDataset(
            size=1,
            seed=42,
            image_size=4,
        )
