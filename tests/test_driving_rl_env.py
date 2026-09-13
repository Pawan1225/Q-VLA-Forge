"""Deterministic tests for the autonomous-driving RL proxy."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.rl.driving_env import (
    DrivingEnvConfig,
    DrivingRLEnv,
)


def test_reset_is_deterministic_for_same_seed() -> None:
    env = DrivingRLEnv()

    first_observation, first_info = env.reset(seed=42)
    second_observation, second_info = env.reset(seed=42)

    np.testing.assert_allclose(
        first_observation,
        second_observation,
    )

    assert first_info == second_info


def test_different_reset_seeds_change_state() -> None:
    env = DrivingRLEnv()

    first_observation, _ = env.reset(seed=42)
    second_observation, _ = env.reset(seed=123)

    assert not np.allclose(
        first_observation,
        second_observation,
    )


def test_reset_observation_is_inside_space() -> None:
    env = DrivingRLEnv()

    observation, _ = env.reset(seed=42)

    assert env.observation_space.contains(observation)


def test_action_space_matches_frozen_contract() -> None:
    env = DrivingRLEnv()

    assert env.action_space.shape == (3,)

    np.testing.assert_allclose(
        env.action_space.low,
        np.array(
            [-1.0, -1.0, 0.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        env.action_space.high,
        np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_invalid_action_shape_is_rejected() -> None:
    env = DrivingRLEnv()
    env.reset(seed=42)

    with pytest.raises(
        ValueError,
        match="driving action must have shape",
    ):
        env.step(
            np.zeros(
                2,
                dtype=np.float32,
            )
        )


def test_frozen_dynamics_are_deterministic() -> None:
    env = DrivingRLEnv()

    initial_observation, _ = env.reset(seed=42)

    action = np.array(
        [0.5, 1.0, 0.0],
        dtype=np.float32,
    )

    observation, _, terminated, truncated, _ = env.step(action)

    speed = float(initial_observation[0])
    lane_offset = float(initial_observation[1])
    heading_error = float(initial_observation[2])
    obstacle_distance = float(initial_observation[3])

    expected_heading = heading_error + 0.12 * 0.5 - 0.08 * heading_error

    expected_lane = lane_offset + 0.10 * expected_heading

    expected_speed = min(
        max(
            speed + 0.08,
            0.0,
        ),
        1.0,
    )

    expected_progress = 0.05 * expected_speed

    expected_obstacle = obstacle_distance - expected_progress

    expected = np.array(
        [
            expected_speed,
            expected_lane,
            expected_heading,
            expected_obstacle,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        observation,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    assert terminated is False
    assert truncated is False


def test_action_values_are_clipped_to_contract() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            initial_speed_min=0.50,
            initial_speed_max=0.50,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    observation, _, _, _, _ = env.step(
        np.array(
            [10.0, 10.0, -10.0],
            dtype=np.float32,
        )
    )

    expected_heading = 0.12
    expected_lane = 0.012
    expected_speed = 0.58

    assert observation[0] == pytest.approx(
        expected_speed,
    )

    assert observation[1] == pytest.approx(
        expected_lane,
    )

    assert observation[2] == pytest.approx(
        expected_heading,
    )


def test_normal_step_is_not_terminal() -> None:
    env = DrivingRLEnv()

    env.reset(seed=42)

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert terminated is False
    assert truncated is False
    assert info["collision"] is False
    assert info["lane_departure"] is False
    assert info["success"] is False


def test_horizon_completion_is_successful_truncation() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=3,
            initial_obstacle_min=10.0,
            initial_obstacle_max=10.0,
            initial_speed_min=0.20,
            initial_speed_max=0.20,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    terminated = False
    truncated = False
    reward = 0.0
    info: dict[str, object] = {}

    for _ in range(3):
        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )

    assert terminated is False
    assert truncated is True
    assert info["success"] is True
    assert env.last_success is True
    assert reward > 4.0


def test_step_info_contains_provenance() -> None:
    env = DrivingRLEnv()

    env.reset(seed=42)

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["domain"] == "autonomous_driving"
    assert info["environment"] == "synthetic_proxy"
    assert info["step_count"] == 1
    assert info["horizon"] == 100
    assert info["terminated"] is terminated
    assert info["truncated"] is truncated
    assert info["terminal_lane_offset"] == 2.0
    assert info["collision_distance"] == 0.0


def test_collision_causes_termination_and_penalty() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            initial_obstacle_min=0.01,
            initial_obstacle_max=0.01,
            initial_speed_min=1.0,
            initial_speed_max=1.0,
        )
    )

    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["collision"] is True
    assert info["lane_departure"] is False
    assert info["success"] is False

    assert terminated is True
    assert truncated is False

    assert reward < -9.0

    assert env.observation_space.contains(observation)


def test_lane_departure_causes_termination_and_penalty() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            terminal_lane_offset=0.20,
            initial_lane_offset_limit=0.19,
            initial_obstacle_min=10.0,
            initial_obstacle_max=10.0,
        )
    )

    env.reset(seed=42)

    terminated = False
    truncated = False
    reward = 0.0
    observation = np.zeros(
        4,
        dtype=np.float32,
    )
    info: dict[str, object] = {}

    for _ in range(50):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )

        if terminated:
            break

    assert info["lane_departure"] is True
    assert info["collision"] is False
    assert info["success"] is False

    assert terminated is True
    assert truncated is False

    assert reward < -5.0

    assert env.observation_space.contains(observation)


def test_exact_positive_lane_boundary_is_departure() -> None:
    env = DrivingRLEnv()

    env.reset(seed=42)

    env._state = np.array(
        [
            0.0,
            2.0,
            0.0,
            5.0,
        ],
        dtype=np.float32,
    )

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["lane_departure"] is True
    assert terminated is True
    assert truncated is False


def test_exact_negative_lane_boundary_is_departure() -> None:
    env = DrivingRLEnv()

    env.reset(seed=42)

    env._state = np.array(
        [
            0.0,
            -2.0,
            0.0,
            5.0,
        ],
        dtype=np.float32,
    )

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["lane_departure"] is True
    assert terminated is True
    assert truncated is False


def test_success_bonus_applies_only_on_successful_horizon() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=1,
            initial_obstacle_min=10.0,
            initial_obstacle_max=10.0,
            initial_speed_min=0.20,
            initial_speed_max=0.20,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert terminated is False
    assert truncated is True
    assert info["success"] is True
    assert reward == pytest.approx(
        5.02,
        abs=1e-6,
    )


def test_collision_penalty_does_not_receive_success_bonus() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=1,
            initial_obstacle_min=0.01,
            initial_obstacle_max=0.01,
            initial_speed_min=1.0,
            initial_speed_max=1.0,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert info["success"] is False

    assert reward == pytest.approx(
        -9.9,
        abs=1e-6,
    )


def test_lane_departure_penalty_does_not_receive_success_bonus() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=1,
            terminal_lane_offset=0.10,
            initial_obstacle_min=10.0,
            initial_obstacle_max=10.0,
            initial_speed_min=0.0,
            initial_speed_max=0.0,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert terminated is False
    assert truncated is True
    assert info["lane_departure"] is False
    assert info["success"] is True

    assert reward == pytest.approx(
        4.878,
        abs=1e-6,
    )

    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=1,
            terminal_lane_offset=0.01,
            initial_obstacle_min=10.0,
            initial_obstacle_max=10.0,
            initial_speed_min=0.0,
            initial_speed_max=0.0,
            initial_lane_offset_limit=0.0,
            initial_heading_error_limit=0.0,
        )
    )

    env.reset(seed=42)

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert terminated is True
    assert truncated is False
    assert info["lane_departure"] is True
    assert info["success"] is False

    assert reward < -5.0


def test_terminal_event_takes_priority_over_horizon_truncation() -> None:
    env = DrivingRLEnv(
        DrivingEnvConfig(
            horizon=1,
            initial_obstacle_min=0.01,
            initial_obstacle_max=0.01,
            initial_speed_min=1.0,
            initial_speed_max=1.0,
        )
    )

    env.reset(seed=42)

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert env.step_count == 1
    assert terminated is True
    assert truncated is False
    assert info["success"] is False
