"""Core deterministic tests for the robotics RL proxy."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.rl.robotics_controller import (
    deterministic_robotics_action,
)
from q_vla_forge.rl.robotics_env import (
    RoboticsEnvConfig,
    RoboticsRLEnv,
)


def test_default_config_contract() -> None:
    config = RoboticsEnvConfig()

    assert config.horizon == 100
    assert config.workspace_limit == 1.0
    assert config.movement_scale == 0.08
    assert config.grasp_distance == 0.12
    assert config.target_tolerance == 0.12
    assert config.close_threshold == 0.5
    assert config.open_threshold == -0.5
    assert config.minimum_initial_separation == 0.35


def test_observation_space_contract() -> None:
    env = RoboticsRLEnv()

    assert env.observation_space.shape == (6,)
    assert env.observation_space.dtype == np.float32

    np.testing.assert_allclose(
        env.observation_space.low,
        np.full(
            6,
            -1.0,
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        env.observation_space.high,
        np.full(
            6,
            1.0,
            dtype=np.float32,
        ),
    )


def test_action_space_contract() -> None:
    env = RoboticsRLEnv()

    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32

    np.testing.assert_allclose(
        env.action_space.low,
        np.array(
            [-1.0, -1.0, -1.0],
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


def test_reset_same_seed_is_deterministic() -> None:
    env = RoboticsRLEnv()

    first, first_info = env.reset(
        seed=42,
    )

    second, second_info = env.reset(
        seed=42,
    )

    np.testing.assert_allclose(
        first,
        second,
    )

    assert first_info == second_info


def test_reset_different_seed_changes_state() -> None:
    env = RoboticsRLEnv()

    first, _ = env.reset(
        seed=42,
    )

    second, _ = env.reset(
        seed=123,
    )

    assert not np.allclose(
        first,
        second,
    )


def test_reset_initial_separation_contract() -> None:
    env = RoboticsRLEnv()

    observation, _ = env.reset(
        seed=42,
    )

    robot = observation[0:2]
    object_position = observation[2:4]
    target = observation[4:6]

    assert (
        float(np.linalg.norm(robot - object_position))
        >= env.config.minimum_initial_separation
    )

    assert (
        float(np.linalg.norm(robot - target)) >= env.config.minimum_initial_separation
    )

    assert (
        float(np.linalg.norm(object_position - target))
        >= env.config.minimum_initial_separation
    )


def test_reset_observation_is_valid() -> None:
    env = RoboticsRLEnv()

    observation, _ = env.reset(
        seed=42,
    )

    assert observation.shape == (6,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)


def test_state_property_returns_copy() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    state = env.state
    original = env.state

    state[0] = 999.0

    np.testing.assert_allclose(
        env.state,
        original,
    )


def test_basic_robot_movement() -> None:
    env = RoboticsRLEnv()

    observation, _ = env.reset(
        seed=42,
    )

    before = observation[0:2].copy()

    observation, _, terminated, truncated, _ = env.step(
        np.array(
            [1.0, -1.0, 0.0],
            dtype=np.float32,
        )
    )

    expected = before + np.array(
        [0.08, -0.08],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        observation[0:2],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    assert terminated is False
    assert truncated is False


def test_robot_movement_is_clipped_to_workspace() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.98, -0.98],
        dtype=np.float32,
    )

    observation, _, _, _, _ = env.step(
        np.array(
            [1.0, -1.0, 0.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        observation[0:2],
        np.array(
            [1.0, -1.0],
            dtype=np.float32,
        ),
    )


def test_action_is_clipped_before_movement() -> None:
    env = RoboticsRLEnv()

    observation, _ = env.reset(
        seed=42,
    )

    before = observation[0:2].copy()

    observation, _, _, _, _ = env.step(
        np.array(
            [10.0, -10.0, 0.0],
            dtype=np.float32,
        )
    )

    expected = np.clip(
        before
        + np.array(
            [0.08, -0.08],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )

    np.testing.assert_allclose(
        observation[0:2],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_malformed_action_shape_is_rejected() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match=r"robotics action must have shape \(3,\)",
    ):
        env.step(
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            )
        )


def test_invalid_grasp_does_not_grasp_object() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    _, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["invalid_grasp"] is True
    assert info["grasped_this_step"] is False
    assert env.object_grasped is False


def test_valid_grasp_sets_object_grasped() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    observation, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["grasped_this_step"] is True
    assert info["invalid_grasp"] is False
    assert env.object_grasped is True

    np.testing.assert_allclose(
        observation[0:2],
        observation[2:4],
    )


def test_grasped_object_follows_robot() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    observation, _, _, _, _ = env.step(
        np.array(
            [1.0, -0.5, 0.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        observation[0:2],
        observation[2:4],
    )

    np.testing.assert_allclose(
        observation[0:2],
        np.array(
            [0.08, -0.04],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_horizon_failure_truncates() -> None:
    env = RoboticsRLEnv(
        RoboticsEnvConfig(
            horizon=2,
        )
    )

    env.reset(
        seed=42,
    )

    _, _, terminated, truncated, _ = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert terminated is False
    assert truncated is True
    assert info["success"] is False


def test_controller_moves_toward_object() -> None:
    observation = np.array(
        [
            0.0,
            0.0,
            0.5,
            0.25,
            -0.5,
            -0.5,
        ],
        dtype=np.float32,
    )

    action = deterministic_robotics_action(
        observation,
        object_grasped=False,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [1.0, 0.5, 0.0],
            dtype=np.float32,
        ),
    )


def test_controller_grasps_when_close() -> None:
    observation = np.array(
        [
            0.0,
            0.0,
            0.05,
            0.0,
            0.8,
            0.8,
        ],
        dtype=np.float32,
    )

    action = deterministic_robotics_action(
        observation,
        object_grasped=False,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_controller_releases_near_target() -> None:
    observation = np.array(
        [
            0.02,
            0.02,
            0.02,
            0.02,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    action = deterministic_robotics_action(
        observation,
        object_grasped=True,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        ),
    )


def test_controller_rejects_bad_observation_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"robotics observation must have shape \(6,\)",
    ):
        deterministic_robotics_action(
            np.zeros(
                5,
                dtype=np.float32,
            ),
            object_grasped=False,
        )


def test_grasp_at_exact_distance_threshold_succeeds() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.12, 0.0],
        dtype=np.float32,
    )

    _, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["grasped_this_step"] is True
    assert info["invalid_grasp"] is False
    assert env.object_grasped is True


def test_grasp_just_outside_threshold_fails() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.1201, 0.0],
        dtype=np.float32,
    )

    _, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["grasped_this_step"] is False
    assert info["invalid_grasp"] is True
    assert env.object_grasped is False


def test_gripper_equal_to_close_threshold_does_not_grasp() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    _, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 0.5],
            dtype=np.float32,
        )
    )

    assert info["grasped_this_step"] is False
    assert info["invalid_grasp"] is False
    assert env.object_grasped is False


def test_gripper_equal_to_open_threshold_does_not_release() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, _, _, _, info = env.step(
        np.array(
            [0.0, 0.0, -0.5],
            dtype=np.float32,
        )
    )

    assert info["released_this_step"] is False
    assert env.object_grasped is True


def test_release_without_grasp_is_not_release_event() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    _, _, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert info["released_this_step"] is False
    assert info["failed_release"] is False
    assert info["success"] is False
    assert env.object_grasped is False
    assert terminated is False
    assert truncated is False


def test_success_at_exact_target_tolerance() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.12, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert info["success"] is True
    assert info["failed_release"] is False
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(
        9.99,
        abs=1e-6,
    )


def test_release_just_outside_target_tolerance_fails() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.1201, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert info["success"] is False
    assert info["failed_release"] is True
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(
        -0.11,
        abs=1e-6,
    )


def test_success_on_horizon_takes_priority_over_truncation() -> None:
    env = RoboticsRLEnv(
        RoboticsEnvConfig(
            horizon=2,
        )
    )

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, _, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert env.step_count == 2
    assert info["success"] is True
    assert terminated is True
    assert truncated is False


def test_reset_clears_grasp_and_success_state() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert env.last_success is True

    observation, info = env.reset(
        seed=123,
    )

    assert env.step_count == 0
    assert env.object_grasped is False
    assert env.last_success is False
    assert info["success"] is False
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert env.observation_space.contains(observation)


def test_progress_toward_object_produces_expected_reward() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert info["progress"] == pytest.approx(
        0.08,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        0.14,
        abs=1e-6,
    )

    assert terminated is False
    assert truncated is False


def test_moving_away_from_object_produces_negative_progress() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    _, reward, _, _, info = env.step(
        np.array(
            [-1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert info["progress"] == pytest.approx(
        -0.08,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        -0.18,
        abs=1e-6,
    )


def test_valid_grasp_reward_has_no_progress_artifact() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    _, reward, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["grasped_this_step"] is True

    assert info["progress"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        0.99,
        abs=1e-6,
    )


def test_invalid_grasp_penalty_is_applied() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    _, reward, _, _, info = env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    assert info["invalid_grasp"] is True
    assert info["progress"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        -0.06,
        abs=1e-6,
    )


def test_grasped_motion_uses_target_progress() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    observation, reward, _, _, info = env.step(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert env.object_grasped is True

    np.testing.assert_allclose(
        observation[0:2],
        observation[2:4],
    )

    assert info["progress"] == pytest.approx(
        0.08,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        0.14,
        abs=1e-6,
    )


def test_failed_release_penalty_has_no_objective_switch_artifact() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.8, 0.8],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert info["released_this_step"] is True
    assert info["failed_release"] is True
    assert info["success"] is False

    assert info["progress"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        -0.11,
        abs=1e-6,
    )

    assert terminated is False
    assert truncated is False


def test_success_reward_has_no_release_progress_artifact() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.05, 0.0],
        dtype=np.float32,
    )

    env._state[4:6] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    _, reward, terminated, truncated, info = env.step(
        np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )
    )

    assert info["success"] is True

    assert info["progress"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert reward == pytest.approx(
        9.99,
        abs=1e-6,
    )

    assert terminated is True
    assert truncated is False


def test_action_cost_uses_clipped_action() -> None:
    env = RoboticsRLEnv()

    env.reset(
        seed=42,
    )

    env._state[0:2] = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env._state[2:4] = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    _, _, _, _, info = env.step(
        np.array(
            [10.0, -10.0, 0.0],
            dtype=np.float32,
        )
    )

    assert info["action_cost"] == pytest.approx(
        2.0,
        abs=1e-6,
    )


def test_horizon_failure_and_success_remain_mutually_exclusive() -> None:
    env = RoboticsRLEnv(
        RoboticsEnvConfig(
            horizon=1,
        )
    )

    env.reset(
        seed=42,
    )

    _, _, terminated, truncated, info = env.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["success"] is False
    assert terminated is False
    assert truncated is True

    assert not (terminated and truncated)
