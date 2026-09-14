"""Exact one-step state predictors for Sprint 5.7 safety filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q_vla_forge.rl.driving_env import DrivingEnvConfig
from q_vla_forge.rl.robotics_env import RoboticsEnvConfig


@dataclass(frozen=True)
class RoboticsPredictionContext:
    """Hidden simulator context required for exact robotics prediction."""

    object_grasped: bool


@dataclass(frozen=True)
class RoboticsPrediction:
    """Predicted robotics transition state and hidden-context outcome."""

    next_state: np.ndarray
    next_context: RoboticsPredictionContext

    grasped_this_step: bool
    released_this_step: bool
    invalid_grasp: bool


def _validated_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array.copy()


def predict_driving_next_state(
    state: np.ndarray,
    action: np.ndarray,
    *,
    config: DrivingEnvConfig | None = None,
) -> np.ndarray:
    """Predict one frozen DrivingRLEnv state transition exactly."""

    env_config = config if config is not None else DrivingEnvConfig()

    state_array = _validated_array(
        state,
        shape=(4,),
        name="driving state",
    )

    action_array = _validated_array(
        action,
        shape=(3,),
        name="driving action",
    )

    action_low = np.array(
        [
            -1.0,
            -1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    action_high = np.array(
        [
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    clipped_action = np.clip(
        action_array,
        action_low,
        action_high,
    )

    steering = float(clipped_action[0])

    acceleration = float(clipped_action[1])

    braking = float(clipped_action[2])

    speed = float(state_array[0])

    lane_offset = float(state_array[1])

    heading_error = float(state_array[2])

    obstacle_distance = float(state_array[3])

    heading_error = (
        heading_error
        + env_config.steering_gain * steering
        - env_config.heading_damping * heading_error
    )

    lane_offset = lane_offset + env_config.lane_gain * heading_error

    speed = float(
        np.clip(
            speed
            + env_config.acceleration_gain * acceleration
            - env_config.braking_gain * braking,
            0.0,
            env_config.maximum_speed,
        )
    )

    progress = env_config.progress_scale * speed

    obstacle_distance = obstacle_distance - progress

    return np.array(
        [
            speed,
            lane_offset,
            heading_error,
            obstacle_distance,
        ],
        dtype=np.float32,
    )


def predict_robotics_next_state(
    state: np.ndarray,
    action: np.ndarray,
    *,
    context: RoboticsPredictionContext,
    config: RoboticsEnvConfig | None = None,
) -> RoboticsPrediction:
    """Predict one frozen RoboticsRLEnv transition exactly."""

    env_config = config if config is not None else RoboticsEnvConfig()

    state_array = _validated_array(
        state,
        shape=(6,),
        name="robotics state",
    )

    action_array = _validated_array(
        action,
        shape=(3,),
        name="robotics action",
    )

    action_low = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
        ],
        dtype=np.float32,
    )

    action_high = np.array(
        [
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    clipped_action = np.clip(
        action_array,
        action_low,
        action_high,
    )

    delta_x = float(clipped_action[0])

    delta_y = float(clipped_action[1])

    gripper = float(clipped_action[2])

    next_state = state_array.copy()

    robot_before = next_state[0:2].copy()

    movement = env_config.movement_scale * np.array(
        [
            delta_x,
            delta_y,
        ],
        dtype=np.float32,
    )

    robot_after = np.clip(
        robot_before + movement,
        -env_config.workspace_limit,
        env_config.workspace_limit,
    ).astype(np.float32)

    next_state[0:2] = robot_after

    object_grasped = bool(context.object_grasped)

    if object_grasped:
        next_state[2:4] = robot_after

    grasped_this_step = False
    released_this_step = False
    invalid_grasp = False

    if gripper > env_config.close_threshold and not object_grasped:
        robot_to_object = float(np.linalg.norm(next_state[0:2] - next_state[2:4]))

        if robot_to_object <= env_config.grasp_distance:
            object_grasped = True

            next_state[2:4] = next_state[0:2]

            grasped_this_step = True

        else:
            invalid_grasp = True

    if gripper < env_config.open_threshold and object_grasped:
        object_grasped = False
        released_this_step = True

    return RoboticsPrediction(
        next_state=(
            next_state.astype(
                np.float32,
                copy=True,
            )
        ),
        next_context=(RoboticsPredictionContext(object_grasped=(object_grasped))),
        grasped_this_step=(grasped_this_step),
        released_this_step=(released_this_step),
        invalid_grasp=(invalid_grasp),
    )
