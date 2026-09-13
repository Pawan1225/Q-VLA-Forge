"""Simple baseline controllers for the driving RL proxy."""

from __future__ import annotations

import numpy as np


def zero_action_controller(
    observation: np.ndarray,
) -> np.ndarray:
    """Return a deterministic no-op driving action."""
    del observation

    return np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def heuristic_driving_action(
    observation: np.ndarray,
) -> np.ndarray:
    """Return a task-aware deterministic driving action."""
    observation = np.asarray(
        observation,
        dtype=np.float32,
    )

    if observation.shape != (4,):
        raise ValueError("driving observation must have shape (4,)")

    speed = float(observation[0])

    lane_offset = float(observation[1])

    heading_error = float(observation[2])

    obstacle_distance = float(observation[3])

    steering = float(
        np.clip(
            -1.5 * lane_offset - heading_error,
            -1.0,
            1.0,
        )
    )

    if obstacle_distance < 0.75:
        acceleration = -0.5
        braking = 1.0

    elif obstacle_distance < 1.5:
        acceleration = 0.0
        braking = 0.5

    else:
        target_speed = 0.60

        acceleration = float(
            np.clip(
                2.0 * (target_speed - speed),
                -1.0,
                1.0,
            )
        )

        braking = 0.0

    return np.array(
        [
            steering,
            acceleration,
            braking,
        ],
        dtype=np.float32,
    )
