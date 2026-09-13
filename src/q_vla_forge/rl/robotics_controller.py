"""Deterministic verification controller for the robotics proxy."""

from __future__ import annotations

import numpy as np


def _normalized_direction(
    source: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    """Return a bounded 2-D direction toward a destination."""
    delta = np.asarray(
        destination,
        dtype=np.float32,
    ) - np.asarray(
        source,
        dtype=np.float32,
    )

    max_abs = float(np.max(np.abs(delta)))

    if max_abs <= 1e-8:
        return np.zeros(
            2,
            dtype=np.float32,
        )

    direction = delta / max_abs

    return np.clip(
        direction,
        -1.0,
        1.0,
    ).astype(np.float32)


def deterministic_robotics_action(
    observation: np.ndarray,
    *,
    object_grasped: bool,
    grasp_distance: float = 0.12,
    target_tolerance: float = 0.12,
) -> np.ndarray:
    """Return one deterministic pick-and-place control action."""
    observation = np.asarray(
        observation,
        dtype=np.float32,
    )

    if observation.shape != (6,):
        raise ValueError("robotics observation must have shape (6,)")

    robot = observation[0:2]

    object_position = observation[2:4]

    target = observation[4:6]

    if not object_grasped:
        robot_to_object = float(np.linalg.norm(robot - object_position))

        if robot_to_object <= grasp_distance:
            return np.array(
                [
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )

        movement = _normalized_direction(
            robot,
            object_position,
        )

        return np.array(
            [
                movement[0],
                movement[1],
                0.0,
            ],
            dtype=np.float32,
        )

    robot_to_target = float(np.linalg.norm(robot - target))

    if robot_to_target <= target_tolerance:
        return np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float32,
        )

    movement = _normalized_direction(
        robot,
        target,
    )

    return np.array(
        [
            movement[0],
            movement[1],
            0.0,
        ],
        dtype=np.float32,
    )
