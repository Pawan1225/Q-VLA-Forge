"""Synthetic robotics manipulation environment for Sprint 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RoboticsEnvConfig:
    """Frozen configuration for the robotics RL proxy."""

    horizon: int = 100
    workspace_limit: float = 1.0

    movement_scale: float = 0.08

    grasp_distance: float = 0.12
    target_tolerance: float = 0.12

    close_threshold: float = 0.5
    open_threshold: float = -0.5

    minimum_initial_separation: float = 0.35


class RoboticsRLEnv(gym.Env[np.ndarray, np.ndarray]):
    """Compact continuous-control pick-and-place proxy."""

    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": [],
    }

    def __init__(
        self,
        config: RoboticsEnvConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else RoboticsEnvConfig()

        limit = self.config.workspace_limit

        self.observation_space: spaces.Box = spaces.Box(
            low=np.full(
                6,
                -limit,
                dtype=np.float32,
            ),
            high=np.full(
                6,
                limit,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.action_space: spaces.Box = spaces.Box(
            low=np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._state = np.zeros(
            6,
            dtype=np.float32,
        )

        self._step_count = 0
        self._object_grasped = False
        self._last_success = False

    @property
    def state(self) -> np.ndarray:
        """Return a defensive copy of the current state."""
        return self._state.copy()

    @property
    def step_count(self) -> int:
        """Return the current episode step count."""
        return self._step_count

    @property
    def object_grasped(self) -> bool:
        """Return whether the object is currently grasped."""
        return self._object_grasped

    @property
    def last_success(self) -> bool:
        """Return whether the latest episode ended successfully."""
        return self._last_success

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the robotics proxy deterministically."""
        super().reset(seed=seed)

        del options

        robot = self._sample_point()

        object_position = self._sample_separated_point(references=(robot,))

        target = self._sample_separated_point(
            references=(
                robot,
                object_position,
            )
        )

        self._state = np.array(
            [
                robot[0],
                robot[1],
                object_position[0],
                object_position[1],
                target[0],
                target[1],
            ],
            dtype=np.float32,
        )

        self._step_count = 0
        self._object_grasped = False
        self._last_success = False

        info = {
            "domain": "robotics",
            "environment": "synthetic_proxy",
            "task": "2d_pick_and_place",
            "step_count": 0,
            "horizon": self.config.horizon,
            "success": False,
            "object_grasped": False,
            "grasped_this_step": False,
            "released_this_step": False,
            "invalid_grasp": False,
            "failed_release": False,
            "grasp_distance": self.config.grasp_distance,
            "target_tolerance": self.config.target_tolerance,
            "movement_scale": self.config.movement_scale,
            "workspace_limit": self.config.workspace_limit,
            "close_threshold": self.config.close_threshold,
            "open_threshold": self.config.open_threshold,
            "minimum_initial_separation": (self.config.minimum_initial_separation),
            "progress": 0.0,
            "action_cost": 0.0,
            "terminated": False,
            "truncated": False,
            "robot_position": self._robot_position().tolist(),
            "object_position": self._object_position().tolist(),
            "target_position": self._target_position().tolist(),
        }

        return self.state, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Advance one robotics proxy manipulation step."""
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (3,):
            raise ValueError("robotics action must have shape (3,)")

        clipped_action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        delta_x = float(clipped_action[0])

        delta_y = float(clipped_action[1])

        gripper = float(clipped_action[2])

        robot_before = self._robot_position()
        object_before = self._object_position()
        target = self._target_position()

        was_grasped = self._object_grasped

        if was_grasped:
            distance_before = float(np.linalg.norm(robot_before - target))
        else:
            distance_before = float(np.linalg.norm(robot_before - object_before))

        movement = self.config.movement_scale * np.array(
            [
                delta_x,
                delta_y,
            ],
            dtype=np.float32,
        )

        robot_after = np.clip(
            robot_before + movement,
            -self.config.workspace_limit,
            self.config.workspace_limit,
        ).astype(np.float32)

        self._state[0:2] = robot_after

        if self._object_grasped:
            self._state[2:4] = robot_after

        grasped_this_step = False
        released_this_step = False
        invalid_grasp = False

        if gripper > self.config.close_threshold and not self._object_grasped:
            robot_to_object = float(
                np.linalg.norm(self._robot_position() - self._object_position())
            )

            if robot_to_object <= self.config.grasp_distance:
                self._object_grasped = True

                self._state[2:4] = self._robot_position()

                grasped_this_step = True

            else:
                invalid_grasp = True

        if gripper < self.config.open_threshold and self._object_grasped:
            self._object_grasped = False
            released_this_step = True

        object_after = self._object_position()

        success = bool(
            released_this_step
            and float(np.linalg.norm(object_after - target))
            <= self.config.target_tolerance
        )

        if grasped_this_step:
            progress = 0.0

        elif released_this_step:
            distance_after = float(np.linalg.norm(object_after - target))

            progress = distance_before - distance_after

        elif self._object_grasped:
            distance_after = float(np.linalg.norm(self._robot_position() - target))

            progress = distance_before - distance_after

        else:
            distance_after = float(
                np.linalg.norm(self._robot_position() - self._object_position())
            )

            progress = distance_before - distance_after

        failed_release = bool(released_this_step and not success)

        action_cost = delta_x**2 + delta_y**2

        reward = 2.0 * progress - 0.01 - 0.01 * action_cost

        if grasped_this_step:
            reward += 1.0

        if invalid_grasp:
            reward -= 0.05

        if failed_release:
            reward -= 0.10

        if success:
            reward += 10.0

        self._step_count += 1

        terminated = success

        truncated = bool(self._step_count >= self.config.horizon and not terminated)

        self._last_success = success

        info = {
            "domain": "robotics",
            "environment": "synthetic_proxy",
            "task": "2d_pick_and_place",
            "step_count": self._step_count,
            "horizon": self.config.horizon,
            "success": success,
            "object_grasped": self._object_grasped,
            "grasped_this_step": grasped_this_step,
            "released_this_step": released_this_step,
            "invalid_grasp": invalid_grasp,
            "failed_release": failed_release,
            "grasp_distance": self.config.grasp_distance,
            "target_tolerance": self.config.target_tolerance,
            "movement_scale": self.config.movement_scale,
            "workspace_limit": self.config.workspace_limit,
            "close_threshold": self.config.close_threshold,
            "open_threshold": self.config.open_threshold,
            "minimum_initial_separation": (self.config.minimum_initial_separation),
            "progress": progress,
            "action_cost": action_cost,
            "terminated": terminated,
            "truncated": truncated,
            "robot_position": self._robot_position().tolist(),
            "object_position": self._object_position().tolist(),
            "target_position": self._target_position().tolist(),
        }

        return (
            self.state,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _sample_point(
        self,
    ) -> np.ndarray:
        """Sample one point inside the initialization region."""
        return self.np_random.uniform(
            low=-0.8,
            high=0.8,
            size=2,
        ).astype(np.float32)

    def _sample_separated_point(
        self,
        *,
        references: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Sample a point separated from all supplied references."""
        for _ in range(1_000):
            candidate = self._sample_point()

            if all(
                float(np.linalg.norm(candidate - reference))
                >= self.config.minimum_initial_separation
                for reference in references
            ):
                return candidate

        raise RuntimeError(
            "unable to sample sufficiently separated " "robotics initial state"
        )

    def _robot_position(
        self,
    ) -> np.ndarray:
        """Return the current robot position."""
        return self._state[0:2].copy()

    def _object_position(
        self,
    ) -> np.ndarray:
        """Return the current object position."""
        return self._state[2:4].copy()

    def _target_position(
        self,
    ) -> np.ndarray:
        """Return the current target position."""
        return self._state[4:6].copy()
