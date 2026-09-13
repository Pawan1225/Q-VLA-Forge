"""Compact autonomous-driving RL proxy environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class DrivingEnvConfig:
    """Frozen configuration for the driving RL proxy."""

    horizon: int = 100
    maximum_speed: float = 1.0

    steering_gain: float = 0.12
    heading_damping: float = 0.08
    lane_gain: float = 0.10

    acceleration_gain: float = 0.08
    braking_gain: float = 0.12

    progress_scale: float = 0.05

    collision_distance: float = 0.0
    terminal_lane_offset: float = 2.0

    initial_obstacle_min: float = 2.5
    initial_obstacle_max: float = 5.0

    initial_speed_min: float = 0.15
    initial_speed_max: float = 0.35

    initial_lane_offset_limit: float = 0.25
    initial_heading_error_limit: float = 0.15


class DrivingRLEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple continuous-control autonomous-driving environment."""

    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": [],
    }

    def __init__(
        self,
        config: DrivingEnvConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else DrivingEnvConfig()

        self.observation_space: spaces.Box = spaces.Box(
            low=np.array(
                [
                    0.0,
                    -np.inf,
                    -np.pi,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.config.maximum_speed,
                    np.inf,
                    np.pi,
                    self.config.initial_obstacle_max,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.action_space: spaces.Box = spaces.Box(
            low=np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
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
            4,
            dtype=np.float32,
        )

        self._step_count = 0
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
    def last_success(self) -> bool:
        """Return whether the most recent episode ended successfully."""
        return self._last_success

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment using Gymnasium deterministic seeding."""
        super().reset(seed=seed)

        del options

        speed = self.np_random.uniform(
            self.config.initial_speed_min,
            self.config.initial_speed_max,
        )

        lane_offset = self.np_random.uniform(
            -self.config.initial_lane_offset_limit,
            self.config.initial_lane_offset_limit,
        )

        heading_error = self.np_random.uniform(
            -self.config.initial_heading_error_limit,
            self.config.initial_heading_error_limit,
        )

        obstacle_distance = self.np_random.uniform(
            self.config.initial_obstacle_min,
            self.config.initial_obstacle_max,
        )

        self._state = np.array(
            [
                speed,
                lane_offset,
                heading_error,
                obstacle_distance,
            ],
            dtype=np.float32,
        )

        self._step_count = 0
        self._last_success = False

        info = {
            "domain": "autonomous_driving",
            "environment": "synthetic_proxy",
            "step_count": 0,
            "horizon": self.config.horizon,
            "success": False,
            "collision": False,
            "lane_departure": False,
            "terminated": False,
            "truncated": False,
            "terminal_lane_offset": self.config.terminal_lane_offset,
            "collision_distance": self.config.collision_distance,
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
        """Advance the frozen proxy dynamics by one environment step."""
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (3,):
            raise ValueError("driving action must have shape (3,)")

        clipped_action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        steering = float(clipped_action[0])
        acceleration = float(clipped_action[1])
        braking = float(clipped_action[2])

        speed = float(self._state[0])
        lane_offset = float(self._state[1])
        heading_error = float(self._state[2])
        obstacle_distance = float(self._state[3])

        heading_error = (
            heading_error
            + self.config.steering_gain * steering
            - self.config.heading_damping * heading_error
        )

        lane_offset = lane_offset + self.config.lane_gain * heading_error

        speed = float(
            np.clip(
                speed
                + self.config.acceleration_gain * acceleration
                - self.config.braking_gain * braking,
                0.0,
                self.config.maximum_speed,
            )
        )

        progress = self.config.progress_scale * speed

        obstacle_distance = obstacle_distance - progress

        self._step_count += 1

        lane_departure = bool(abs(lane_offset) >= self.config.terminal_lane_offset)

        collision = bool(obstacle_distance <= self.config.collision_distance)

        terminated = bool(collision or lane_departure)

        truncated = bool(self._step_count >= self.config.horizon and not terminated)

        success = bool(truncated and not collision and not lane_departure)

        reward = self._reward(
            speed=speed,
            lane_offset=lane_offset,
            heading_error=heading_error,
            steering=steering,
            acceleration=acceleration,
            braking=braking,
            collision=collision,
            lane_departure=lane_departure,
            success=success,
        )

        self._state = np.array(
            [
                speed,
                lane_offset,
                heading_error,
                obstacle_distance,
            ],
            dtype=np.float32,
        )

        self._last_success = success

        info = {
            "domain": "autonomous_driving",
            "environment": "synthetic_proxy",
            "step_count": self._step_count,
            "horizon": self.config.horizon,
            "progress": float(progress),
            "speed": float(speed),
            "lane_offset": float(lane_offset),
            "heading_error": float(heading_error),
            "obstacle_distance": float(obstacle_distance),
            "success": success,
            "collision": collision,
            "lane_departure": lane_departure,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_lane_offset": (self.config.terminal_lane_offset),
            "collision_distance": (self.config.collision_distance),
        }

        return (
            self.state,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _reward(
        self,
        *,
        speed: float,
        lane_offset: float,
        heading_error: float,
        steering: float,
        acceleration: float,
        braking: float,
        collision: bool,
        lane_departure: bool,
        success: bool,
    ) -> float:
        """Return the frozen Sprint 4.2 driving reward."""
        reward = 0.0

        reward += 0.10 * speed

        reward -= 1.00 * abs(lane_offset)

        reward -= 0.50 * abs(heading_error)

        reward -= 0.05 * steering**2

        reward -= 0.02 * acceleration**2

        reward -= 0.02 * braking**2

        if collision:
            reward -= 10.0

        if lane_departure:
            reward -= 5.0

        if success:
            reward += 5.0

        return reward
