from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class Domain(str, Enum):
    """Supported Q-VLA Forge task domains."""

    AUTONOMOUS_DRIVING = "autonomous_driving"
    ROBOTICS = "robotics"


@dataclass(frozen=True)
class Observation:
    """Shared multimodal observation used by both domains."""

    visual: FloatArray
    state: FloatArray
    language_goal: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.visual, np.ndarray):
            raise TypeError("visual must be a NumPy array")

        if not isinstance(self.state, np.ndarray):
            raise TypeError("state must be a NumPy array")

        if self.visual.dtype != np.float32:
            raise TypeError("visual must use dtype float32")

        if self.state.dtype != np.float32:
            raise TypeError("state must use dtype float32")

        if self.visual.size == 0:
            raise ValueError("visual must not be empty")

        if self.state.size == 0:
            raise ValueError("state must not be empty")

        if not self.language_goal.strip():
            raise ValueError("language_goal must not be empty")


@dataclass(frozen=True)
class Action:
    """Continuous action vector emitted by the policy."""

    values: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array")

        if self.values.dtype != np.float32:
            raise TypeError("values must use dtype float32")

        if self.values.ndim != 1:
            raise ValueError("action values must be one-dimensional")

        if self.values.size == 0:
            raise ValueError("action values must not be empty")


@dataclass(frozen=True)
class SafetyConstraints:
    """Generic lower and upper limits for an action vector."""

    lower_bounds: FloatArray
    upper_bounds: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.lower_bounds, np.ndarray):
            raise TypeError("lower_bounds must be a NumPy array")

        if not isinstance(self.upper_bounds, np.ndarray):
            raise TypeError("upper_bounds must be a NumPy array")

        if self.lower_bounds.dtype != np.float32:
            raise TypeError("lower_bounds must use dtype float32")

        if self.upper_bounds.dtype != np.float32:
            raise TypeError("upper_bounds must use dtype float32")

        if self.lower_bounds.ndim != 1:
            raise ValueError("lower_bounds must be one-dimensional")

        if self.upper_bounds.ndim != 1:
            raise ValueError("upper_bounds must be one-dimensional")

        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have the same shape")

        if self.lower_bounds.size == 0:
            raise ValueError("safety bounds must not be empty")

        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError(
                "every lower bound must be less than or equal to "
                "the corresponding upper bound"
            )


@dataclass(frozen=True)
class TaskSample:
    """One supervised multimodal sample."""

    sample_id: str
    domain: Domain
    observation: Observation
    target_action: Action
    safety_constraints: SafetyConstraints

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")

        if (
            self.target_action.values.shape
            != self.safety_constraints.lower_bounds.shape
        ):
            raise ValueError("target action dimensions must match safety constraints")


@dataclass(frozen=True)
class Transition:
    """One reinforcement-learning transition."""

    domain: Domain
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    safety_constraints: SafetyConstraints
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.values.shape != self.safety_constraints.lower_bounds.shape:
            raise ValueError("action dimensions must match safety constraints")

        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
