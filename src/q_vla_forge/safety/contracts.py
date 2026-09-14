"""Safety contracts for Q-VLA Forge Sprint 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SafetyMethod(str, Enum):
    """Frozen Sprint 5 safety methods."""

    NONE = "none"
    CLIPPING = "clipping"
    LYAPUNOV = "lyapunov"


class RobustnessCondition(str, Enum):
    """Frozen Sprint 5 robustness conditions."""

    CLEAN = "clean"
    GAUSSIAN_STATE_PERTURBATION = "gaussian_state_perturbation"
    STATE_PERTURBATION = "state_perturbation"
    ACTION_PERTURBATION = "action_perturbation"


class ViolationSeverity(str, Enum):
    """Pilot violation severity."""

    WARNING = "warning"
    CRITICAL = "critical"


class InterventionReason(str, Enum):
    """Reason a proposed action was modified."""

    NONE = "none"
    ACTION_BOUND = "action_bound"
    DOMAIN_CONSTRAINT = "domain_constraint"
    LYAPUNOV_DECREASE = "lyapunov_decrease"
    EMERGENCY_FALLBACK = "emergency_fallback"


@dataclass(frozen=True)
class ViolationRecord:
    """One evaluated safety constraint."""

    name: str
    severity: ViolationSeverity
    value: float
    threshold: float
    violated: bool


@dataclass(frozen=True)
class SafetyDecision:
    """Result of applying a safety method to one proposed action."""

    method: SafetyMethod

    proposed_action: np.ndarray
    executed_action: np.ndarray

    intervened: bool
    intervention_reason: InterventionReason

    correction_l2: float

    violations_before: tuple[ViolationRecord, ...] = ()
    violations_after: tuple[ViolationRecord, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyStepRecord:
    """Safety information for one environment step."""

    step_index: int

    true_state: np.ndarray
    observed_state: np.ndarray

    proposed_action: np.ndarray
    executed_action: np.ndarray

    decision: SafetyDecision

    step_has_violation: bool
    constraint_violation_count: int


@dataclass(frozen=True)
class SafetyEpisodeRecord:
    """Safety summary for one evaluation episode."""

    domain: str
    seed: int
    evaluation_seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    reward: float
    success: bool
    episode_length: int

    violation_step_count: int
    constraint_violation_count: int

    intervention_count: int

    mean_action_correction_l2: float
    max_action_correction_l2: float

    recovered_to_safe_region: bool | None = None
    steps_to_recovery: int | None = None


@dataclass(frozen=True)
class SafetyEvaluationRecord:
    """Aggregated safety evaluation for one experiment cell."""

    domain: str
    seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    episode_count: int
    total_environment_steps: int

    mean_reward: float
    reward_standard_deviation: float
    success_rate: float
    mean_episode_length: float

    violation_step_rate: float
    constraint_violation_rate: float

    intervention_rate: float

    mean_action_correction_l2: float
    p95_action_correction_l2: float
    max_action_correction_l2: float
