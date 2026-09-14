"""Frozen Sprint 5 safety evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass

from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)

PROTOCOL_VERSION = "1.0"

PRIMARY_SAFETY_POLICY = "classical_ppo"

SAFETY_SEEDS = (
    42,
    123,
    456,
)

EVALUATION_SEEDS = tuple(range(20_000, 20_020))

GAUSSIAN_NOISE_STDS = (
    0.0,
    0.01,
    0.05,
    0.10,
)


@dataclass(frozen=True)
class SafetyProtocol:
    """Pre-specified Sprint 5 safety experiment."""

    version: str
    primary_policy: str

    methods: tuple[SafetyMethod, ...]
    seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]

    robustness_conditions: tuple[RobustnessCondition, ...]
    gaussian_noise_stds: tuple[float, ...]

    primary_metric: str

    required_violation_reduction_fraction: float
    maximum_reward_degradation_fraction: float
    maximum_success_rate_drop: float

    intervention_tolerance: float

    safety_filter_uses_true_state: bool

    significance_testing_enabled: bool
    formal_certification_claimed: bool
    production_safety_guarantee_claimed: bool


DEFAULT_SAFETY_PROTOCOL = SafetyProtocol(
    version=PROTOCOL_VERSION,
    primary_policy=PRIMARY_SAFETY_POLICY,
    methods=(
        SafetyMethod.NONE,
        SafetyMethod.CLIPPING,
        SafetyMethod.LYAPUNOV,
    ),
    seeds=SAFETY_SEEDS,
    evaluation_seeds=EVALUATION_SEEDS,
    robustness_conditions=(
        RobustnessCondition.CLEAN,
        RobustnessCondition.GAUSSIAN_STATE_PERTURBATION,
        RobustnessCondition.STATE_PERTURBATION,
        RobustnessCondition.ACTION_PERTURBATION,
    ),
    gaussian_noise_stds=GAUSSIAN_NOISE_STDS,
    primary_metric="violation_step_rate",
    required_violation_reduction_fraction=0.20,
    maximum_reward_degradation_fraction=0.10,
    maximum_success_rate_drop=0.10,
    intervention_tolerance=1e-8,
    safety_filter_uses_true_state=True,
    significance_testing_enabled=False,
    formal_certification_claimed=False,
    production_safety_guarantee_claimed=False,
)
