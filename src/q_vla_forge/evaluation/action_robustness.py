"""Sprint 5.12 action-perturbation robustness contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean, stdev

import numpy as np


@dataclass(frozen=True)
class ActionRobustnessEpisodeEvidence:
    """One Sprint 5.12 action-perturbation episode."""

    domain: str
    method: str
    perturbation_name: str
    perturbation_family: str

    principal_seed: int
    evaluation_seed: int

    reward: float
    success: bool
    episode_length: int

    perturbed_violation_step_count: int
    executed_violation_step_count: int

    perturbed_constraint_violation_count: int
    executed_constraint_violation_count: int

    critical_violation_step_count: int

    unsafe_perturbed_steps: int
    recovered_unsafe_steps: int
    unresolved_unsafe_steps: int

    intervention_count: int

    action_perturbation_l2_sum: float
    action_perturbation_linf_max: float

    safety_correction_l2_sum: float = 0.0
    safety_correction_l2_values: tuple[float, ...] = ()
    safety_correction_l2_max: float = 0.0

    total_action_displacement_l2_sum: float = 0.0

    environment_interface_adjustment_count: int = 0
    environment_interface_adjustment_l2_sum: float = 0.0
    environment_interface_adjustment_l2_max: float = 0.0

    perturbed_category_violation_counts: Mapping[str, int] = field(default_factory=dict)

    executed_category_violation_counts: Mapping[str, int] = field(default_factory=dict)

    recovered_category_counts: Mapping[str, int] = field(default_factory=dict)

    intervention_reason_counts: Mapping[str, int] = field(default_factory=dict)

    selected_candidate_source_counts: Mapping[str, int] = field(default_factory=dict)

    strict_lyapunov_decrease_count: int = 0
    lyapunov_nonincrease_count: int = 0
    selected_lower_than_perturbed_count: int = 0
    emergency_fallback_count: int = 0

    steps_object_grasped: int = 0
    episodes_with_object_grasped: int = 0
    perturbed_unsafe_steps_while_grasped: int = 0
    interventions_while_grasped: int = 0

    proposed_to_perturbed_gripper_semantic_changes: int = 0
    perturbed_to_executed_gripper_semantic_changes: int = 0

    def __post_init__(self) -> None:
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")

        counts = (
            self.perturbed_violation_step_count,
            self.executed_violation_step_count,
            self.perturbed_constraint_violation_count,
            self.executed_constraint_violation_count,
            self.critical_violation_step_count,
            self.unsafe_perturbed_steps,
            self.recovered_unsafe_steps,
            self.unresolved_unsafe_steps,
            self.intervention_count,
            self.environment_interface_adjustment_count,
            self.strict_lyapunov_decrease_count,
            self.lyapunov_nonincrease_count,
            self.selected_lower_than_perturbed_count,
            self.emergency_fallback_count,
            self.steps_object_grasped,
            self.episodes_with_object_grasped,
            self.perturbed_unsafe_steps_while_grasped,
            self.interventions_while_grasped,
            self.proposed_to_perturbed_gripper_semantic_changes,
            self.perturbed_to_executed_gripper_semantic_changes,
        )

        if any(value < 0 for value in counts):
            raise ValueError("episode counts must be nonnegative")

        if (
            self.recovered_unsafe_steps + self.unresolved_unsafe_steps
            != self.unsafe_perturbed_steps
        ):
            raise ValueError("unsafe action recovery accounting mismatch")


@dataclass(frozen=True)
class ActionRobustnessSeedSummary:
    """Per-seed summary for one action perturbation cell."""

    domain: str
    method: str
    perturbation_name: str
    perturbation_family: str
    principal_seed: int

    episode_count: int
    total_environment_steps: int

    mean_reward: float
    reward_sample_sd: float
    success_rate: float
    mean_episode_length: float

    perturbed_violation_step_rate: float
    executed_violation_step_rate: float

    perturbed_constraint_violation_rate: float
    executed_constraint_violation_rate: float

    critical_violation_step_rate: float

    unsafe_perturbed_steps: int
    recovered_unsafe_steps: int
    unresolved_unsafe_steps: int

    recovery_rate: float | None
    within_filter_violation_reduction: float | None

    intervention_rate: float

    mean_action_perturbation_l2: float
    max_action_perturbation_linf: float

    mean_safety_correction_l2: float
    p95_safety_correction_l2: float
    max_safety_correction_l2: float

    mean_total_action_displacement_l2: float

    environment_interface_adjustment_rate: float
    mean_environment_interface_adjustment_l2: float
    max_environment_interface_adjustment_l2: float

    perturbed_category_violation_rates: Mapping[str, float]
    executed_category_violation_rates: Mapping[str, float]
    recovered_category_rates: Mapping[str, float]

    intervention_reason_rates: Mapping[str, float]
    selected_candidate_source_rates: Mapping[str, float]

    strict_lyapunov_decrease_rate: float
    lyapunov_nonincrease_rate: float
    selected_lower_than_perturbed_rate: float
    emergency_fallback_rate: float

    steps_object_grasped: int
    episodes_with_object_grasped: int
    perturbed_unsafe_steps_while_grasped: int
    interventions_while_grasped: int

    proposed_to_perturbed_gripper_semantic_change_rate: float
    perturbed_to_executed_gripper_semantic_change_rate: float


@dataclass(frozen=True)
class ActionCleanDelta:
    """Delta from matching frozen clean evidence."""

    executed_violation_delta: float
    critical_violation_delta: float
    constraint_violation_delta: float
    reward_delta: float
    success_delta: float
    episode_length_delta: float


@dataclass(frozen=True)
class ActionAggregateMetric:
    """Mean and sample SD across principal seeds."""

    mean: float
    sample_sd: float


def gripper_semantic(
    value: float,
) -> str:
    """Classify frozen robotics gripper semantics."""

    if value > 0.50:
        return "close"

    if value < -0.50:
        return "open"

    return "hold"


def action_safety_recovery_rate(
    *,
    unsafe_perturbed_steps: int,
    recovered_unsafe_steps: int,
) -> float | None:
    """Fraction of unsafe perturbed steps recovered explicitly by filtering."""

    if unsafe_perturbed_steps < 0:
        raise ValueError("unsafe_perturbed_steps must be nonnegative")

    if recovered_unsafe_steps < 0:
        raise ValueError("recovered_unsafe_steps must be nonnegative")

    if recovered_unsafe_steps > unsafe_perturbed_steps:
        raise ValueError("recovered steps cannot exceed unsafe perturbed steps")

    if unsafe_perturbed_steps == 0:
        return None

    return recovered_unsafe_steps / unsafe_perturbed_steps


def within_filter_violation_reduction(
    *,
    perturbed_violation_rate: float,
    executed_violation_rate: float,
) -> float | None:
    """Relative explicit-filter reduction from perturbed to executed."""

    if perturbed_violation_rate < 0.0:
        raise ValueError("perturbed violation rate must be nonnegative")

    if executed_violation_rate < 0.0:
        raise ValueError("executed violation rate must be nonnegative")

    if perturbed_violation_rate == 0.0:
        return None

    return (
        perturbed_violation_rate - executed_violation_rate
    ) / perturbed_violation_rate


def environment_effective_action(
    *,
    executed_action: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Reconstruct the frozen proxy environments' internal np.clip."""

    action = np.asarray(
        executed_action,
        dtype=np.float32,
    )

    lower = np.asarray(
        lower_bounds,
        dtype=np.float32,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=np.float32,
    )

    if action.ndim != 1:
        raise ValueError("executed_action must be one-dimensional")

    if action.shape != lower.shape or action.shape != upper.shape:
        raise ValueError("action and bound shapes must match")

    if not np.all(np.isfinite(action)):
        raise ValueError("executed_action must be finite")

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("action bounds must be finite")

    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")

    return np.clip(
        action,
        lower,
        upper,
    ).astype(
        np.float32,
        copy=False,
    )


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _p95(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    index = max(
        0,
        min(
            len(ordered) - 1,
            int(0.95 * len(ordered) + 0.999999999) - 1,
        ),
    )

    return float(ordered[index])


def _merge_counts(
    mappings: list[Mapping[str, int]],
) -> Counter[str]:
    merged: Counter[str] = Counter()

    for mapping in mappings:
        merged.update({key: int(value) for key, value in mapping.items()})

    return merged


def summarize_action_robustness_seed(
    episodes: list[ActionRobustnessEpisodeEvidence],
) -> ActionRobustnessSeedSummary:
    """Summarize one method/seed/action-perturbation cell."""

    if not episodes:
        raise ValueError("episodes must be non-empty")

    first = episodes[0]

    identity = (
        first.domain,
        first.method,
        first.perturbation_name,
        first.perturbation_family,
        first.principal_seed,
    )

    for episode in episodes:
        candidate = (
            episode.domain,
            episode.method,
            episode.perturbation_name,
            episode.perturbation_family,
            episode.principal_seed,
        )

        if candidate != identity:
            raise ValueError("all episodes must belong to one action cell")

    total_steps = sum(episode.episode_length for episode in episodes)

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    perturbed_violation_steps = sum(
        episode.perturbed_violation_step_count for episode in episodes
    )

    executed_violation_steps = sum(
        episode.executed_violation_step_count for episode in episodes
    )

    unsafe_steps = sum(episode.unsafe_perturbed_steps for episode in episodes)

    recovered_steps = sum(episode.recovered_unsafe_steps for episode in episodes)

    unresolved_steps = sum(episode.unresolved_unsafe_steps for episode in episodes)

    if recovered_steps + unresolved_steps != unsafe_steps:
        raise ValueError("aggregate recovery accounting mismatch")

    correction_values = [
        float(value)
        for episode in episodes
        for value in episode.safety_correction_l2_values
    ]

    perturbed_categories = _merge_counts(
        [episode.perturbed_category_violation_counts for episode in episodes]
    )

    executed_categories = _merge_counts(
        [episode.executed_category_violation_counts for episode in episodes]
    )

    recovered_categories = _merge_counts(
        [episode.recovered_category_counts for episode in episodes]
    )

    reason_counts = _merge_counts(
        [episode.intervention_reason_counts for episode in episodes]
    )

    source_counts = _merge_counts(
        [episode.selected_candidate_source_counts for episode in episodes]
    )

    perturbed_rate = perturbed_violation_steps / total_steps

    executed_rate = executed_violation_steps / total_steps

    return ActionRobustnessSeedSummary(
        domain=first.domain,
        method=first.method,
        perturbation_name=first.perturbation_name,
        perturbation_family=first.perturbation_family,
        principal_seed=first.principal_seed,
        episode_count=len(episodes),
        total_environment_steps=total_steps,
        mean_reward=float(mean([episode.reward for episode in episodes])),
        reward_sample_sd=_sample_sd([episode.reward for episode in episodes]),
        success_rate=(
            sum(int(episode.success) for episode in episodes) / len(episodes)
        ),
        mean_episode_length=(total_steps / len(episodes)),
        perturbed_violation_step_rate=perturbed_rate,
        executed_violation_step_rate=executed_rate,
        perturbed_constraint_violation_rate=(
            sum(episode.perturbed_constraint_violation_count for episode in episodes)
            / total_steps
        ),
        executed_constraint_violation_rate=(
            sum(episode.executed_constraint_violation_count for episode in episodes)
            / total_steps
        ),
        critical_violation_step_rate=(
            sum(episode.critical_violation_step_count for episode in episodes)
            / total_steps
        ),
        unsafe_perturbed_steps=unsafe_steps,
        recovered_unsafe_steps=recovered_steps,
        unresolved_unsafe_steps=unresolved_steps,
        recovery_rate=action_safety_recovery_rate(
            unsafe_perturbed_steps=unsafe_steps,
            recovered_unsafe_steps=recovered_steps,
        ),
        within_filter_violation_reduction=(
            within_filter_violation_reduction(
                perturbed_violation_rate=perturbed_rate,
                executed_violation_rate=executed_rate,
            )
        ),
        intervention_rate=(
            sum(episode.intervention_count for episode in episodes) / total_steps
        ),
        mean_action_perturbation_l2=(
            sum(episode.action_perturbation_l2_sum for episode in episodes)
            / total_steps
        ),
        max_action_perturbation_linf=max(
            episode.action_perturbation_linf_max for episode in episodes
        ),
        mean_safety_correction_l2=(
            sum(episode.safety_correction_l2_sum for episode in episodes) / total_steps
        ),
        p95_safety_correction_l2=_p95(correction_values),
        max_safety_correction_l2=max(
            episode.safety_correction_l2_max for episode in episodes
        ),
        mean_total_action_displacement_l2=(
            sum(episode.total_action_displacement_l2_sum for episode in episodes)
            / total_steps
        ),
        environment_interface_adjustment_rate=(
            sum(episode.environment_interface_adjustment_count for episode in episodes)
            / total_steps
        ),
        mean_environment_interface_adjustment_l2=(
            sum(episode.environment_interface_adjustment_l2_sum for episode in episodes)
            / total_steps
        ),
        max_environment_interface_adjustment_l2=max(
            episode.environment_interface_adjustment_l2_max for episode in episodes
        ),
        perturbed_category_violation_rates={
            key: value / total_steps
            for key, value in sorted(perturbed_categories.items())
        },
        executed_category_violation_rates={
            key: value / total_steps
            for key, value in sorted(executed_categories.items())
        },
        recovered_category_rates={
            key: value / total_steps
            for key, value in sorted(recovered_categories.items())
        },
        intervention_reason_rates={
            key: value / total_steps for key, value in sorted(reason_counts.items())
        },
        selected_candidate_source_rates={
            key: value / total_steps for key, value in sorted(source_counts.items())
        },
        strict_lyapunov_decrease_rate=(
            sum(episode.strict_lyapunov_decrease_count for episode in episodes)
            / total_steps
        ),
        lyapunov_nonincrease_rate=(
            sum(episode.lyapunov_nonincrease_count for episode in episodes)
            / total_steps
        ),
        selected_lower_than_perturbed_rate=(
            sum(episode.selected_lower_than_perturbed_count for episode in episodes)
            / total_steps
        ),
        emergency_fallback_rate=(
            sum(episode.emergency_fallback_count for episode in episodes) / total_steps
        ),
        steps_object_grasped=sum(episode.steps_object_grasped for episode in episodes),
        episodes_with_object_grasped=sum(
            episode.episodes_with_object_grasped for episode in episodes
        ),
        perturbed_unsafe_steps_while_grasped=sum(
            episode.perturbed_unsafe_steps_while_grasped for episode in episodes
        ),
        interventions_while_grasped=sum(
            episode.interventions_while_grasped for episode in episodes
        ),
        proposed_to_perturbed_gripper_semantic_change_rate=(
            sum(
                episode.proposed_to_perturbed_gripper_semantic_changes
                for episode in episodes
            )
            / total_steps
        ),
        perturbed_to_executed_gripper_semantic_change_rate=(
            sum(
                episode.perturbed_to_executed_gripper_semantic_changes
                for episode in episodes
            )
            / total_steps
        ),
    )


def aggregate_action_metric(
    values: list[float],
) -> ActionAggregateMetric:
    """Aggregate one metric across principal seeds."""

    if not values:
        raise ValueError("values must be non-empty")

    return ActionAggregateMetric(
        mean=float(mean(values)),
        sample_sd=_sample_sd(values),
    )
