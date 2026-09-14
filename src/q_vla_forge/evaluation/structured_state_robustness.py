"""Sprint 5.11 structured-state robustness evidence contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass(frozen=True)
class StructuredStateEpisodeEvidence:
    """One structured-state robustness evaluation episode."""

    domain: str
    method: str
    perturbation_name: str
    perturbation_family: str
    principal_seed: int
    evaluation_seed: int

    reward: float
    success: bool
    episode_length: int

    proposed_violation_step_count: int
    executed_violation_step_count: int

    proposed_constraint_violation_count: int
    executed_constraint_violation_count: int

    critical_violation_step_count: int

    intervention_count: int

    perturbation_l1_sum: float
    perturbation_l2_sum: float
    perturbation_linf_max: float

    action_correction_l2_sum: float = 0.0
    action_correction_l2_max: float = 0.0

    category_violation_counts: Mapping[str, int] = field(default_factory=dict)

    intervention_reason_counts: Mapping[str, int] = field(default_factory=dict)

    selected_candidate_source_counts: Mapping[str, int] = field(default_factory=dict)

    intervened_candidate_source_counts: Mapping[str, int] = field(default_factory=dict)

    strict_lyapunov_decrease_count: int = 0
    lyapunov_nonincrease_count: int = 0
    selected_lower_than_proposed_count: int = 0
    emergency_fallback_count: int = 0

    steps_object_grasped: int = 0
    steps_object_not_grasped: int = 0
    interventions_while_grasped: int = 0

    def __post_init__(self) -> None:
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")

        integer_counts = (
            self.proposed_violation_step_count,
            self.executed_violation_step_count,
            self.proposed_constraint_violation_count,
            self.executed_constraint_violation_count,
            self.critical_violation_step_count,
            self.intervention_count,
            self.strict_lyapunov_decrease_count,
            self.lyapunov_nonincrease_count,
            self.selected_lower_than_proposed_count,
            self.emergency_fallback_count,
            self.steps_object_grasped,
            self.steps_object_not_grasped,
            self.interventions_while_grasped,
        )

        if any(value < 0 for value in integer_counts):
            raise ValueError("episode counts must be nonnegative")

        if self.steps_object_grasped + self.steps_object_not_grasped not in (
            0,
            self.episode_length,
        ):
            raise ValueError(
                "robotics grasp-state accounting " "must equal episode length"
            )


@dataclass(frozen=True)
class StructuredStateSeedSummary:
    """Per-seed summary for one perturbation/method cell."""

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

    proposed_violation_step_rate: float
    executed_violation_step_rate: float

    proposed_constraint_violation_rate: float
    executed_constraint_violation_rate: float

    critical_violation_step_rate: float

    intervention_rate: float

    mean_action_correction_l2: float
    max_action_correction_l2: float

    mean_perturbation_l1: float
    mean_perturbation_l2: float
    max_perturbation_linf: float

    category_violation_rates: Mapping[str, float]
    intervention_reason_rates: Mapping[str, float]

    selected_candidate_source_rates: Mapping[str, float]
    intervened_candidate_source_rates: Mapping[str, float]

    strict_lyapunov_decrease_rate: float
    lyapunov_nonincrease_rate: float
    selected_lower_than_proposed_rate: float
    emergency_fallback_rate: float

    steps_object_grasped: int
    steps_object_not_grasped: int
    interventions_while_grasped: int


@dataclass(frozen=True)
class StructuredCleanDelta:
    """Per-seed delta from the matching frozen clean reference."""

    violation_delta: float
    critical_violation_delta: float
    constraint_violation_delta: float
    reward_delta: float
    success_delta: float
    episode_length_delta: float
    relative_violation_degradation: float | None


@dataclass(frozen=True)
class StructuredAggregateMetric:
    """Mean and sample SD across principal seeds."""

    mean: float
    sample_sd: float


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _merge_counts(
    mappings: list[Mapping[str, int]],
) -> Counter[str]:
    merged: Counter[str] = Counter()

    for mapping in mappings:
        merged.update({key: int(value) for key, value in mapping.items()})

    return merged


def summarize_structured_state_seed(
    episodes: list[StructuredStateEpisodeEvidence],
) -> StructuredStateSeedSummary:
    """Summarize twenty held-out episodes for one cell."""

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
            raise ValueError("all episodes must belong " "to the same structured cell")

    total_steps = sum(episode.episode_length for episode in episodes)

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    rewards = [episode.reward for episode in episodes]

    category_counts = _merge_counts(
        [episode.category_violation_counts for episode in episodes]
    )

    reason_counts = _merge_counts(
        [episode.intervention_reason_counts for episode in episodes]
    )

    selected_counts = _merge_counts(
        [episode.selected_candidate_source_counts for episode in episodes]
    )

    intervened_counts = _merge_counts(
        [episode.intervened_candidate_source_counts for episode in episodes]
    )

    return StructuredStateSeedSummary(
        domain=first.domain,
        method=first.method,
        perturbation_name=(first.perturbation_name),
        perturbation_family=(first.perturbation_family),
        principal_seed=(first.principal_seed),
        episode_count=len(episodes),
        total_environment_steps=(total_steps),
        mean_reward=float(mean(rewards)),
        reward_sample_sd=(_sample_sd(rewards)),
        success_rate=(
            sum(int(episode.success) for episode in episodes) / len(episodes)
        ),
        mean_episode_length=(
            sum(episode.episode_length for episode in episodes) / len(episodes)
        ),
        proposed_violation_step_rate=(
            sum(episode.proposed_violation_step_count for episode in episodes)
            / total_steps
        ),
        executed_violation_step_rate=(
            sum(episode.executed_violation_step_count for episode in episodes)
            / total_steps
        ),
        proposed_constraint_violation_rate=(
            sum(episode.proposed_constraint_violation_count for episode in episodes)
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
        intervention_rate=(
            sum(episode.intervention_count for episode in episodes) / total_steps
        ),
        mean_action_correction_l2=(
            sum(episode.action_correction_l2_sum for episode in episodes) / total_steps
        ),
        max_action_correction_l2=max(
            episode.action_correction_l2_max for episode in episodes
        ),
        mean_perturbation_l1=(
            sum(episode.perturbation_l1_sum for episode in episodes) / total_steps
        ),
        mean_perturbation_l2=(
            sum(episode.perturbation_l2_sum for episode in episodes) / total_steps
        ),
        max_perturbation_linf=max(
            episode.perturbation_linf_max for episode in episodes
        ),
        category_violation_rates={
            key: value / total_steps for key, value in sorted(category_counts.items())
        },
        intervention_reason_rates={
            key: value / total_steps for key, value in sorted(reason_counts.items())
        },
        selected_candidate_source_rates={
            key: value / total_steps for key, value in sorted(selected_counts.items())
        },
        intervened_candidate_source_rates={
            key: value / total_steps for key, value in sorted(intervened_counts.items())
        },
        strict_lyapunov_decrease_rate=(
            sum(episode.strict_lyapunov_decrease_count for episode in episodes)
            / total_steps
        ),
        lyapunov_nonincrease_rate=(
            sum(episode.lyapunov_nonincrease_count for episode in episodes)
            / total_steps
        ),
        selected_lower_than_proposed_rate=(
            sum(episode.selected_lower_than_proposed_count for episode in episodes)
            / total_steps
        ),
        emergency_fallback_rate=(
            sum(episode.emergency_fallback_count for episode in episodes) / total_steps
        ),
        steps_object_grasped=sum(episode.steps_object_grasped for episode in episodes),
        steps_object_not_grasped=sum(
            episode.steps_object_not_grasped for episode in episodes
        ),
        interventions_while_grasped=sum(
            episode.interventions_while_grasped for episode in episodes
        ),
    )


def compute_structured_clean_delta(
    *,
    structured: StructuredStateSeedSummary,
    clean: StructuredStateSeedSummary,
) -> StructuredCleanDelta:
    """Compute paired per-seed change relative to frozen clean."""

    if structured.domain != clean.domain:
        raise ValueError("domain mismatch")

    if structured.method != clean.method:
        raise ValueError("method mismatch")

    if structured.principal_seed != clean.principal_seed:
        raise ValueError("principal seed mismatch")

    clean_violation = clean.executed_violation_step_rate

    violation_delta = structured.executed_violation_step_rate - clean_violation

    relative: float | None

    if clean_violation > 0.0:
        relative = violation_delta / clean_violation
    else:
        relative = None

    return StructuredCleanDelta(
        violation_delta=(violation_delta),
        critical_violation_delta=(
            structured.critical_violation_step_rate - clean.critical_violation_step_rate
        ),
        constraint_violation_delta=(
            structured.executed_constraint_violation_rate
            - clean.executed_constraint_violation_rate
        ),
        reward_delta=(structured.mean_reward - clean.mean_reward),
        success_delta=(structured.success_rate - clean.success_rate),
        episode_length_delta=(
            structured.mean_episode_length - clean.mean_episode_length
        ),
        relative_violation_degradation=(relative),
    )


def aggregate_structured_metric(
    values: list[float],
) -> StructuredAggregateMetric:
    """Aggregate one metric across principal seeds."""

    if not values:
        raise ValueError("values must be non-empty")

    return StructuredAggregateMetric(
        mean=float(mean(values)),
        sample_sd=(_sample_sd(values)),
    )
