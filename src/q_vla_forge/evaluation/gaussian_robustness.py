"""Sprint 5.10 Gaussian robustness contracts and aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass(frozen=True)
class GaussianEpisodeEvidence:
    """Evidence for one Gaussian-noise evaluation episode."""

    domain: str
    method: str
    principal_seed: int
    evaluation_seed: int
    sigma: float
    noise_seed: int

    reward: float
    success: bool
    episode_length: int

    proposed_violation_step_count: int
    executed_violation_step_count: int
    proposed_constraint_violation_count: int
    executed_constraint_violation_count: int
    critical_violation_step_count: int

    intervention_count: int

    noise_sample_count: int
    noise_l2_sum: float
    noise_l2_max: float

    action_correction_l2_sum: float = 0.0
    action_correction_l2_max: float = 0.0

    intervention_reason_counts: Mapping[str, int] = field(default_factory=dict)
    category_violation_counts: Mapping[str, int] = field(default_factory=dict)
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

        if self.noise_sample_count <= 0:
            raise ValueError("noise_sample_count must be positive")

        if self.sigma < 0.0 or not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite and non-negative")

        integer_fields = (
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

        if any(value < 0 for value in integer_fields):
            raise ValueError("count fields must be non-negative")

        if self.steps_object_grasped + self.steps_object_not_grasped not in (
            0,
            self.episode_length,
        ):
            raise ValueError(
                "grasp-context counts must sum to " "episode length when supplied"
            )


@dataclass(frozen=True)
class GaussianSeedSummary:
    """Aggregated evidence for one seed/method/sigma cell."""

    domain: str
    method: str
    principal_seed: int
    sigma: float

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

    mean_noise_l2: float
    max_noise_l2: float

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
class CleanDelta:
    """Difference between noisy and clean seed-level evidence."""

    violation_step_delta: float
    critical_violation_step_delta: float
    constraint_violation_rate_delta: float
    reward_delta: float
    success_rate_delta: float
    episode_length_delta: float
    relative_violation_degradation: float | None


@dataclass(frozen=True)
class AggregateMetric:
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
) -> dict[str, int]:
    result: dict[str, int] = {}

    for mapping in mappings:
        for key, value in mapping.items():
            result[key] = result.get(key, 0) + int(value)

    return result


def summarize_gaussian_seed(
    episodes: list[GaussianEpisodeEvidence],
) -> GaussianSeedSummary:
    """Aggregate one principal-seed robustness cell."""

    if not episodes:
        raise ValueError("episodes must not be empty")

    first = episodes[0]

    for episode in episodes:
        if episode.domain != first.domain:
            raise ValueError("all episodes must share domain")

        if episode.method != first.method:
            raise ValueError("all episodes must share method")

        if episode.principal_seed != first.principal_seed:
            raise ValueError("all episodes must share principal_seed")

        if episode.sigma != first.sigma:
            raise ValueError("all episodes must share sigma")

    total_steps = sum(episode.episode_length for episode in episodes)

    total_noise_samples = sum(episode.noise_sample_count for episode in episodes)

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    if total_noise_samples <= 0:
        raise ValueError("total noise samples must be positive")

    rewards = [episode.reward for episode in episodes]

    category_counts = _merge_counts(
        [episode.category_violation_counts for episode in episodes]
    )

    reason_counts = _merge_counts(
        [episode.intervention_reason_counts for episode in episodes]
    )

    selected_candidate_counts = _merge_counts(
        [episode.selected_candidate_source_counts for episode in episodes]
    )

    intervened_candidate_counts = _merge_counts(
        [episode.intervened_candidate_source_counts for episode in episodes]
    )

    category_rates = {
        key: value / total_steps for key, value in sorted(category_counts.items())
    }

    reason_rates = {
        key: value / total_steps for key, value in sorted(reason_counts.items())
    }

    selected_candidate_rates = {
        key: value / total_steps
        for key, value in sorted(selected_candidate_counts.items())
    }

    intervened_candidate_rates = {
        key: value / total_steps
        for key, value in sorted(intervened_candidate_counts.items())
    }

    return GaussianSeedSummary(
        domain=first.domain,
        method=first.method,
        principal_seed=first.principal_seed,
        sigma=first.sigma,
        episode_count=len(episodes),
        total_environment_steps=total_steps,
        mean_reward=float(mean(rewards)),
        reward_sample_sd=_sample_sd(rewards),
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
        mean_noise_l2=(
            sum(episode.noise_l2_sum for episode in episodes) / total_noise_samples
        ),
        max_noise_l2=max(episode.noise_l2_max for episode in episodes),
        category_violation_rates=category_rates,
        intervention_reason_rates=reason_rates,
        selected_candidate_source_rates=selected_candidate_rates,
        intervened_candidate_source_rates=intervened_candidate_rates,
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


def compute_clean_delta(
    *,
    noisy: GaussianSeedSummary,
    clean: GaussianSeedSummary,
) -> CleanDelta:
    """Compute noisy-minus-clean robustness deltas."""

    if noisy.domain != clean.domain:
        raise ValueError("domain mismatch")

    if noisy.method != clean.method:
        raise ValueError("method mismatch")

    if noisy.principal_seed != clean.principal_seed:
        raise ValueError("principal seed mismatch")

    if clean.executed_violation_step_rate > 0.0:
        relative = (
            noisy.executed_violation_step_rate - clean.executed_violation_step_rate
        ) / clean.executed_violation_step_rate
    else:
        relative = None

    return CleanDelta(
        violation_step_delta=(
            noisy.executed_violation_step_rate - clean.executed_violation_step_rate
        ),
        critical_violation_step_delta=(
            noisy.critical_violation_step_rate - clean.critical_violation_step_rate
        ),
        constraint_violation_rate_delta=(
            noisy.executed_constraint_violation_rate
            - clean.executed_constraint_violation_rate
        ),
        reward_delta=(noisy.mean_reward - clean.mean_reward),
        success_rate_delta=(noisy.success_rate - clean.success_rate),
        episode_length_delta=(noisy.mean_episode_length - clean.mean_episode_length),
        relative_violation_degradation=relative,
    )


def aggregate_across_principal_seeds(
    values: Mapping[int, float],
) -> AggregateMetric:
    """Aggregate one metric across principal seeds."""

    if not values:
        raise ValueError("values must not be empty")

    ordered_values = [float(values[key]) for key in sorted(values)]

    return AggregateMetric(
        mean=float(mean(ordered_values)),
        sample_sd=_sample_sd(ordered_values),
    )
