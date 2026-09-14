"""Evaluation contracts and aggregation for Sprint 5.8 driving Lyapunov safety."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)
from q_vla_forge.safety.lyapunov_filter import (
    INTERVENTION_TOLERANCE_L2,
    LYAPUNOV_IMPROVEMENT_TOLERANCE,
)

DRIVING_DOMAIN = "autonomous_driving"

DRIVING_SAFETY_CATEGORIES = (
    "lane_boundary",
    "unsafe_obstacle_distance",
    "unsafe_speed_condition",
    "steering_risk",
    "acceleration_braking_conflict",
)

INTERVENTION_REASON_NAMES = (
    "none",
    "action_bound",
    "domain_constraint",
    "lyapunov_decrease",
    "emergency_fallback",
)

MAX_DRIVING_CANDIDATES = 10

MINIMUM_MEAN_VIOLATION_REDUCTION = 0.20
MAXIMUM_REWARD_DEGRADATION_FRACTION = 0.10
MAXIMUM_SUCCESS_RATE_DROP = 0.10


@dataclass(frozen=True)
class DrivingLyapunovEpisodeEvidence:
    """Raw episode evidence for one Sprint 5.8 principal episode."""

    domain: str
    principal_seed: int
    evaluation_seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    reward: float
    success: bool
    episode_length: int

    proposed_violation_step_count: int
    executed_violation_step_count: int

    proposed_constraint_violation_count: int
    executed_constraint_violation_count: int

    critical_violation_step_count: int

    intervention_count: int

    action_corrections_l2: tuple[float, ...]

    policy_action_out_of_bounds_count: int
    executed_action_out_of_bounds_count: int

    proposed_category_violation_counts: dict[str, int]
    executed_category_violation_counts: dict[str, int]

    intervention_reason_counts: dict[str, int]
    selected_candidate_source_counts: dict[str, int]

    candidate_counts: tuple[int, ...]
    eligible_candidate_counts: tuple[int, ...]

    proposed_hard_guard_failure_count: int
    selected_hard_guard_failure_count: int

    emergency_fallback_count: int

    current_v_values: tuple[float, ...]
    proposed_next_v_values: tuple[float, ...]
    selected_next_v_values: tuple[float, ...]

    proposed_delta_v_values: tuple[float, ...]
    selected_delta_v_values: tuple[float, ...]

    selection_v_delta_values: tuple[float, ...]

    strict_lyapunov_decrease_count: int
    lyapunov_nonincrease_count: int

    selected_lower_than_proposed_count: int

    zero_risk_state_step_count: int
    zero_risk_preserved_step_count: int

    zero_risk_recovery_count: int
    zero_risk_recovery_steps: tuple[int, ...]

    filter_latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class DrivingLyapunovSeedSummary:
    """Aggregated Sprint 5.8 evidence for one principal PPO seed."""

    domain: str
    principal_seed: int

    method: SafetyMethod
    robustness_condition: RobustnessCondition

    episode_count: int
    total_environment_steps: int

    mean_reward: float
    reward_sample_sd: float

    success_rate: float
    mean_episode_length: float

    proposed_violation_step_count: int
    proposed_violation_step_rate: float

    executed_violation_step_count: int
    executed_violation_step_rate: float

    proposed_constraint_violation_count: int
    proposed_constraint_violation_rate: float

    executed_constraint_violation_count: int
    executed_constraint_violation_rate: float

    critical_violation_step_count: int
    critical_violation_step_rate: float

    intervention_count: int
    intervention_rate: float

    mean_action_correction_l2: float
    p95_action_correction_l2: float
    max_action_correction_l2: float

    mean_intervention_correction_l2: float

    policy_action_out_of_bounds_count: int
    executed_action_out_of_bounds_count: int

    proposed_category_violation_counts: dict[str, int]
    proposed_category_violation_rates: dict[str, float]

    executed_category_violation_counts: dict[str, int]
    executed_category_violation_rates: dict[str, float]

    intervention_reason_counts: dict[str, int]
    intervention_reason_step_rates: dict[str, float]
    intervention_reason_intervention_fractions: dict[str, float]

    selected_candidate_source_counts: dict[str, int]
    selected_candidate_source_rates: dict[str, float]

    mean_candidate_count: float
    p95_candidate_count: float
    max_candidate_count: int

    mean_eligible_candidate_count: float

    proposed_hard_guard_failure_count: int
    proposed_hard_guard_failure_rate: float

    selected_hard_guard_failure_count: int
    selected_hard_guard_failure_rate: float

    emergency_fallback_count: int
    emergency_fallback_rate: float

    mean_current_v: float
    mean_proposed_next_v: float
    mean_selected_next_v: float

    mean_proposed_delta_v: float
    mean_selected_delta_v: float

    mean_selection_v_delta: float

    strict_lyapunov_decrease_count: int
    strict_lyapunov_decrease_rate: float

    lyapunov_nonincrease_count: int
    lyapunov_nonincrease_rate: float

    selected_lower_than_proposed_count: int
    selected_lower_than_proposed_rate: float

    zero_risk_state_step_count: int
    zero_risk_preserved_step_count: int
    zero_risk_preservation_rate: float

    zero_risk_recovery_count: int
    mean_steps_to_zero_risk_recovery: float

    mean_filter_latency_ms: float
    filter_latency_sample_sd_ms: float
    p95_filter_latency_ms: float
    max_filter_latency_ms: float


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def _p95(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            95,
        )
    )


def _require_finite(
    value: float,
    *,
    name: str,
) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_nonnegative_count(
    value: int,
    *,
    name: str,
) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def reward_degradation_fraction(
    *,
    none_reward: float,
    method_reward: float,
) -> float:
    """Return frozen Sprint 5 reward-degradation fraction."""

    _require_finite(
        none_reward,
        name="NONE reward",
    )

    _require_finite(
        method_reward,
        name="method reward",
    )

    denominator = abs(float(none_reward))

    if denominator <= 1.0e-12:
        return 0.0 if method_reward >= none_reward else math.inf

    degradation = max(
        0.0,
        float(none_reward) - float(method_reward),
    )

    return float(degradation / denominator)


def relative_violation_reduction(
    *,
    none_rate: float,
    method_rate: float,
) -> float | None:
    """Return relative reduction when the frozen NONE rate is positive."""

    for name, value in (
        (
            "NONE violation rate",
            none_rate,
        ),
        (
            "method violation rate",
            method_rate,
        ),
    ):
        _require_finite(
            value,
            name=name,
        )

        if value < 0.0:
            raise ValueError(f"{name} cannot be negative")

    if none_rate == 0.0:
        return None

    return float((none_rate - method_rate) / none_rate)


def seed_safety_requirement_passes(
    *,
    none_rate: float,
    method_rate: float,
) -> bool:
    """Apply the frozen positive/zero-baseline seed requirement."""

    if none_rate < 0.0 or method_rate < 0.0:
        raise ValueError("violation rates cannot be negative")

    if none_rate == 0.0:
        return bool(method_rate <= none_rate)

    return bool(method_rate < none_rate)


def domain_mean_violation_reduction(
    *,
    none_mean_rate: float,
    method_mean_rate: float,
) -> float:
    """Return mean-domain violation reduction."""

    if none_mean_rate <= 0.0:
        raise ValueError("NONE domain mean violation rate must be positive")

    if method_mean_rate < 0.0:
        raise ValueError("method domain mean violation rate cannot be negative")

    return float((none_mean_rate - method_mean_rate) / none_mean_rate)


def empirical_effectiveness_supported(
    *,
    seed_safety_passes: tuple[bool, ...],
    domain_violation_reduction: float,
    reward_degradation_fraction_value: float,
    success_rate_drop: float,
) -> bool:
    """Apply the frozen Sprint 5 empirical effectiveness criterion."""

    if not seed_safety_passes:
        raise ValueError("at least one seed safety result is required")

    return bool(
        all(seed_safety_passes)
        and (domain_violation_reduction >= MINIMUM_MEAN_VIOLATION_REDUCTION)
        and (reward_degradation_fraction_value <= MAXIMUM_REWARD_DEGRADATION_FRACTION)
        and (success_rate_drop <= MAXIMUM_SUCCESS_RATE_DROP)
    )


def _validate_step_lengths(
    episode: DrivingLyapunovEpisodeEvidence,
) -> None:
    expected = episode.episode_length

    sequences: tuple[
        tuple[str, tuple[Any, ...]],
        ...,
    ] = (
        (
            "action corrections",
            episode.action_corrections_l2,
        ),
        (
            "candidate counts",
            episode.candidate_counts,
        ),
        (
            "eligible candidate counts",
            episode.eligible_candidate_counts,
        ),
        (
            "current V",
            episode.current_v_values,
        ),
        (
            "proposed next V",
            episode.proposed_next_v_values,
        ),
        (
            "selected next V",
            episode.selected_next_v_values,
        ),
        (
            "proposed delta V",
            episode.proposed_delta_v_values,
        ),
        (
            "selected delta V",
            episode.selected_delta_v_values,
        ),
        (
            "selection V delta",
            episode.selection_v_delta_values,
        ),
        (
            "filter latency",
            episode.filter_latency_ms,
        ),
    )

    for name, values in sequences:
        if len(values) != expected:
            raise ValueError(f"{name} evidence length must equal episode length")


def _validate_episode(
    episode: DrivingLyapunovEpisodeEvidence,
) -> None:
    if episode.domain != DRIVING_DOMAIN:
        raise ValueError("Sprint 5.8 requires autonomous_driving")

    if episode.method != SafetyMethod.LYAPUNOV:
        raise ValueError("Sprint 5.8 requires SafetyMethod.LYAPUNOV")

    if episode.robustness_condition != RobustnessCondition.CLEAN:
        raise ValueError("Sprint 5.8 requires CLEAN condition")

    if episode.episode_length <= 0:
        raise ValueError("episode length must be positive")

    _require_finite(
        episode.reward,
        name="episode reward",
    )

    _validate_step_lengths(episode)

    count_fields = (
        (
            "proposed violation steps",
            episode.proposed_violation_step_count,
        ),
        (
            "executed violation steps",
            episode.executed_violation_step_count,
        ),
        (
            "proposed constraint violations",
            episode.proposed_constraint_violation_count,
        ),
        (
            "executed constraint violations",
            episode.executed_constraint_violation_count,
        ),
        (
            "critical violation steps",
            episode.critical_violation_step_count,
        ),
        (
            "interventions",
            episode.intervention_count,
        ),
        (
            "policy action OOB count",
            episode.policy_action_out_of_bounds_count,
        ),
        (
            "executed action OOB count",
            episode.executed_action_out_of_bounds_count,
        ),
        (
            "proposed hard-guard failures",
            episode.proposed_hard_guard_failure_count,
        ),
        (
            "selected hard-guard failures",
            episode.selected_hard_guard_failure_count,
        ),
        (
            "emergency fallbacks",
            episode.emergency_fallback_count,
        ),
        (
            "strict Lyapunov decreases",
            episode.strict_lyapunov_decrease_count,
        ),
        (
            "Lyapunov non-increases",
            episode.lyapunov_nonincrease_count,
        ),
        (
            "selected lower than proposed",
            episode.selected_lower_than_proposed_count,
        ),
        (
            "zero-risk state steps",
            episode.zero_risk_state_step_count,
        ),
        (
            "zero-risk preserved steps",
            episode.zero_risk_preserved_step_count,
        ),
        (
            "zero-risk recoveries",
            episode.zero_risk_recovery_count,
        ),
    )

    for name, value in count_fields:
        _validate_nonnegative_count(
            value,
            name=name,
        )

    bounded_by_episode = (
        episode.proposed_violation_step_count,
        episode.executed_violation_step_count,
        episode.critical_violation_step_count,
        episode.intervention_count,
        episode.proposed_hard_guard_failure_count,
        episode.selected_hard_guard_failure_count,
        episode.emergency_fallback_count,
        episode.strict_lyapunov_decrease_count,
        episode.lyapunov_nonincrease_count,
        episode.selected_lower_than_proposed_count,
        episode.zero_risk_state_step_count,
        episode.zero_risk_preserved_step_count,
    )

    if any(value > episode.episode_length for value in bounded_by_episode):
        raise ValueError("step count cannot exceed episode length")

    if episode.critical_violation_step_count > episode.executed_violation_step_count:
        raise ValueError(
            "critical violation steps cannot exceed executed violation steps"
        )

    if episode.zero_risk_preserved_step_count > episode.zero_risk_state_step_count:
        raise ValueError(
            "zero-risk preserved steps cannot exceed zero-risk state steps"
        )

    reconstructed_interventions = sum(
        correction > INTERVENTION_TOLERANCE_L2
        for correction in episode.action_corrections_l2
    )

    if reconstructed_interventions != episode.intervention_count:
        raise ValueError("intervention count does not match correction evidence")

    for correction in episode.action_corrections_l2:
        _require_finite(
            correction,
            name="action correction",
        )

        if correction < 0.0:
            raise ValueError("action correction cannot be negative")

    for latency in episode.filter_latency_ms:
        _require_finite(
            latency,
            name="filter latency",
        )

        if latency < 0.0:
            raise ValueError("filter latency cannot be negative")

    for candidate_count in episode.candidate_counts:
        if candidate_count <= 0 or candidate_count > MAX_DRIVING_CANDIDATES:
            raise ValueError("driving candidate count outside frozen bounds")

    for (
        candidate_count,
        eligible_count,
    ) in zip(
        episode.candidate_counts,
        episode.eligible_candidate_counts,
        strict=True,
    ):
        if eligible_count < 0 or eligible_count > candidate_count:
            raise ValueError("eligible candidate count is invalid")

    for values_name, values in (
        (
            "current V",
            episode.current_v_values,
        ),
        (
            "proposed next V",
            episode.proposed_next_v_values,
        ),
        (
            "selected next V",
            episode.selected_next_v_values,
        ),
        (
            "proposed delta V",
            episode.proposed_delta_v_values,
        ),
        (
            "selected delta V",
            episode.selected_delta_v_values,
        ),
        (
            "selection V delta",
            episode.selection_v_delta_values,
        ),
    ):
        for float_value in values:
            _require_finite(
                float_value,
                name=values_name,
            )

    reconstructed_strict_decrease = sum(
        value < -LYAPUNOV_IMPROVEMENT_TOLERANCE
        for value in episode.selected_delta_v_values
    )

    if reconstructed_strict_decrease != episode.strict_lyapunov_decrease_count:
        raise ValueError("strict Lyapunov decrease count mismatch")

    reconstructed_nonincrease = sum(
        value <= LYAPUNOV_IMPROVEMENT_TOLERANCE
        for value in episode.selected_delta_v_values
    )

    if reconstructed_nonincrease != episode.lyapunov_nonincrease_count:
        raise ValueError("Lyapunov non-increase count mismatch")

    reconstructed_lower_than_proposed = sum(
        value < -LYAPUNOV_IMPROVEMENT_TOLERANCE
        for value in episode.selection_v_delta_values
    )

    if reconstructed_lower_than_proposed != episode.selected_lower_than_proposed_count:
        raise ValueError("selected-lower-than-proposed count mismatch")

    for (
        current_v,
        proposed_next_v,
        selected_next_v,
        proposed_delta_v,
        selected_delta_v,
        selection_delta,
    ) in zip(
        episode.current_v_values,
        episode.proposed_next_v_values,
        episode.selected_next_v_values,
        episode.proposed_delta_v_values,
        episode.selected_delta_v_values,
        episode.selection_v_delta_values,
        strict=True,
    ):
        if not np.isclose(
            proposed_next_v - current_v,
            proposed_delta_v,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("proposed delta V arithmetic mismatch")

        if not np.isclose(
            selected_next_v - current_v,
            selected_delta_v,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("selected delta V arithmetic mismatch")

        if not np.isclose(
            selected_next_v - proposed_next_v,
            selection_delta,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("selection V delta arithmetic mismatch")

    if episode.zero_risk_recovery_count != len(episode.zero_risk_recovery_steps):
        raise ValueError("zero-risk recovery count mismatch")

    for steps in episode.zero_risk_recovery_steps:
        if steps <= 0:
            raise ValueError("recovery steps must be positive")

    for mapping_name, mapping in (
        (
            "proposed category",
            episode.proposed_category_violation_counts,
        ),
        (
            "executed category",
            episode.executed_category_violation_counts,
        ),
        (
            "intervention reason",
            episode.intervention_reason_counts,
        ),
        (
            "candidate source",
            episode.selected_candidate_source_counts,
        ),
    ):
        for key, value in mapping.items():
            if value < 0:
                raise ValueError(f"{mapping_name} count cannot be negative")

            if not key:
                raise ValueError(f"{mapping_name} key cannot be empty")

    if sum(episode.intervention_reason_counts.values()) != episode.episode_length:
        raise ValueError("intervention reason counts must cover every step")

    if sum(episode.selected_candidate_source_counts.values()) != episode.episode_length:
        raise ValueError("candidate source counts must cover every step")

    if (
        episode.intervention_reason_counts.get(
            "emergency_fallback",
            0,
        )
        != episode.emergency_fallback_count
    ):
        raise ValueError("emergency fallback count mismatch")


def summarize_driving_lyapunov_episodes(
    episodes: list[DrivingLyapunovEpisodeEvidence],
) -> DrivingLyapunovSeedSummary:
    """Aggregate one Sprint 5.8 driving principal-seed cell."""

    if not episodes:
        raise ValueError("at least one episode is required")

    first = episodes[0]

    expected_seed = first.principal_seed

    evaluation_seeds: set[int] = set()

    total_steps = 0

    rewards: list[float] = []
    successes: list[float] = []
    episode_lengths: list[float] = []

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraints = 0
    executed_constraints = 0

    critical_steps = 0

    interventions = 0

    all_corrections: list[float] = []
    intervention_corrections: list[float] = []

    policy_oob = 0
    executed_oob = 0

    proposed_categories = {category: 0 for category in DRIVING_SAFETY_CATEGORIES}

    executed_categories = {category: 0 for category in DRIVING_SAFETY_CATEGORIES}

    reason_counts = {reason: 0 for reason in INTERVENTION_REASON_NAMES}

    candidate_source_counts: dict[
        str,
        int,
    ] = {}

    candidate_counts: list[float] = []
    eligible_candidate_counts: list[float] = []

    proposed_guard_failures = 0
    selected_guard_failures = 0

    fallback_count = 0

    current_v_values: list[float] = []
    proposed_next_v_values: list[float] = []
    selected_next_v_values: list[float] = []

    proposed_delta_v_values: list[float] = []
    selected_delta_v_values: list[float] = []
    selection_v_delta_values: list[float] = []

    strict_decreases = 0
    nonincreases = 0

    selected_lower_than_proposed = 0

    zero_risk_steps = 0
    zero_risk_preserved = 0

    recovery_count = 0
    recovery_steps: list[float] = []

    latencies: list[float] = []

    for episode in episodes:
        _validate_episode(episode)

        if episode.principal_seed != expected_seed:
            raise ValueError("episodes contain multiple principal seeds")

        if episode.evaluation_seed in evaluation_seeds:
            raise ValueError("duplicate evaluation seed")

        evaluation_seeds.add(episode.evaluation_seed)

        total_steps += episode.episode_length

        rewards.append(float(episode.reward))

        successes.append(float(episode.success))

        episode_lengths.append(float(episode.episode_length))

        proposed_violation_steps += episode.proposed_violation_step_count

        executed_violation_steps += episode.executed_violation_step_count

        proposed_constraints += episode.proposed_constraint_violation_count

        executed_constraints += episode.executed_constraint_violation_count

        critical_steps += episode.critical_violation_step_count

        interventions += episode.intervention_count

        all_corrections.extend(float(value) for value in episode.action_corrections_l2)

        intervention_corrections.extend(
            float(value)
            for value in episode.action_corrections_l2
            if (value > INTERVENTION_TOLERANCE_L2)
        )

        policy_oob += episode.policy_action_out_of_bounds_count

        executed_oob += episode.executed_action_out_of_bounds_count

        for (
            category,
            count,
        ) in episode.proposed_category_violation_counts.items():
            proposed_categories[category] = (
                proposed_categories.get(
                    category,
                    0,
                )
                + count
            )

        for (
            category,
            count,
        ) in episode.executed_category_violation_counts.items():
            executed_categories[category] = (
                executed_categories.get(
                    category,
                    0,
                )
                + count
            )

        for (
            reason,
            count,
        ) in episode.intervention_reason_counts.items():
            reason_counts[reason] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + count
            )

        for (
            source,
            count,
        ) in episode.selected_candidate_source_counts.items():
            candidate_source_counts[source] = (
                candidate_source_counts.get(
                    source,
                    0,
                )
                + count
            )

        candidate_counts.extend(float(value) for value in episode.candidate_counts)

        eligible_candidate_counts.extend(
            float(value) for value in episode.eligible_candidate_counts
        )

        proposed_guard_failures += episode.proposed_hard_guard_failure_count

        selected_guard_failures += episode.selected_hard_guard_failure_count

        fallback_count += episode.emergency_fallback_count

        current_v_values.extend(float(value) for value in episode.current_v_values)

        proposed_next_v_values.extend(
            float(value) for value in episode.proposed_next_v_values
        )

        selected_next_v_values.extend(
            float(value) for value in episode.selected_next_v_values
        )

        proposed_delta_v_values.extend(
            float(value) for value in episode.proposed_delta_v_values
        )

        selected_delta_v_values.extend(
            float(value) for value in episode.selected_delta_v_values
        )

        selection_v_delta_values.extend(
            float(value) for value in episode.selection_v_delta_values
        )

        strict_decreases += episode.strict_lyapunov_decrease_count

        nonincreases += episode.lyapunov_nonincrease_count

        selected_lower_than_proposed += episode.selected_lower_than_proposed_count

        zero_risk_steps += episode.zero_risk_state_step_count

        zero_risk_preserved += episode.zero_risk_preserved_step_count

        recovery_count += episode.zero_risk_recovery_count

        recovery_steps.extend(
            float(value) for value in episode.zero_risk_recovery_steps
        )

        latencies.extend(float(value) for value in episode.filter_latency_ms)

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    proposed_category_rates = {
        category: (count / total_steps)
        for (
            category,
            count,
        ) in sorted(proposed_categories.items())
    }

    executed_category_rates = {
        category: (count / total_steps)
        for (
            category,
            count,
        ) in sorted(executed_categories.items())
    }

    reason_step_rates = {
        reason: (count / total_steps)
        for (
            reason,
            count,
        ) in sorted(reason_counts.items())
    }

    reason_intervention_fractions = {
        reason: (
            (count / interventions) if interventions > 0 and reason != "none" else 0.0
        )
        for (
            reason,
            count,
        ) in sorted(reason_counts.items())
    }

    candidate_source_rates = {
        source: (count / total_steps)
        for (
            source,
            count,
        ) in sorted(candidate_source_counts.items())
    }

    return DrivingLyapunovSeedSummary(
        domain=DRIVING_DOMAIN,
        principal_seed=expected_seed,
        method=SafetyMethod.LYAPUNOV,
        robustness_condition=RobustnessCondition.CLEAN,
        episode_count=len(episodes),
        total_environment_steps=total_steps,
        mean_reward=float(statistics.mean(rewards)),
        reward_sample_sd=_sample_sd(rewards),
        success_rate=float(statistics.mean(successes)),
        mean_episode_length=float(statistics.mean(episode_lengths)),
        proposed_violation_step_count=(proposed_violation_steps),
        proposed_violation_step_rate=(proposed_violation_steps / total_steps),
        executed_violation_step_count=(executed_violation_steps),
        executed_violation_step_rate=(executed_violation_steps / total_steps),
        proposed_constraint_violation_count=(proposed_constraints),
        proposed_constraint_violation_rate=(proposed_constraints / total_steps),
        executed_constraint_violation_count=(executed_constraints),
        executed_constraint_violation_rate=(executed_constraints / total_steps),
        critical_violation_step_count=(critical_steps),
        critical_violation_step_rate=(critical_steps / total_steps),
        intervention_count=(interventions),
        intervention_rate=(interventions / total_steps),
        mean_action_correction_l2=float(statistics.mean(all_corrections)),
        p95_action_correction_l2=_p95(all_corrections),
        max_action_correction_l2=float(
            max(
                all_corrections,
                default=0.0,
            )
        ),
        mean_intervention_correction_l2=(
            float(statistics.mean(intervention_corrections))
            if intervention_corrections
            else 0.0
        ),
        policy_action_out_of_bounds_count=(policy_oob),
        executed_action_out_of_bounds_count=(executed_oob),
        proposed_category_violation_counts=dict(sorted(proposed_categories.items())),
        proposed_category_violation_rates=(proposed_category_rates),
        executed_category_violation_counts=dict(sorted(executed_categories.items())),
        executed_category_violation_rates=(executed_category_rates),
        intervention_reason_counts=dict(sorted(reason_counts.items())),
        intervention_reason_step_rates=(reason_step_rates),
        intervention_reason_intervention_fractions=(reason_intervention_fractions),
        selected_candidate_source_counts=dict(sorted(candidate_source_counts.items())),
        selected_candidate_source_rates=(candidate_source_rates),
        mean_candidate_count=float(statistics.mean(candidate_counts)),
        p95_candidate_count=_p95(candidate_counts),
        max_candidate_count=int(
            max(
                candidate_counts,
                default=0.0,
            )
        ),
        mean_eligible_candidate_count=float(statistics.mean(eligible_candidate_counts)),
        proposed_hard_guard_failure_count=(proposed_guard_failures),
        proposed_hard_guard_failure_rate=(proposed_guard_failures / total_steps),
        selected_hard_guard_failure_count=(selected_guard_failures),
        selected_hard_guard_failure_rate=(selected_guard_failures / total_steps),
        emergency_fallback_count=(fallback_count),
        emergency_fallback_rate=(fallback_count / total_steps),
        mean_current_v=float(statistics.mean(current_v_values)),
        mean_proposed_next_v=float(statistics.mean(proposed_next_v_values)),
        mean_selected_next_v=float(statistics.mean(selected_next_v_values)),
        mean_proposed_delta_v=float(statistics.mean(proposed_delta_v_values)),
        mean_selected_delta_v=float(statistics.mean(selected_delta_v_values)),
        mean_selection_v_delta=float(statistics.mean(selection_v_delta_values)),
        strict_lyapunov_decrease_count=(strict_decreases),
        strict_lyapunov_decrease_rate=(strict_decreases / total_steps),
        lyapunov_nonincrease_count=(nonincreases),
        lyapunov_nonincrease_rate=(nonincreases / total_steps),
        selected_lower_than_proposed_count=(selected_lower_than_proposed),
        selected_lower_than_proposed_rate=(selected_lower_than_proposed / total_steps),
        zero_risk_state_step_count=(zero_risk_steps),
        zero_risk_preserved_step_count=(zero_risk_preserved),
        zero_risk_preservation_rate=(
            (zero_risk_preserved / zero_risk_steps) if zero_risk_steps > 0 else 0.0
        ),
        zero_risk_recovery_count=(recovery_count),
        mean_steps_to_zero_risk_recovery=(
            float(statistics.mean(recovery_steps)) if recovery_steps else 0.0
        ),
        mean_filter_latency_ms=float(statistics.mean(latencies)),
        filter_latency_sample_sd_ms=(_sample_sd(latencies)),
        p95_filter_latency_ms=_p95(latencies),
        max_filter_latency_ms=float(
            max(
                latencies,
                default=0.0,
            )
        ),
    )


def driving_lyapunov_episode_to_dict(
    episode: DrivingLyapunovEpisodeEvidence,
) -> dict[str, Any]:
    """Convert episode evidence to JSON-safe form."""

    payload = asdict(episode)

    payload["method"] = episode.method.value

    payload["robustness_condition"] = episode.robustness_condition.value

    return payload


def driving_lyapunov_summary_to_dict(
    summary: DrivingLyapunovSeedSummary,
) -> dict[str, Any]:
    """Convert seed summary to JSON-safe form."""

    payload = asdict(summary)

    payload["method"] = summary.method.value

    payload["robustness_condition"] = summary.robustness_condition.value

    return payload


@dataclass(frozen=True)
class DrivingLyapunovAuditSnapshot:
    """Deterministic audit snapshot for one evaluated driving step."""

    evaluation_seed: int
    step_index: int

    true_state: tuple[float, ...]
    observed_state: tuple[float, ...]

    proposed_action: tuple[float, ...]
    executed_action: tuple[float, ...]

    current_v: float
    proposed_next_v: float
    selected_next_v: float

    proposed_delta_v: float
    selected_delta_v: float
    selection_v_delta: float

    intervened: bool
    intervention_reason: str

    correction_l2: float

    candidate_count: int
    eligible_candidate_count: int

    selected_candidate_index: int
    selected_candidate_source: str

    proposed_hard_guard_passed: bool
    selected_hard_guard_passed: bool

    emergency_fallback: bool


@dataclass(frozen=True)
class DrivingLyapunovEpisodeRun:
    """One episode plus bounded deterministic audit evidence."""

    episode: DrivingLyapunovEpisodeEvidence

    intervened_audit_snapshots: tuple[
        DrivingLyapunovAuditSnapshot,
        ...,
    ]

    nonintervened_audit_snapshots: tuple[
        DrivingLyapunovAuditSnapshot,
        ...,
    ]


def driving_lyapunov_audit_to_dict(
    snapshot: DrivingLyapunovAuditSnapshot,
) -> dict[str, Any]:
    """Convert an audit snapshot to JSON-safe form."""

    return asdict(snapshot)


from time import perf_counter_ns

from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    LoadedPPOPolicy,
    action_is_within_bounds,
    deterministic_policy_action,
    environment_for_domain,
)
from q_vla_forge.safety.contracts import (
    ViolationSeverity,
)
from q_vla_forge.safety.driving_constraints import (
    evaluate_driving_violations,
)
from q_vla_forge.safety.lyapunov import (
    evaluate_lyapunov,
)
from q_vla_forge.safety.lyapunov_filter import (
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.predictors import (
    predict_driving_next_state,
)

AUDIT_SNAPSHOT_LIMIT_PER_CLASS = 5


def _potential(
    state: np.ndarray,
) -> float:
    return float(
        evaluate_lyapunov(
            domain=DRIVING_DOMAIN,
            state=state,
        ).total
    )


def _audit_snapshot(
    *,
    evaluation_seed: int,
    step_index: int,
    true_state: np.ndarray,
    observed_state: np.ndarray,
    proposed_action: np.ndarray,
    executed_action: np.ndarray,
    current_v: float,
    proposed_next_v: float,
    selected_next_v: float,
    proposed_delta_v: float,
    selected_delta_v: float,
    selection_v_delta: float,
    intervened: bool,
    intervention_reason: str,
    correction_l2: float,
    candidate_count: int,
    eligible_candidate_count: int,
    selected_candidate_index: int,
    selected_candidate_source: str,
    proposed_hard_guard_passed: bool,
    selected_hard_guard_passed: bool,
    emergency_fallback: bool,
) -> DrivingLyapunovAuditSnapshot:
    return DrivingLyapunovAuditSnapshot(
        evaluation_seed=evaluation_seed,
        step_index=step_index,
        true_state=tuple(float(value) for value in true_state),
        observed_state=tuple(float(value) for value in observed_state),
        proposed_action=tuple(float(value) for value in proposed_action),
        executed_action=tuple(float(value) for value in executed_action),
        current_v=float(current_v),
        proposed_next_v=float(proposed_next_v),
        selected_next_v=float(selected_next_v),
        proposed_delta_v=float(proposed_delta_v),
        selected_delta_v=float(selected_delta_v),
        selection_v_delta=float(selection_v_delta),
        intervened=bool(intervened),
        intervention_reason=(intervention_reason),
        correction_l2=float(correction_l2),
        candidate_count=int(candidate_count),
        eligible_candidate_count=int(eligible_candidate_count),
        selected_candidate_index=int(selected_candidate_index),
        selected_candidate_source=(selected_candidate_source),
        proposed_hard_guard_passed=bool(proposed_hard_guard_passed),
        selected_hard_guard_passed=bool(selected_hard_guard_passed),
        emergency_fallback=bool(emergency_fallback),
    )


def run_driving_lyapunov_episode(
    *,
    policy: LoadedPPOPolicy,
    evaluation_seed: int,
) -> DrivingLyapunovEpisodeRun:
    """Run one deterministic CLEAN + LYAPUNOV driving episode."""

    if policy.domain != DRIVING_DOMAIN:
        raise ValueError("Sprint 5.8 runner requires autonomous_driving policy")

    if evaluation_seed not in SAFETY_EVALUATION_SEEDS:
        raise ValueError("evaluation seed is outside the frozen Sprint 5.1 set")

    env = environment_for_domain(DRIVING_DOMAIN)

    observation, _ = env.reset(seed=evaluation_seed)

    total_reward = 0.0
    episode_length = 0

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraint_violations = 0
    executed_constraint_violations = 0

    critical_violation_steps = 0

    intervention_count = 0

    corrections: list[float] = []

    policy_oob_count = 0
    executed_oob_count = 0

    proposed_category_counts: dict[
        str,
        int,
    ] = {}

    executed_category_counts: dict[
        str,
        int,
    ] = {}

    reason_counts: dict[
        str,
        int,
    ] = {}

    candidate_source_counts: dict[
        str,
        int,
    ] = {}

    candidate_counts: list[int] = []
    eligible_candidate_counts: list[int] = []

    proposed_hard_guard_failures = 0
    selected_hard_guard_failures = 0

    emergency_fallback_count = 0

    current_v_values: list[float] = []
    proposed_next_v_values: list[float] = []
    selected_next_v_values: list[float] = []

    proposed_delta_v_values: list[float] = []
    selected_delta_v_values: list[float] = []
    selection_v_delta_values: list[float] = []

    strict_decrease_count = 0
    nonincrease_count = 0

    selected_lower_than_proposed_count = 0

    zero_risk_state_steps = 0
    zero_risk_preserved_steps = 0

    recovery_count = 0
    recovery_steps: list[int] = []

    active_recovery_steps: int | None = None

    latencies_ms: list[float] = []

    intervened_audits: list[DrivingLyapunovAuditSnapshot] = []

    nonintervened_audits: list[DrivingLyapunovAuditSnapshot] = []

    final_success = False

    try:
        while True:
            true_state = np.asarray(
                observation,
                dtype=np.float64,
            ).copy()

            observed_state = true_state.copy()

            proposed_action = deterministic_policy_action(
                policy=policy,
                observation=observed_state,
            )

            if not action_is_within_bounds(
                action=proposed_action,
                action_low=policy.action_low,
                action_high=policy.action_high,
            ):
                policy_oob_count += 1

            proposed_violations = tuple(
                record
                for record in evaluate_driving_violations(
                    true_state,
                    proposed_action,
                )
                if record.violated
            )

            start_ns = perf_counter_ns()

            decision = apply_lyapunov_safety_filter(
                domain=DRIVING_DOMAIN,
                true_state=true_state,
                proposed_action=proposed_action,
            )

            end_ns = perf_counter_ns()

            latency_ms = (end_ns - start_ns) / 1_000_000.0

            latencies_ms.append(float(latency_ms))

            if decision.method != SafetyMethod.LYAPUNOV:
                raise RuntimeError("Sprint 5.8 runner received non-Lyapunov decision")

            executed_action = np.asarray(
                decision.executed_action,
                dtype=np.float64,
            ).copy()

            if not action_is_within_bounds(
                action=executed_action,
                action_low=policy.action_low,
                action_high=policy.action_high,
            ):
                executed_oob_count += 1

            executed_violations = tuple(
                record
                for record in evaluate_driving_violations(
                    true_state,
                    executed_action,
                )
                if record.violated
            )

            if proposed_violations:
                proposed_violation_steps += 1

            if executed_violations:
                executed_violation_steps += 1

            proposed_constraint_violations += len(proposed_violations)

            executed_constraint_violations += len(executed_violations)

            if any(
                record.severity == ViolationSeverity.CRITICAL
                for record in executed_violations
            ):
                critical_violation_steps += 1

            for record in proposed_violations:
                proposed_category_counts[record.name] = (
                    proposed_category_counts.get(
                        record.name,
                        0,
                    )
                    + 1
                )

            for record in executed_violations:
                executed_category_counts[record.name] = (
                    executed_category_counts.get(
                        record.name,
                        0,
                    )
                    + 1
                )

            if decision.intervened:
                intervention_count += 1

            corrections.append(float(decision.correction_l2))

            reason = decision.intervention_reason.value

            reason_counts[reason] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )

            metadata = decision.metadata

            candidate_count = int(metadata["candidate_count"])

            eligible_count = int(metadata["eligible_candidate_count"])

            selected_index = int(metadata["selected_candidate_index"])

            selected_source = str(metadata["selected_candidate_source"])

            proposed_guard_passed = bool(metadata["proposed_hard_guard_passed"])

            selected_guard_passed = bool(metadata["selected_hard_guard_passed"])

            emergency_fallback = bool(metadata["emergency_fallback"])

            candidate_counts.append(candidate_count)

            eligible_candidate_counts.append(eligible_count)

            candidate_source_counts[selected_source] = (
                candidate_source_counts.get(
                    selected_source,
                    0,
                )
                + 1
            )

            if not proposed_guard_passed:
                proposed_hard_guard_failures += 1

            if not selected_guard_passed:
                selected_hard_guard_failures += 1

            if emergency_fallback:
                emergency_fallback_count += 1

            current_v = _potential(true_state)

            proposed_next_state = predict_driving_next_state(
                true_state,
                proposed_action,
            )

            selected_next_state = predict_driving_next_state(
                true_state,
                executed_action,
            )

            proposed_next_v = _potential(
                np.asarray(
                    proposed_next_state,
                    dtype=np.float64,
                )
            )

            selected_next_v = _potential(
                np.asarray(
                    selected_next_state,
                    dtype=np.float64,
                )
            )

            proposed_delta_v = proposed_next_v - current_v

            selected_delta_v = selected_next_v - current_v

            selection_v_delta = selected_next_v - proposed_next_v

            current_v_values.append(current_v)

            proposed_next_v_values.append(proposed_next_v)

            selected_next_v_values.append(selected_next_v)

            proposed_delta_v_values.append(proposed_delta_v)

            selected_delta_v_values.append(selected_delta_v)

            selection_v_delta_values.append(selection_v_delta)

            if selected_delta_v < -LYAPUNOV_IMPROVEMENT_TOLERANCE:
                strict_decrease_count += 1

            if selected_delta_v <= LYAPUNOV_IMPROVEMENT_TOLERANCE:
                nonincrease_count += 1

            if selection_v_delta < -LYAPUNOV_IMPROVEMENT_TOLERANCE:
                selected_lower_than_proposed_count += 1

            if current_v <= LYAPUNOV_IMPROVEMENT_TOLERANCE:
                zero_risk_state_steps += 1

                if selected_next_v <= LYAPUNOV_IMPROVEMENT_TOLERANCE:
                    zero_risk_preserved_steps += 1

            if current_v > LYAPUNOV_IMPROVEMENT_TOLERANCE:
                if active_recovery_steps is None:
                    active_recovery_steps = 0

                active_recovery_steps += 1

                if selected_next_v <= LYAPUNOV_IMPROVEMENT_TOLERANCE:
                    recovery_count += 1

                    recovery_steps.append(active_recovery_steps)

                    active_recovery_steps = None

            elif active_recovery_steps is not None:
                active_recovery_steps = None

            snapshot = _audit_snapshot(
                evaluation_seed=(evaluation_seed),
                step_index=(episode_length),
                true_state=true_state,
                observed_state=observed_state,
                proposed_action=proposed_action,
                executed_action=executed_action,
                current_v=current_v,
                proposed_next_v=(proposed_next_v),
                selected_next_v=(selected_next_v),
                proposed_delta_v=(proposed_delta_v),
                selected_delta_v=(selected_delta_v),
                selection_v_delta=(selection_v_delta),
                intervened=(decision.intervened),
                intervention_reason=reason,
                correction_l2=(decision.correction_l2),
                candidate_count=(candidate_count),
                eligible_candidate_count=(eligible_count),
                selected_candidate_index=(selected_index),
                selected_candidate_source=(selected_source),
                proposed_hard_guard_passed=(proposed_guard_passed),
                selected_hard_guard_passed=(selected_guard_passed),
                emergency_fallback=(emergency_fallback),
            )

            if (
                decision.intervened
                and len(intervened_audits) < AUDIT_SNAPSHOT_LIMIT_PER_CLASS
            ):
                intervened_audits.append(snapshot)

            if (
                not decision.intervened
                and len(nonintervened_audits) < AUDIT_SNAPSHOT_LIMIT_PER_CLASS
            ):
                nonintervened_audits.append(snapshot)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(executed_action)

            total_reward += float(reward)

            episode_length += 1

            if terminated or truncated:
                final_success = bool(info["success"])
                break

    finally:
        env.close()

    episode = DrivingLyapunovEpisodeEvidence(
        domain=DRIVING_DOMAIN,
        principal_seed=(policy.principal_seed),
        evaluation_seed=(evaluation_seed),
        method=SafetyMethod.LYAPUNOV,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=total_reward,
        success=final_success,
        episode_length=episode_length,
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_constraint_violation_count=(proposed_constraint_violations),
        executed_constraint_violation_count=(executed_constraint_violations),
        critical_violation_step_count=(critical_violation_steps),
        intervention_count=(intervention_count),
        action_corrections_l2=tuple(corrections),
        policy_action_out_of_bounds_count=(policy_oob_count),
        executed_action_out_of_bounds_count=(executed_oob_count),
        proposed_category_violation_counts=dict(
            sorted(proposed_category_counts.items())
        ),
        executed_category_violation_counts=dict(
            sorted(executed_category_counts.items())
        ),
        intervention_reason_counts=dict(sorted(reason_counts.items())),
        selected_candidate_source_counts=dict(sorted(candidate_source_counts.items())),
        candidate_counts=tuple(candidate_counts),
        eligible_candidate_counts=tuple(eligible_candidate_counts),
        proposed_hard_guard_failure_count=(proposed_hard_guard_failures),
        selected_hard_guard_failure_count=(selected_hard_guard_failures),
        emergency_fallback_count=(emergency_fallback_count),
        current_v_values=tuple(current_v_values),
        proposed_next_v_values=tuple(proposed_next_v_values),
        selected_next_v_values=tuple(selected_next_v_values),
        proposed_delta_v_values=tuple(proposed_delta_v_values),
        selected_delta_v_values=tuple(selected_delta_v_values),
        selection_v_delta_values=tuple(selection_v_delta_values),
        strict_lyapunov_decrease_count=(strict_decrease_count),
        lyapunov_nonincrease_count=(nonincrease_count),
        selected_lower_than_proposed_count=(selected_lower_than_proposed_count),
        zero_risk_state_step_count=(zero_risk_state_steps),
        zero_risk_preserved_step_count=(zero_risk_preserved_steps),
        zero_risk_recovery_count=(recovery_count),
        zero_risk_recovery_steps=tuple(recovery_steps),
        filter_latency_ms=tuple(latencies_ms),
    )

    _validate_episode(episode)

    return DrivingLyapunovEpisodeRun(
        episode=episode,
        intervened_audit_snapshots=tuple(intervened_audits),
        nonintervened_audit_snapshots=tuple(nonintervened_audits),
    )
