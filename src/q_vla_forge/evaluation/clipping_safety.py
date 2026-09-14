"""Evaluation utilities for Sprint 5.5 heuristic clipping."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    LoadedPPOPolicy,
    action_is_within_bounds,
    deterministic_policy_action,
    environment_for_domain,
)
from q_vla_forge.safety.clipping import (
    INTERVENTION_TOLERANCE_L2,
    apply_clipping_safety_filter,
)
from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
    ViolationSeverity,
)


@dataclass(frozen=True)
class ClippingEpisodeEvidence:
    """Episode evidence for one clean-condition clipping evaluation."""

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

    total_action_correction_l2: float
    max_action_correction_l2: float
    action_corrections_l2: tuple[float, ...]

    policy_action_out_of_bounds_count: int
    executed_action_out_of_bounds_count: int

    proposed_category_violation_counts: dict[str, int]
    executed_category_violation_counts: dict[str, int]

    triggered_rule_counts: dict[str, int]


@dataclass(frozen=True)
class ClippingSeedSummary:
    """Aggregated evidence for one clipping domain/seed cell."""

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

    triggered_rule_counts: dict[str, int]
    triggered_rule_rates: dict[str, float]


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


def run_clipping_episode(
    *,
    policy: LoadedPPOPolicy,
    evaluation_seed: int,
) -> ClippingEpisodeEvidence:
    """Run one deterministic CLEAN + CLIPPING episode."""

    if evaluation_seed not in SAFETY_EVALUATION_SEEDS:
        raise ValueError("evaluation seed is outside the frozen Sprint 5.1 set")

    env = environment_for_domain(policy.domain)

    observation, _ = env.reset(seed=evaluation_seed)

    total_reward = 0.0
    episode_length = 0

    proposed_violation_step_count = 0
    executed_violation_step_count = 0

    proposed_constraint_violation_count = 0
    executed_constraint_violation_count = 0

    critical_violation_step_count = 0

    intervention_count = 0

    corrections: list[float] = []

    policy_action_out_of_bounds_count = 0
    executed_action_out_of_bounds_count = 0

    proposed_category_counts: dict[
        str,
        int,
    ] = {}

    executed_category_counts: dict[
        str,
        int,
    ] = {}

    triggered_rule_counts: dict[
        str,
        int,
    ] = {}

    final_success = False

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
            policy_action_out_of_bounds_count += 1

        decision = apply_clipping_safety_filter(
            domain=policy.domain,
            true_state=true_state,
            proposed_action=proposed_action,
        )

        if decision.method != SafetyMethod.CLIPPING:
            raise RuntimeError("clipping evaluator received non-clipping decision")

        if not action_is_within_bounds(
            action=decision.executed_action,
            action_low=policy.action_low,
            action_high=policy.action_high,
        ):
            executed_action_out_of_bounds_count += 1

        proposed_violations = tuple(
            record for record in decision.violations_before if record.violated
        )

        executed_violations = tuple(
            record for record in decision.violations_after if record.violated
        )

        if proposed_violations:
            proposed_violation_step_count += 1

        if executed_violations:
            executed_violation_step_count += 1

        proposed_constraint_violation_count += len(proposed_violations)

        executed_constraint_violation_count += len(executed_violations)

        if any(
            record.severity == ViolationSeverity.CRITICAL
            for record in executed_violations
        ):
            critical_violation_step_count += 1

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

        for rule in decision.metadata["triggered_rules"]:
            triggered_rule_counts[rule] = (
                triggered_rule_counts.get(
                    rule,
                    0,
                )
                + 1
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(decision.executed_action)

        total_reward += float(reward)

        episode_length += 1

        if terminated or truncated:
            final_success = bool(info["success"])
            break

    env.close()

    return ClippingEpisodeEvidence(
        domain=policy.domain,
        principal_seed=(policy.principal_seed),
        evaluation_seed=(evaluation_seed),
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=total_reward,
        success=final_success,
        episode_length=(episode_length),
        proposed_violation_step_count=(proposed_violation_step_count),
        executed_violation_step_count=(executed_violation_step_count),
        proposed_constraint_violation_count=(proposed_constraint_violation_count),
        executed_constraint_violation_count=(executed_constraint_violation_count),
        critical_violation_step_count=(critical_violation_step_count),
        intervention_count=(intervention_count),
        total_action_correction_l2=float(sum(corrections)),
        max_action_correction_l2=float(
            max(
                corrections,
                default=0.0,
            )
        ),
        action_corrections_l2=tuple(float(value) for value in corrections),
        policy_action_out_of_bounds_count=(policy_action_out_of_bounds_count),
        executed_action_out_of_bounds_count=(executed_action_out_of_bounds_count),
        proposed_category_violation_counts=dict(
            sorted(proposed_category_counts.items())
        ),
        executed_category_violation_counts=dict(
            sorted(executed_category_counts.items())
        ),
        triggered_rule_counts=dict(sorted(triggered_rule_counts.items())),
    )


def summarize_clipping_episodes(
    episodes: list[ClippingEpisodeEvidence],
) -> ClippingSeedSummary:
    """Aggregate one clipping domain/seed cell."""

    if not episodes:
        raise ValueError("at least one clipping episode is required")

    first = episodes[0]

    domain = first.domain
    principal_seed = first.principal_seed

    evaluation_seeds: set[int] = set()

    total_steps = 0

    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[float] = []

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraints = 0
    executed_constraints = 0

    critical_steps = 0

    interventions = 0

    all_step_corrections: list[float] = []

    intervention_corrections: list[float] = []

    policy_oob = 0
    executed_oob = 0

    proposed_categories: dict[
        str,
        int,
    ] = {}

    executed_categories: dict[
        str,
        int,
    ] = {}

    rule_counts: dict[
        str,
        int,
    ] = {}

    for episode in episodes:
        if episode.domain != domain:
            raise ValueError("episodes contain multiple domains")

        if episode.principal_seed != principal_seed:
            raise ValueError("episodes contain multiple principal seeds")

        if episode.method != SafetyMethod.CLIPPING:
            raise ValueError("non-clipping episode in clipping summary")

        if episode.robustness_condition != RobustnessCondition.CLEAN:
            raise ValueError("non-clean episode in clipping summary")

        if episode.evaluation_seed in evaluation_seeds:
            raise ValueError("duplicate evaluation seed")

        evaluation_seeds.add(episode.evaluation_seed)

        if episode.episode_length <= 0:
            raise ValueError("episode length must be positive")

        if len(episode.action_corrections_l2) != episode.episode_length:
            raise ValueError(
                "step correction evidence length must equal episode length"
            )

        if any(correction < 0.0 for correction in episode.action_corrections_l2):
            raise ValueError("action correction cannot be negative")

        reconstructed_interventions = sum(
            correction > INTERVENTION_TOLERANCE_L2
            for correction in episode.action_corrections_l2
        )

        if reconstructed_interventions != episode.intervention_count:
            raise ValueError(
                "intervention count does not match step correction evidence"
            )

        reconstructed_total = float(sum(episode.action_corrections_l2))

        if not np.isclose(
            reconstructed_total,
            episode.total_action_correction_l2,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("total correction does not match step correction evidence")

        reconstructed_max = float(
            max(
                episode.action_corrections_l2,
                default=0.0,
            )
        )

        if not np.isclose(
            reconstructed_max,
            episode.max_action_correction_l2,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "maximum correction does not match step correction evidence"
            )

        total_steps += episode.episode_length

        rewards.append(float(episode.reward))

        successes.append(float(episode.success))

        lengths.append(float(episode.episode_length))

        proposed_violation_steps += episode.proposed_violation_step_count

        executed_violation_steps += episode.executed_violation_step_count

        proposed_constraints += episode.proposed_constraint_violation_count

        executed_constraints += episode.executed_constraint_violation_count

        critical_steps += episode.critical_violation_step_count

        interventions += episode.intervention_count

        policy_oob += episode.policy_action_out_of_bounds_count

        executed_oob += episode.executed_action_out_of_bounds_count

        all_step_corrections.extend(
            float(correction) for correction in episode.action_corrections_l2
        )

        intervention_corrections.extend(
            float(correction)
            for correction in episode.action_corrections_l2
            if (correction > INTERVENTION_TOLERANCE_L2)
        )

        for (
            category,
            count,
        ) in episode.proposed_category_violation_counts.items():
            if count < 0:
                raise ValueError("category violation count cannot be negative")

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
            if count < 0:
                raise ValueError("category violation count cannot be negative")

            executed_categories[category] = (
                executed_categories.get(
                    category,
                    0,
                )
                + count
            )

        for (
            rule,
            count,
        ) in episode.triggered_rule_counts.items():
            if count < 0:
                raise ValueError("triggered rule count cannot be negative")

            rule_counts[rule] = (
                rule_counts.get(
                    rule,
                    0,
                )
                + count
            )

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

    rule_rates = {
        rule: (count / total_steps)
        for (
            rule,
            count,
        ) in sorted(rule_counts.items())
    }

    return ClippingSeedSummary(
        domain=domain,
        principal_seed=(principal_seed),
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        episode_count=len(episodes),
        total_environment_steps=(total_steps),
        mean_reward=float(statistics.mean(rewards)),
        reward_sample_sd=(_sample_sd(rewards)),
        success_rate=float(statistics.mean(successes)),
        mean_episode_length=float(statistics.mean(lengths)),
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
        mean_action_correction_l2=float(statistics.mean(all_step_corrections)),
        p95_action_correction_l2=(_p95(all_step_corrections)),
        max_action_correction_l2=float(
            max(
                all_step_corrections,
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
        triggered_rule_counts=dict(sorted(rule_counts.items())),
        triggered_rule_rates=(rule_rates),
    )


def clipping_episode_to_dict(
    episode: ClippingEpisodeEvidence,
) -> dict[str, Any]:
    """Convert clipping episode evidence into JSON-safe form."""

    payload = asdict(episode)

    payload["method"] = episode.method.value

    payload["robustness_condition"] = episode.robustness_condition.value

    payload["action_corrections_l2"] = list(episode.action_corrections_l2)

    return payload


def clipping_summary_to_dict(
    summary: ClippingSeedSummary,
) -> dict[str, Any]:
    """Convert clipping seed summary into JSON-safe form."""

    payload = asdict(summary)

    payload["method"] = summary.method.value

    payload["robustness_condition"] = summary.robustness_condition.value

    return payload
