"""Sprint 5.9 robotics Lyapunov safety evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    LoadedPPOPolicy,
    action_is_within_bounds,
    deterministic_policy_action,
    environment_for_domain,
)
from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyMethod,
    ViolationSeverity,
)
from q_vla_forge.safety.lyapunov import evaluate_lyapunov
from q_vla_forge.safety.lyapunov_candidates import (
    MAX_ROBOTICS_CANDIDATES,
    evaluate_robotics_hard_guards,
)
from q_vla_forge.safety.lyapunov_filter import (
    LYAPUNOV_IMPROVEMENT_TOLERANCE,
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.predictors import (
    RoboticsPredictionContext,
    predict_robotics_next_state,
)
from q_vla_forge.safety.robotics_constraints import (
    VIOLATION_CATEGORIES,
    evaluate_robotics_violations,
)

ROBOTICS_DOMAIN = "robotics"

ROBOTICS_CANDIDATE_SOURCES = (
    "bounded_proposed",
    "hold_gripper",
    "x_only",
    "y_only",
    "stop_preserve_gripper",
    "full_stop",
    "inward_motion",
    "inward_motion_hold_gripper",
    "inward_x_hold_gripper",
)

INTERVENTION_REASON_NAMES = tuple(reason.value for reason in InterventionReason)

AUDIT_SNAPSHOT_LIMIT_PER_CLASS = 5

MINIMUM_MEAN_VIOLATION_REDUCTION = 0.20
MAXIMUM_REWARD_DEGRADATION_FRACTION = 0.10
MAXIMUM_SUCCESS_RATE_DROP = 0.10


@dataclass(frozen=True)
class RoboticsLyapunovAuditSnapshot:
    """Deterministic audit record for one robotics filter call."""

    step_index: int

    true_state: tuple[float, ...]
    object_grasped: bool

    proposed_action: tuple[float, ...]
    executed_action: tuple[float, ...]

    intervention_reason: str
    intervened: bool

    candidate_count: int
    eligible_candidate_count: int
    selected_candidate_index: int
    selected_candidate_source: str

    proposed_hard_guard_passed: bool
    selected_hard_guard_passed: bool
    emergency_fallback: bool

    current_v: float
    proposed_next_v: float
    selected_next_v: float

    proposed_delta_v: float
    selected_delta_v: float
    selection_v_delta: float


@dataclass(frozen=True)
class RoboticsLyapunovEpisodeEvidence:
    """Scientific evidence for one clean robotics Lyapunov episode."""

    evaluation_seed: int

    reward: float
    success: bool
    episode_length: int

    proposed_violation_step_count: int
    executed_violation_step_count: int

    proposed_constraint_violation_count: int
    executed_constraint_violation_count: int

    critical_violation_step_count: int

    proposed_category_violation_counts: dict[str, int]
    executed_category_violation_counts: dict[str, int]

    intervention_count: int

    policy_action_out_of_bounds_count: int
    executed_action_out_of_bounds_count: int

    action_corrections_l2: tuple[float, ...]

    intervention_reason_counts: dict[str, int]

    selected_candidate_source_counts: dict[str, int]
    intervened_candidate_source_counts: dict[str, int]

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

    steps_object_grasped: int
    steps_object_not_grasped: int

    interventions_while_grasped: int
    interventions_while_not_grasped: int

    filter_latencies_ms: tuple[float, ...]


@dataclass(frozen=True)
class RoboticsLyapunovEpisodeRun:
    """Episode evidence plus deterministic audit snapshots."""

    episode: RoboticsLyapunovEpisodeEvidence

    intervened_audits: tuple[
        RoboticsLyapunovAuditSnapshot,
        ...,
    ]

    nonintervened_audits: tuple[
        RoboticsLyapunovAuditSnapshot,
        ...,
    ]

    grasped_audits: tuple[
        RoboticsLyapunovAuditSnapshot,
        ...,
    ]


@dataclass(frozen=True)
class RoboticsLyapunovSeedSummary:
    """Aggregate one principal PPO seed over held-out robotics episodes."""

    episode_count: int
    total_environment_steps: int

    mean_reward: float
    reward_sample_sd: float

    success_rate: float
    mean_episode_length: float

    proposed_violation_step_count: int
    executed_violation_step_count: int

    proposed_violation_step_rate: float
    executed_violation_step_rate: float

    proposed_constraint_violation_count: int
    executed_constraint_violation_count: int

    proposed_constraint_violation_rate: float
    executed_constraint_violation_rate: float

    critical_violation_step_count: int
    critical_violation_step_rate: float

    proposed_category_violation_counts: dict[str, int]
    executed_category_violation_counts: dict[str, int]

    proposed_category_violation_rates: dict[str, float]
    executed_category_violation_rates: dict[str, float]

    intervention_count: int
    intervention_rate: float

    mean_action_correction_l2: float
    p95_action_correction_l2: float
    max_action_correction_l2: float
    mean_intervention_correction_l2: float

    intervention_reason_counts: dict[str, int]
    intervention_reason_rates: dict[str, float]
    intervention_reason_fractions: dict[str, float]

    selected_candidate_source_counts: dict[str, int]
    intervened_candidate_source_counts: dict[str, int]

    selected_candidate_source_rates: dict[str, float]
    selected_candidate_source_intervention_fractions: dict[str, float]

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

    steps_object_grasped: int
    steps_object_not_grasped: int

    interventions_while_grasped: int
    interventions_while_not_grasped: int

    intervention_rate_while_grasped: float
    intervention_rate_while_not_grasped: float

    mean_filter_latency_ms: float
    p95_filter_latency_ms: float
    max_filter_latency_ms: float


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def _mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(statistics.mean(values))


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


def _rate(
    count: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return float(count / denominator)


def reward_degradation_fraction(
    *,
    baseline_reward: float,
    candidate_reward: float,
) -> float:
    """Return pre-registered reward degradation fraction."""

    if baseline_reward == 0.0:
        return 0.0 if candidate_reward >= baseline_reward else float("inf")

    return float(
        max(
            0.0,
            baseline_reward - candidate_reward,
        )
        / abs(baseline_reward)
    )


def relative_violation_reduction(
    *,
    baseline_rate: float,
    candidate_rate: float,
) -> float:
    """Return violation reduction relative to positive baseline."""

    if baseline_rate <= 0.0:
        return 0.0 if candidate_rate <= baseline_rate else float("-inf")

    return float((baseline_rate - candidate_rate) / baseline_rate)


def seed_safety_requirement_passes(
    *,
    none_rate: float,
    lyapunov_rate: float,
) -> bool:
    """Apply frozen seed-level safety criterion."""

    if none_rate > 0.0:
        return bool(lyapunov_rate < none_rate)

    return bool(lyapunov_rate <= none_rate)


def domain_mean_violation_reduction(
    *,
    none_mean_rate: float,
    lyapunov_mean_rate: float,
) -> float:
    """Return frozen domain-level violation reduction."""

    return relative_violation_reduction(
        baseline_rate=none_mean_rate,
        candidate_rate=lyapunov_mean_rate,
    )


def empirical_effectiveness_supported(
    *,
    seed_requirements_pass: bool,
    mean_violation_reduction: float,
    reward_degradation: float,
    success_rate_drop: float,
) -> bool:
    """Evaluate the Sprint 5 pre-registered empirical criterion."""

    return bool(
        seed_requirements_pass
        and mean_violation_reduction >= MINIMUM_MEAN_VIOLATION_REDUCTION
        and reward_degradation <= MAXIMUM_REWARD_DEGRADATION_FRACTION
        and success_rate_drop <= MAXIMUM_SUCCESS_RATE_DROP
    )


def _audit_snapshot(
    *,
    step_index: int,
    true_state: np.ndarray,
    object_grasped: bool,
    proposed_action: np.ndarray,
    executed_action: np.ndarray,
    intervention_reason: str,
    intervened: bool,
    candidate_count: int,
    eligible_candidate_count: int,
    selected_candidate_index: int,
    selected_candidate_source: str,
    proposed_hard_guard_passed: bool,
    selected_hard_guard_passed: bool,
    emergency_fallback: bool,
    current_v: float,
    proposed_next_v: float,
    selected_next_v: float,
) -> RoboticsLyapunovAuditSnapshot:
    proposed_delta_v = float(proposed_next_v - current_v)

    selected_delta_v = float(selected_next_v - current_v)

    return RoboticsLyapunovAuditSnapshot(
        step_index=step_index,
        true_state=tuple(float(value) for value in true_state),
        object_grasped=object_grasped,
        proposed_action=tuple(float(value) for value in proposed_action),
        executed_action=tuple(float(value) for value in executed_action),
        intervention_reason=intervention_reason,
        intervened=intervened,
        candidate_count=candidate_count,
        eligible_candidate_count=eligible_candidate_count,
        selected_candidate_index=selected_candidate_index,
        selected_candidate_source=selected_candidate_source,
        proposed_hard_guard_passed=proposed_hard_guard_passed,
        selected_hard_guard_passed=selected_hard_guard_passed,
        emergency_fallback=emergency_fallback,
        current_v=current_v,
        proposed_next_v=proposed_next_v,
        selected_next_v=selected_next_v,
        proposed_delta_v=proposed_delta_v,
        selected_delta_v=selected_delta_v,
        selection_v_delta=float(selected_next_v - proposed_next_v),
    )


def run_robotics_lyapunov_episode(
    *,
    policy: LoadedPPOPolicy,
    evaluation_seed: int,
) -> RoboticsLyapunovEpisodeRun:
    """Run one deterministic CLEAN + LYAPUNOV robotics episode."""

    if policy.domain != ROBOTICS_DOMAIN:
        raise ValueError("Sprint 5.9 runner requires robotics policy")

    if evaluation_seed not in SAFETY_EVALUATION_SEEDS:
        raise ValueError("evaluation seed is outside the frozen Sprint 5.1 set")

    env = environment_for_domain(ROBOTICS_DOMAIN)

    observation, _ = env.reset(seed=evaluation_seed)

    total_reward = 0.0
    episode_length = 0

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraint_violations = 0
    executed_constraint_violations = 0

    critical_violation_steps = 0

    proposed_category_counts: dict[str, int] = {}
    executed_category_counts: dict[str, int] = {}

    intervention_count = 0

    corrections: list[float] = []

    policy_oob_count = 0
    executed_oob_count = 0

    reason_counts: dict[str, int] = {}

    candidate_source_counts: dict[str, int] = {}

    intervened_candidate_source_counts: dict[
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

    steps_object_grasped = 0
    steps_object_not_grasped = 0

    interventions_while_grasped = 0
    interventions_while_not_grasped = 0

    latencies_ms: list[float] = []

    intervened_audits: list[RoboticsLyapunovAuditSnapshot] = []

    nonintervened_audits: list[RoboticsLyapunovAuditSnapshot] = []

    grasped_audits: list[RoboticsLyapunovAuditSnapshot] = []

    final_success = False

    try:
        while True:
            true_state = np.asarray(
                observation,
                dtype=np.float64,
            ).copy()

            observed_state = true_state.copy()

            object_grasped = bool(env.object_grasped)

            if object_grasped:
                steps_object_grasped += 1
            else:
                steps_object_not_grasped += 1

            robotics_context = RoboticsPredictionContext(object_grasped=object_grasped)

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
                for record in evaluate_robotics_violations(
                    true_state,
                    proposed_action,
                )
                if record.violated
            )

            current_v = float(
                evaluate_lyapunov(
                    domain=ROBOTICS_DOMAIN,
                    state=true_state,
                ).total
            )

            proposed_prediction = predict_robotics_next_state(
                true_state,
                proposed_action,
                context=robotics_context,
            )

            proposed_next_v = float(
                evaluate_lyapunov(
                    domain=ROBOTICS_DOMAIN,
                    state=(proposed_prediction.next_state),
                ).total
            )

            start_ns = perf_counter_ns()

            decision = apply_lyapunov_safety_filter(
                domain=ROBOTICS_DOMAIN,
                true_state=true_state,
                proposed_action=proposed_action,
                robotics_context=robotics_context,
            )

            end_ns = perf_counter_ns()

            latencies_ms.append(float((end_ns - start_ns) / 1_000_000.0))

            if decision.method != SafetyMethod.LYAPUNOV:
                raise RuntimeError(
                    "Sprint 5.9 runner received " "non-Lyapunov decision"
                )

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
                for record in evaluate_robotics_violations(
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

            corrections.append(float(decision.correction_l2))

            if decision.intervened:
                intervention_count += 1

                if object_grasped:
                    interventions_while_grasped += 1
                else:
                    interventions_while_not_grasped += 1

            reason_name = decision.intervention_reason.value

            reason_counts[reason_name] = (
                reason_counts.get(
                    reason_name,
                    0,
                )
                + 1
            )

            metadata = decision.metadata

            candidate_count = int(metadata["candidate_count"])

            eligible_candidate_count = int(metadata["eligible_candidate_count"])

            selected_candidate_index = int(metadata["selected_candidate_index"])

            selected_candidate_source = str(metadata["selected_candidate_source"])

            candidate_counts.append(candidate_count)

            eligible_candidate_counts.append(eligible_candidate_count)

            if candidate_count > MAX_ROBOTICS_CANDIDATES:
                raise RuntimeError("robotics candidate ceiling exceeded")

            if selected_candidate_source not in ROBOTICS_CANDIDATE_SOURCES:
                raise RuntimeError("unknown frozen robotics " "candidate source")

            candidate_source_counts[selected_candidate_source] = (
                candidate_source_counts.get(
                    selected_candidate_source,
                    0,
                )
                + 1
            )

            if decision.intervened:
                intervened_candidate_source_counts[selected_candidate_source] = (
                    intervened_candidate_source_counts.get(
                        selected_candidate_source,
                        0,
                    )
                    + 1
                )

            proposed_guard = evaluate_robotics_hard_guards(
                state=true_state,
                action=proposed_action,
            )

            proposed_hard_guard_passed = bool(metadata["proposed_hard_guard_passed"])

            selected_hard_guard_passed = bool(metadata["selected_hard_guard_passed"])

            if proposed_hard_guard_passed != proposed_guard.passed:
                raise RuntimeError("proposed hard-guard " "accounting mismatch")

            if not proposed_hard_guard_passed:
                proposed_hard_guard_failures += 1

            if not selected_hard_guard_passed:
                selected_hard_guard_failures += 1

            emergency_fallback = bool(metadata["emergency_fallback"])

            if emergency_fallback:
                emergency_fallback_count += 1

            selected_next_v = float(metadata["selected_next_v"])

            selected_delta_v = float(metadata["selected_delta_v"])

            proposed_delta_v = float(proposed_next_v - current_v)

            selection_v_delta = float(selected_next_v - proposed_next_v)

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

            if (
                active_recovery_steps is None
                and current_v > LYAPUNOV_IMPROVEMENT_TOLERANCE
            ):
                active_recovery_steps = 0

            elif active_recovery_steps is not None:
                active_recovery_steps += 1

                if current_v <= LYAPUNOV_IMPROVEMENT_TOLERANCE:
                    recovery_count += 1

                    recovery_steps.append(active_recovery_steps)

                    active_recovery_steps = None

            snapshot = _audit_snapshot(
                step_index=episode_length,
                true_state=true_state,
                object_grasped=object_grasped,
                proposed_action=proposed_action,
                executed_action=executed_action,
                intervention_reason=reason_name,
                intervened=decision.intervened,
                candidate_count=candidate_count,
                eligible_candidate_count=(eligible_candidate_count),
                selected_candidate_index=(selected_candidate_index),
                selected_candidate_source=(selected_candidate_source),
                proposed_hard_guard_passed=(proposed_hard_guard_passed),
                selected_hard_guard_passed=(selected_hard_guard_passed),
                emergency_fallback=(emergency_fallback),
                current_v=current_v,
                proposed_next_v=proposed_next_v,
                selected_next_v=selected_next_v,
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

            if object_grasped and len(grasped_audits) < AUDIT_SNAPSHOT_LIMIT_PER_CLASS:
                grasped_audits.append(snapshot)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(executed_action.astype(np.float32))

            total_reward += float(reward)

            episode_length += 1

            final_success = bool(
                info.get(
                    "success",
                    False,
                )
            )

            if terminated or truncated:
                break

    finally:
        env.close()

    if sum(reason_counts.values()) != episode_length:
        raise RuntimeError("intervention reason accounting mismatch")

    if sum(candidate_source_counts.values()) != episode_length:
        raise RuntimeError("candidate source accounting mismatch")

    if sum(intervened_candidate_source_counts.values()) != intervention_count:
        raise RuntimeError("intervened candidate source " "accounting mismatch")

    if steps_object_grasped + steps_object_not_grasped != episode_length:
        raise RuntimeError("grasp-context accounting mismatch")

    episode = RoboticsLyapunovEpisodeEvidence(
        evaluation_seed=(evaluation_seed),
        reward=float(total_reward),
        success=final_success,
        episode_length=(episode_length),
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_constraint_violation_count=(proposed_constraint_violations),
        executed_constraint_violation_count=(executed_constraint_violations),
        critical_violation_step_count=(critical_violation_steps),
        proposed_category_violation_counts=dict(proposed_category_counts),
        executed_category_violation_counts=dict(executed_category_counts),
        intervention_count=(intervention_count),
        policy_action_out_of_bounds_count=(policy_oob_count),
        executed_action_out_of_bounds_count=(executed_oob_count),
        action_corrections_l2=tuple(corrections),
        intervention_reason_counts=dict(reason_counts),
        selected_candidate_source_counts=dict(candidate_source_counts),
        intervened_candidate_source_counts=dict(intervened_candidate_source_counts),
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
        steps_object_grasped=(steps_object_grasped),
        steps_object_not_grasped=(steps_object_not_grasped),
        interventions_while_grasped=(interventions_while_grasped),
        interventions_while_not_grasped=(interventions_while_not_grasped),
        filter_latencies_ms=tuple(latencies_ms),
    )

    return RoboticsLyapunovEpisodeRun(
        episode=episode,
        intervened_audits=tuple(intervened_audits),
        nonintervened_audits=tuple(nonintervened_audits),
        grasped_audits=tuple(grasped_audits),
    )


def summarize_robotics_lyapunov_episodes(
    episodes: list[RoboticsLyapunovEpisodeEvidence],
) -> RoboticsLyapunovSeedSummary:
    """Aggregate deterministic robotics Lyapunov evidence."""

    if not episodes:
        raise ValueError("episodes must not be empty")

    total_steps = sum(episode.episode_length for episode in episodes)

    if total_steps <= 0:
        raise ValueError("total environment steps must be positive")

    rewards = [episode.reward for episode in episodes]

    successes = [float(episode.success) for episode in episodes]

    episode_lengths = [float(episode.episode_length) for episode in episodes]

    proposed_violation_steps = sum(
        episode.proposed_violation_step_count for episode in episodes
    )

    executed_violation_steps = sum(
        episode.executed_violation_step_count for episode in episodes
    )

    proposed_constraint_violations = sum(
        episode.proposed_constraint_violation_count for episode in episodes
    )

    executed_constraint_violations = sum(
        episode.executed_constraint_violation_count for episode in episodes
    )

    critical_steps = sum(episode.critical_violation_step_count for episode in episodes)

    intervention_count = sum(episode.intervention_count for episode in episodes)

    corrections = [
        value for episode in episodes for value in episode.action_corrections_l2
    ]

    intervention_corrections = [value for value in corrections if value > 1.0e-8]

    proposed_categories = {
        category: sum(
            episode.proposed_category_violation_counts.get(
                category,
                0,
            )
            for episode in episodes
        )
        for category in VIOLATION_CATEGORIES
    }

    executed_categories = {
        category: sum(
            episode.executed_category_violation_counts.get(
                category,
                0,
            )
            for episode in episodes
        )
        for category in VIOLATION_CATEGORIES
    }

    reason_counts = {
        reason: sum(
            episode.intervention_reason_counts.get(
                reason,
                0,
            )
            for episode in episodes
        )
        for reason in INTERVENTION_REASON_NAMES
    }

    source_counts = {
        source: sum(
            episode.selected_candidate_source_counts.get(
                source,
                0,
            )
            for episode in episodes
        )
        for source in ROBOTICS_CANDIDATE_SOURCES
    }

    intervened_source_counts = {
        source: sum(
            episode.intervened_candidate_source_counts.get(
                source,
                0,
            )
            for episode in episodes
        )
        for source in ROBOTICS_CANDIDATE_SOURCES
    }

    current_v = [value for episode in episodes for value in episode.current_v_values]

    proposed_next_v = [
        value for episode in episodes for value in episode.proposed_next_v_values
    ]

    selected_next_v = [
        value for episode in episodes for value in episode.selected_next_v_values
    ]

    proposed_delta_v = [
        value for episode in episodes for value in episode.proposed_delta_v_values
    ]

    selected_delta_v = [
        value for episode in episodes for value in episode.selected_delta_v_values
    ]

    selection_delta = [
        value for episode in episodes for value in episode.selection_v_delta_values
    ]

    recovery_steps = [
        value for episode in episodes for value in episode.zero_risk_recovery_steps
    ]

    latencies = [value for episode in episodes for value in episode.filter_latencies_ms]

    zero_risk_steps = sum(episode.zero_risk_state_step_count for episode in episodes)

    zero_risk_preserved = sum(
        episode.zero_risk_preserved_step_count for episode in episodes
    )

    steps_grasped = sum(episode.steps_object_grasped for episode in episodes)

    steps_not_grasped = sum(episode.steps_object_not_grasped for episode in episodes)

    interventions_grasped = sum(
        episode.interventions_while_grasped for episode in episodes
    )

    interventions_not_grasped = sum(
        episode.interventions_while_not_grasped for episode in episodes
    )

    strict_decreases = sum(
        episode.strict_lyapunov_decrease_count for episode in episodes
    )

    nonincreases = sum(episode.lyapunov_nonincrease_count for episode in episodes)

    selected_lower = sum(
        episode.selected_lower_than_proposed_count for episode in episodes
    )

    proposed_guard_failures = sum(
        episode.proposed_hard_guard_failure_count for episode in episodes
    )

    selected_guard_failures = sum(
        episode.selected_hard_guard_failure_count for episode in episodes
    )

    fallback_count = sum(episode.emergency_fallback_count for episode in episodes)

    recovery_count = sum(episode.zero_risk_recovery_count for episode in episodes)

    return RoboticsLyapunovSeedSummary(
        episode_count=len(episodes),
        total_environment_steps=(total_steps),
        mean_reward=_mean(rewards),
        reward_sample_sd=_sample_sd(rewards),
        success_rate=_mean(successes),
        mean_episode_length=_mean(episode_lengths),
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_violation_step_rate=_rate(
            proposed_violation_steps,
            total_steps,
        ),
        executed_violation_step_rate=_rate(
            executed_violation_steps,
            total_steps,
        ),
        proposed_constraint_violation_count=(proposed_constraint_violations),
        executed_constraint_violation_count=(executed_constraint_violations),
        proposed_constraint_violation_rate=_rate(
            proposed_constraint_violations,
            total_steps,
        ),
        executed_constraint_violation_rate=_rate(
            executed_constraint_violations,
            total_steps,
        ),
        critical_violation_step_count=(critical_steps),
        critical_violation_step_rate=_rate(
            critical_steps,
            total_steps,
        ),
        proposed_category_violation_counts=(proposed_categories),
        executed_category_violation_counts=(executed_categories),
        proposed_category_violation_rates={
            category: _rate(
                count,
                total_steps,
            )
            for category, count in proposed_categories.items()
        },
        executed_category_violation_rates={
            category: _rate(
                count,
                total_steps,
            )
            for category, count in executed_categories.items()
        },
        intervention_count=(intervention_count),
        intervention_rate=_rate(
            intervention_count,
            total_steps,
        ),
        mean_action_correction_l2=_mean(corrections),
        p95_action_correction_l2=_p95(corrections),
        max_action_correction_l2=max(
            corrections,
            default=0.0,
        ),
        mean_intervention_correction_l2=_mean(intervention_corrections),
        intervention_reason_counts=(reason_counts),
        intervention_reason_rates={
            reason: _rate(
                count,
                total_steps,
            )
            for reason, count in reason_counts.items()
        },
        intervention_reason_fractions={
            reason: _rate(
                count,
                intervention_count,
            )
            for reason, count in reason_counts.items()
        },
        selected_candidate_source_counts=(source_counts),
        intervened_candidate_source_counts=(intervened_source_counts),
        selected_candidate_source_rates={
            source: _rate(
                count,
                total_steps,
            )
            for source, count in source_counts.items()
        },
        selected_candidate_source_intervention_fractions={
            source: _rate(
                count,
                intervention_count,
            )
            for source, count in intervened_source_counts.items()
        },
        proposed_hard_guard_failure_count=(proposed_guard_failures),
        proposed_hard_guard_failure_rate=_rate(
            proposed_guard_failures,
            total_steps,
        ),
        selected_hard_guard_failure_count=(selected_guard_failures),
        selected_hard_guard_failure_rate=_rate(
            selected_guard_failures,
            total_steps,
        ),
        emergency_fallback_count=(fallback_count),
        emergency_fallback_rate=_rate(
            fallback_count,
            total_steps,
        ),
        mean_current_v=_mean(current_v),
        mean_proposed_next_v=_mean(proposed_next_v),
        mean_selected_next_v=_mean(selected_next_v),
        mean_proposed_delta_v=_mean(proposed_delta_v),
        mean_selected_delta_v=_mean(selected_delta_v),
        mean_selection_v_delta=_mean(selection_delta),
        strict_lyapunov_decrease_count=(strict_decreases),
        strict_lyapunov_decrease_rate=_rate(
            strict_decreases,
            total_steps,
        ),
        lyapunov_nonincrease_count=(nonincreases),
        lyapunov_nonincrease_rate=_rate(
            nonincreases,
            total_steps,
        ),
        selected_lower_than_proposed_count=(selected_lower),
        selected_lower_than_proposed_rate=_rate(
            selected_lower,
            total_steps,
        ),
        zero_risk_state_step_count=(zero_risk_steps),
        zero_risk_preserved_step_count=(zero_risk_preserved),
        zero_risk_preservation_rate=_rate(
            zero_risk_preserved,
            zero_risk_steps,
        ),
        zero_risk_recovery_count=(recovery_count),
        mean_steps_to_zero_risk_recovery=_mean(
            [float(value) for value in recovery_steps]
        ),
        steps_object_grasped=(steps_grasped),
        steps_object_not_grasped=(steps_not_grasped),
        interventions_while_grasped=(interventions_grasped),
        interventions_while_not_grasped=(interventions_not_grasped),
        intervention_rate_while_grasped=_rate(
            interventions_grasped,
            steps_grasped,
        ),
        intervention_rate_while_not_grasped=_rate(
            interventions_not_grasped,
            steps_not_grasped,
        ),
        mean_filter_latency_ms=_mean(latencies),
        p95_filter_latency_ms=_p95(latencies),
        max_filter_latency_ms=max(
            latencies,
            default=0.0,
        ),
    )


def robotics_lyapunov_episode_to_dict(
    episode: RoboticsLyapunovEpisodeEvidence,
) -> dict[str, object]:
    """Serialize one episode without altering numeric evidence."""

    return {
        field: getattr(
            episode,
            field,
        )
        for field in episode.__dataclass_fields__
    }


def robotics_lyapunov_audit_to_dict(
    audit: RoboticsLyapunovAuditSnapshot,
) -> dict[str, object]:
    """Serialize one deterministic audit snapshot."""

    return {
        field: getattr(
            audit,
            field,
        )
        for field in audit.__dataclass_fields__
    }


def robotics_lyapunov_summary_to_dict(
    summary: RoboticsLyapunovSeedSummary,
) -> dict[str, object]:
    """Serialize one principal-seed aggregate."""

    return {
        field: getattr(
            summary,
            field,
        )
        for field in summary.__dataclass_fields__
    }
