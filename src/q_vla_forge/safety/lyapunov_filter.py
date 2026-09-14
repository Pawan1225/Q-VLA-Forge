"""Deterministic Lyapunov safety filter for Sprint 5.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyDecision,
    SafetyMethod,
    ViolationRecord,
)
from q_vla_forge.safety.driving_constraints import (
    evaluate_driving_violations,
)
from q_vla_forge.safety.lyapunov import (
    evaluate_lyapunov,
)
from q_vla_forge.safety.lyapunov_candidates import (
    HardGuardResult,
    LyapunovActionCandidate,
    evaluate_hard_guards,
    generate_driving_candidates,
    generate_robotics_candidates,
)
from q_vla_forge.safety.predictors import (
    RoboticsPredictionContext,
    predict_driving_next_state,
    predict_robotics_next_state,
)
from q_vla_forge.safety.robotics_constraints import (
    evaluate_robotics_violations,
)

INTERVENTION_TOLERANCE_L2 = 1.0e-8
LYAPUNOV_IMPROVEMENT_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class LyapunovCandidateScore:
    """Fully evaluated candidate used by the deterministic selector."""

    candidate: LyapunovActionCandidate
    hard_guard: HardGuardResult

    next_state: np.ndarray

    current_value: float
    next_value: float
    delta: float

    correction_from_bounded_proposed_l2: float


def _validated_vector(
    value: np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.size != expected_size:
        raise ValueError(f"{name} must contain exactly {expected_size} values")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array.copy()


def _evaluate_domain_violations(
    *,
    domain: str,
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[ViolationRecord, ...]:
    if domain == "autonomous_driving":
        return evaluate_driving_violations(
            state,
            action,
        )

    if domain == "robotics":
        return evaluate_robotics_violations(
            state,
            action,
        )

    raise ValueError(f"unsupported domain: {domain}")


def _predict_next_state(
    *,
    domain: str,
    state: np.ndarray,
    action: np.ndarray,
    robotics_context: RoboticsPredictionContext | None,
) -> np.ndarray:
    if domain == "autonomous_driving":
        return predict_driving_next_state(
            state,
            action,
        ).astype(np.float64)

    if domain == "robotics":
        if robotics_context is None:
            raise ValueError(
                "robotics_context is required for robotics Lyapunov filtering"
            )

        prediction = predict_robotics_next_state(
            state,
            action,
            context=robotics_context,
        )

        return prediction.next_state.astype(np.float64)

    raise ValueError(f"unsupported domain: {domain}")


def _candidate_sort_key(
    score: LyapunovCandidateScore,
) -> tuple[
    int,
    float,
    float,
    float,
    int,
]:
    return (
        0 if score.hard_guard.passed else 1,
        score.next_value,
        score.delta,
        score.correction_from_bounded_proposed_l2,
        score.candidate.index,
    )


def _fallback_sort_key(
    score: LyapunovCandidateScore,
) -> tuple[
    float,
    float,
    float,
    int,
]:
    return (
        score.next_value,
        score.delta,
        score.correction_from_bounded_proposed_l2,
        score.candidate.index,
    )


def _score_candidates(
    *,
    domain: str,
    state: np.ndarray,
    candidates: tuple[
        LyapunovActionCandidate,
        ...,
    ],
    robotics_context: RoboticsPredictionContext | None,
) -> tuple[
    LyapunovCandidateScore,
    ...,
]:
    if not candidates:
        raise ValueError("candidate set must not be empty")

    current_value = float(
        evaluate_lyapunov(
            domain=domain,
            state=state,
        ).total
    )

    bounded_proposed = candidates[0].action

    scored: list[LyapunovCandidateScore] = []

    for candidate in candidates:
        hard_guard = evaluate_hard_guards(
            domain=domain,
            state=state,
            action=candidate.action,
        )

        next_state = _predict_next_state(
            domain=domain,
            state=state,
            action=candidate.action,
            robotics_context=robotics_context,
        )

        next_value = float(
            evaluate_lyapunov(
                domain=domain,
                state=next_state,
            ).total
        )

        delta = float(next_value - current_value)

        correction = float(np.linalg.norm(candidate.action - bounded_proposed))

        scored.append(
            LyapunovCandidateScore(
                candidate=candidate,
                hard_guard=hard_guard,
                next_state=next_state.copy(),
                current_value=current_value,
                next_value=next_value,
                delta=delta,
                correction_from_bounded_proposed_l2=correction,
            )
        )

    return tuple(scored)


def _select_candidate(
    scores: tuple[
        LyapunovCandidateScore,
        ...,
    ],
) -> tuple[
    LyapunovCandidateScore,
    bool,
]:
    """Return selected candidate and whether emergency fallback was required."""

    if not scores:
        raise ValueError("candidate scores must not be empty")

    bounded_proposed = scores[0]

    passing = tuple(score for score in scores if score.hard_guard.passed)

    if not passing:
        return (
            min(
                scores,
                key=_fallback_sort_key,
            ),
            True,
        )

    if bounded_proposed.hard_guard.passed:
        strictly_better = tuple(
            score
            for score in passing
            if (
                score.next_value
                < bounded_proposed.next_value - LYAPUNOV_IMPROVEMENT_TOLERANCE
            )
        )

        if not strictly_better:
            return (
                bounded_proposed,
                False,
            )

        return (
            min(
                strictly_better,
                key=_candidate_sort_key,
            ),
            False,
        )

    return (
        min(
            passing,
            key=_candidate_sort_key,
        ),
        False,
    )


def _score_metadata(
    score: LyapunovCandidateScore,
) -> dict[str, Any]:
    return {
        "index": score.candidate.index,
        "source": score.candidate.source,
        "action": score.candidate.action.tolist(),
        "hard_guard_passed": score.hard_guard.passed,
        "violated_guard_names": list(score.hard_guard.violated_guard_names),
        "current_v": score.current_value,
        "next_v": score.next_value,
        "delta_v": score.delta,
        "correction_from_bounded_proposed_l2": (
            score.correction_from_bounded_proposed_l2
        ),
    }


def apply_lyapunov_safety_filter(
    *,
    domain: str,
    true_state: np.ndarray,
    proposed_action: np.ndarray,
    robotics_context: RoboticsPredictionContext | None = None,
) -> SafetyDecision:
    """Apply deterministic one-step Lyapunov candidate filtering."""

    if domain == "autonomous_driving":
        state = _validated_vector(
            true_state,
            expected_size=4,
            name="driving state",
        )

        proposed = _validated_vector(
            proposed_action,
            expected_size=3,
            name="driving proposed action",
        )

        candidates = generate_driving_candidates(
            state=state,
            proposed_action=proposed,
        )

    elif domain == "robotics":
        state = _validated_vector(
            true_state,
            expected_size=6,
            name="robotics state",
        )

        proposed = _validated_vector(
            proposed_action,
            expected_size=3,
            name="robotics proposed action",
        )

        if robotics_context is None:
            raise ValueError(
                "robotics_context is required for robotics Lyapunov filtering"
            )

        candidates = generate_robotics_candidates(
            state=state,
            proposed_action=proposed,
        )

    else:
        raise ValueError(f"unsupported domain: {domain}")

    scores = _score_candidates(
        domain=domain,
        state=state,
        candidates=candidates,
        robotics_context=robotics_context,
    )

    selected, emergency_fallback = _select_candidate(scores)

    executed = selected.candidate.action.copy()

    violations_before = _evaluate_domain_violations(
        domain=domain,
        state=state,
        action=proposed,
    )

    violations_after = _evaluate_domain_violations(
        domain=domain,
        state=state,
        action=executed,
    )

    correction_l2 = float(np.linalg.norm(executed - proposed))

    intervened = bool(correction_l2 > INTERVENTION_TOLERANCE_L2)

    bounded_proposed = scores[0]

    proposed_hard_guard_passed = bounded_proposed.hard_guard.passed

    action_bound_changed = bool(
        np.linalg.norm(bounded_proposed.candidate.action - proposed)
        > INTERVENTION_TOLERANCE_L2
    )

    if emergency_fallback:
        reason = InterventionReason.EMERGENCY_FALLBACK

    elif not intervened:
        reason = InterventionReason.NONE

    elif selected.candidate.index == 0 and action_bound_changed:
        reason = InterventionReason.ACTION_BOUND

    elif not proposed_hard_guard_passed:
        reason = InterventionReason.DOMAIN_CONSTRAINT

    else:
        reason = InterventionReason.LYAPUNOV_DECREASE

    before_count = sum(int(record.violated) for record in violations_before)

    after_count = sum(int(record.violated) for record in violations_after)

    eligible_count = sum(int(score.hard_guard.passed) for score in scores)

    metadata: dict[
        str,
        Any,
    ] = {
        "candidate_count": len(scores),
        "eligible_candidate_count": eligible_count,
        "selected_candidate_index": (selected.candidate.index),
        "selected_candidate_source": (selected.candidate.source),
        "current_v": (selected.current_value),
        "selected_next_v": (selected.next_value),
        "selected_delta_v": (selected.delta),
        "selected_hard_guard_passed": (selected.hard_guard.passed),
        "selected_violated_guard_names": list(selected.hard_guard.violated_guard_names),
        "proposed_hard_guard_passed": (proposed_hard_guard_passed),
        "emergency_fallback": (emergency_fallback),
        "lyapunov_improvement_tolerance": (LYAPUNOV_IMPROVEMENT_TOLERANCE),
        "intervention_tolerance_l2": (INTERVENTION_TOLERANCE_L2),
        "pre_filter_violation_count": (before_count),
        "post_filter_violation_count": (after_count),
        "immediate_violation_reduction": (before_count - after_count),
        "candidate_scores": [_score_metadata(score) for score in scores],
    }

    return SafetyDecision(
        method=SafetyMethod.LYAPUNOV,
        proposed_action=proposed.copy(),
        executed_action=executed.copy(),
        intervened=intervened,
        intervention_reason=reason,
        correction_l2=correction_l2,
        violations_before=violations_before,
        violations_after=violations_after,
        metadata=metadata,
    )
