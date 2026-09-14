from __future__ import annotations

import numpy as np
import pytest

import q_vla_forge.safety.lyapunov_filter as lyapunov_filter_module
from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyMethod,
)
from q_vla_forge.safety.lyapunov_candidates import (
    HardGuardResult,
)
from q_vla_forge.safety.lyapunov_filter import (
    INTERVENTION_TOLERANCE_L2,
    LYAPUNOV_IMPROVEMENT_TOLERANCE,
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.predictors import (
    RoboticsPredictionContext,
)


def test_filter_constants_are_frozen() -> None:
    assert INTERVENTION_TOLERANCE_L2 == 1.0e-8

    assert LYAPUNOV_IMPROVEMENT_TOLERANCE == 1.0e-12


def test_driving_safe_zero_risk_keeps_proposed_action() -> None:
    state = np.array(
        [
            0.30,
            0.00,
            0.00,
            2.00,
        ]
    )

    proposed = np.array(
        [
            0.10,
            0.20,
            0.00,
        ]
    )

    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=state,
        proposed_action=proposed,
    )

    assert decision.method == SafetyMethod.LYAPUNOV

    np.testing.assert_array_equal(
        decision.executed_action,
        proposed,
    )

    assert decision.intervened is False

    assert decision.intervention_reason == InterventionReason.NONE

    assert decision.metadata["selected_candidate_source"] == "bounded_proposed"


def test_driving_action_bounds_use_action_bound_reason() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.30,
                0.00,
                0.00,
                2.00,
            ]
        ),
        proposed_action=np.array(
            [
                2.00,
                0.00,
                0.00,
            ]
        ),
    )

    np.testing.assert_array_equal(
        decision.executed_action,
        np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.ACTION_BOUND


def test_driving_hard_guard_failure_uses_domain_constraint_reason() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.40,
                0.70,
                0.60,
                1.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.90,
                0.00,
                0.00,
            ]
        ),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.DOMAIN_CONSTRAINT

    assert decision.metadata["selected_hard_guard_passed"] is True


def test_driving_can_select_strictly_lower_lyapunov_candidate() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.40,
                0.80,
                0.60,
                1.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                0.00,
            ]
        ),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.LYAPUNOV_DECREASE

    assert (
        decision.metadata["selected_next_v"]
        < decision.metadata["candidate_scores"][0]["next_v"]
    )


def test_equal_lyapunov_value_does_not_create_unnecessary_intervention() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.20,
                0.00,
                0.00,
                3.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.20,
                0.20,
                0.00,
            ]
        ),
    )

    assert decision.metadata["selected_candidate_index"] == 0

    assert decision.intervened is False


def test_driving_metadata_contains_candidate_scores() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.50,
                0.80,
                0.50,
                0.40,
            ]
        ),
        proposed_action=np.array(
            [
                0.50,
                0.20,
                0.00,
            ]
        ),
    )

    scores = decision.metadata["candidate_scores"]

    assert isinstance(
        scores,
        list,
    )

    assert len(scores) == decision.metadata["candidate_count"]

    assert all(
        "next_v" in score and "delta_v" in score and "hard_guard_passed" in score
        for score in scores
    )


def test_robotics_requires_prediction_context() -> None:
    with pytest.raises(
        ValueError,
        match="robotics_context",
    ):
        apply_lyapunov_safety_filter(
            domain="robotics",
            true_state=np.zeros(6),
            proposed_action=np.zeros(3),
        )


def test_robotics_safe_zero_risk_keeps_proposed_action() -> None:
    proposed = np.array(
        [
            0.20,
            0.00,
            0.00,
        ]
    )

    decision = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.00,
                0.00,
                0.20,
                0.20,
                0.50,
                0.50,
            ]
        ),
        proposed_action=proposed,
        robotics_context=(RoboticsPredictionContext(object_grasped=False)),
    )

    np.testing.assert_array_equal(
        decision.executed_action,
        proposed,
    )

    assert decision.intervened is False

    assert decision.intervention_reason == InterventionReason.NONE


def test_robotics_outward_motion_is_rejected_by_hard_guard() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.89,
                0.00,
                0.00,
                0.00,
                0.50,
                0.50,
            ]
        ),
        proposed_action=np.array(
            [
                1.00,
                0.00,
                0.00,
            ]
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=False)),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.DOMAIN_CONSTRAINT

    assert decision.metadata["selected_hard_guard_passed"] is True


def test_robotics_inward_motion_can_reduce_lyapunov_value() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.95,
                0.95,
                0.95,
                0.95,
                0.00,
                0.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                0.00,
            ]
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=True)),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.LYAPUNOV_DECREASE

    assert (
        decision.metadata["selected_next_v"]
        < decision.metadata["candidate_scores"][0]["next_v"]
    )


def test_robotics_unsafe_gripper_close_can_be_replaced() -> None:
    decision = apply_lyapunov_safety_filter(
        domain="robotics",
        true_state=np.array(
            [
                0.00,
                0.00,
                0.80,
                0.80,
                0.00,
                0.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                1.00,
            ]
        ),
        robotics_context=(RoboticsPredictionContext(object_grasped=False)),
    )

    assert decision.intervened is True

    assert decision.intervention_reason == InterventionReason.DOMAIN_CONSTRAINT

    assert decision.executed_action[2] <= 0.50


def test_filter_is_deterministic() -> None:
    kwargs = {
        "domain": "autonomous_driving",
        "true_state": np.array(
            [
                0.50,
                0.80,
                0.60,
                0.50,
            ]
        ),
        "proposed_action": np.array(
            [
                0.40,
                0.30,
                0.00,
            ]
        ),
    }

    first = apply_lyapunov_safety_filter(**kwargs)

    second = apply_lyapunov_safety_filter(**kwargs)

    np.testing.assert_array_equal(
        first.executed_action,
        second.executed_action,
    )

    assert first.intervention_reason == second.intervention_reason

    assert (
        first.metadata["selected_candidate_index"]
        == second.metadata["selected_candidate_index"]
    )


def test_filter_does_not_mutate_inputs() -> None:
    state = np.array(
        [
            0.50,
            0.80,
            0.60,
            0.50,
        ]
    )

    proposed = np.array(
        [
            0.40,
            0.30,
            0.00,
        ]
    )

    state_before = state.copy()

    proposed_before = proposed.copy()

    apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=state,
        proposed_action=proposed,
    )

    np.testing.assert_array_equal(
        state,
        state_before,
    )

    np.testing.assert_array_equal(
        proposed,
        proposed_before,
    )


def test_emergency_fallback_path_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_all_guards(
        *,
        domain: str,
        state: np.ndarray,
        action: np.ndarray,
    ) -> HardGuardResult:
        del domain
        del state
        del action

        return HardGuardResult(
            passed=False,
            violated_guard_names=("synthetic_guard",),
            violations=(),
        )

    monkeypatch.setattr(
        lyapunov_filter_module,
        "evaluate_hard_guards",
        reject_all_guards,
    )

    decision = apply_lyapunov_safety_filter(
        domain="autonomous_driving",
        true_state=np.array(
            [
                0.40,
                0.80,
                0.60,
                1.00,
            ]
        ),
        proposed_action=np.array(
            [
                0.00,
                0.00,
                0.00,
            ]
        ),
    )

    assert decision.metadata["emergency_fallback"] is True

    assert decision.intervention_reason == InterventionReason.EMERGENCY_FALLBACK


def test_unknown_domain_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported domain",
    ):
        apply_lyapunov_safety_filter(
            domain="unknown",
            true_state=np.zeros(4),
            proposed_action=np.zeros(3),
        )
