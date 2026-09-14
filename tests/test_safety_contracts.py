from __future__ import annotations

import numpy as np

from q_vla_forge.safety.contracts import (
    InterventionReason,
    RobustnessCondition,
    SafetyDecision,
    SafetyMethod,
    ViolationRecord,
    ViolationSeverity,
)


def test_safety_methods_are_frozen() -> None:
    assert tuple(method.value for method in SafetyMethod) == (
        "none",
        "clipping",
        "lyapunov",
    )


def test_robustness_conditions_are_frozen() -> None:
    assert tuple(condition.value for condition in RobustnessCondition) == (
        "clean",
        "gaussian_state_perturbation",
        "state_perturbation",
        "action_perturbation",
    )


def test_safety_decision_preserves_both_actions() -> None:
    proposed = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    executed = np.array([0.8, 0.5, 0.1], dtype=np.float32)

    decision = SafetyDecision(
        method=SafetyMethod.CLIPPING,
        proposed_action=proposed,
        executed_action=executed,
        intervened=True,
        intervention_reason=InterventionReason.DOMAIN_CONSTRAINT,
        correction_l2=float(np.linalg.norm(executed - proposed)),
    )

    np.testing.assert_array_equal(decision.proposed_action, proposed)
    np.testing.assert_array_equal(decision.executed_action, executed)
    assert decision.intervened
    assert decision.correction_l2 > 0.0


def test_non_intervention_preserves_action() -> None:
    action = np.array([0.2, 0.0, 0.1], dtype=np.float32)

    decision = SafetyDecision(
        method=SafetyMethod.NONE,
        proposed_action=action,
        executed_action=action.copy(),
        intervened=False,
        intervention_reason=InterventionReason.NONE,
        correction_l2=0.0,
    )

    assert decision.intervened is False
    assert decision.correction_l2 == 0.0


def test_violation_record() -> None:
    record = ViolationRecord(
        name="unsafe_distance",
        severity=ViolationSeverity.CRITICAL,
        value=0.1,
        threshold=0.2,
        violated=True,
    )

    assert record.violated
    assert record.severity is ViolationSeverity.CRITICAL
