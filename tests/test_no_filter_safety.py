from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyMethod,
    ViolationRecord,
    ViolationSeverity,
)
from q_vla_forge.safety.no_filter import (
    no_filter_decision,
)


def _violations() -> tuple[
    ViolationRecord,
    ...,
]:
    return (
        ViolationRecord(
            name="example",
            severity=(ViolationSeverity.WARNING),
            value=1.0,
            threshold=0.5,
            violated=True,
        ),
    )


def test_method_is_none() -> None:
    decision = no_filter_decision(
        proposed_action=np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
        violations=_violations(),
    )

    assert decision.method == SafetyMethod.NONE


def test_no_intervention() -> None:
    decision = no_filter_decision(
        proposed_action=np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
        violations=_violations(),
    )

    assert not decision.intervened

    assert decision.intervention_reason == InterventionReason.NONE


def test_proposed_equals_executed() -> None:
    action = np.array(
        [-1.0, 0.25, 1.0],
        dtype=np.float32,
    )

    decision = no_filter_decision(
        proposed_action=action,
        violations=_violations(),
    )

    np.testing.assert_array_equal(
        decision.proposed_action,
        decision.executed_action,
    )


def test_action_preserved_numerically() -> None:
    action = np.array(
        [-3.0, 2.5, 9.0],
        dtype=np.float64,
    )

    decision = no_filter_decision(
        proposed_action=action,
        violations=(),
    )

    np.testing.assert_array_equal(
        decision.executed_action,
        action,
    )


def test_correction_is_exactly_zero() -> None:
    decision = no_filter_decision(
        proposed_action=np.array(
            [0.1, 0.2],
            dtype=np.float32,
        ),
        violations=(),
    )

    assert decision.correction_l2 == 0.0


def test_violations_before_equal_after() -> None:
    violations = _violations()

    decision = no_filter_decision(
        proposed_action=np.array(
            [0.1],
            dtype=np.float32,
        ),
        violations=violations,
    )

    assert decision.violations_before == violations

    assert decision.violations_after == violations


def test_input_action_not_mutated() -> None:
    action = np.array(
        [0.5, -0.5],
        dtype=np.float32,
    )

    before = action.copy()

    decision = no_filter_decision(
        proposed_action=action,
        violations=(),
    )

    decision.executed_action[0] = 99.0

    np.testing.assert_array_equal(
        action,
        before,
    )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_action_rejected(
    bad_value: float,
) -> None:
    action = np.array(
        [
            0.0,
            bad_value,
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        no_filter_decision(
            proposed_action=action,
            violations=(),
        )


def test_empty_action_rejected() -> None:
    with pytest.raises(ValueError):
        no_filter_decision(
            proposed_action=np.array(
                [],
                dtype=np.float64,
            ),
            violations=(),
        )


def test_multidimensional_action_rejected() -> None:
    with pytest.raises(ValueError):
        no_filter_decision(
            proposed_action=np.zeros(
                (2, 2),
                dtype=np.float64,
            ),
            violations=(),
        )


def test_metadata_marks_no_explicit_filter() -> None:
    decision = no_filter_decision(
        proposed_action=np.array(
            [0.0],
            dtype=np.float32,
        ),
        violations=(),
    )

    assert decision.metadata["explicit_safety_filter"] is False
