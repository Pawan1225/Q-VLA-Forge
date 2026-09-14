"""No-filter safety method for Sprint 5.4."""

from __future__ import annotations

import numpy as np

from q_vla_forge.safety.contracts import (
    InterventionReason,
    SafetyDecision,
    SafetyMethod,
    ViolationRecord,
)


def _validated_action(
    action: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        action,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError("proposed action must be one-dimensional")

    if array.size == 0:
        raise ValueError("proposed action must not be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError("proposed action must contain only finite values")

    return array


def no_filter_decision(
    *,
    proposed_action: np.ndarray,
    violations: tuple[
        ViolationRecord,
        ...,
    ],
) -> SafetyDecision:
    """Return the unmodified Sprint 5 NONE safety decision."""

    action = _validated_action(proposed_action).copy()

    return SafetyDecision(
        method=SafetyMethod.NONE,
        proposed_action=action.copy(),
        executed_action=action.copy(),
        intervened=False,
        intervention_reason=(InterventionReason.NONE),
        correction_l2=0.0,
        violations_before=violations,
        violations_after=violations,
        metadata={
            "explicit_safety_filter": False,
            "sprint": "5.4",
        },
    )
