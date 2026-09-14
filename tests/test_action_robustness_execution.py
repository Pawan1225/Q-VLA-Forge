"""Execution-order tests for Sprint 5.12 action robustness."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.evaluation.action_robustness import (
    action_safety_recovery_rate,
    environment_effective_action,
    gripper_semantic,
)
from q_vla_forge.safety.perturbations import (
    action_perturbation_by_name,
    apply_structured_action_perturbation,
)


def test_action_perturbation_occurs_without_pre_clipping() -> None:
    proposed = np.asarray(
        [
            0.90,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    perturbation = action_perturbation_by_name(
        domain="autonomous_driving",
        name="steering_plus_0p25",
    )

    perturbed, delta = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=perturbation,
    )

    np.testing.assert_allclose(
        perturbed,
        proposed + delta,
        rtol=0.0,
        atol=1.0e-7,
    )

    assert float(perturbed[0]) > 1.0


def test_environment_clipping_is_after_explicit_execution() -> None:
    executed = np.asarray(
        [
            1.20,
            0.0,
            -0.20,
        ],
        dtype=np.float32,
    )

    effective = environment_effective_action(
        executed_action=executed,
        lower_bounds=np.asarray(
            [
                -1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        upper_bounds=np.asarray(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    assert float(executed[0]) == pytest.approx(1.20)

    assert float(effective[0]) == pytest.approx(1.0)

    assert float(effective[2]) == pytest.approx(0.0)


def test_environment_clipping_is_not_filter_recovery() -> None:
    recovery = action_safety_recovery_rate(
        unsafe_perturbed_steps=8,
        recovered_unsafe_steps=0,
    )

    assert recovery == 0.0


def test_gripper_plus_0p50_can_change_semantics() -> None:
    proposed = np.asarray(
        [
            0.0,
            0.0,
            0.10,
        ],
        dtype=np.float32,
    )

    perturbation = action_perturbation_by_name(
        domain="robotics",
        name="gripper_plus_0p50",
    )

    perturbed, _ = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=perturbation,
    )

    assert gripper_semantic(float(proposed[2])) == "hold"

    assert gripper_semantic(float(perturbed[2])) == "close"


def test_exact_gripper_thresholds_remain_hold() -> None:
    assert gripper_semantic(0.50) == "hold"

    assert gripper_semantic(-0.50) == "hold"
