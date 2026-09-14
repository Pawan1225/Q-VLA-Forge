from __future__ import annotations

from q_vla_forge.safety.contracts import SafetyMethod
from q_vla_forge.safety.protocol import DEFAULT_SAFETY_PROTOCOL


def test_primary_policy_is_frozen_ppo() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.primary_policy == "classical_ppo"


def test_principal_seeds() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.seeds == (
        42,
        123,
        456,
    )


def test_evaluation_seeds() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.evaluation_seeds == tuple(range(20_000, 20_020))


def test_three_safety_methods() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.methods == (
        SafetyMethod.NONE,
        SafetyMethod.CLIPPING,
        SafetyMethod.LYAPUNOV,
    )


def test_primary_metric() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.primary_metric == "violation_step_rate"


def test_acceptance_thresholds() -> None:
    protocol = DEFAULT_SAFETY_PROTOCOL

    assert protocol.required_violation_reduction_fraction == 0.20
    assert protocol.maximum_reward_degradation_fraction == 0.10
    assert protocol.maximum_success_rate_drop == 0.10


def test_gaussian_noise_levels() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.gaussian_noise_stds == (
        0.0,
        0.01,
        0.05,
        0.10,
    )


def test_intervention_tolerance() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.intervention_tolerance == 1e-8


def test_filter_uses_true_state() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.safety_filter_uses_true_state is True


def test_no_significance_testing() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.significance_testing_enabled is False


def test_no_formal_certification_claim() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.formal_certification_claimed is False


def test_no_production_safety_guarantee() -> None:
    assert DEFAULT_SAFETY_PROTOCOL.production_safety_guarantee_claimed is False
