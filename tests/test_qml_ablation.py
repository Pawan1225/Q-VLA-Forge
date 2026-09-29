import math
from pathlib import Path

import pytest

from q_vla_forge.evaluation.qml_ablation import (
    ABLATION_SOURCE,
    CLAIM_CONTROLS,
    DOMAINS,
    EXECUTION_BACKEND,
    POLICIES,
    QUANTUM_HARDWARE_USED,
    REQUIRED_SEEDS,
    QMLAblationRecord,
    claim_controls,
    compactness_record,
    load_canonical_qml_ablation,
    parameter_reduction_percent,
    validate_required_seeds,
    validate_target_attainment,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ABLATION_SOURCE


def make_record(
    **overrides,
) -> QMLAblationRecord:
    payload = {
        "domain": "autonomous_driving",
        "policy": "ppo_mlp",
        "seed": 42,
        "target_reached": True,
        "environment_steps_to_target": 1000,
        "episodes_to_target": 10,
        "final_evaluation_reward": 1.0,
        "final_success_rate": 1.0,
        "training_seconds": 10.0,
        "actor_parameters": 1318,
        "source_artifacts": (SOURCE,),
    }

    payload.update(overrides)

    return QMLAblationRecord(**payload)


def test_required_seeds_are_locked() -> None:
    assert REQUIRED_SEEDS == (
        42,
        123,
        456,
    )


def test_required_seed_set_validation() -> None:
    validate_required_seeds(
        [
            456,
            42,
            123,
        ]
    )


def test_missing_required_seed_rejected() -> None:
    with pytest.raises(ValueError):
        validate_required_seeds(
            [
                42,
                123,
            ]
        )


def test_domains_are_locked() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_policies_are_locked() -> None:
    assert POLICIES == (
        "ppo_mlp",
        "ppo_pqc",
    )


def test_target_reached_boolean_required() -> None:
    with pytest.raises(TypeError):
        validate_target_attainment(
            target_reached=1,
            environment_steps_to_target=1000,
        )


def test_non_reaching_policy_has_no_steps_to_target() -> None:
    record = make_record(
        policy="ppo_pqc",
        target_reached=False,
        environment_steps_to_target=None,
        episodes_to_target=None,
        actor_parameters=54,
    )

    assert not record.target_reached
    assert record.environment_steps_to_target is None


def test_non_reaching_policy_cannot_use_budget_as_steps() -> None:
    with pytest.raises(ValueError):
        make_record(
            policy="ppo_pqc",
            target_reached=False,
            environment_steps_to_target=20_000,
            episodes_to_target=None,
            actor_parameters=54,
        )


def test_reaching_policy_requires_steps() -> None:
    with pytest.raises(ValueError):
        make_record(
            target_reached=True,
            environment_steps_to_target=None,
        )


def test_reaching_policy_requires_positive_steps() -> None:
    with pytest.raises(ValueError):
        make_record(
            target_reached=True,
            environment_steps_to_target=0,
        )


def test_missing_reward_is_not_zero() -> None:
    record = make_record(
        final_evaluation_reward=None,
    )

    assert record.final_evaluation_reward is None


def test_missing_success_is_not_zero() -> None:
    record = make_record(
        final_success_rate=None,
    )

    assert record.final_success_rate is None


def test_success_rate_must_be_valid_rate() -> None:
    with pytest.raises(ValueError):
        make_record(
            final_success_rate=1.1,
        )


def test_parameter_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_record(
            actor_parameters=0,
        )


def test_parameter_reduction_driving_target() -> None:
    reduction = parameter_reduction_percent(
        classical_parameters=1318,
        qml_parameters=54,
    )

    assert reduction == pytest.approx(95.9028831563)


def test_parameter_reduction_robotics_target() -> None:
    reduction = parameter_reduction_percent(
        classical_parameters=1382,
        qml_parameters=62,
    )

    assert reduction == pytest.approx(95.5137481910)


def test_parameter_reduction_is_deterministic() -> None:
    record = compactness_record(
        domain="autonomous_driving",
        classical_parameters=1318,
        qml_parameters=54,
        source_artifacts=(SOURCE,),
    )

    assert record["statistical"] is False
    assert "sample_std" not in record


def test_source_provenance_required() -> None:
    with pytest.raises(ValueError):
        make_record(
            source_artifacts=(),
        )


def test_execution_backend_is_simulator() -> None:
    record = make_record()

    assert record.execution_backend == "simulator"
    assert EXECUTION_BACKEND == "simulator"


def test_non_simulator_backend_rejected() -> None:
    with pytest.raises(ValueError):
        make_record(
            execution_backend="qpu",
        )


def test_quantum_hardware_used_is_false() -> None:
    record = make_record()

    assert record.quantum_hardware_used is False
    assert QUANTUM_HARDWARE_USED is False


def test_quantum_hardware_true_rejected() -> None:
    with pytest.raises(ValueError):
        make_record(
            quantum_hardware_used=True,
        )


def test_quantum_advantage_claim_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


def test_quantum_speedup_claim_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_speedup_claim_allowed"] is False


def test_qml_sample_efficiency_claim_blocked() -> None:
    assert CLAIM_CONTROLS["qml_sample_efficiency_advantage_claim_allowed"] is False


def test_qml_performance_superiority_claim_blocked() -> None:
    assert CLAIM_CONTROLS["qml_performance_superiority_claim_allowed"] is False


def test_claim_controls_copy_is_safe() -> None:
    controls = claim_controls()

    controls["quantum_advantage_claim_allowed"] = True

    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


def test_non_finite_reward_rejected() -> None:
    with pytest.raises(ValueError):
        make_record(
            final_evaluation_reward=math.nan,
        )


def test_negative_training_seconds_rejected() -> None:
    with pytest.raises(ValueError):
        make_record(
            training_seconds=-1.0,
        )


def test_record_to_dict_preserves_null_non_attainment() -> None:
    record = make_record(
        policy="ppo_pqc",
        target_reached=False,
        environment_steps_to_target=None,
        episodes_to_target=None,
        final_evaluation_reward=None,
        final_success_rate=None,
        actor_parameters=54,
    )

    payload = record.to_dict()

    assert payload["environment_steps_to_target"] is None

    assert payload["target_reached"] is False


def test_canonical_qml_ablation_has_twelve_primary_records() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    assert len(evidence.primary_records) == 12


def test_canonical_primary_matrix_complete() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    observed = {
        (
            record.domain,
            record.policy,
            record.seed,
        )
        for record in evidence.primary_records
    }

    expected = {
        (
            domain,
            policy,
            seed,
        )
        for domain in DOMAINS
        for policy in POLICIES
        for seed in REQUIRED_SEEDS
    }

    assert observed == expected


def test_canonical_ppo_reaches_target_six_of_six() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    ppo = [record for record in evidence.primary_records if record.policy == "ppo_mlp"]

    assert len(ppo) == 6

    assert all(record.target_reached for record in ppo)

    assert all(record.environment_steps_to_target is not None for record in ppo)


def test_canonical_qml_reaches_zero_of_six() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    qml = [record for record in evidence.primary_records if record.policy == "ppo_pqc"]

    assert len(qml) == 6

    assert all(not record.target_reached for record in qml)

    assert all(record.environment_steps_to_target is None for record in qml)

    assert all(record.episodes_to_target is None for record in qml)


def test_canonical_matched_control_is_one_of_six() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    controls = evidence.matched_classical_controls

    assert len(controls) == 6

    assert sum(record.target_reached for record in controls) == 1


def test_matched_control_is_not_primary_ppo() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    primary_policies = {record.policy for record in evidence.primary_records}

    assert primary_policies == {
        "ppo_mlp",
        "ppo_pqc",
    }

    assert "matched_classical" not in primary_policies


def test_driving_actor_compactness_matches_frozen_evidence() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    driving = next(
        record
        for record in evidence.compactness
        if record["domain"] == "autonomous_driving"
    )

    assert driving["classical_actor_parameters"] == 1318

    assert driving["qml_actor_parameters"] == 54

    assert driving["parameter_reduction_percent"] == pytest.approx(95.90288315629742)


def test_robotics_actor_compactness_matches_frozen_evidence() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    robotics = next(
        record for record in evidence.compactness if record["domain"] == "robotics"
    )

    assert robotics["classical_actor_parameters"] == 1382

    assert robotics["qml_actor_parameters"] == 62

    assert robotics["parameter_reduction_percent"] == pytest.approx(95.5137481910275)


def test_canonical_qml_provenance_is_frozen_rl_evidence() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    for record in evidence.primary_records:
        assert record.source_artifacts

        assert all(
            source.startswith("results/rl/") for source in record.source_artifacts
        )

        assert not any(
            "final-validation" in source for source in record.source_artifacts
        )


def test_canonical_control_provenance_is_frozen_rl_evidence() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    for record in evidence.matched_classical_controls:
        assert record.source_artifacts
        assert ABLATION_SOURCE in record.source_artifacts


def test_qml_actor_counts_equal_matched_actor_counts() -> None:
    evidence = load_canonical_qml_ablation(ROOT)

    qml = {
        (
            record.domain,
            record.seed,
        ): record.actor_parameters
        for record in evidence.primary_records
        if record.policy == "ppo_pqc"
    }

    matched = {
        (
            record.domain,
            record.seed,
        ): record.actor_parameters
        for record in evidence.matched_classical_controls
    }

    assert qml == matched
