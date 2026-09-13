from __future__ import annotations

import pytest

from q_vla_forge.rl.contracts import (
    EvaluationPoint,
    QuantumResourceRecord,
    RLDomain,
    RLMethod,
    RLRunRecord,
    RLTargetDefinition,
    RLTargetReach,
    run_record_to_dict,
)


def target() -> RLTargetDefinition:
    return RLTargetDefinition(
        domain=RLDomain.AUTONOMOUS_DRIVING,
        seed=42,
        reference_method=RLMethod.CLASSICAL_PPO,
        reference_best_evaluation_reward=100.0,
        target_evaluation_reward=95.0,
        target_fraction=0.95,
        derivation=("95% of best completed classical PPO evaluation"),
    )


def test_evaluation_point() -> None:
    point = EvaluationPoint(
        environment_steps=1_000,
        mean_reward=10.0,
        reward_standard_deviation=2.0,
        success_rate=0.75,
        evaluation_episodes=20,
    )

    assert point.environment_steps == 1_000


def test_invalid_success_rate_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationPoint(
            environment_steps=1_000,
            mean_reward=10.0,
            reward_standard_deviation=2.0,
            success_rate=1.5,
            evaluation_episodes=20,
        )


def test_failed_target_keeps_none_metrics() -> None:
    reach = RLTargetReach(
        reached_target=False,
        environment_steps_to_target=None,
        episodes_to_target=None,
        seconds_to_target=None,
    )

    assert not reach.reached_target


def test_failed_target_cannot_be_censored() -> None:
    with pytest.raises(ValueError):
        RLTargetReach(
            reached_target=False,
            environment_steps_to_target=20_000,
            episodes_to_target=None,
            seconds_to_target=None,
        )


def test_reached_target_requires_metrics() -> None:
    with pytest.raises(ValueError):
        RLTargetReach(
            reached_target=True,
            environment_steps_to_target=5_000,
            episodes_to_target=None,
            seconds_to_target=None,
        )


def test_classical_run_rejects_quantum_resources() -> None:
    resources = QuantumResourceRecord(
        qubits=4,
        circuit_layers=2,
        trainable_circuit_parameters=16,
        circuit_evaluations=100,
        execution_backend="simulator",
        hardware_used=False,
    )

    with pytest.raises(ValueError):
        RLRunRecord(
            experiment_id="test",
            domain=RLDomain.AUTONOMOUS_DRIVING,
            method=RLMethod.CLASSICAL_PPO,
            seed=42,
            training_environment_steps=20_000,
            completed_episodes=200,
            evaluation_history=(),
            target=target(),
            target_reach=RLTargetReach(
                reached_target=False,
                environment_steps_to_target=None,
                episodes_to_target=None,
                seconds_to_target=None,
            ),
            final_evaluation_reward=10.0,
            final_success_rate=0.5,
            policy_parameters=1_000,
            training_seconds=1.0,
            quantum_resources=resources,
            notes=(),
        )


def test_qml_run_requires_quantum_resources() -> None:
    with pytest.raises(ValueError):
        RLRunRecord(
            experiment_id="test",
            domain=RLDomain.AUTONOMOUS_DRIVING,
            method=RLMethod.HYBRID_QML_PPO,
            seed=42,
            training_environment_steps=20_000,
            completed_episodes=200,
            evaluation_history=(),
            target=target(),
            target_reach=RLTargetReach(
                reached_target=False,
                environment_steps_to_target=None,
                episodes_to_target=None,
                seconds_to_target=None,
            ),
            final_evaluation_reward=10.0,
            final_success_rate=0.5,
            policy_parameters=1_000,
            training_seconds=1.0,
            quantum_resources=None,
            notes=(),
        )


def test_run_record_serialization() -> None:
    record = RLRunRecord(
        experiment_id="classical-test",
        domain=RLDomain.AUTONOMOUS_DRIVING,
        method=RLMethod.CLASSICAL_PPO,
        seed=42,
        training_environment_steps=20_000,
        completed_episodes=200,
        evaluation_history=(
            EvaluationPoint(
                environment_steps=1_000,
                mean_reward=10.0,
                reward_standard_deviation=1.0,
                success_rate=0.5,
                evaluation_episodes=20,
            ),
        ),
        target=target(),
        target_reach=RLTargetReach(
            reached_target=False,
            environment_steps_to_target=None,
            episodes_to_target=None,
            seconds_to_target=None,
        ),
        final_evaluation_reward=10.0,
        final_success_rate=0.5,
        policy_parameters=1_000,
        training_seconds=1.0,
        quantum_resources=None,
        notes=("pilot",),
    )

    payload = run_record_to_dict(record)

    assert payload["domain"] == "autonomous_driving"
    assert payload["method"] == "classical_ppo"
    assert payload["quantum_resources"] is None
