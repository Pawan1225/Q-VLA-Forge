from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.rl.contracts import RLDomain
from q_vla_forge.rl.protocol import (
    DEFAULT_RL_PROTOCOL,
    RLProtocol,
    action_semantics,
    protocol_to_dict,
)

PROTOCOL_ARTIFACT = Path("results") / "rl" / "sprint4-rl-protocol.json"


def test_default_protocol_is_frozen() -> None:
    protocol = DEFAULT_RL_PROTOCOL

    assert protocol.seeds == (
        42,
        123,
        456,
    )
    assert protocol.episode_horizon == 100
    assert protocol.maximum_environment_steps == 20_000
    assert protocol.evaluation_frequency_steps == 1_000
    assert protocol.evaluation_episodes == 20
    assert protocol.action_dimension == 3
    assert protocol.target_fraction == 0.95
    assert protocol.robust_efficiency_threshold_percent == 10.0
    assert protocol.qml_qubits == 4
    assert protocol.qml_circuit_layers == 2
    assert protocol.qml_execution_backend == "simulator"


def test_seed_protocol_cannot_change() -> None:
    with pytest.raises(ValueError):
        RLProtocol(
            seeds=(
                1,
                2,
                3,
            )
        )


def test_qml_qubits_frozen_to_four() -> None:
    with pytest.raises(ValueError):
        RLProtocol(qml_qubits=8)


def test_evaluation_frequency_must_divide_budget() -> None:
    with pytest.raises(ValueError):
        RLProtocol(evaluation_frequency_steps=3_000)


def test_driving_action_semantics() -> None:
    assert action_semantics(RLDomain.AUTONOMOUS_DRIVING) == (
        "steering",
        "acceleration",
        "braking",
    )


def test_robotics_action_semantics() -> None:
    assert action_semantics(RLDomain.ROBOTICS) == (
        "delta_x",
        "delta_y",
        "gripper",
    )


def test_protocol_serialization() -> None:
    payload = protocol_to_dict()

    assert payload["domains"] == [
        "autonomous_driving",
        "robotics",
    ]

    assert payload["primary_efficiency_metric"] == "environment_steps_to_target"

    assert payload["target_rule"]["reference"] == "classical_ppo"

    assert payload["target_rule"]["baseline_reference"] == "random_policy"

    assert payload["target_rule"]["formula"] == (
        "random_reference + target_fraction * "
        "(classical_ppo_best - random_reference)"
    )

    assert payload["target_rule"]["supports_negative_reward_scales"] is True

    assert payload["target_rule"]["random_reference_required"] is True

    assert payload["target_rule"]["numerical_targets_frozen_before_qml"] is True

    assert payload["target_rule"]["numerical_targets_available_in_4_1"] is False

    driving_success = payload["success_definitions"]["autonomous_driving"]

    assert driving_success["collision_free_required"] is True

    assert driving_success["terminal_lane_departure_forbidden"] is True

    assert driving_success["exact_lane_departure_tolerance_frozen_in"] == "sprint_4_2"

    assert driving_success["tolerance_frozen_before_classical_ppo"] is True

    assert driving_success["same_success_rule_for_all_methods"] is True

    robotics_success = payload["success_definitions"]["robotics"]

    assert robotics_success["task_completion_required"] is True

    assert robotics_success["exact_completion_tolerance_frozen_in"] == "sprint_4_3"

    assert robotics_success["tolerance_frozen_before_classical_ppo"] is True

    assert robotics_success["same_success_rule_for_all_methods"] is True

    comparison = payload["comparison_hierarchy"]

    assert comparison["primary"] == [
        "environment_steps_to_target",
    ]

    assert comparison["secondary"] == [
        "episodes_to_target",
        "final_evaluation_reward",
        "final_success_rate",
    ]

    assert comparison["resource_metrics"] == [
        "policy_parameters",
        "qubits",
        "trainable_circuit_parameters",
        "circuit_evaluations",
        "circuit_layers",
    ]

    assert comparison["descriptive_only"] == [
        "training_seconds",
    ]

    assert comparison["primary_metric_must_drive_efficiency_claims"] is True

    assert comparison["wall_clock_must_not_define_qml_advantage"] is True

    assert comparison["simulator_runtime_must_not_imply_hardware_performance"] is True

    assert comparison["failed_target_uses_none_not_budget_cap"] is True

    robust = payload["robust_qml_efficiency_rule"]

    assert robust["paired_seed_comparison_required"] is True

    assert robust["paired_seeds"] == [
        42,
        123,
        456,
    ]

    assert robust["all_three_qml_seeds_must_reach_target"] is True

    assert robust["all_three_classical_seeds_must_have_valid_targets"] is True

    assert robust["aggregation"] == "mean_and_sample_standard_deviation"

    assert robust["minimum_required_mean_reduction_percent"] == 10.0

    assert robust["failed_qml_target_invalidates_robust_improvement"] is True

    assert robust["failed_target_must_remain_none"] is True

    assert robust["budget_cap_must_not_replace_failed_target"] is True

    assert payload["qml"]["quantum_hardware_used"] is False

    assert payload["claim_controls"]["quantum_advantage_claimed"] is False


def test_generated_protocol_artifact() -> None:
    if not PROTOCOL_ARTIFACT.exists():
        pytest.skip("Sprint 4 protocol artifact not generated")

    payload = json.loads(PROTOCOL_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]

    assert payload["episode_horizon"] == 100
    assert payload["maximum_environment_steps"] == 20_000
    assert payload["evaluation_frequency_steps"] == 1_000
    assert payload["evaluation_episodes"] == 20
    assert payload["action_dimension"] == 3
    assert payload["target_fraction"] == 0.95

    assert payload["robust_efficiency_threshold_percent"] == 10.0

    assert payload["primary_efficiency_metric"] == "environment_steps_to_target"

    assert payload["target_rule"]["reference"] == "classical_ppo"

    assert payload["target_rule"]["baseline_reference"] == "random_policy"

    assert payload["target_rule"]["formula"] == (
        "random_reference + target_fraction * "
        "(classical_ppo_best - random_reference)"
    )

    assert payload["target_rule"]["supports_negative_reward_scales"] is True

    assert payload["target_rule"]["random_reference_required"] is True

    assert payload["target_rule"]["numerical_targets_frozen_before_qml"] is True

    assert payload["target_rule"]["numerical_targets_available_in_4_1"] is False

    assert (
        payload["success_definitions"]["autonomous_driving"]["collision_free_required"]
        is True
    )

    assert (
        payload["success_definitions"]["autonomous_driving"][
            "terminal_lane_departure_forbidden"
        ]
        is True
    )

    assert (
        payload["success_definitions"]["autonomous_driving"][
            "exact_lane_departure_tolerance_frozen_in"
        ]
        == "sprint_4_2"
    )

    assert (
        payload["success_definitions"]["autonomous_driving"][
            "same_success_rule_for_all_methods"
        ]
        is True
    )

    assert (
        payload["success_definitions"]["robotics"]["task_completion_required"] is True
    )

    assert (
        payload["success_definitions"]["robotics"][
            "exact_completion_tolerance_frozen_in"
        ]
        == "sprint_4_3"
    )

    assert (
        payload["success_definitions"]["robotics"]["same_success_rule_for_all_methods"]
        is True
    )

    comparison = payload["comparison_hierarchy"]

    assert comparison["primary"] == [
        "environment_steps_to_target",
    ]

    assert comparison["secondary"] == [
        "episodes_to_target",
        "final_evaluation_reward",
        "final_success_rate",
    ]

    assert comparison["resource_metrics"] == [
        "policy_parameters",
        "qubits",
        "trainable_circuit_parameters",
        "circuit_evaluations",
        "circuit_layers",
    ]

    assert comparison["descriptive_only"] == [
        "training_seconds",
    ]

    assert comparison["primary_metric_must_drive_efficiency_claims"] is True

    assert comparison["wall_clock_must_not_define_qml_advantage"] is True

    assert comparison["simulator_runtime_must_not_imply_hardware_performance"] is True

    assert comparison["failed_target_uses_none_not_budget_cap"] is True

    robust = payload["robust_qml_efficiency_rule"]

    assert robust["paired_seed_comparison_required"] is True

    assert robust["paired_seeds"] == [
        42,
        123,
        456,
    ]

    assert robust["all_three_qml_seeds_must_reach_target"] is True

    assert robust["all_three_classical_seeds_must_have_valid_targets"] is True

    assert robust["minimum_required_mean_reduction_percent"] == 10.0

    assert robust["failed_qml_target_invalidates_robust_improvement"] is True

    assert robust["failed_target_must_remain_none"] is True

    assert robust["budget_cap_must_not_replace_failed_target"] is True

    assert payload["qml"]["encoding"] == "angle_encoding"
    assert payload["qml"]["qubits"] == 4
    assert payload["qml"]["circuit_layers"] == 2

    assert payload["qml"]["trainable_rotations"] == [
        "RY",
        "RZ",
    ]

    assert payload["qml"]["entanglement"] == "CNOT"
    assert payload["qml"]["execution"] == "simulator"

    assert payload["qml"]["quantum_hardware_used"] is False

    assert payload["claim_controls"]["pilot_scale_proxy"] is True

    assert payload["claim_controls"]["quantum_advantage_claimed"] is False

    assert payload["claim_controls"]["quantum_speedup_claimed"] is False

    assert payload["claim_controls"]["production_vla_claimed"] is False

    assert payload["claim_controls"]["functional_safety_certification_claimed"] is False

    assert payload["claim_controls"]["wall_clock_is_primary_efficiency_metric"] is False
