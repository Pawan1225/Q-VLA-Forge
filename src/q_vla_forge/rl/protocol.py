"""Frozen experimental protocol for Sprint 4 RL/QML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from q_vla_forge.rl.contracts import RLDomain

SPRINT4_SEEDS = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class RLProtocol:
    """Scientific protocol shared by Sprint 4 experiments."""

    seeds: tuple[int, ...] = SPRINT4_SEEDS
    episode_horizon: int = 100
    maximum_environment_steps: int = 20_000
    evaluation_frequency_steps: int = 1_000
    evaluation_episodes: int = 20
    action_dimension: int = 3
    target_fraction: float = 0.95
    robust_efficiency_threshold_percent: float = 10.0
    qml_qubits: int = 4
    qml_circuit_layers: int = 2
    qml_execution_backend: str = "simulator"

    def __post_init__(self) -> None:
        if self.seeds != SPRINT4_SEEDS:
            raise ValueError("Sprint 4 seeds are frozen to (42, 123, 456)")

        if self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")

        if self.maximum_environment_steps <= 0:
            raise ValueError("maximum_environment_steps must be positive")

        if self.evaluation_frequency_steps <= 0:
            raise ValueError("evaluation_frequency_steps must be positive")

        if self.maximum_environment_steps % self.evaluation_frequency_steps != 0:
            raise ValueError("evaluation frequency must divide " "the training budget")

        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")

        if self.action_dimension != 3:
            raise ValueError("Sprint 4 action dimension is frozen to 3")

        if not 0.0 < self.target_fraction <= 1.0:
            raise ValueError("target_fraction must be in (0, 1]")

        if self.robust_efficiency_threshold_percent != 10.0:
            raise ValueError("robust efficiency threshold is frozen to 10%")

        if self.qml_qubits != 4:
            raise ValueError("initial Sprint 4 QML experiment " "is frozen to 4 qubits")

        if self.qml_circuit_layers <= 0:
            raise ValueError("qml_circuit_layers must be positive")

        if self.qml_execution_backend != "simulator":
            raise ValueError("Sprint 4 uses simulator execution")


DEFAULT_RL_PROTOCOL = RLProtocol()


DRIVING_ACTION_SEMANTICS = (
    "steering",
    "acceleration",
    "braking",
)

ROBOTICS_ACTION_SEMANTICS = (
    "delta_x",
    "delta_y",
    "gripper",
)


def action_semantics(
    domain: RLDomain,
) -> tuple[str, ...]:
    """Return the frozen action meaning for one domain."""
    if domain == RLDomain.AUTONOMOUS_DRIVING:
        return DRIVING_ACTION_SEMANTICS

    if domain == RLDomain.ROBOTICS:
        return ROBOTICS_ACTION_SEMANTICS

    raise ValueError(f"unsupported RL domain: {domain}")


def protocol_to_dict(
    protocol: RLProtocol = DEFAULT_RL_PROTOCOL,
) -> dict[str, Any]:
    """Return a JSON-safe frozen protocol representation."""
    payload = asdict(protocol)

    payload["seeds"] = list(protocol.seeds)

    payload["domains"] = [
        RLDomain.AUTONOMOUS_DRIVING.value,
        RLDomain.ROBOTICS.value,
    ]

    payload["primary_efficiency_metric"] = "environment_steps_to_target"

    payload["secondary_metrics"] = [
        "episodes_to_target",
        "final_evaluation_reward",
        "final_success_rate",
        "training_seconds",
        "policy_parameters",
    ]

    payload["target_rule"] = {
        "reference": "classical_ppo",
        "baseline_reference": "random_policy",
        "performance_source": ("best_completed_reference_evaluation"),
        "fraction": protocol.target_fraction,
        "formula": (
            "random_reference + target_fraction * "
            "(classical_ppo_best - random_reference)"
        ),
        "comparison_direction": "higher_reward_is_better",
        "supports_negative_reward_scales": True,
        "random_reference_required": True,
        "numerical_targets_frozen_before_qml": True,
        "numerical_targets_available_in_4_1": False,
    }

    payload["success_definitions"] = {
        "autonomous_driving": {
            "semantic_definition": (
                "episode reaches intended horizon or task completion "
                "without collision and without terminal lane departure"
            ),
            "collision_free_required": True,
            "terminal_lane_departure_forbidden": True,
            "exact_lane_departure_tolerance_frozen_in": ("sprint_4_2"),
            "tolerance_frozen_before_classical_ppo": True,
            "same_success_rule_for_all_methods": True,
        },
        "robotics": {
            "semantic_definition": (
                "task or object reaches the target condition "
                "within the environment completion tolerance"
            ),
            "task_completion_required": True,
            "exact_completion_tolerance_frozen_in": ("sprint_4_3"),
            "tolerance_frozen_before_classical_ppo": True,
            "same_success_rule_for_all_methods": True,
        },
    }

    payload["comparison_hierarchy"] = {
        "primary": [
            "environment_steps_to_target",
        ],
        "secondary": [
            "episodes_to_target",
            "final_evaluation_reward",
            "final_success_rate",
        ],
        "resource_metrics": [
            "policy_parameters",
            "qubits",
            "trainable_circuit_parameters",
            "circuit_evaluations",
            "circuit_layers",
        ],
        "descriptive_only": [
            "training_seconds",
        ],
        "primary_metric_must_drive_efficiency_claims": True,
        "wall_clock_must_not_define_qml_advantage": True,
        "simulator_runtime_must_not_imply_hardware_performance": True,
        "failed_target_uses_none_not_budget_cap": True,
    }

    payload["robust_qml_efficiency_rule"] = {
        "paired_seed_comparison_required": True,
        "paired_seeds": list(protocol.seeds),
        "all_three_qml_seeds_must_reach_target": True,
        "all_three_classical_seeds_must_have_valid_targets": True,
        "per_seed_formula": (
            "100 * "
            "(classical_steps_to_target - qml_steps_to_target) "
            "/ classical_steps_to_target"
        ),
        "aggregation": "mean_and_sample_standard_deviation",
        "mean_environment_step_reduction_percent": (
            protocol.robust_efficiency_threshold_percent
        ),
        "minimum_required_mean_reduction_percent": (
            protocol.robust_efficiency_threshold_percent
        ),
        "failed_qml_target_invalidates_robust_improvement": True,
        "failed_target_must_remain_none": True,
        "budget_cap_must_not_replace_failed_target": True,
        "claim_if_rule_passes": (
            "pilot-scale evidence of improved RL sample efficiency"
        ),
        "claim_if_rule_fails": ("no robust sample-efficiency improvement established"),
    }

    payload["qml"] = {
        "encoding": "angle_encoding",
        "qubits": protocol.qml_qubits,
        "circuit_layers": protocol.qml_circuit_layers,
        "trainable_rotations": [
            "RY",
            "RZ",
        ],
        "entanglement": "CNOT",
        "execution": protocol.qml_execution_backend,
        "quantum_hardware_used": False,
    }

    payload["claim_controls"] = {
        "pilot_scale_proxy": True,
        "allowed_qml_terms": [
            "hybrid_classical_quantum_policy",
            "parameterized_quantum_circuit",
            "variational_quantum_circuit",
            "angle_encoded_pqc_vqc",
            "four_qubit_simulator",
            "qml_policy",
            "quantum_circuit_simulation",
        ],
        "conditionally_allowed_after_evidence": [
            "pilot_scale_evidence_of_improved_rl_sample_efficiency",
        ],
        "disallowed_without_additional_evidence": [
            "quantum_advantage",
            "quantum_supremacy",
            "quantum_speedup",
            "quantum_enhanced_autonomous_driving_performance",
            "production_quantum_controller",
        ],
        "quantum_hardware_used": False,
        "quantum_advantage_claimed": False,
        "quantum_speedup_claimed": False,
        "quantum_supremacy_claimed": False,
        "production_vla_claimed": False,
        "production_autonomous_driving_claimed": False,
        "production_robotics_claimed": False,
        "functional_safety_certification_claimed": False,
        "simulator_execution_must_be_disclosed": True,
        "proxy_environment_must_be_disclosed": True,
        "wall_clock_is_primary_efficiency_metric": False,
        "sample_efficiency_claim_requires_robust_rule": True,
    }

    payload["claim_registry_handoff"] = {
        "source_sprint": "sprint_3_13",
        "source_registry_path": ("results/pilot-readiness/claim-registry.json"),
        "historical_registry_must_not_be_rewritten": True,
        "sprint_3_13_statuses_preserved": {
            "ppo_sample_efficiency": "NOT_YET_TESTED",
            "pqc_vqc_sample_efficiency": "NOT_YET_TESTED",
            "hybrid_qml_policy_advantage": "NOT_YET_TESTED",
            "cross_domain_rl_consistency": "NOT_YET_TESTED",
        },
        "sprint_4_may_generate_new_evidence": True,
        "proposal_level_claims_may_be_updated_after_sprint_4": True,
        "prior_audit_history_must_remain_immutable": True,
    }

    return payload
