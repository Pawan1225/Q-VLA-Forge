"""Independent Sprint 4.7 hybrid QML policy verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from q_vla_forge.evaluation.hybrid_qml_policy import (
    build_hybrid_policy_resource_summary,
    hybrid_policy_summary_to_dict,
)
from q_vla_forge.rl.hybrid_qml_policy import (
    DEFAULT_HYBRID_QML_POLICY_CONFIG,
    HybridQuantumActorCritic,
)

OUTPUT = Path("results") / "rl" / "hybrid-policy" / "sprint4-hybrid-qml-policy.json"


def build_driving() -> HybridQuantumActorCritic:
    """Build the frozen driving hybrid policy."""
    return HybridQuantumActorCritic(
        observation_dim=4,
        action_low=np.array(
            [
                -1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        action_high=np.array(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def build_robotics() -> HybridQuantumActorCritic:
    """Build the frozen robotics hybrid policy."""
    return HybridQuantumActorCritic(
        observation_dim=6,
        action_low=np.array(
            [
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=np.float32,
        ),
        action_high=np.array(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def verify_model(
    model: HybridQuantumActorCritic,
    observation_dim: int,
) -> dict[str, object]:
    """Verify one domain-specific hybrid actor-critic."""

    observations = torch.tensor(
        [
            [0.1] * observation_dim,
            [-0.2] * observation_dim,
            [0.3] * observation_dim,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    angles = model.quantum_angles(observations)

    quantum_features = model.quantum_features(observations)

    distribution = model.distribution(observations)

    values = model.value(observations)

    deterministic_actions = model.deterministic_action(observations)

    loss = distribution.mean.pow(2).mean() + 0.1 * values.pow(2).mean()

    loss.backward()

    projection_gradient = model.quantum_projection.weight.grad

    quantum_gradient = model.quantum_circuit.weights.grad

    action_head_gradient = model.action_mean_head.weight.grad

    if projection_gradient is None:
        raise RuntimeError("projection gradient missing")

    if quantum_gradient is None:
        raise RuntimeError("quantum gradient missing")

    if action_head_gradient is None:
        raise RuntimeError("action-head gradient missing")

    checks = {
        "angles_shape": (
            list(angles.shape)
            == [
                3,
                4,
            ]
        ),
        "angles_bounded": bool(
            torch.all(angles <= torch.pi) and torch.all(angles >= -torch.pi)
        ),
        "quantum_features_shape": (
            list(quantum_features.shape)
            == [
                3,
                4,
            ]
        ),
        "quantum_outputs_finite": bool(torch.isfinite(quantum_features).all()),
        "distribution_shape": (
            list(distribution.mean.shape)
            == [
                3,
                3,
            ]
        ),
        "values_shape": (
            list(values.shape)
            == [
                3,
            ]
        ),
        "actions_shape": (
            list(deterministic_actions.shape)
            == [
                3,
                3,
            ]
        ),
        "projection_gradient_finite": bool(torch.isfinite(projection_gradient).all()),
        "quantum_gradient_finite": bool(torch.isfinite(quantum_gradient).all()),
        "action_head_gradient_finite": bool(torch.isfinite(action_head_gradient).all()),
        "quantum_gradient_nonzero": (
            float(torch.linalg.vector_norm(quantum_gradient).item()) > 0.0
        ),
    }

    if not all(checks.values()):
        raise RuntimeError("hybrid QML policy verification failed")

    return {
        "checks": checks,
        "quantum_gradient_norm": float(
            torch.linalg.vector_norm(quantum_gradient).item()
        ),
        "projection_gradient_norm": float(
            torch.linalg.vector_norm(projection_gradient).item()
        ),
        "action_head_gradient_norm": float(
            torch.linalg.vector_norm(action_head_gradient).item()
        ),
    }


def main() -> None:
    """Run Sprint 4.7 independent verification."""

    driving = build_driving()
    robotics = build_robotics()

    driving_verification = verify_model(
        driving,
        4,
    )

    robotics_verification = verify_model(
        robotics,
        6,
    )

    driving_resources = build_hybrid_policy_resource_summary(
        domain="autonomous_driving",
        model=driving,
        classical_reference_actor_parameters=1318,
        classical_reference_total_parameters=2567,
    )

    robotics_resources = build_hybrid_policy_resource_summary(
        domain="robotics",
        model=robotics,
        classical_reference_actor_parameters=1382,
        classical_reference_total_parameters=2695,
    )

    payload = {
        "sprint": "4.7",
        "component": ("hybrid_quantum_classical_ppo_policy"),
        "architecture": {
            "classical_projection": ("observation_dim -> 4"),
            "angle_mapping": ("pi * tanh(projection)"),
            "quantum_circuit": ("Sprint 4.6 frozen 4-qubit PQC"),
            "quantum_outputs": ("4 Pauli-Z expectations"),
            "classical_action_head": ("4 -> 3"),
            "policy_distribution": ("Gaussian"),
            "critic": ("classical 32-32 MLP"),
        },
        "configuration": {
            "quantum_features": (DEFAULT_HYBRID_QML_POLICY_CONFIG.quantum_features),
            "angle_scale": (DEFAULT_HYBRID_QML_POLICY_CONFIG.angle_scale),
            "critic_hidden_dim": (DEFAULT_HYBRID_QML_POLICY_CONFIG.critic_hidden_dim),
        },
        "domains": {
            "autonomous_driving": {
                "resources": (hybrid_policy_summary_to_dict(driving_resources)),
                "verification": (driving_verification),
            },
            "robotics": {
                "resources": (hybrid_policy_summary_to_dict(robotics_resources)),
                "verification": (robotics_verification),
            },
        },
        "scientific_boundary": {
            "rl_training_performed": False,
            "principal_qml_runs_performed": False,
            "ppo_vs_qml_comparison_performed": False,
            "sample_efficiency_tested": False,
            "quantum_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_used": False,
            "frozen_ppo_targets_modified": False,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===========================================")
    print(" Q-VLA FORGE — SPRINT 4.7 HYBRID QML POLICY")
    print("===========================================")

    print()
    print("DRIVING")
    print(
        "Actor parameters:",
        driving.actor_parameter_count,
    )
    print(
        "Quantum parameters:",
        driving.quantum_parameter_count,
    )
    print(
        "Critic parameters:",
        driving.critic_parameter_count,
    )
    print(
        "Total parameters:",
        driving.total_parameter_count,
    )

    print()
    print("ROBOTICS")
    print(
        "Actor parameters:",
        robotics.actor_parameter_count,
    )
    print(
        "Quantum parameters:",
        robotics.quantum_parameter_count,
    )
    print(
        "Critic parameters:",
        robotics.critic_parameter_count,
    )
    print(
        "Total parameters:",
        robotics.total_parameter_count,
    )

    print()
    print(
        "Principal RL training performed:",
        False,
    )
    print(
        "Frozen PPO targets modified:",
        False,
    )

    print()
    print("SPRINT 4.7 HYBRID QML POLICY: PASS")
    print(
        "Artifact:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
