"""Evidence utilities for the hybrid QML PPO policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from q_vla_forge.rl.hybrid_qml_policy import (
    HybridQuantumActorCritic,
)


@dataclass(frozen=True)
class HybridPolicyResourceSummary:
    """Transparent parameter accounting for one hybrid policy."""

    domain: str

    observation_dim: int
    action_dim: int

    projection_parameters: int
    quantum_parameters: int
    action_head_parameters: int
    log_std_parameters: int

    actor_parameters: int
    critic_parameters: int
    total_parameters: int

    classical_reference_actor_parameters: int
    classical_reference_total_parameters: int

    actor_parameter_reduction_percent: float
    total_parameter_reduction_percent: float


def build_hybrid_policy_resource_summary(
    *,
    domain: str,
    model: HybridQuantumActorCritic,
    classical_reference_actor_parameters: int,
    classical_reference_total_parameters: int,
) -> HybridPolicyResourceSummary:
    """Calculate hybrid versus classical parameter accounting."""

    if classical_reference_actor_parameters <= 0:
        raise ValueError("classical reference actor parameter count must be positive")

    if classical_reference_total_parameters <= 0:
        raise ValueError("classical reference total parameter count must be positive")

    actor_reduction = (
        100.0
        * (classical_reference_actor_parameters - model.actor_parameter_count)
        / classical_reference_actor_parameters
    )

    total_reduction = (
        100.0
        * (classical_reference_total_parameters - model.total_parameter_count)
        / classical_reference_total_parameters
    )

    return HybridPolicyResourceSummary(
        domain=domain,
        observation_dim=model.observation_dim,
        action_dim=model.action_dim,
        projection_parameters=(model.projection_parameter_count),
        quantum_parameters=(model.quantum_parameter_count),
        action_head_parameters=(model.action_head_parameter_count),
        log_std_parameters=int(model.log_std.numel()),
        actor_parameters=(model.actor_parameter_count),
        critic_parameters=(model.critic_parameter_count),
        total_parameters=(model.total_parameter_count),
        classical_reference_actor_parameters=(classical_reference_actor_parameters),
        classical_reference_total_parameters=(classical_reference_total_parameters),
        actor_parameter_reduction_percent=(actor_reduction),
        total_parameter_reduction_percent=(total_reduction),
    )


def hybrid_policy_summary_to_dict(
    summary: HybridPolicyResourceSummary,
) -> dict[str, Any]:
    """Serialize hybrid-policy resource evidence."""
    return asdict(summary)
