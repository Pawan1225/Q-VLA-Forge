"""Core contracts for Sprint 4 reinforcement-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RLDomain(str, Enum):
    """Supported Sprint 4 proxy domains."""

    AUTONOMOUS_DRIVING = "autonomous_driving"
    ROBOTICS = "robotics"


class RLMethod(str, Enum):
    """Principal Sprint 4 policy families."""

    CLASSICAL_PPO = "classical_ppo"
    HYBRID_QML_PPO = "hybrid_qml_ppo"


class TargetStatus(str, Enum):
    """Status of one policy relative to its frozen target."""

    REACHED = "reached"
    NOT_REACHED = "not_reached"


@dataclass(frozen=True)
class EvaluationPoint:
    """Evaluation result at one training interaction count."""

    environment_steps: int
    mean_reward: float
    reward_standard_deviation: float
    success_rate: float
    evaluation_episodes: int

    def __post_init__(self) -> None:
        if self.environment_steps < 0:
            raise ValueError("environment_steps must be non-negative")

        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")

        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be in [0, 1]")


@dataclass(frozen=True)
class RLTargetDefinition:
    """Frozen reward target derived from a classical PPO reference."""

    domain: RLDomain
    seed: int
    reference_method: RLMethod
    reference_best_evaluation_reward: float
    target_evaluation_reward: float
    target_fraction: float
    derivation: str

    def __post_init__(self) -> None:
        if self.reference_method != RLMethod.CLASSICAL_PPO:
            raise ValueError("Sprint 4 targets must be derived from classical PPO")

        if not 0.0 < self.target_fraction <= 1.0:
            raise ValueError("target_fraction must be in (0, 1]")


@dataclass(frozen=True)
class RLTargetReach:
    """First point at which a run reaches its frozen reward target."""

    reached_target: bool
    environment_steps_to_target: int | None
    episodes_to_target: int | None
    seconds_to_target: float | None

    def __post_init__(self) -> None:
        values = (
            self.environment_steps_to_target,
            self.episodes_to_target,
            self.seconds_to_target,
        )

        if self.reached_target:
            if any(value is None for value in values):
                raise ValueError(
                    "reached target requires complete target-reach metrics"
                )
        else:
            if any(value is not None for value in values):
                raise ValueError(
                    "failed target must retain None for target-reach metrics"
                )


@dataclass(frozen=True)
class QuantumResourceRecord:
    """Resource accounting for the hybrid QML policy."""

    qubits: int
    circuit_layers: int
    trainable_circuit_parameters: int
    circuit_evaluations: int
    execution_backend: str
    hardware_used: bool

    def __post_init__(self) -> None:
        if self.qubits <= 0:
            raise ValueError("qubits must be positive")

        if self.circuit_layers <= 0:
            raise ValueError("circuit_layers must be positive")

        if self.trainable_circuit_parameters < 0:
            raise ValueError("trainable_circuit_parameters must be non-negative")

        if self.circuit_evaluations < 0:
            raise ValueError("circuit_evaluations must be non-negative")


@dataclass(frozen=True)
class RLRunRecord:
    """Normalized evidence record for one principal RL run."""

    experiment_id: str
    domain: RLDomain
    method: RLMethod
    seed: int
    training_environment_steps: int
    completed_episodes: int
    evaluation_history: tuple[EvaluationPoint, ...]
    target: RLTargetDefinition
    target_reach: RLTargetReach
    final_evaluation_reward: float
    final_success_rate: float
    policy_parameters: int
    training_seconds: float
    quantum_resources: QuantumResourceRecord | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.training_environment_steps < 0:
            raise ValueError("training_environment_steps must be non-negative")

        if self.completed_episodes < 0:
            raise ValueError("completed_episodes must be non-negative")

        if self.policy_parameters <= 0:
            raise ValueError("policy_parameters must be positive")

        if self.training_seconds < 0.0:
            raise ValueError("training_seconds must be non-negative")

        if not 0.0 <= self.final_success_rate <= 1.0:
            raise ValueError("final_success_rate must be in [0, 1]")

        if self.target.domain != self.domain:
            raise ValueError("target domain must match run domain")

        if self.target.seed != self.seed:
            raise ValueError("target seed must match run seed")

        if self.method == RLMethod.HYBRID_QML_PPO and self.quantum_resources is None:
            raise ValueError("hybrid QML run requires quantum resources")

        if self.method == RLMethod.CLASSICAL_PPO and self.quantum_resources is not None:
            raise ValueError("classical PPO must not report quantum resources")


def run_record_to_dict(
    record: RLRunRecord,
) -> dict[str, Any]:
    """Convert a normalized run record into JSON-safe data."""
    return {
        "experiment_id": record.experiment_id,
        "domain": record.domain.value,
        "method": record.method.value,
        "seed": record.seed,
        "training_environment_steps": (record.training_environment_steps),
        "completed_episodes": record.completed_episodes,
        "evaluation_history": [
            {
                "environment_steps": point.environment_steps,
                "mean_reward": point.mean_reward,
                "reward_standard_deviation": (point.reward_standard_deviation),
                "success_rate": point.success_rate,
                "evaluation_episodes": (point.evaluation_episodes),
            }
            for point in record.evaluation_history
        ],
        "target": {
            "domain": record.target.domain.value,
            "seed": record.target.seed,
            "reference_method": (record.target.reference_method.value),
            "reference_best_evaluation_reward": (
                record.target.reference_best_evaluation_reward
            ),
            "target_evaluation_reward": (record.target.target_evaluation_reward),
            "target_fraction": record.target.target_fraction,
            "derivation": record.target.derivation,
        },
        "target_reach": {
            "reached_target": (record.target_reach.reached_target),
            "environment_steps_to_target": (
                record.target_reach.environment_steps_to_target
            ),
            "episodes_to_target": (record.target_reach.episodes_to_target),
            "seconds_to_target": (record.target_reach.seconds_to_target),
        },
        "final_evaluation_reward": (record.final_evaluation_reward),
        "final_success_rate": record.final_success_rate,
        "policy_parameters": record.policy_parameters,
        "training_seconds": record.training_seconds,
        "quantum_resources": (
            None
            if record.quantum_resources is None
            else {
                "qubits": record.quantum_resources.qubits,
                "circuit_layers": (record.quantum_resources.circuit_layers),
                "trainable_circuit_parameters": (
                    record.quantum_resources.trainable_circuit_parameters
                ),
                "circuit_evaluations": (record.quantum_resources.circuit_evaluations),
                "execution_backend": (record.quantum_resources.execution_backend),
                "hardware_used": (record.quantum_resources.hardware_used),
            }
        ),
        "notes": list(record.notes),
    }
