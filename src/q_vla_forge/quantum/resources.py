"""Quantum-resource accounting for Sprint 4.6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from q_vla_forge.quantum.pqc import PQCConfig


@dataclass(frozen=True)
class QuantumResourceSummary:
    """Static resource accounting for the frozen PQC."""

    qubits: int
    encoding_ry_gates: int
    variational_ry_gates: int
    variational_rz_gates: int
    cnot_gates: int
    measurements: int
    trainable_quantum_parameters: int
    variational_layers: int
    simulator: str
    analytic_expectation_values: bool

    @property
    def total_parameterized_rotations(
        self,
    ) -> int:
        return (
            self.encoding_ry_gates
            + self.variational_ry_gates
            + self.variational_rz_gates
        )

    @property
    def total_gates(
        self,
    ) -> int:
        return self.total_parameterized_rotations + self.cnot_gates


def build_quantum_resource_summary(
    config: PQCConfig,
) -> QuantumResourceSummary:
    """Calculate frozen circuit resources analytically."""
    encoding_ry = config.qubits

    variational_ry = config.layers * config.qubits

    variational_rz = config.layers * config.qubits

    cnot = config.layers * (config.qubits - 1)

    trainable_parameters = config.layers * config.qubits * 2

    return QuantumResourceSummary(
        qubits=config.qubits,
        encoding_ry_gates=encoding_ry,
        variational_ry_gates=variational_ry,
        variational_rz_gates=variational_rz,
        cnot_gates=cnot,
        measurements=config.qubits,
        trainable_quantum_parameters=(trainable_parameters),
        variational_layers=config.layers,
        simulator="PennyLane default.qubit",
        analytic_expectation_values=True,
    )


def resource_summary_to_dict(
    summary: QuantumResourceSummary,
) -> dict[str, Any]:
    """Serialize the resource summary."""
    return asdict(summary) | {
        "total_parameterized_rotations": (summary.total_parameterized_rotations),
        "total_gates": (summary.total_gates),
    }
