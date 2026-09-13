from __future__ import annotations

from q_vla_forge.quantum.pqc import (
    DEFAULT_PQC_CONFIG,
)
from q_vla_forge.quantum.resources import (
    build_quantum_resource_summary,
)


def test_frozen_quantum_resources() -> None:
    resources = build_quantum_resource_summary(DEFAULT_PQC_CONFIG)

    assert resources.qubits == 4

    assert resources.encoding_ry_gates == 4

    assert resources.variational_ry_gates == 8

    assert resources.variational_rz_gates == 8

    assert resources.cnot_gates == 6

    assert resources.measurements == 4

    assert resources.trainable_quantum_parameters == 16

    assert resources.total_parameterized_rotations == 20

    assert resources.total_gates == 26


def test_simulator_disclosure() -> None:
    resources = build_quantum_resource_summary(DEFAULT_PQC_CONFIG)

    assert resources.simulator == "PennyLane default.qubit"

    assert resources.analytic_expectation_values is True
