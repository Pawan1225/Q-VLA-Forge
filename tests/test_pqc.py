from __future__ import annotations

import pytest
import torch

from q_vla_forge.quantum.pqc import (
    DEFAULT_PQC_CONFIG,
    VariationalQuantumCircuit,
)


def test_frozen_pqc_configuration() -> None:
    assert DEFAULT_PQC_CONFIG.qubits == 4
    assert DEFAULT_PQC_CONFIG.layers == 2
    assert DEFAULT_PQC_CONFIG.input_features == 4


def test_quantum_parameter_count() -> None:
    circuit = VariationalQuantumCircuit()

    assert circuit.quantum_parameter_count == 16


def test_single_output_shape() -> None:
    circuit = VariationalQuantumCircuit()

    inputs = torch.zeros(
        4,
        dtype=torch.float32,
    )

    output = circuit(inputs)

    assert output.shape == (4,)


def test_batch_output_shape() -> None:
    circuit = VariationalQuantumCircuit()

    inputs = torch.zeros(
        (
            3,
            4,
        ),
        dtype=torch.float32,
    )

    output = circuit(inputs)

    assert output.shape == (
        3,
        4,
    )


def test_expectation_values_are_bounded() -> None:
    circuit = VariationalQuantumCircuit()

    inputs = torch.randn(
        (
            5,
            4,
        ),
        dtype=torch.float32,
    )

    output = circuit(inputs)

    assert torch.all(output <= 1.0 + 1e-6)

    assert torch.all(output >= -1.0 - 1e-6)


def test_same_seed_same_initialization() -> None:
    first = VariationalQuantumCircuit(seed=123)

    second = VariationalQuantumCircuit(seed=123)

    torch.testing.assert_close(
        first.weights,
        second.weights,
    )


def test_different_seed_changes_initialization() -> None:
    first = VariationalQuantumCircuit(seed=123)

    second = VariationalQuantumCircuit(seed=456)

    assert not torch.equal(
        first.weights,
        second.weights,
    )


def test_forward_is_deterministic() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            0.1,
            -0.2,
            0.3,
            -0.4,
        ],
        dtype=torch.float32,
    )

    first = circuit(inputs)

    second = circuit(inputs)

    torch.testing.assert_close(
        first,
        second,
    )


def test_quantum_weights_receive_gradients() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            0.1,
            -0.2,
            0.3,
            -0.4,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = circuit(inputs)

    loss = output.sum()

    loss.backward()

    assert circuit.weights.grad is not None

    assert torch.isfinite(circuit.weights.grad).all()


def test_inputs_receive_gradients() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            0.1,
            -0.2,
            0.3,
            -0.4,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = circuit(inputs)

    output.sum().backward()

    assert inputs.grad is not None

    assert torch.isfinite(inputs.grad).all()


def test_rejects_wrong_single_shape() -> None:
    circuit = VariationalQuantumCircuit()

    with pytest.raises(ValueError):
        circuit(torch.zeros(3))


def test_rejects_wrong_batch_shape() -> None:
    circuit = VariationalQuantumCircuit()

    with pytest.raises(ValueError):
        circuit(
            torch.zeros(
                (
                    2,
                    5,
                )
            )
        )


def test_optimizer_updates_quantum_weights() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    optimizer = torch.optim.Adam(
        circuit.parameters(),
        lr=1e-2,
    )

    inputs = torch.tensor(
        [
            [0.10, -0.20, 0.30, -0.40],
            [0.40, 0.30, -0.20, -0.10],
        ],
        dtype=torch.float32,
    )

    before = circuit.weights.detach().clone()

    output = circuit(inputs)

    loss = output.pow(2).mean()

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    after = circuit.weights.detach()

    assert not torch.equal(
        before,
        after,
    )


def test_batch_matches_individual_execution() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.4, -0.3, -0.2, -0.1],
        ],
        dtype=torch.float32,
    )

    batched = circuit(inputs)

    first = circuit(inputs[0])

    second = circuit(inputs[1])

    torch.testing.assert_close(
        batched[0],
        first,
    )

    torch.testing.assert_close(
        batched[1],
        second,
    )


def test_circuit_structure_and_entanglement() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.zeros(
        4,
        dtype=torch.float32,
    )

    tape = circuit._circuit.construct(
        [
            inputs,
            circuit.weights,
        ],
        {},
    )

    operations = tape.operations

    operation_names = [operation.name for operation in operations]

    assert operation_names.count("RY") == 12

    assert operation_names.count("RZ") == 8

    assert operation_names.count("CNOT") == 6

    cnot_wires = [
        tuple(operation.wires.tolist())
        for operation in operations
        if operation.name == "CNOT"
    ]

    assert cnot_wires == [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 1),
        (1, 2),
        (2, 3),
    ]

    assert len(tape.measurements) == 4

    measurement_wires = [
        tuple(measurement.wires.tolist()) for measurement in tape.measurements
    ]

    assert measurement_wires == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]


def test_numerical_stability_across_input_scales() -> None:
    circuit = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-6, -1e-6, 1e-6, -1e-6],
            [0.1, -0.2, 0.3, -0.4],
            [1.0, -1.0, 2.0, -2.0],
            [3.1415927, -3.1415927, 6.2831855, -6.2831855],
            [10.0, -10.0, 20.0, -20.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = circuit(inputs)

    assert output.shape == (
        6,
        4,
    )

    assert torch.isfinite(output).all()

    assert torch.all(output <= 1.0 + 1e-6)

    assert torch.all(output >= -1.0 - 1e-6)

    loss = output.pow(2).mean()

    loss.backward()

    assert inputs.grad is not None

    assert torch.isfinite(inputs.grad).all()

    assert circuit.weights.grad is not None

    assert torch.isfinite(circuit.weights.grad).all()
