"""Independent Sprint 4.6 PQC foundation verification."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from q_vla_forge.quantum.pqc import (
    DEFAULT_PQC_CONFIG,
    VariationalQuantumCircuit,
)
from q_vla_forge.quantum.resources import (
    build_quantum_resource_summary,
    resource_summary_to_dict,
)

OUTPUT = Path("results") / "rl" / "qml-foundation" / "sprint4-qml-foundation.json"


def main() -> None:
    first = VariationalQuantumCircuit(seed=42)

    second = VariationalQuantumCircuit(seed=42)

    inputs = torch.tensor(
        [
            [0.10, -0.20, 0.30, -0.40],
            [0.25, 0.50, -0.25, -0.50],
            [-0.75, 0.10, 0.80, -0.15],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = first(inputs)

    loss = output.pow(2).mean()

    loss.backward()

    weight_gradient = first.weights.grad
    input_gradient = inputs.grad

    if weight_gradient is None:
        raise RuntimeError("PQC weight gradient is missing")

    if input_gradient is None:
        raise RuntimeError("PQC input gradient is missing")

    repeated_output = second(inputs.detach())

    deterministic = torch.allclose(
        output.detach(),
        repeated_output.detach(),
        atol=1e-7,
        rtol=1e-6,
    )

    outputs_finite = bool(torch.isfinite(output).all())

    weight_gradients_finite = bool(torch.isfinite(weight_gradient).all())

    input_gradients_finite = bool(torch.isfinite(input_gradient).all())

    bounded = bool(
        torch.all(output.detach() <= 1.0 + 1e-6)
        and torch.all(output.detach() >= -1.0 - 1e-6)
    )

    weight_gradient_norm = float(torch.linalg.vector_norm(weight_gradient).item())

    input_gradient_norm = float(torch.linalg.vector_norm(input_gradient).item())

    resources = build_quantum_resource_summary(DEFAULT_PQC_CONFIG)

    payload = {
        "sprint": "4.6",
        "component": ("variational_quantum_circuit"),
        "purpose": ("pre-RL QML circuit foundation"),
        "configuration": {
            "qubits": (DEFAULT_PQC_CONFIG.qubits),
            "layers": (DEFAULT_PQC_CONFIG.layers),
            "input_features": (DEFAULT_PQC_CONFIG.input_features),
            "encoding": ("RY angle encoding"),
            "variational_gates": [
                "RY",
                "RZ",
            ],
            "entanglement": ("linear nearest-neighbor CNOT"),
            "measurements": ("Pauli-Z expectation values"),
            "shots": None,
            "simulator": ("PennyLane default.qubit"),
        },
        "resources": (resource_summary_to_dict(resources)),
        "verification": {
            "output_shape": list(output.shape),
            "outputs_finite": (outputs_finite),
            "outputs_bounded": (bounded),
            "deterministic_same_seed": (deterministic),
            "weight_gradients_present": True,
            "weight_gradients_finite": (weight_gradients_finite),
            "input_gradients_present": True,
            "input_gradients_finite": (input_gradients_finite),
            "weight_gradient_norm": (weight_gradient_norm),
            "input_gradient_norm": (input_gradient_norm),
        },
        "scientific_boundary": {
            "rl_training_performed": False,
            "ppo_comparison_performed": False,
            "qml_sample_efficiency_tested": False,
            "quantum_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_used": False,
        },
    }

    checks = [
        outputs_finite,
        bounded,
        deterministic,
        weight_gradients_finite,
        input_gradients_finite,
        weight_gradient_norm > 0.0,
        input_gradient_norm > 0.0,
        resources.qubits == 4,
        (resources.trainable_quantum_parameters == 16),
        resources.cnot_gates == 6,
        resources.total_gates == 26,
    ]

    if not all(checks):
        raise RuntimeError("Sprint 4.6 PQC foundation verification failed")

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
    print("==========================================")
    print(" Q-VLA FORGE — SPRINT 4.6 PQC FOUNDATION")
    print("==========================================")

    print(
        "Qubits:",
        resources.qubits,
    )

    print(
        "Layers:",
        resources.variational_layers,
    )

    print(
        "Trainable quantum parameters:",
        resources.trainable_quantum_parameters,
    )

    print(
        "Total gates:",
        resources.total_gates,
    )

    print(
        "Output shape:",
        tuple(output.shape),
    )

    print(
        "Weight gradient norm:",
        weight_gradient_norm,
    )

    print(
        "Input gradient norm:",
        input_gradient_norm,
    )

    print(
        "Deterministic:",
        deterministic,
    )

    print(
        "Quantum hardware used:",
        False,
    )

    print()
    print("SPRINT 4.6 PQC FOUNDATION: PASS")

    print(
        "Artifact:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
