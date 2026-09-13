"""PQC foundation for the Q-VLA Forge hybrid QML policy."""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml  # type: ignore[import-untyped]
import torch
from torch import nn


@dataclass(frozen=True)
class PQCConfig:
    """Frozen Sprint 4.6 variational-circuit configuration."""

    qubits: int = 4
    layers: int = 2
    input_features: int = 4

    def __post_init__(self) -> None:
        if self.qubits != 4:
            raise ValueError("Sprint 4.6 freezes the PQC to 4 qubits")

        if self.layers != 2:
            raise ValueError("Sprint 4.6 freezes the PQC to 2 layers")

        if self.input_features != self.qubits:
            raise ValueError("angle encoding requires one feature per qubit")


DEFAULT_PQC_CONFIG = PQCConfig()


class VariationalQuantumCircuit(nn.Module):
    """Four-qubit differentiable variational quantum circuit."""

    def __init__(
        self,
        *,
        config: PQCConfig = DEFAULT_PQC_CONFIG,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.config = config

        self.device = qml.device(
            "default.qubit",
            wires=config.qubits,
            shots=None,
        )

        generator = torch.Generator().manual_seed(seed)

        initial_weights = 0.01 * torch.randn(
            (
                config.layers,
                config.qubits,
                2,
            ),
            generator=generator,
            dtype=torch.float32,
        )

        self.weights = nn.Parameter(initial_weights)

        @qml.qnode(
            self.device,
            interface="torch",
            diff_method="backprop",
        )
        def circuit(
            inputs: torch.Tensor,
            weights: torch.Tensor,
        ) -> list[torch.Tensor]:
            for wire in range(config.qubits):
                qml.RY(
                    inputs[wire],
                    wires=wire,
                )

            for layer in range(config.layers):
                for wire in range(config.qubits):
                    qml.RY(
                        weights[
                            layer,
                            wire,
                            0,
                        ],
                        wires=wire,
                    )

                    qml.RZ(
                        weights[
                            layer,
                            wire,
                            1,
                        ],
                        wires=wire,
                    )

                for wire in range(config.qubits - 1):
                    qml.CNOT(
                        wires=[
                            wire,
                            wire + 1,
                        ]
                    )

            return [qml.expval(qml.PauliZ(wire)) for wire in range(config.qubits)]

        self._circuit = circuit

    @property
    def quantum_parameter_count(
        self,
    ) -> int:
        return int(self.weights.numel())

    def forward_single(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one four-feature circuit input."""
        if inputs.ndim != 1:
            raise ValueError("single PQC input must be rank 1")

        if inputs.shape[0] != self.config.input_features:
            raise ValueError(
                "single PQC input must contain exactly "
                f"{self.config.input_features} features"
            )

        outputs = self._circuit(
            inputs,
            self.weights,
        )

        return torch.stack(tuple(outputs)).to(dtype=inputs.dtype)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a batch of PQC inputs."""
        if inputs.ndim == 1:
            return self.forward_single(inputs)

        if inputs.ndim != 2:
            raise ValueError("PQC input must have shape (4,) or (batch, 4)")

        if inputs.shape[1] != self.config.input_features:
            raise ValueError("PQC batch must have four input features")

        outputs = [self.forward_single(row) for row in inputs]

        return torch.stack(
            outputs,
            dim=0,
        )
