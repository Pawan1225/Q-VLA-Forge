from __future__ import annotations

import pytest
import torch

from q_vla_forge.data import Domain
from q_vla_forge.models import DomainStateEncoder


def test_default_state_encoder_configuration() -> None:
    encoder = DomainStateEncoder()

    assert encoder.output_dim == 16
    assert encoder.driving_input_dim == 4
    assert encoder.robotics_input_dim == 6
    assert encoder.adapter_dim == 16


def test_driving_state_output_shape() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        4,
        4,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.AUTONOMOUS_DRIVING,
    )

    assert output.shape == (4, 16)


def test_robotics_state_output_shape() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        4,
        6,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.ROBOTICS,
    )

    assert output.shape == (4, 16)


def test_custom_output_dimension() -> None:
    encoder = DomainStateEncoder(
        output_dim=32,
    )

    driving_state = torch.zeros(
        2,
        4,
        dtype=torch.float32,
    )

    robotics_state = torch.zeros(
        2,
        6,
        dtype=torch.float32,
    )

    driving_output = encoder(
        driving_state,
        Domain.AUTONOMOUS_DRIVING,
    )

    robotics_output = encoder(
        robotics_state,
        Domain.ROBOTICS,
    )

    assert driving_output.shape == (2, 32)
    assert robotics_output.shape == (2, 32)


def test_outputs_are_finite_for_driving() -> None:
    torch.manual_seed(42)

    encoder = DomainStateEncoder()

    state = torch.rand(
        8,
        4,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.AUTONOMOUS_DRIVING,
    )

    assert torch.isfinite(output).all()


def test_outputs_are_finite_for_robotics() -> None:
    torch.manual_seed(42)

    encoder = DomainStateEncoder()

    state = torch.rand(
        8,
        6,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.ROBOTICS,
    )

    assert torch.isfinite(output).all()


def test_driving_gradient_flow() -> None:
    torch.manual_seed(42)

    encoder = DomainStateEncoder()

    state = torch.rand(
        4,
        4,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.AUTONOMOUS_DRIVING,
    )

    loss = output.square().mean()
    loss.backward()

    assert encoder.driving_adapter[0].weight.grad is not None
    assert encoder.shared_encoder[0].weight.grad is not None


def test_robotics_gradient_flow() -> None:
    torch.manual_seed(42)

    encoder = DomainStateEncoder()

    state = torch.rand(
        4,
        6,
        dtype=torch.float32,
    )

    output = encoder(
        state,
        Domain.ROBOTICS,
    )

    loss = output.square().mean()
    loss.backward()

    assert encoder.robotics_adapter[0].weight.grad is not None
    assert encoder.shared_encoder[0].weight.grad is not None


def test_driving_rejects_wrong_feature_count() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        1,
        6,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        encoder(
            state,
            Domain.AUTONOMOUS_DRIVING,
        )


def test_robotics_rejects_wrong_feature_count() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        1,
        4,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        encoder(
            state,
            Domain.ROBOTICS,
        )


def test_encoder_rejects_missing_batch_dimension() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        4,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        encoder(
            state,
            Domain.AUTONOMOUS_DRIVING,
        )


def test_encoder_rejects_integer_input() -> None:
    encoder = DomainStateEncoder()

    state = torch.zeros(
        1,
        4,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        encoder(
            state,
            Domain.AUTONOMOUS_DRIVING,
        )


def test_encoder_rejects_invalid_output_dimension() -> None:
    with pytest.raises(ValueError):
        DomainStateEncoder(
            output_dim=0,
        )


def test_encoder_rejects_invalid_driving_dimension() -> None:
    with pytest.raises(ValueError):
        DomainStateEncoder(
            driving_input_dim=0,
        )


def test_encoder_rejects_invalid_robotics_dimension() -> None:
    with pytest.raises(ValueError):
        DomainStateEncoder(
            robotics_input_dim=0,
        )


def test_encoder_rejects_invalid_adapter_dimension() -> None:
    with pytest.raises(ValueError):
        DomainStateEncoder(
            adapter_dim=0,
        )
