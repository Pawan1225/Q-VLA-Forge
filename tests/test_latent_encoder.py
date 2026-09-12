from __future__ import annotations

import pytest
import torch

from q_vla_forge.models import SharedLatentRepresentation


def test_default_latent_configuration() -> None:
    model = SharedLatentRepresentation()

    assert model.input_dim == 64
    assert model.latent_dim == 32
    assert model.hidden_dim == 64


def test_latent_output_shape() -> None:
    model = SharedLatentRepresentation()

    fused = torch.rand(
        4,
        64,
        dtype=torch.float32,
    )

    output = model(fused)

    assert output.shape == (4, 32)


def test_custom_latent_dimensions() -> None:
    model = SharedLatentRepresentation(
        input_dim=32,
        latent_dim=16,
        hidden_dim=24,
    )

    fused = torch.rand(
        3,
        32,
        dtype=torch.float32,
    )

    output = model(fused)

    assert output.shape == (3, 16)


def test_latent_outputs_are_finite() -> None:
    torch.manual_seed(42)

    model = SharedLatentRepresentation()

    fused = torch.rand(
        8,
        64,
        dtype=torch.float32,
    )

    output = model(fused)

    assert torch.isfinite(output).all()


def test_latent_supports_gradient_flow() -> None:
    torch.manual_seed(42)

    model = SharedLatentRepresentation()

    fused = torch.rand(
        4,
        64,
        dtype=torch.float32,
    )

    output = model(fused)

    loss = output.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_different_inputs_produce_different_latents() -> None:
    torch.manual_seed(42)

    model = SharedLatentRepresentation()

    first = torch.zeros(
        1,
        64,
        dtype=torch.float32,
    )

    second = torch.ones(
        1,
        64,
        dtype=torch.float32,
    )

    first_output = model(first)
    second_output = model(second)

    assert not torch.equal(
        first_output,
        second_output,
    )


def test_latent_rejects_wrong_feature_dimension() -> None:
    model = SharedLatentRepresentation()

    fused = torch.rand(
        2,
        63,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        model(fused)


def test_latent_rejects_missing_batch_dimension() -> None:
    model = SharedLatentRepresentation()

    fused = torch.rand(
        64,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        model(fused)


def test_latent_rejects_integer_input() -> None:
    model = SharedLatentRepresentation()

    fused = torch.zeros(
        2,
        64,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(fused)


def test_rejects_invalid_input_dimension() -> None:
    with pytest.raises(ValueError):
        SharedLatentRepresentation(
            input_dim=0,
        )


def test_rejects_invalid_latent_dimension() -> None:
    with pytest.raises(ValueError):
        SharedLatentRepresentation(
            latent_dim=0,
        )


def test_rejects_invalid_hidden_dimension() -> None:
    with pytest.raises(ValueError):
        SharedLatentRepresentation(
            hidden_dim=0,
        )
