from __future__ import annotations

import pytest
import torch

from q_vla_forge.models import SharedActionHead


def test_default_action_head_configuration() -> None:
    model = SharedActionHead()

    assert model.input_dim == 32
    assert model.action_dim == 3
    assert model.hidden_dim == 32


def test_action_output_shape() -> None:
    model = SharedActionHead()

    latent = torch.rand(
        4,
        32,
        dtype=torch.float32,
    )

    output = model(latent)

    assert output.shape == (4, 3)


def test_custom_action_dimensions() -> None:
    model = SharedActionHead(
        input_dim=16,
        action_dim=4,
        hidden_dim=24,
    )

    latent = torch.rand(
        3,
        16,
        dtype=torch.float32,
    )

    output = model(latent)

    assert output.shape == (3, 4)


def test_action_outputs_are_finite() -> None:
    torch.manual_seed(42)

    model = SharedActionHead()

    latent = torch.rand(
        8,
        32,
        dtype=torch.float32,
    )

    output = model(latent)

    assert torch.isfinite(output).all()


def test_action_outputs_are_bounded() -> None:
    torch.manual_seed(42)

    model = SharedActionHead()

    latent = (
        torch.randn(
            100,
            32,
            dtype=torch.float32,
        )
        * 10.0
    )

    output = model(latent)

    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_action_head_supports_gradient_flow() -> None:
    torch.manual_seed(42)

    model = SharedActionHead()

    latent = torch.rand(
        4,
        32,
        dtype=torch.float32,
    )

    output = model(latent)

    loss = output.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_different_latents_produce_different_actions() -> None:
    torch.manual_seed(42)

    model = SharedActionHead()

    first = torch.zeros(
        1,
        32,
        dtype=torch.float32,
    )

    second = torch.ones(
        1,
        32,
        dtype=torch.float32,
    )

    first_output = model(first)
    second_output = model(second)

    assert not torch.equal(
        first_output,
        second_output,
    )


def test_action_head_rejects_wrong_feature_dimension() -> None:
    model = SharedActionHead()

    latent = torch.rand(
        2,
        31,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        model(latent)


def test_action_head_rejects_missing_batch_dimension() -> None:
    model = SharedActionHead()

    latent = torch.rand(
        32,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        model(latent)


def test_action_head_rejects_integer_input() -> None:
    model = SharedActionHead()

    latent = torch.zeros(
        2,
        32,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(latent)


def test_rejects_invalid_input_dimension() -> None:
    with pytest.raises(ValueError):
        SharedActionHead(
            input_dim=0,
        )


def test_rejects_invalid_action_dimension() -> None:
    with pytest.raises(ValueError):
        SharedActionHead(
            action_dim=0,
        )


def test_rejects_invalid_hidden_dimension() -> None:
    with pytest.raises(ValueError):
        SharedActionHead(
            hidden_dim=0,
        )
