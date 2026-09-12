from __future__ import annotations

import pytest
import torch

from q_vla_forge.models import SharedVisionEncoder


def test_default_vision_encoder_configuration() -> None:
    encoder = SharedVisionEncoder()

    assert encoder.input_channels == 3
    assert encoder.output_dim == 64


def test_vision_encoder_output_shape() -> None:
    encoder = SharedVisionEncoder(
        output_dim=64,
    )

    visual = torch.zeros(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    output = encoder(visual)

    assert output.shape == (4, 64)


def test_custom_output_dimension() -> None:
    encoder = SharedVisionEncoder(
        output_dim=32,
    )

    visual = torch.zeros(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    output = encoder(visual)

    assert output.shape == (2, 32)


def test_encoder_supports_different_image_sizes() -> None:
    encoder = SharedVisionEncoder(
        output_dim=64,
    )

    visual = torch.zeros(
        2,
        3,
        64,
        64,
        dtype=torch.float32,
    )

    output = encoder(visual)

    assert output.shape == (2, 64)


def test_encoder_outputs_finite_values() -> None:
    torch.manual_seed(42)

    encoder = SharedVisionEncoder()

    visual = torch.rand(
        3,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    output = encoder(visual)

    assert torch.isfinite(output).all()


def test_encoder_supports_gradient_flow() -> None:
    torch.manual_seed(42)

    encoder = SharedVisionEncoder()

    visual = torch.rand(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    output = encoder(visual)

    loss = output.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]

    assert trainable_parameters

    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_batch_items_produce_independent_embeddings() -> None:
    torch.manual_seed(42)

    encoder = SharedVisionEncoder()

    visual = torch.zeros(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    visual[1] = 1.0

    output = encoder(visual)

    assert not torch.equal(
        output[0],
        output[1],
    )


def test_encoder_rejects_missing_batch_dimension() -> None:
    encoder = SharedVisionEncoder()

    visual = torch.zeros(
        3,
        32,
        32,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        encoder(visual)


def test_encoder_rejects_wrong_channel_count() -> None:
    encoder = SharedVisionEncoder()

    visual = torch.zeros(
        1,
        1,
        32,
        32,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        encoder(visual)


def test_encoder_rejects_integer_input() -> None:
    encoder = SharedVisionEncoder()

    visual = torch.zeros(
        1,
        3,
        32,
        32,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        encoder(visual)


def test_encoder_rejects_invalid_output_dimension() -> None:
    with pytest.raises(ValueError):
        SharedVisionEncoder(
            output_dim=0,
        )


def test_encoder_rejects_invalid_channel_count() -> None:
    with pytest.raises(ValueError):
        SharedVisionEncoder(
            input_channels=0,
        )
