from __future__ import annotations

import pytest
import torch

from q_vla_forge.models import SharedLanguageEncoder


def test_default_language_encoder_configuration() -> None:
    encoder = SharedLanguageEncoder()

    assert encoder.output_dim == 32
    assert encoder.vocab_size == 512
    assert encoder.token_dim == 32
    assert encoder.max_tokens == 8


def test_language_encoder_output_shape() -> None:
    encoder = SharedLanguageEncoder()

    output = encoder(
        [
            "keep lane",
            "move object to target",
        ]
    )

    assert output.shape == (2, 32)


def test_custom_output_dimension() -> None:
    encoder = SharedLanguageEncoder(
        output_dim=16,
    )

    output = encoder(
        [
            "slow down",
        ]
    )

    assert output.shape == (1, 16)


def test_tokenization_shape() -> None:
    encoder = SharedLanguageEncoder(
        max_tokens=6,
    )

    token_ids = encoder.tokenize(
        [
            "keep lane",
            "close gripper",
        ]
    )

    assert token_ids.shape == (2, 6)
    assert token_ids.dtype == torch.long


def test_tokenization_is_deterministic() -> None:
    encoder = SharedLanguageEncoder()

    first = encoder.tokenize(
        [
            "move object to target",
        ]
    )

    second = encoder.tokenize(
        [
            "move object to target",
        ]
    )

    assert torch.equal(
        first,
        second,
    )


def test_tokenization_is_case_insensitive() -> None:
    encoder = SharedLanguageEncoder()

    first = encoder.tokenize(
        [
            "KEEP LANE",
        ]
    )

    second = encoder.tokenize(
        [
            "keep lane",
        ]
    )

    assert torch.equal(
        first,
        second,
    )


def test_different_instructions_generate_different_tokens() -> None:
    encoder = SharedLanguageEncoder()

    first = encoder.tokenize(
        [
            "keep lane",
        ]
    )

    second = encoder.tokenize(
        [
            "close gripper",
        ]
    )

    assert not torch.equal(
        first,
        second,
    )


def test_output_contains_finite_values() -> None:
    torch.manual_seed(42)

    encoder = SharedLanguageEncoder()

    output = encoder(
        [
            "keep lane",
            "move right",
            "open gripper",
        ]
    )

    assert torch.isfinite(output).all()


def test_language_encoder_supports_gradient_flow() -> None:
    torch.manual_seed(42)

    encoder = SharedLanguageEncoder()

    output = encoder(
        [
            "keep lane",
            "move object to target",
        ]
    )

    loss = output.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]

    assert trainable_parameters

    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_encoder_rejects_empty_batch() -> None:
    encoder = SharedLanguageEncoder()

    with pytest.raises(ValueError):
        encoder([])


def test_encoder_rejects_empty_instruction() -> None:
    encoder = SharedLanguageEncoder()

    with pytest.raises(ValueError):
        encoder(
            [
                "   ",
            ]
        )


def test_encoder_rejects_non_string_instruction() -> None:
    encoder = SharedLanguageEncoder()

    with pytest.raises(TypeError):
        encoder(
            [
                "keep lane",
                123,  # type: ignore[list-item]
            ]
        )


def test_encoder_rejects_invalid_output_dimension() -> None:
    with pytest.raises(ValueError):
        SharedLanguageEncoder(
            output_dim=0,
        )


def test_encoder_rejects_invalid_vocab_size() -> None:
    with pytest.raises(ValueError):
        SharedLanguageEncoder(
            vocab_size=1,
        )


def test_encoder_rejects_invalid_token_dimension() -> None:
    with pytest.raises(ValueError):
        SharedLanguageEncoder(
            token_dim=0,
        )


def test_encoder_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError):
        SharedLanguageEncoder(
            max_tokens=0,
        )
