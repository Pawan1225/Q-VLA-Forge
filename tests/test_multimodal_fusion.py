from __future__ import annotations

import pytest
import torch

from q_vla_forge.fusion import MultimodalFusion


def make_inputs(
    batch_size: int = 4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    vision = torch.rand(
        batch_size,
        64,
        dtype=torch.float32,
    )

    language = torch.rand(
        batch_size,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        batch_size,
        16,
        dtype=torch.float32,
    )

    return vision, language, state


def test_default_fusion_configuration() -> None:
    fusion = MultimodalFusion()

    assert fusion.vision_dim == 64
    assert fusion.language_dim == 32
    assert fusion.state_dim == 16
    assert fusion.input_dim == 112
    assert fusion.hidden_dim == 128
    assert fusion.output_dim == 64


def test_fusion_output_shape() -> None:
    fusion = MultimodalFusion()

    vision, language, state = make_inputs(
        batch_size=4,
    )

    output = fusion(
        vision,
        language,
        state,
    )

    assert output.shape == (4, 64)


def test_custom_fusion_dimensions() -> None:
    fusion = MultimodalFusion(
        vision_dim=32,
        language_dim=16,
        state_dim=8,
        output_dim=24,
        hidden_dim=48,
    )

    vision = torch.rand(
        2,
        32,
        dtype=torch.float32,
    )

    language = torch.rand(
        2,
        16,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        8,
        dtype=torch.float32,
    )

    output = fusion(
        vision,
        language,
        state,
    )

    assert fusion.input_dim == 56
    assert output.shape == (2, 24)


def test_fusion_outputs_are_finite() -> None:
    torch.manual_seed(42)

    fusion = MultimodalFusion()

    vision, language, state = make_inputs(
        batch_size=8,
    )

    output = fusion(
        vision,
        language,
        state,
    )

    assert torch.isfinite(output).all()


def test_fusion_supports_gradient_flow() -> None:
    torch.manual_seed(42)

    fusion = MultimodalFusion()

    vision, language, state = make_inputs(
        batch_size=4,
    )

    output = fusion(
        vision,
        language,
        state,
    )

    loss = output.square().mean()
    loss.backward()

    trainable_parameters = [
        parameter for parameter in fusion.parameters() if parameter.requires_grad
    ]

    assert trainable_parameters

    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_fusion_rejects_wrong_vision_dimension() -> None:
    fusion = MultimodalFusion()

    vision = torch.rand(
        2,
        63,
        dtype=torch.float32,
    )

    language = torch.rand(
        2,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        16,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        fusion(
            vision,
            language,
            state,
        )


def test_fusion_rejects_wrong_language_dimension() -> None:
    fusion = MultimodalFusion()

    vision = torch.rand(
        2,
        64,
        dtype=torch.float32,
    )

    language = torch.rand(
        2,
        31,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        16,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        fusion(
            vision,
            language,
            state,
        )


def test_fusion_rejects_wrong_state_dimension() -> None:
    fusion = MultimodalFusion()

    vision = torch.rand(
        2,
        64,
        dtype=torch.float32,
    )

    language = torch.rand(
        2,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        15,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        fusion(
            vision,
            language,
            state,
        )


def test_fusion_rejects_batch_size_mismatch() -> None:
    fusion = MultimodalFusion()

    vision = torch.rand(
        2,
        64,
        dtype=torch.float32,
    )

    language = torch.rand(
        3,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        16,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        fusion(
            vision,
            language,
            state,
        )


def test_fusion_rejects_non_matrix_input() -> None:
    fusion = MultimodalFusion()

    vision = torch.rand(
        64,
        dtype=torch.float32,
    )

    language = torch.rand(
        1,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        1,
        16,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        fusion(
            vision,
            language,
            state,
        )


def test_fusion_rejects_integer_input() -> None:
    fusion = MultimodalFusion()

    vision = torch.zeros(
        2,
        64,
        dtype=torch.int64,
    )

    language = torch.rand(
        2,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        2,
        16,
        dtype=torch.float32,
    )

    with pytest.raises(TypeError):
        fusion(
            vision,
            language,
            state,
        )


def test_rejects_invalid_vision_dimension() -> None:
    with pytest.raises(ValueError):
        MultimodalFusion(
            vision_dim=0,
        )


def test_rejects_invalid_language_dimension() -> None:
    with pytest.raises(ValueError):
        MultimodalFusion(
            language_dim=0,
        )


def test_rejects_invalid_state_dimension() -> None:
    with pytest.raises(ValueError):
        MultimodalFusion(
            state_dim=0,
        )


def test_rejects_invalid_output_dimension() -> None:
    with pytest.raises(ValueError):
        MultimodalFusion(
            output_dim=0,
        )


def test_rejects_invalid_hidden_dimension() -> None:
    with pytest.raises(ValueError):
        MultimodalFusion(
            hidden_dim=0,
        )
