from __future__ import annotations

import pytest
import torch

from q_vla_forge.data import Domain
from q_vla_forge.models import SharedVLAModel


def test_driving_vla_output_shape() -> None:
    model = SharedVLAModel()

    visual = torch.rand(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        4,
        4,
        dtype=torch.float32,
    )

    language = [
        "keep lane",
        "slow down",
        "move left",
        "stop safely",
    ]

    output = model(
        visual,
        state,
        language,
        Domain.AUTONOMOUS_DRIVING,
    )

    assert output.shape == (4, 3)


def test_robotics_vla_output_shape() -> None:
    model = SharedVLAModel()

    visual = torch.rand(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    state = torch.rand(
        4,
        6,
        dtype=torch.float32,
    )

    language = [
        "move object to target",
        "move left",
        "close gripper",
        "open gripper",
    ]

    output = model(
        visual,
        state,
        language,
        Domain.ROBOTICS,
    )

    assert output.shape == (4, 3)


def test_vla_outputs_are_bounded() -> None:
    torch.manual_seed(42)

    model = SharedVLAModel()

    visual = torch.rand(
        2,
        3,
        32,
        32,
    )

    state = torch.rand(
        2,
        4,
    )

    output = model(
        visual,
        state,
        [
            "keep lane",
            "slow down",
        ],
        Domain.AUTONOMOUS_DRIVING,
    )

    assert torch.all(output >= -1.0)

    assert torch.all(output <= 1.0)


def test_vla_supports_end_to_end_gradients() -> None:
    torch.manual_seed(42)

    model = SharedVLAModel()

    visual = torch.rand(
        2,
        3,
        32,
        32,
    )

    state = torch.rand(
        2,
        4,
    )

    output = model(
        visual,
        state,
        [
            "keep lane",
            "stop safely",
        ],
        Domain.AUTONOMOUS_DRIVING,
    )

    loss = output.square().mean()
    loss.backward()

    assert next(model.vision_encoder.parameters()).grad is not None

    assert next(model.language_encoder.parameters()).grad is not None

    assert next(model.state_encoder.parameters()).grad is not None

    assert next(model.fusion.parameters()).grad is not None

    assert next(model.latent.parameters()).grad is not None

    assert next(model.action_head.parameters()).grad is not None


def test_vla_rejects_visual_state_batch_mismatch() -> None:
    model = SharedVLAModel()

    visual = torch.rand(
        2,
        3,
        32,
        32,
    )

    state = torch.rand(
        3,
        4,
    )

    with pytest.raises(ValueError):
        model(
            visual,
            state,
            [
                "keep lane",
                "slow down",
            ],
            Domain.AUTONOMOUS_DRIVING,
        )


def test_vla_rejects_language_batch_mismatch() -> None:
    model = SharedVLAModel()

    visual = torch.rand(
        2,
        3,
        32,
        32,
    )

    state = torch.rand(
        2,
        4,
    )

    with pytest.raises(ValueError):
        model(
            visual,
            state,
            [
                "keep lane",
            ],
            Domain.AUTONOMOUS_DRIVING,
        )
