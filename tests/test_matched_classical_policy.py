from __future__ import annotations

import numpy as np
import torch

from q_vla_forge.rl.matched_classical_policy import (
    MatchedClassicalActorCritic,
)


def build_driving() -> MatchedClassicalActorCritic:
    return MatchedClassicalActorCritic(
        observation_dim=4,
        action_low=np.array(
            [-1.0, -1.0, 0.0],
            dtype=np.float32,
        ),
        action_high=np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )


def build_robotics() -> MatchedClassicalActorCritic:
    return MatchedClassicalActorCritic(
        observation_dim=6,
        action_low=np.array(
            [-1.0, -1.0, -1.0],
            dtype=np.float32,
        ),
        action_high=np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_driving_exact_parameter_match() -> None:
    model = build_driving()

    assert model.projection_parameter_count == 20
    assert model.core_parameter_count == 16
    assert model.action_head_parameter_count == 15
    assert model.log_std.numel() == 3

    assert model.actor_parameter_count == 54
    assert model.critic_parameter_count == 1249
    assert model.total_parameter_count == 1303


def test_robotics_exact_parameter_match() -> None:
    model = build_robotics()

    assert model.projection_parameter_count == 28
    assert model.core_parameter_count == 16
    assert model.action_head_parameter_count == 15
    assert model.log_std.numel() == 3

    assert model.actor_parameter_count == 62
    assert model.critic_parameter_count == 1313
    assert model.total_parameter_count == 1375


def test_classical_core_has_exactly_16_parameters() -> None:
    model = build_driving()

    assert (
        sum(parameter.numel() for parameter in model.classical_core.parameters()) == 16
    )


def test_classical_core_has_no_bias() -> None:
    model = build_driving()

    assert model.classical_core.bias is None


def test_actor_features_are_bounded() -> None:
    model = build_robotics()

    observations = torch.randn(
        8,
        6,
    )

    features = model.actor_features(observations)

    assert features.shape == (
        8,
        4,
    )

    assert torch.all(features <= 1.0)

    assert torch.all(features >= -1.0)


def test_driving_action_bounds() -> None:
    model = build_driving()

    observations = torch.randn(
        8,
        4,
    )

    actions = model.deterministic_action(observations)

    assert torch.all(actions[:, 0] >= -1.0)
    assert torch.all(actions[:, 0] <= 1.0)

    assert torch.all(actions[:, 1] >= -1.0)
    assert torch.all(actions[:, 1] <= 1.0)

    assert torch.all(actions[:, 2] >= 0.0)
    assert torch.all(actions[:, 2] <= 1.0)


def test_robotics_action_bounds() -> None:
    model = build_robotics()

    observations = torch.randn(
        8,
        6,
    )

    actions = model.deterministic_action(observations)

    assert torch.all(actions >= -1.0)

    assert torch.all(actions <= 1.0)


def test_raw_action_log_probability_reconstructs() -> None:
    torch.manual_seed(42)

    model = build_driving()

    observation = torch.zeros(
        1,
        4,
    )

    raw_action, original_log_prob, _ = model.sample_action(observation)

    distribution = model.distribution(observation)

    reconstructed = distribution.log_prob(raw_action).sum(dim=-1)

    torch.testing.assert_close(
        original_log_prob,
        reconstructed,
    )


def test_gradients_reach_classical_core() -> None:
    model = build_robotics()

    observations = torch.randn(
        4,
        6,
    )

    distribution = model.distribution(observations)

    loss = distribution.mean.pow(2).mean()

    loss.backward()

    gradient = model.classical_core.weight.grad

    assert gradient is not None

    assert torch.isfinite(gradient).all()

    assert torch.linalg.vector_norm(gradient).item() > 0.0
