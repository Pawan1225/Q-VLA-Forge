from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from q_vla_forge.rl.hybrid_qml_policy import (
    DEFAULT_HYBRID_QML_POLICY_CONFIG,
    HybridQuantumActorCritic,
)


def build_driving_model(
    *,
    seed: int = 42,
) -> HybridQuantumActorCritic:
    config = type(DEFAULT_HYBRID_QML_POLICY_CONFIG)(pqc_seed=seed)

    return HybridQuantumActorCritic(
        observation_dim=4,
        action_low=np.array(
            [
                -1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        action_high=np.array(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        config=config,
    )


def build_robotics_model(
    *,
    seed: int = 42,
) -> HybridQuantumActorCritic:
    config = type(DEFAULT_HYBRID_QML_POLICY_CONFIG)(pqc_seed=seed)

    return HybridQuantumActorCritic(
        observation_dim=6,
        action_low=np.array(
            [
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=np.float32,
        ),
        action_high=np.array(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        config=config,
    )


def test_frozen_hybrid_configuration() -> None:
    config = DEFAULT_HYBRID_QML_POLICY_CONFIG

    assert config.quantum_features == 4
    assert config.critic_hidden_dim == 32

    assert math.isclose(
        config.angle_scale,
        math.pi,
    )


def test_driving_actor_parameter_count() -> None:
    model = build_driving_model()

    assert model.projection_parameter_count == 20
    assert model.quantum_parameter_count == 16
    assert model.action_head_parameter_count == 15
    assert model.log_std.numel() == 3
    assert model.actor_parameter_count == 54
    assert model.critic_parameter_count == 1249
    assert model.total_parameter_count == 1303


def test_robotics_actor_parameter_count() -> None:
    model = build_robotics_model()

    assert model.projection_parameter_count == 28
    assert model.quantum_parameter_count == 16
    assert model.action_head_parameter_count == 15
    assert model.log_std.numel() == 3
    assert model.actor_parameter_count == 62
    assert model.critic_parameter_count == 1313
    assert model.total_parameter_count == 1375


def test_driving_distribution_shape() -> None:
    model = build_driving_model()

    observations = torch.zeros(
        (
            3,
            4,
        ),
        dtype=torch.float32,
    )

    distribution = model.distribution(observations)

    assert distribution.mean.shape == (
        3,
        3,
    )

    assert distribution.stddev.shape == (
        3,
        3,
    )


def test_robotics_distribution_shape() -> None:
    model = build_robotics_model()

    observations = torch.zeros(
        (
            3,
            6,
        ),
        dtype=torch.float32,
    )

    distribution = model.distribution(observations)

    assert distribution.mean.shape == (
        3,
        3,
    )


def test_value_shape() -> None:
    model = build_robotics_model()

    observations = torch.zeros(
        (
            5,
            6,
        ),
        dtype=torch.float32,
    )

    values = model.value(observations)

    assert values.shape == (5,)


def test_quantum_angles_are_bounded() -> None:
    model = build_robotics_model()

    observations = 100.0 * torch.randn(
        (
            10,
            6,
        )
    )

    angles = model.quantum_angles(observations)

    assert torch.all(angles <= math.pi)

    assert torch.all(angles >= -math.pi)


def test_quantum_features_are_bounded() -> None:
    model = build_robotics_model()

    observations = torch.randn(
        (
            4,
            6,
        )
    )

    features = model.quantum_features(observations)

    assert features.shape == (
        4,
        4,
    )

    assert torch.all(features <= 1.0 + 1e-6)

    assert torch.all(features >= -1.0 - 1e-6)


def test_driving_actions_respect_bounds() -> None:
    model = build_driving_model()

    observations = torch.randn(
        (
            4,
            4,
        )
    )

    actions = model.deterministic_action(observations)

    assert torch.all(actions[:, 0] >= -1.0)

    assert torch.all(actions[:, 0] <= 1.0)

    assert torch.all(actions[:, 1] >= -1.0)

    assert torch.all(actions[:, 1] <= 1.0)

    assert torch.all(actions[:, 2] >= 0.0)

    assert torch.all(actions[:, 2] <= 1.0)


def test_robotics_actions_respect_bounds() -> None:
    model = build_robotics_model()

    observations = torch.randn(
        (
            4,
            6,
        )
    )

    actions = model.deterministic_action(observations)

    assert torch.all(actions >= -1.0)

    assert torch.all(actions <= 1.0)


def test_gradients_reach_projection_and_pqc() -> None:
    model = build_robotics_model(seed=42)

    observations = torch.randn(
        (
            3,
            6,
        ),
        dtype=torch.float32,
        requires_grad=True,
    )

    distribution = model.distribution(observations)

    loss = distribution.mean.pow(2).mean()

    loss.backward()

    projection_gradient = model.quantum_projection.weight.grad

    quantum_gradient = model.quantum_circuit.weights.grad

    action_gradient = model.action_mean_head.weight.grad

    assert projection_gradient is not None
    assert quantum_gradient is not None
    assert action_gradient is not None

    assert torch.isfinite(projection_gradient).all()

    assert torch.isfinite(quantum_gradient).all()

    assert torch.isfinite(action_gradient).all()

    assert torch.linalg.vector_norm(quantum_gradient).item() > 0.0


def test_policy_optimizer_updates_quantum_weights() -> None:
    model = build_robotics_model(seed=42)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    observations = torch.randn(
        (
            4,
            6,
        )
    )

    before = model.quantum_circuit.weights.detach().clone()

    distribution = model.distribution(observations)

    loss = distribution.mean.pow(2).mean()

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    after = model.quantum_circuit.weights.detach()

    assert not torch.equal(
        before,
        after,
    )


def test_same_seed_same_quantum_initialization() -> None:
    first = build_robotics_model(seed=123)

    second = build_robotics_model(seed=123)

    torch.testing.assert_close(
        first.quantum_circuit.weights,
        second.quantum_circuit.weights,
    )


def test_wrong_observation_dimension_rejected() -> None:
    model = build_robotics_model()

    with pytest.raises(ValueError):
        model.distribution(
            torch.zeros(
                (
                    2,
                    5,
                )
            )
        )


def test_raw_action_log_probability_is_reproducible() -> None:
    torch.manual_seed(42)

    model = build_driving_model(seed=42)

    observation = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    (
        raw_action,
        original_log_probability,
        _,
    ) = model.sample_action(observation)

    distribution = model.distribution(observation)

    recomputed = distribution.log_prob(raw_action).sum(dim=-1)

    torch.testing.assert_close(
        original_log_probability,
        recomputed,
    )


def test_critic_does_not_require_quantum_circuit() -> None:
    model = build_robotics_model()

    observations = torch.randn(
        (
            2,
            6,
        ),
        dtype=torch.float32,
    )

    values = model.value(observations)

    assert values.shape == (2,)

    assert torch.isfinite(values).all()


def test_ppo_style_loss_updates_hybrid_actor() -> None:
    torch.manual_seed(42)

    model = build_robotics_model(seed=42)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    observations = torch.randn(
        (
            4,
            6,
        ),
        dtype=torch.float32,
    )

    distribution = model.distribution(observations)

    raw_actions = distribution.sample().detach()

    old_log_probabilities = distribution.log_prob(raw_actions).sum(dim=-1).detach()

    advantages = torch.tensor(
        [
            1.0,
            -0.5,
            0.25,
            0.75,
        ],
        dtype=torch.float32,
    )

    before = model.quantum_circuit.weights.detach().clone()

    new_distribution = model.distribution(observations)

    new_log_probabilities = new_distribution.log_prob(raw_actions).sum(dim=-1)

    ratios = torch.exp(new_log_probabilities - old_log_probabilities)

    loss = -(ratios * advantages).mean()

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    quantum_gradient = model.quantum_circuit.weights.grad

    assert quantum_gradient is not None

    assert torch.isfinite(quantum_gradient).all()

    assert torch.linalg.vector_norm(quantum_gradient).item() > 0.0

    optimizer.step()

    after = model.quantum_circuit.weights.detach()

    assert not torch.equal(
        before,
        after,
    )
