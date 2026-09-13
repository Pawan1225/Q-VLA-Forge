from __future__ import annotations

import numpy as np
import torch

from q_vla_forge.rl.ppo import (
    DEFAULT_PPO_CONFIG,
    GaussianActorCritic,
    count_actor_parameters,
    count_critic_parameters,
)


def build_model() -> GaussianActorCritic:
    return GaussianActorCritic(
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
    )


def test_frozen_ppo_budget() -> None:
    assert DEFAULT_PPO_CONFIG.total_environment_steps == 20_000

    assert DEFAULT_PPO_CONFIG.evaluation_frequency_steps == 1_000

    assert DEFAULT_PPO_CONFIG.evaluation_episodes == 20


def test_actor_output_shape() -> None:
    model = build_model()

    observation = torch.zeros(
        (
            2,
            4,
        )
    )

    distribution = model.distribution(observation)

    assert distribution.mean.shape == (
        2,
        3,
    )


def test_value_output_shape() -> None:
    model = build_model()

    observation = torch.zeros(
        (
            2,
            4,
        )
    )

    value = model.value(observation)

    assert value.shape == (2,)


def test_scaled_action_respects_bounds() -> None:
    model = build_model()

    raw = torch.tensor(
        [
            [
                100.0,
                -100.0,
                0.0,
            ]
        ]
    )

    action = model.scale_raw_action(raw)[0].detach().numpy()

    assert -1.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0
    assert 0.0 <= action[2] <= 1.0


def test_parameter_accounting() -> None:
    model = build_model()

    actor = count_actor_parameters(model)

    critic = count_critic_parameters(model)

    total = sum(parameter.numel() for parameter in model.parameters())

    assert actor > 0
    assert critic > 0
    assert actor + critic == total


def test_deterministic_action_is_repeatable() -> None:
    model = build_model()

    observation = torch.zeros(
        (
            1,
            4,
        )
    )

    first = model.deterministic_action(observation)

    second = model.deterministic_action(observation)

    torch.testing.assert_close(
        first,
        second,
    )


def test_sampled_log_probability_can_be_recomputed() -> None:
    model = build_model()

    observation = torch.zeros(
        (
            1,
            4,
        )
    )

    (
        raw_action,
        sampled_log_probability,
        _,
    ) = model.sample_action(observation)

    distribution = model.distribution(observation)

    recomputed = distribution.log_prob(raw_action).sum(dim=-1)

    torch.testing.assert_close(
        sampled_log_probability,
        recomputed,
    )


def test_raw_action_transformation_preserves_shape() -> None:
    model = build_model()

    raw = torch.zeros(
        (
            5,
            3,
        )
    )

    transformed = model.scale_raw_action(raw)

    assert transformed.shape == (
        5,
        3,
    )
