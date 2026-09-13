from __future__ import annotations

from q_vla_forge.rl.ppo import (
    DEFAULT_PPO_CONFIG,
)


def test_hybrid_ppo_uses_frozen_budget() -> None:
    assert DEFAULT_PPO_CONFIG.total_environment_steps == 20_000


def test_hybrid_ppo_uses_frozen_evaluation_frequency() -> None:
    assert DEFAULT_PPO_CONFIG.evaluation_frequency_steps == 1_000


def test_hybrid_ppo_uses_frozen_evaluation_episode_count() -> None:
    assert DEFAULT_PPO_CONFIG.evaluation_episodes == 20


def test_hybrid_ppo_frozen_core_hyperparameters() -> None:
    config = DEFAULT_PPO_CONFIG

    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_coefficient == 0.20
    assert config.learning_rate == 3e-4
    assert config.rollout_steps == 256
    assert config.update_epochs == 10
    assert config.minibatch_size == 64
    assert config.value_coefficient == 0.50
    assert config.entropy_coefficient == 0.01
    assert config.max_gradient_norm == 0.50
