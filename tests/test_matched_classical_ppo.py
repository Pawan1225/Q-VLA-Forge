from __future__ import annotations

from q_vla_forge.rl.ppo import DEFAULT_PPO_CONFIG


def test_ablation_budget_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.total_environment_steps == 20_000


def test_ablation_evaluation_frequency_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.evaluation_frequency_steps == 1_000


def test_ablation_gamma_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.gamma == 0.99


def test_ablation_gae_lambda_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.gae_lambda == 0.95


def test_ablation_clip_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.clip_coefficient == 0.20


def test_ablation_learning_rate_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.learning_rate == 3e-4


def test_ablation_rollout_size_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.rollout_steps == 256


def test_ablation_update_epochs_are_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.update_epochs == 10


def test_ablation_minibatch_size_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.minibatch_size == 64


def test_ablation_value_coefficient_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.value_coefficient == 0.50


def test_ablation_entropy_coefficient_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.entropy_coefficient == 0.01


def test_ablation_gradient_norm_is_frozen() -> None:
    assert DEFAULT_PPO_CONFIG.max_gradient_norm == 0.50
