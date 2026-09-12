from __future__ import annotations

import math

import pytest
import torch

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    build_tensor_batch,
    evaluate_supervised,
    train_supervised,
)
from q_vla_forge.utils.reproducibility import set_seed


def test_default_training_configuration() -> None:
    config = TrainingConfig()

    assert config.epochs == 20
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.min_learning_rate == 1e-5
    assert config.seed == 42


def test_build_driving_tensor_batch() -> None:
    dataset = SyntheticDrivingDataset(
        size=4,
        seed=42,
    )

    batch = build_tensor_batch(
        list(dataset),
        torch.device("cpu"),
    )

    assert batch.visual.shape == (
        4,
        3,
        32,
        32,
    )

    assert batch.state.shape == (
        4,
        4,
    )

    assert batch.targets.shape == (
        4,
        3,
    )

    assert len(batch.language_goals) == 4


def test_build_robotics_tensor_batch() -> None:
    dataset = SyntheticRoboticsDataset(
        size=4,
        seed=42,
    )

    batch = build_tensor_batch(
        list(dataset),
        torch.device("cpu"),
    )

    assert batch.visual.shape == (
        4,
        3,
        32,
        32,
    )

    assert batch.state.shape == (
        4,
        6,
    )

    assert batch.targets.shape == (
        4,
        3,
    )


def test_evaluate_driving_returns_finite_loss() -> None:
    set_seed(42)

    model = SharedVLAModel()

    dataset = SyntheticDrivingDataset(
        size=8,
        seed=42,
    )

    loss = evaluate_supervised(
        model,
        list(dataset),
        Domain.AUTONOMOUS_DRIVING,
        batch_size=4,
    )

    assert math.isfinite(loss)
    assert loss >= 0.0


def test_evaluate_robotics_returns_finite_loss() -> None:
    set_seed(42)

    model = SharedVLAModel()

    dataset = SyntheticRoboticsDataset(
        size=8,
        seed=42,
    )

    loss = evaluate_supervised(
        model,
        list(dataset),
        Domain.ROBOTICS,
        batch_size=4,
    )

    assert math.isfinite(loss)
    assert loss >= 0.0


def test_training_produces_epoch_history() -> None:
    set_seed(42)

    model = SharedVLAModel()

    train_dataset = SyntheticDrivingDataset(
        size=16,
        seed=42,
    )

    validation_dataset = SyntheticDrivingDataset(
        size=8,
        seed=123,
    )

    result = train_supervised(
        model=model,
        train_samples=list(train_dataset),
        validation_samples=list(validation_dataset),
        domain=Domain.AUTONOMOUS_DRIVING,
        config=TrainingConfig(
            epochs=2,
            batch_size=8,
            seed=42,
        ),
    )

    assert len(result.history) == 2

    assert math.isfinite(result.initial_validation_loss)

    assert math.isfinite(result.final_validation_loss)

    assert math.isfinite(result.best_validation_loss)


def test_cosine_learning_rate_changes() -> None:
    set_seed(42)

    model = SharedVLAModel()

    train_dataset = SyntheticDrivingDataset(
        size=16,
        seed=42,
    )

    validation_dataset = SyntheticDrivingDataset(
        size=8,
        seed=123,
    )

    result = train_supervised(
        model=model,
        train_samples=list(train_dataset),
        validation_samples=list(validation_dataset),
        domain=Domain.AUTONOMOUS_DRIVING,
        config=TrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            seed=42,
        ),
    )

    learning_rates = [item.learning_rate for item in result.history]

    assert learning_rates[0] != learning_rates[-1]


def test_training_rejects_wrong_domain() -> None:
    model = SharedVLAModel()

    dataset = SyntheticDrivingDataset(
        size=4,
        seed=42,
    )

    with pytest.raises(ValueError):
        train_supervised(
            model=model,
            train_samples=list(dataset),
            validation_samples=list(dataset),
            domain=Domain.ROBOTICS,
            config=TrainingConfig(
                epochs=1,
            ),
        )


def test_training_config_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            epochs=0,
        )


def test_training_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            batch_size=0,
        )


def test_training_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            learning_rate=0.0,
        )


def test_training_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            weight_decay=-1.0,
        )


def test_training_config_rejects_invalid_minimum_lr() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            learning_rate=1e-4,
            min_learning_rate=1e-3,
        )
