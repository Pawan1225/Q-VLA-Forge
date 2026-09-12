from __future__ import annotations

import torch

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    train_supervised,
    train_supervised_instrumented,
)
from q_vla_forge.utils.reproducibility import set_seed


def _datasets() -> tuple[list, list]:
    train = list(
        SyntheticDrivingDataset(
            size=64,
            seed=42,
        )
    )

    validation = list(
        SyntheticDrivingDataset(
            size=32,
            seed=1042,
        )
    )

    return train, validation


def _config() -> TrainingConfig:
    return TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=42,
    )


def test_instrumented_history_length() -> None:
    train, validation = _datasets()

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_config(),
    )

    assert len(result.history) == 3


def test_steps_are_cumulative() -> None:
    train, validation = _datasets()

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_config(),
    )

    assert [item.optimizer_steps for item in result.history] == [
        4,
        8,
        12,
    ]


def test_samples_are_cumulative() -> None:
    train, validation = _datasets()

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_config(),
    )

    assert [item.samples_processed for item in result.history] == [
        64,
        128,
        192,
    ]


def test_epoch_times_are_non_negative() -> None:
    train, validation = _datasets()

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_config(),
    )

    assert all(item.epoch_seconds >= 0.0 for item in result.history)


def test_cumulative_time_is_monotonic() -> None:
    train, validation = _datasets()

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_config(),
    )

    cumulative = [item.cumulative_seconds for item in result.history]

    assert cumulative == sorted(cumulative)


def test_optimizer_behavior_matches_frozen_trainer() -> None:
    train, validation = _datasets()
    config = _config()

    set_seed(42)
    original_model = SharedVLAModel()

    original_result = train_supervised(
        model=original_model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=config,
    )

    set_seed(42)
    instrumented_model = SharedVLAModel()

    instrumented_result = train_supervised_instrumented(
        model=instrumented_model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=config,
    )

    assert (
        original_result.initial_validation_loss
        == instrumented_result.initial_validation_loss
    )

    assert (
        original_result.best_validation_loss == instrumented_result.best_validation_loss
    )

    assert (
        original_result.final_validation_loss
        == instrumented_result.final_validation_loss
    )

    assert len(original_result.history) == len(instrumented_result.history)

    for original, instrumented in zip(
        original_result.history,
        instrumented_result.history,
        strict=True,
    ):
        assert original.epoch == instrumented.epoch
        assert original.train_loss == instrumented.train_loss
        assert original.validation_loss == instrumented.validation_loss
        assert original.learning_rate == instrumented.learning_rate

    for name, parameter in original_model.state_dict().items():
        assert torch.equal(
            parameter,
            instrumented_model.state_dict()[name],
        )


def test_full_protocol_step_count() -> None:
    train = list(
        SyntheticDrivingDataset(
            size=512,
            seed=42,
        )
    )

    validation = list(
        SyntheticDrivingDataset(
            size=32,
            seed=1042,
        )
    )

    set_seed(42)
    model = SharedVLAModel()

    result = train_supervised_instrumented(
        model=model,
        train_samples=train,
        validation_samples=validation,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=TrainingConfig(
            epochs=1,
            batch_size=32,
            seed=42,
        ),
    )

    assert result.optimizer_steps == 16
    assert result.samples_processed == 512
