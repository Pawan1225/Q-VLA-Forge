"""Instrumented supervised training for Sprint 3 efficiency analysis."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from q_vla_forge.data import Domain, TaskSample
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training.efficiency import EpochEfficiencyRecord
from q_vla_forge.training.supervised import (
    TrainingConfig,
    build_tensor_batch,
    evaluate_supervised,
)
from q_vla_forge.utils.reproducibility import set_seed


@dataclass(frozen=True)
class InstrumentedTrainingResult:
    """Complete instrumented supervised training result."""

    history: tuple[EpochEfficiencyRecord, ...]
    initial_validation_loss: float
    best_validation_loss: float
    best_epoch: int
    final_validation_loss: float
    total_training_seconds: float
    mean_epoch_seconds: float
    optimizer_steps: int
    samples_processed: int


def _validate_dataset_domain(
    samples: Sequence[TaskSample],
    domain: Domain,
) -> None:
    if not samples:
        raise ValueError("dataset must not be empty")

    if any(sample.domain != domain for sample in samples):
        raise ValueError("all samples must match the requested domain")


def train_supervised_instrumented(
    model: SharedVLAModel,
    train_samples: Sequence[TaskSample],
    validation_samples: Sequence[TaskSample],
    domain: Domain,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
) -> InstrumentedTrainingResult:
    """
    Train with the frozen Sprint 1 protocol while collecting
    Sprint 3 efficiency measurements.

    Optimization behavior intentionally mirrors train_supervised().
    """
    active_config = config if config is not None else TrainingConfig()

    _validate_dataset_domain(
        train_samples,
        domain,
    )

    _validate_dataset_domain(
        validation_samples,
        domain,
    )

    set_seed(active_config.seed)

    active_device = device if device is not None else torch.device("cpu")

    model.to(active_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=active_config.epochs,
        eta_min=active_config.min_learning_rate,
    )

    criterion = nn.MSELoss()

    initial_validation_loss = evaluate_supervised(
        model=model,
        samples=validation_samples,
        domain=domain,
        batch_size=active_config.batch_size,
        device=active_device,
    )

    history: list[EpochEfficiencyRecord] = []

    best_validation_loss = initial_validation_loss
    best_epoch = 0

    generator = torch.Generator()
    generator.manual_seed(active_config.seed)

    cumulative_seconds = 0.0
    optimizer_steps = 0
    samples_processed = 0

    for epoch in range(
        1,
        active_config.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        model.train()

        permutation = torch.randperm(
            len(train_samples),
            generator=generator,
        ).tolist()

        total_train_loss = 0.0
        total_train_items = 0

        for start in range(
            0,
            len(permutation),
            active_config.batch_size,
        ):
            batch_indices = permutation[start : start + active_config.batch_size]

            batch_samples = [train_samples[index] for index in batch_indices]

            batch = build_tensor_batch(
                batch_samples,
                active_device,
            )

            optimizer.zero_grad(set_to_none=True)

            prediction = model(
                batch.visual,
                batch.state,
                batch.language_goals,
                domain,
            )

            loss = criterion(
                prediction,
                batch.targets,
            )

            loss.backward()

            optimizer.step()

            actual_batch_size = len(batch_samples)

            total_train_loss += loss.item() * actual_batch_size
            total_train_items += actual_batch_size

            optimizer_steps += 1
            samples_processed += actual_batch_size

        scheduler.step()

        train_loss = total_train_loss / total_train_items

        validation_loss = evaluate_supervised(
            model=model,
            samples=validation_samples,
            domain=domain,
            batch_size=active_config.batch_size,
            device=active_device,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

        learning_rate = float(optimizer.param_groups[0]["lr"])

        epoch_seconds = time.perf_counter() - epoch_start
        cumulative_seconds += epoch_seconds

        history.append(
            EpochEfficiencyRecord(
                epoch=epoch,
                train_loss=float(train_loss),
                validation_loss=float(validation_loss),
                learning_rate=learning_rate,
                epoch_seconds=float(epoch_seconds),
                cumulative_seconds=float(cumulative_seconds),
                optimizer_steps=optimizer_steps,
                samples_processed=samples_processed,
            )
        )

    epoch_durations = [item.epoch_seconds for item in history]

    return InstrumentedTrainingResult(
        history=tuple(history),
        initial_validation_loss=float(initial_validation_loss),
        best_validation_loss=float(best_validation_loss),
        best_epoch=best_epoch,
        final_validation_loss=float(history[-1].validation_loss),
        total_training_seconds=float(cumulative_seconds),
        mean_epoch_seconds=float(statistics.mean(epoch_durations)),
        optimizer_steps=optimizer_steps,
        samples_processed=samples_processed,
    )
