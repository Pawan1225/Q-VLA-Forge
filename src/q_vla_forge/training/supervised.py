from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from q_vla_forge.data import Domain, TaskSample
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.utils.reproducibility import set_seed


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for supervised VLA baseline training."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")

        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")

        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must not be negative")

        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured after one training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    """Complete result of one supervised training run."""

    history: tuple[EpochMetrics, ...]
    initial_validation_loss: float
    best_validation_loss: float
    final_validation_loss: float


@dataclass(frozen=True)
class TensorBatch:
    """Tensor representation of a homogeneous-domain sample batch."""

    visual: torch.Tensor
    state: torch.Tensor
    language_goals: tuple[str, ...]
    targets: torch.Tensor


def build_tensor_batch(
    samples: Sequence[TaskSample],
    device: torch.device,
) -> TensorBatch:
    """Convert task samples into model-ready tensors."""
    if not samples:
        raise ValueError("samples must not be empty")

    visual = torch.stack(
        [torch.from_numpy(sample.observation.visual) for sample in samples]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    state = torch.stack(
        [torch.from_numpy(sample.observation.state) for sample in samples]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    targets = torch.stack(
        [torch.from_numpy(sample.target_action.values) for sample in samples]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    language_goals = tuple(sample.observation.language_goal for sample in samples)

    return TensorBatch(
        visual=visual,
        state=state,
        language_goals=language_goals,
        targets=targets,
    )


def _validate_dataset_domain(
    samples: Sequence[TaskSample],
    domain: Domain,
) -> None:
    if not samples:
        raise ValueError("dataset must not be empty")

    if any(sample.domain != domain for sample in samples):
        raise ValueError("all samples must match the requested domain")


def evaluate_supervised(
    model: SharedVLAModel,
    samples: Sequence[TaskSample],
    domain: Domain,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> float:
    """Evaluate mean action MSE over a dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    _validate_dataset_domain(
        samples,
        domain,
    )

    active_device = device if device is not None else torch.device("cpu")

    model.eval()

    criterion = nn.MSELoss(reduction="sum")

    total_squared_error = 0.0
    total_action_values = 0

    with torch.no_grad():
        for start in range(
            0,
            len(samples),
            batch_size,
        ):
            batch_samples = samples[start : start + batch_size]

            batch = build_tensor_batch(
                batch_samples,
                active_device,
            )

            prediction = model(
                batch.visual,
                batch.state,
                batch.language_goals,
                domain,
            )

            total_squared_error += criterion(
                prediction,
                batch.targets,
            ).item()

            total_action_values += batch.targets.numel()

    return total_squared_error / total_action_values


def train_supervised(
    model: SharedVLAModel,
    train_samples: Sequence[TaskSample],
    validation_samples: Sequence[TaskSample],
    domain: Domain,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
) -> TrainingResult:
    """Train the VLA baseline with AdamW and cosine scheduling."""
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

    history: list[EpochMetrics] = []

    best_validation_loss = initial_validation_loss

    generator = torch.Generator()
    generator.manual_seed(active_config.seed)

    for epoch in range(
        1,
        active_config.epochs + 1,
    ):
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

            batch_size = len(batch_samples)

            total_train_loss += loss.item() * batch_size

            total_train_items += batch_size

        scheduler.step()

        train_loss = total_train_loss / total_train_items

        validation_loss = evaluate_supervised(
            model=model,
            samples=validation_samples,
            domain=domain,
            batch_size=active_config.batch_size,
            device=active_device,
        )

        best_validation_loss = min(
            best_validation_loss,
            validation_loss,
        )

        learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(learning_rate),
            )
        )

    return TrainingResult(
        history=tuple(history),
        initial_validation_loss=(initial_validation_loss),
        best_validation_loss=(best_validation_loss),
        final_validation_loss=(history[-1].validation_loss),
    )
