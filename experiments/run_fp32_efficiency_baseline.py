from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    TrainingEfficiencySummary,
    build_target_definition,
    find_target_reach,
    train_supervised_instrumented,
)
from q_vla_forge.utils.reproducibility import (
    DEFAULT_SEEDS,
    set_seed,
)

OUTPUT_DIR = Path("results") / "training" / "fp32"

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32


def _dataset_class(
    domain: Domain,
):
    if domain == Domain.AUTONOMOUS_DRIVING:
        return SyntheticDrivingDataset

    if domain == Domain.ROBOTICS:
        return SyntheticRoboticsDataset

    raise ValueError(f"unsupported domain: {domain}")


def _run(
    domain: Domain,
    seed: int,
) -> TrainingEfficiencySummary:
    set_seed(seed)

    dataset_class = _dataset_class(domain)

    train_samples = list(
        dataset_class(
            size=TRAIN_SIZE,
            seed=seed,
        )
    )

    validation_samples = list(
        dataset_class(
            size=VALIDATION_SIZE,
            seed=seed + 1000,
        )
    )

    # Created now so the split itself is frozen,
    # even though test metrics are not the primary
    # output of Sprint 3.2.
    test_samples = list(
        dataset_class(
            size=TEST_SIZE,
            seed=seed + 2000,
        )
    )

    if len(test_samples) != TEST_SIZE:
        raise RuntimeError("unexpected test set size")

    model = SharedVLAModel()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    result = train_supervised_instrumented(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=domain,
        config=TrainingConfig(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_learning_rate=1e-5,
            seed=seed,
        ),
    )

    completed_best = min(item.validation_loss for item in result.history)

    completed_best_epoch = next(
        item.epoch for item in result.history if item.validation_loss == completed_best
    )

    target = build_target_definition(
        completed_best,
        tolerance_fraction=0.05,
    )

    target_reach = find_target_reach(
        result.history,
        target.target_validation_loss,
    )

    if not target_reach.reached_target:
        raise RuntimeError(
            "FP32 reference failed to reach " "its own paired convergence target"
        )

    return TrainingEfficiencySummary(
        domain=domain.value,
        method="shared_vla_fp32",
        seed=seed,
        trainable_parameters=parameter_count,
        effective_parameters=parameter_count,
        total_training_seconds=result.total_training_seconds,
        mean_epoch_seconds=result.mean_epoch_seconds,
        best_validation_loss=completed_best,
        best_epoch=completed_best_epoch,
        final_validation_loss=result.final_validation_loss,
        target=target,
        target_reach=target_reach,
        history=result.history,
    )


def _save(
    summary: TrainingEfficiencySummary,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    domain_label = (
        "driving" if summary.domain == Domain.AUTONOMOUS_DRIVING.value else "robotics"
    )

    path = OUTPUT_DIR / f"{domain_label}-fp32-seed-{summary.seed}.json"

    path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    seeds = tuple(DEFAULT_SEEDS)

    if seeds != (42, 123, 456):
        raise RuntimeError(f"unexpected default seeds: {seeds}")

    summaries: list[TrainingEfficiencySummary] = []

    for domain in (
        Domain.AUTONOMOUS_DRIVING,
        Domain.ROBOTICS,
    ):
        for seed in seeds:
            print()
            print(
                "Running:",
                domain.value,
                "seed",
                seed,
            )

            summary = _run(
                domain,
                seed,
            )

            path = _save(summary)

            summaries.append(summary)

            print(
                "  best validation:",
                f"{summary.best_validation_loss:.8f}",
            )

            print(
                "  best epoch:",
                summary.best_epoch,
            )

            print(
                "  target:",
                f"{summary.target.target_validation_loss:.8f}",
            )

            print(
                "  epoch to target:",
                summary.target_reach.epoch_to_target,
            )

            print(
                "  steps to target:",
                summary.target_reach.steps_to_target,
            )

            print(
                "  samples to target:",
                summary.target_reach.samples_to_target,
            )

            print(
                "  total seconds:",
                f"{summary.total_training_seconds:.3f}",
            )

            print(
                "  saved:",
                path,
            )

    if len(summaries) != 6:
        raise RuntimeError(f"expected 6 FP32 runs, found {len(summaries)}")

    targets = {
        (
            summary.domain,
            summary.seed,
        ): summary.target.target_validation_loss
        for summary in summaries
    }

    if len(targets) != 6:
        raise RuntimeError("expected six unique domain/seed targets")

    target_path = Path("results") / "training" / "fp32-targets.json"

    target_payload = {
        "protocol": "paired_fp32_best_validation_loss",
        "tolerance_fraction": 0.05,
        "seeds": list(seeds),
        "targets": [
            {
                "domain": summary.domain,
                "seed": summary.seed,
                "reference_best_validation_loss": (
                    summary.target.reference_best_validation_loss
                ),
                "target_validation_loss": (summary.target.target_validation_loss),
                "fp32_epoch_to_target": (summary.target_reach.epoch_to_target),
                "fp32_steps_to_target": (summary.target_reach.steps_to_target),
                "fp32_samples_to_target": (summary.target_reach.samples_to_target),
                "fp32_seconds_to_target": (summary.target_reach.seconds_to_target),
            }
            for summary in summaries
        ],
    }

    target_path.write_text(
        json.dumps(
            target_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=====================================")
    print(" FP32 CONVERGENCE BASELINE COMPLETE")
    print("=====================================")
    print("Runs:", len(summaries))
    print("Frozen targets:", target_path)


if __name__ == "__main__":
    main()
