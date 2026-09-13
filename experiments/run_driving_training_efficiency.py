from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
)
from q_vla_forge.evaluation.driving_baseline import (
    evaluate_driving_model,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TargetDefinition,
    TargetReachResult,
    TrainingConfig,
    TrainingEfficiencySummary,
    compare_target_efficiency,
    find_target_reach,
    train_supervised_instrumented,
)
from q_vla_forge.training.low_rank import (
    convert_model_to_trainable_svd,
    trainable_parameter_count,
)
from q_vla_forge.training.tensor_network import (
    convert_model_to_trainable_tt,
)
from q_vla_forge.utils.reproducibility import set_seed

SEED = 42

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32

SVD_RANK_FRACTION = 0.75
TT_MAX_RANK = 2
TT_ORDER = 3

MINIMUM_WEIGHT_PARAMETERS = 1024

RESULT_ROOT = Path("results") / "training"

OUTPUT_DIR = RESULT_ROOT / "driving"

FP32_RESULT = RESULT_ROOT / "fp32" / "driving-fp32-seed-42.json"

FP32_TARGETS = RESULT_ROOT / "fp32-targets.json"

SPRINT2_SELECTION = (
    Path("results") / "compression" / "driving-compression-selection.json"
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _validate_sprint2_selection() -> None:
    """Ensure the runner uses the frozen Sprint 2 driving selection."""
    payload = _load_json(SPRINT2_SELECTION)

    if payload["domain"] != Domain.AUTONOMOUS_DRIVING.value:
        raise RuntimeError("Sprint 2 selection has wrong domain")

    if payload["seed"] != SEED:
        raise RuntimeError("Sprint 2 selection has wrong seed")

    svd_label = payload["svd"]["selected"]["configuration_label"]

    tt_label = payload["tensor_network"]["selected"]["configuration_label"]

    expected_svd = f"SVD-{int(SVD_RANK_FRACTION * 100)}%"

    expected_tt = f"TT-rank-{TT_MAX_RANK}"

    if svd_label != expected_svd:
        raise RuntimeError(
            "SVD configuration does not match " f"Sprint 2 selection: {svd_label}"
        )

    if tt_label != expected_tt:
        raise RuntimeError(
            "TT configuration does not match " f"Sprint 2 selection: {tt_label}"
        )


def _load_fp32_target() -> TargetDefinition:
    payload = _load_json(FP32_TARGETS)

    matches = [
        item
        for item in payload["targets"]
        if (item["domain"] == Domain.AUTONOMOUS_DRIVING.value and item["seed"] == SEED)
    ]

    if len(matches) != 1:
        raise RuntimeError("expected exactly one driving " "seed-42 FP32 target")

    item = matches[0]

    reference_best = float(item["reference_best_validation_loss"])

    target_loss = float(item["target_validation_loss"])

    expected_target = reference_best * 1.05

    if abs(expected_target - target_loss) > 1e-12:
        raise RuntimeError("stored FP32 target violates " "frozen 5% rule")

    return TargetDefinition(
        reference_best_validation_loss=(reference_best),
        tolerance_fraction=0.05,
        target_validation_loss=(target_loss),
    )


def _load_fp32_reference() -> tuple[
    TrainingEfficiencySummary,
    TargetReachResult,
]:
    payload = _load_json(FP32_RESULT)

    if payload["seed"] != SEED:
        raise RuntimeError("FP32 reference seed mismatch")

    if payload["domain"] != Domain.AUTONOMOUS_DRIVING.value:
        raise RuntimeError("FP32 reference domain mismatch")

    target = TargetDefinition(**payload["target"])

    target_reach = TargetReachResult(**payload["target_reach"])

    if not target_reach.reached_target:
        raise RuntimeError("FP32 reference did not " "reach its target")

    summary = TrainingEfficiencySummary(
        domain=payload["domain"],
        method=payload["method"],
        seed=payload["seed"],
        trainable_parameters=(payload["trainable_parameters"]),
        effective_parameters=(payload["effective_parameters"]),
        total_training_seconds=(payload["total_training_seconds"]),
        mean_epoch_seconds=(payload["mean_epoch_seconds"]),
        best_validation_loss=(payload["best_validation_loss"]),
        best_epoch=(payload["best_epoch"]),
        final_validation_loss=(payload["final_validation_loss"]),
        target=target,
        target_reach=target_reach,
        history=(),
    )

    return (
        summary,
        target_reach,
    )


def _datasets() -> tuple[
    list,
    list,
    list,
]:
    train_samples = list(
        SyntheticDrivingDataset(
            size=TRAIN_SIZE,
            seed=SEED,
        )
    )

    validation_samples = list(
        SyntheticDrivingDataset(
            size=VALIDATION_SIZE,
            seed=SEED + 1000,
        )
    )

    test_samples = list(
        SyntheticDrivingDataset(
            size=TEST_SIZE,
            seed=SEED + 2000,
        )
    )

    return (
        train_samples,
        validation_samples,
        test_samples,
    )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=SEED,
    )


def _train_candidate(
    *,
    method: str,
    model,
    target: TargetDefinition,
    train_samples: list,
    validation_samples: list,
    test_samples: list,
    effective_parameters: int,
) -> dict[str, Any]:
    result = train_supervised_instrumented(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_training_config(),
    )

    target_reach = find_target_reach(
        result.history,
        target.target_validation_loss,
    )

    completed_best = min(item.validation_loss for item in result.history)

    best_epoch = next(
        item.epoch for item in result.history if item.validation_loss == completed_best
    )

    trainable_parameters = trainable_parameter_count(model)

    summary = TrainingEfficiencySummary(
        domain=(Domain.AUTONOMOUS_DRIVING.value),
        method=method,
        seed=SEED,
        trainable_parameters=(trainable_parameters),
        effective_parameters=(effective_parameters),
        total_training_seconds=(result.total_training_seconds),
        mean_epoch_seconds=(result.mean_epoch_seconds),
        best_validation_loss=(completed_best),
        best_epoch=best_epoch,
        final_validation_loss=(result.final_validation_loss),
        target=target,
        target_reach=target_reach,
        history=result.history,
    )

    test_metrics = evaluate_driving_model(
        model=model,
        samples=test_samples,
        batch_size=BATCH_SIZE,
    )

    return {
        "summary": asdict(summary),
        "test_metrics": asdict(test_metrics),
    }


def main() -> None:
    _validate_sprint2_selection()

    target = _load_fp32_target()

    (
        fp32_summary,
        fp32_target_reach,
    ) = _load_fp32_reference()

    if (
        abs(fp32_summary.target.target_validation_loss - target.target_validation_loss)
        > 1e-12
    ):
        raise RuntimeError("FP32 result and target " "registry disagree")

    (
        train_samples,
        validation_samples,
        test_samples,
    ) = _datasets()

    #
    # Trainable SVD
    #
    set_seed(SEED)

    svd_dense = SharedVLAModel()

    (
        svd_model,
        svd_report,
    ) = convert_model_to_trainable_svd(
        svd_dense,
        rank_fraction=(SVD_RANK_FRACTION),
        minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
    )

    svd_result = _train_candidate(
        method="trainable_svd",
        model=svd_model,
        target=target,
        train_samples=train_samples,
        validation_samples=(validation_samples),
        test_samples=test_samples,
        effective_parameters=(svd_report.structured_trainable_parameters),
    )

    #
    # Trainable TT/MPS
    #
    set_seed(SEED)

    tt_dense = SharedVLAModel()

    (
        tt_model,
        tt_report,
    ) = convert_model_to_trainable_tt(
        tt_dense,
        max_rank=TT_MAX_RANK,
        minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
        tensor_order=TT_ORDER,
    )

    tt_result = _train_candidate(
        method="trainable_tt_mps",
        model=tt_model,
        target=target,
        train_samples=train_samples,
        validation_samples=(validation_samples),
        test_samples=test_samples,
        effective_parameters=(tt_report.structured_trainable_parameters),
    )

    svd_target_reach = TargetReachResult(**svd_result["summary"]["target_reach"])

    tt_target_reach = TargetReachResult(**tt_result["summary"]["target_reach"])

    svd_comparison = compare_target_efficiency(
        domain=(Domain.AUTONOMOUS_DRIVING.value),
        seed=SEED,
        reference_method="shared_vla_fp32",
        candidate_method="trainable_svd",
        reference=fp32_target_reach,
        candidate=svd_target_reach,
    )

    tt_comparison = compare_target_efficiency(
        domain=(Domain.AUTONOMOUS_DRIVING.value),
        seed=SEED,
        reference_method="shared_vla_fp32",
        candidate_method="trainable_tt_mps",
        reference=fp32_target_reach,
        candidate=tt_target_reach,
    )

    baseline_parameters = fp32_summary.trainable_parameters

    payload = {
        "experiment_id": ("driving-training-efficiency-seed-42"),
        "domain": (Domain.AUTONOMOUS_DRIVING.value),
        "seed": SEED,
        "exploratory": True,
        "protocol": {
            "train_size": TRAIN_SIZE,
            "validation_size": (VALIDATION_SIZE),
            "test_size": TEST_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": ("CosineAnnealingLR"),
            "minimum_learning_rate": (1e-5),
            "loss": "MSE",
            "target_source": ("frozen paired FP32 " "seed-42 target"),
        },
        "target": asdict(target),
        "configurations": {
            "fp32": {
                "method": ("shared_vla_fp32"),
            },
            "svd": {
                "method": ("trainable_svd"),
                "rank_fraction": (SVD_RANK_FRACTION),
                "selection_source": ("Sprint 2 driving selection"),
            },
            "tt_mps": {
                "method": ("trainable_tt_mps"),
                "max_rank": TT_MAX_RANK,
                "tensor_order": TT_ORDER,
                "selection_source": ("Sprint 2 driving selection"),
                "quantum_inspired": True,
                "quantum_hardware_used": False,
            },
        },
        "fp32_reference": {
            "trainable_parameters": (baseline_parameters),
            "target_reach": asdict(fp32_target_reach),
            "best_validation_loss": (fp32_summary.best_validation_loss),
            "final_validation_loss": (fp32_summary.final_validation_loss),
        },
        "svd": {
            **svd_result,
            "conversion_report": asdict(svd_report),
            "comparison_vs_fp32": asdict(svd_comparison),
            "parameter_reduction_percent": (
                (
                    1.0
                    - (svd_report.structured_trainable_parameters / baseline_parameters)
                )
                * 100.0
            ),
        },
        "tt_mps": {
            **tt_result,
            "conversion_report": asdict(tt_report),
            "comparison_vs_fp32": asdict(tt_comparison),
            "parameter_reduction_percent": (
                (
                    1.0
                    - (tt_report.structured_trainable_parameters / baseline_parameters)
                )
                * 100.0
            ),
        },
        "limitations": [
            ("Single-seed exploratory result; " "no statistical superiority claim."),
            (
                "CPU wall-clock measurements are "
                "descriptive rather than primary evidence."
            ),
            (
                "TT/MPS forward reconstructs a temporary "
                "differentiable dense weight and is not "
                "a native optimized TT runtime."
            ),
            (
                "SVD and TT configurations were frozen "
                "from Sprint 2 rather than tuned on "
                "Sprint 3 convergence."
            ),
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "driving-training-efficiency-seed-42.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" Q-VLA FORGE — DRIVING TRAINING EFFICIENCY")
    print("============================================")

    print()
    print(
        "FP32 target:",
        f"{target.target_validation_loss:.8f}",
    )

    print()
    print("FP32")
    print(
        "  parameters:",
        baseline_parameters,
    )
    print(
        "  epoch to target:",
        fp32_target_reach.epoch_to_target,
    )
    print(
        "  steps to target:",
        fp32_target_reach.steps_to_target,
    )

    print()
    print("SVD")
    print(
        "  rank fraction:",
        SVD_RANK_FRACTION,
    )
    print(
        "  parameters:",
        (svd_report.structured_trainable_parameters),
    )
    print(
        "  target reached:",
        svd_target_reach.reached_target,
    )
    print(
        "  epoch to target:",
        svd_target_reach.epoch_to_target,
    )
    print(
        "  steps to target:",
        svd_target_reach.steps_to_target,
    )
    print(
        "  test MSE:",
        svd_result["test_metrics"]["test_mse"],
    )

    print()
    print("TT/MPS")
    print(
        "  max rank:",
        TT_MAX_RANK,
    )
    print(
        "  parameters:",
        (tt_report.structured_trainable_parameters),
    )
    print(
        "  target reached:",
        tt_target_reach.reached_target,
    )
    print(
        "  epoch to target:",
        tt_target_reach.epoch_to_target,
    )
    print(
        "  steps to target:",
        tt_target_reach.steps_to_target,
    )
    print(
        "  test MSE:",
        tt_result["test_metrics"]["test_mse"],
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
