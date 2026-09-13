from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.evaluation.driving_baseline import (
    evaluate_driving_model,
)
from q_vla_forge.evaluation.robotics_baseline import (
    evaluate_robotics_model,
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
from q_vla_forge.utils.reproducibility import (
    DEFAULT_SEEDS,
    set_seed,
)

SEEDS = tuple(DEFAULT_SEEDS)

EXPECTED_SEEDS = (
    42,
    123,
    456,
)

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32

MINIMUM_WEIGHT_PARAMETERS = 1024
TT_ORDER = 3

DOMAIN_CONFIGS = {
    Domain.AUTONOMOUS_DRIVING: {
        "svd_rank_fraction": 0.75,
        "tt_max_rank": 2,
    },
    Domain.ROBOTICS: {
        "svd_rank_fraction": 0.50,
        "tt_max_rank": 2,
    },
}

ROOT = Path("results") / "training"

VALIDATION_DIR = ROOT / "validation"

TARGET_FILE = ROOT / "fp32-targets.json"


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _target_for(
    domain: Domain,
    seed: int,
) -> TargetDefinition:
    payload = _load_json(TARGET_FILE)

    matches = [
        item
        for item in payload["targets"]
        if (item["domain"] == domain.value and item["seed"] == seed)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one paired FP32 target " f"for {domain.value} seed {seed}"
        )

    item = matches[0]

    reference_best = float(item["reference_best_validation_loss"])

    target_loss = float(item["target_validation_loss"])

    if abs(target_loss - (reference_best * 1.05)) > 1e-12:
        raise RuntimeError("FP32 target violates frozen 5% rule")

    return TargetDefinition(
        reference_best_validation_loss=(reference_best),
        tolerance_fraction=0.05,
        target_validation_loss=(target_loss),
    )


def _fp32_path(
    domain: Domain,
    seed: int,
) -> Path:
    prefix = "driving" if domain == Domain.AUTONOMOUS_DRIVING else "robotics"

    return ROOT / "fp32" / f"{prefix}-fp32-seed-{seed}.json"


def _load_fp32_reach(
    domain: Domain,
    seed: int,
) -> tuple[
    dict[str, Any],
    TargetReachResult,
]:
    payload = _load_json(
        _fp32_path(
            domain,
            seed,
        )
    )

    if payload["domain"] != domain.value:
        raise RuntimeError("FP32 domain mismatch")

    if payload["seed"] != seed:
        raise RuntimeError("FP32 seed mismatch")

    reach = TargetReachResult(**payload["target_reach"])

    if not reach.reached_target:
        raise RuntimeError("FP32 reference must reach its paired target")

    return (
        payload,
        reach,
    )


def _datasets(
    domain: Domain,
    seed: int,
) -> tuple[
    list,
    list,
    list,
]:
    dataset_class = (
        SyntheticDrivingDataset
        if domain == Domain.AUTONOMOUS_DRIVING
        else SyntheticRoboticsDataset
    )

    return (
        list(
            dataset_class(
                size=TRAIN_SIZE,
                seed=seed,
            )
        ),
        list(
            dataset_class(
                size=VALIDATION_SIZE,
                seed=seed + 1000,
            )
        ),
        list(
            dataset_class(
                size=TEST_SIZE,
                seed=seed + 2000,
            )
        ),
    )


def _config(
    seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=seed,
    )


def _evaluate_test(
    *,
    domain: Domain,
    model,
    test_samples: list,
) -> dict[str, Any]:
    if domain == Domain.AUTONOMOUS_DRIVING:
        metrics = evaluate_driving_model(
            model=model,
            samples=test_samples,
            batch_size=BATCH_SIZE,
        )
    else:
        metrics = evaluate_robotics_model(
            model=model,
            samples=test_samples,
            batch_size=BATCH_SIZE,
        )

    return asdict(metrics)


def _train_candidate(
    *,
    domain: Domain,
    seed: int,
    method: str,
    model,
    target: TargetDefinition,
    fp32_reach: TargetReachResult,
    train_samples: list,
    validation_samples: list,
    test_samples: list,
    conversion_report: Any,
) -> dict[str, Any]:
    result = train_supervised_instrumented(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=domain,
        config=_config(seed),
    )

    reach = find_target_reach(
        result.history,
        target.target_validation_loss,
    )

    completed_best = min(item.validation_loss for item in result.history)

    best_epoch = next(
        item.epoch for item in result.history if item.validation_loss == completed_best
    )

    parameter_count = trainable_parameter_count(model)

    summary = TrainingEfficiencySummary(
        domain=domain.value,
        method=method,
        seed=seed,
        trainable_parameters=(parameter_count),
        effective_parameters=(parameter_count),
        total_training_seconds=(result.total_training_seconds),
        mean_epoch_seconds=(result.mean_epoch_seconds),
        best_validation_loss=(completed_best),
        best_epoch=best_epoch,
        final_validation_loss=(result.final_validation_loss),
        target=target,
        target_reach=reach,
        history=result.history,
    )

    comparison = compare_target_efficiency(
        domain=domain.value,
        seed=seed,
        reference_method="shared_vla_fp32",
        candidate_method=method,
        reference=fp32_reach,
        candidate=reach,
    )

    return {
        "domain": domain.value,
        "method": method,
        "seed": seed,
        "summary": asdict(summary),
        "test_metrics": _evaluate_test(
            domain=domain,
            model=model,
            test_samples=test_samples,
        ),
        "comparison_vs_fp32": asdict(comparison),
        "conversion_report": asdict(conversion_report),
    }


def _output_path(
    domain: Domain,
    method: str,
    seed: int,
) -> Path:
    prefix = "driving" if domain == Domain.AUTONOMOUS_DRIVING else "robotics"

    method_label = "svd" if method == "trainable_svd" else "tt-mps"

    return VALIDATION_DIR / (f"{prefix}-" f"{method_label}-" f"seed-{seed}.json")


def _seed42_source(
    domain: Domain,
) -> Path:
    if domain == Domain.AUTONOMOUS_DRIVING:
        return ROOT / "driving" / "driving-training-efficiency-seed-42.json"

    return ROOT / "robotics" / "robotics-training-efficiency-seed-42.json"


def _reuse_seed42(
    *,
    domain: Domain,
    method: str,
) -> dict[str, Any]:
    payload = _load_json(_seed42_source(domain))

    section_name = "svd" if method == "trainable_svd" else "tt_mps"

    section = payload[section_name]

    normalized = {
        "domain": domain.value,
        "method": method,
        "seed": 42,
        "summary": section["summary"],
        "test_metrics": section["test_metrics"],
        "comparison_vs_fp32": section["comparison_vs_fp32"],
        "conversion_report": section["conversion_report"],
        "source": (str(_seed42_source(domain))),
        "reused_seed42": True,
    }

    return normalized


def _save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_new_seed(
    *,
    domain: Domain,
    seed: int,
    method: str,
) -> dict[str, Any]:
    target = _target_for(
        domain,
        seed,
    )

    (
        fp32_payload,
        fp32_reach,
    ) = _load_fp32_reach(
        domain,
        seed,
    )

    if (
        abs(
            fp32_payload["target"]["target_validation_loss"]
            - target.target_validation_loss
        )
        > 1e-12
    ):
        raise RuntimeError("FP32 target registry mismatch")

    (
        train_samples,
        validation_samples,
        test_samples,
    ) = _datasets(
        domain,
        seed,
    )

    set_seed(seed)

    dense_model = SharedVLAModel()

    domain_config = DOMAIN_CONFIGS[domain]

    if method == "trainable_svd":
        (
            structured_model,
            conversion_report,
        ) = convert_model_to_trainable_svd(
            dense_model,
            rank_fraction=(domain_config["svd_rank_fraction"]),
            minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
        )
    elif method == "trainable_tt_mps":
        (
            structured_model,
            conversion_report,
        ) = convert_model_to_trainable_tt(
            dense_model,
            max_rank=(domain_config["tt_max_rank"]),
            minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
            tensor_order=TT_ORDER,
        )
    else:
        raise ValueError(f"unsupported method: {method}")

    return _train_candidate(
        domain=domain,
        seed=seed,
        method=method,
        model=structured_model,
        target=target,
        fp32_reach=fp32_reach,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        conversion_report=(conversion_report),
    )


def main() -> None:
    if SEEDS != EXPECTED_SEEDS:
        raise RuntimeError(f"expected seeds {EXPECTED_SEEDS}, " f"found {SEEDS}")

    methods = (
        "trainable_svd",
        "trainable_tt_mps",
    )

    domains = (
        Domain.AUTONOMOUS_DRIVING,
        Domain.ROBOTICS,
    )

    produced = []

    for domain in domains:
        for method in methods:
            for seed in SEEDS:
                path = _output_path(
                    domain,
                    method,
                    seed,
                )

                print()
                print(
                    "Running:",
                    domain.value,
                    method,
                    "seed",
                    seed,
                )

                if seed == 42:
                    payload = _reuse_seed42(
                        domain=domain,
                        method=method,
                    )
                else:
                    payload = _run_new_seed(
                        domain=domain,
                        seed=seed,
                        method=method,
                    )

                    payload["reused_seed42"] = False

                _save(
                    payload,
                    path,
                )

                produced.append(path)

                reach = payload["summary"]["target_reach"]

                print(
                    "  target reached:",
                    reach["reached_target"],
                )

                print(
                    "  epoch:",
                    reach["epoch_to_target"],
                )

                print(
                    "  steps:",
                    reach["steps_to_target"],
                )

                print(
                    "  saved:",
                    path,
                )

    if len(produced) != 12:
        raise RuntimeError(
            f"expected 12 structured validation " f"artifacts, found {len(produced)}"
        )

    print()
    print("============================================")
    print(" THREE-SEED TRAINING VALIDATION COMPLETE")
    print("============================================")

    print(
        "Structured validation files:",
        len(produced),
    )

    print(
        "New training runs:",
        8,
    )

    print(
        "Reused seed-42 structured runs:",
        4,
    )

    print(
        "FP32 references reused:",
        6,
    )


if __name__ == "__main__":
    main()
