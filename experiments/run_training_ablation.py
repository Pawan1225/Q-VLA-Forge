from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch import nn

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.evaluation.training_ablation import (
    MatchedParameterPair,
    ParameterBudget,
    select_closest_parameter_pair,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TargetDefinition,
    TargetReachResult,
    TrainingConfig,
    compare_target_efficiency,
    evaluate_supervised,
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
    set_seed,
)

SEED = 42

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32

MINIMUM_WEIGHT_PARAMETERS = 1024
TT_ORDER = 3

SVD_FRACTIONS = (
    0.25,
    0.50,
    0.75,
)

TT_RANKS = (
    2,
    4,
    8,
)

ROOT = Path("results") / "training"

OUTPUT_DIR = ROOT / "ablation"

TARGET_FILE = ROOT / "fp32-targets.json"


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _prefix(
    domain: Domain,
) -> str:
    if domain == Domain.AUTONOMOUS_DRIVING:
        return "driving"

    return "robotics"


def _fp32_path(
    domain: Domain,
) -> Path:
    return ROOT / "fp32" / (f"{_prefix(domain)}-" f"fp32-seed-{SEED}.json")


def _paired_target(
    domain: Domain,
) -> TargetDefinition:
    payload = _load_json(TARGET_FILE)

    matches = [
        item
        for item in payload["targets"]
        if (item["domain"] == domain.value and item["seed"] == SEED)
    ]

    if len(matches) != 1:
        raise RuntimeError("expected exactly one paired target")

    item = matches[0]

    reference_best = float(item["reference_best_validation_loss"])

    target_loss = float(item["target_validation_loss"])

    if abs(target_loss - reference_best * 1.05) > 1e-12:
        raise RuntimeError("paired target violates frozen 5% rule")

    return TargetDefinition(
        reference_best_validation_loss=(reference_best),
        tolerance_fraction=0.05,
        target_validation_loss=(target_loss),
    )


def _fp32_target_reach(
    domain: Domain,
) -> TargetReachResult:
    payload = _load_json(_fp32_path(domain))

    reach = TargetReachResult(**payload["target_reach"])

    if not reach.reached_target:
        raise RuntimeError("FP32 reference must reach target")

    return reach


def _common_layer_names() -> set[str]:
    """
    Return the common structural layer pool.

    Selection depends only on model structure and layer size.
    """
    set_seed(SEED)

    model = SharedVLAModel()

    names = {
        name
        for name, module in model.named_modules()
        if (
            isinstance(
                module,
                nn.Linear,
            )
            and module.weight.numel() >= MINIMUM_WEIGHT_PARAMETERS
        )
    }

    if not names:
        raise RuntimeError("no common ablation layers found")

    return names


def _svd_budget(
    *,
    fraction: float,
    selected_layers: set[str],
) -> tuple[
    ParameterBudget,
    Any,
]:
    set_seed(SEED)

    dense = SharedVLAModel()

    structured, report = convert_model_to_trainable_svd(
        dense,
        rank_fraction=fraction,
        minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
        selected_layer_names=(selected_layers),
    )

    replaced = {item.name for item in report.layers}

    if replaced != selected_layers:
        raise RuntimeError(
            "SVD candidate failed to replace " "the complete matched layer set"
        )

    count = trainable_parameter_count(structured)

    return (
        ParameterBudget(
            method="trainable_svd",
            configuration=(f"SVD-{int(fraction * 100)}%"),
            trainable_parameters=count,
        ),
        report,
    )


def _tt_budget(
    *,
    rank: int,
    selected_layers: set[str],
) -> tuple[
    ParameterBudget,
    Any,
]:
    set_seed(SEED)

    dense = SharedVLAModel()

    structured, report = convert_model_to_trainable_tt(
        dense,
        max_rank=rank,
        minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
        tensor_order=TT_ORDER,
        selected_layer_names=(selected_layers),
    )

    replaced = {item.name for item in report.layers}

    if replaced != selected_layers:
        raise RuntimeError(
            "TT candidate failed to replace " "the complete matched layer set"
        )

    count = trainable_parameter_count(structured)

    return (
        ParameterBudget(
            method="trainable_tt_mps",
            configuration=(f"TT-rank-{rank}"),
            trainable_parameters=count,
        ),
        report,
    )


def _select_matched_pair(
    selected_layers: set[str],
) -> tuple[
    MatchedParameterPair,
    dict[str, float],
]:
    svd_budgets = []

    for fraction in SVD_FRACTIONS:
        budget, _ = _svd_budget(
            fraction=fraction,
            selected_layers=selected_layers,
        )

        svd_budgets.append(budget)

    tt_budgets = []

    for rank in TT_RANKS:
        budget, _ = _tt_budget(
            rank=rank,
            selected_layers=selected_layers,
        )

        tt_budgets.append(budget)

    pair = select_closest_parameter_pair(
        svd_budgets,
        tt_budgets,
    )

    selected_svd_fraction = next(
        fraction
        for fraction in SVD_FRACTIONS
        if (f"SVD-{int(fraction * 100)}%" == pair.svd.configuration)
    )

    selected_tt_rank = next(
        rank for rank in TT_RANKS if (f"TT-rank-{rank}" == pair.tt_mps.configuration)
    )

    return (
        pair,
        {
            "svd_rank_fraction": (selected_svd_fraction),
            "tt_max_rank": float(selected_tt_rank),
        },
    )


def _datasets(
    domain: Domain,
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
                seed=SEED,
            )
        ),
        list(
            dataset_class(
                size=VALIDATION_SIZE,
                seed=SEED + 1000,
            )
        ),
        list(
            dataset_class(
                size=TEST_SIZE,
                seed=SEED + 2000,
            )
        ),
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


def _run_method(
    *,
    domain: Domain,
    method: str,
    selected_layers: set[str],
    svd_fraction: float,
    tt_rank: int,
    target: TargetDefinition,
    reference_reach: TargetReachResult,
    train_samples: list,
    validation_samples: list,
    test_samples: list,
) -> dict[str, Any]:
    set_seed(SEED)

    dense = SharedVLAModel()

    if method == "trainable_svd":
        structured, report = convert_model_to_trainable_svd(
            dense,
            rank_fraction=(svd_fraction),
            minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
            selected_layer_names=(selected_layers),
        )
    elif method == "trainable_tt_mps":
        structured, report = convert_model_to_trainable_tt(
            dense,
            max_rank=tt_rank,
            minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
            tensor_order=TT_ORDER,
            selected_layer_names=(selected_layers),
        )
    else:
        raise ValueError(f"unsupported method: {method}")

    replaced = {item.name for item in report.layers}

    if replaced != selected_layers:
        raise RuntimeError(f"{method} did not preserve " "the matched layer set")

    training = train_supervised_instrumented(
        model=structured,
        train_samples=train_samples,
        validation_samples=(validation_samples),
        domain=domain,
        config=_training_config(),
    )

    reach = find_target_reach(
        training.history,
        target.target_validation_loss,
    )

    comparison = compare_target_efficiency(
        domain=domain.value,
        seed=SEED,
        reference_method=("shared_vla_fp32"),
        candidate_method=method,
        reference=reference_reach,
        candidate=reach,
    )

    test_mse = evaluate_supervised(
        structured,
        test_samples,
        domain,
        batch_size=BATCH_SIZE,
    )

    completed_best = min(item.validation_loss for item in training.history)

    return {
        "method": method,
        "seed": SEED,
        "trainable_parameters": (trainable_parameter_count(structured)),
        "target_reach": asdict(reach),
        "comparison_vs_fp32": asdict(comparison),
        "best_validation_loss": (completed_best),
        "final_validation_loss": (training.final_validation_loss),
        "test_mse": float(test_mse),
        "history": [asdict(item) for item in training.history],
        "conversion_report": asdict(report),
    }


def _run_domain(
    domain: Domain,
) -> dict[str, Any]:
    selected_layers = _common_layer_names()

    (
        matched_pair,
        configuration,
    ) = _select_matched_pair(selected_layers)

    svd_fraction = float(configuration["svd_rank_fraction"])

    tt_rank = int(configuration["tt_max_rank"])

    target = _paired_target(domain)

    reference_reach = _fp32_target_reach(domain)

    (
        train_samples,
        validation_samples,
        test_samples,
    ) = _datasets(domain)

    svd = _run_method(
        domain=domain,
        method="trainable_svd",
        selected_layers=selected_layers,
        svd_fraction=svd_fraction,
        tt_rank=tt_rank,
        target=target,
        reference_reach=(reference_reach),
        train_samples=train_samples,
        validation_samples=(validation_samples),
        test_samples=test_samples,
    )

    tt = _run_method(
        domain=domain,
        method="trainable_tt_mps",
        selected_layers=selected_layers,
        svd_fraction=svd_fraction,
        tt_rank=tt_rank,
        target=target,
        reference_reach=(reference_reach),
        train_samples=train_samples,
        validation_samples=(validation_samples),
        test_samples=test_samples,
    )

    observed_difference = abs(svd["trainable_parameters"] - tt["trainable_parameters"])

    if observed_difference != matched_pair.absolute_parameter_difference:
        raise RuntimeError(
            "trained parameter budgets differ " "from frozen matched selection"
        )

    return {
        "domain": domain.value,
        "seed": SEED,
        "ablation_type": ("matched_layer_matched_parameter_budget"),
        "selection_uses_performance_data": False,
        "selected_layer_names": sorted(selected_layers),
        "matched_pair": asdict(matched_pair),
        "configuration": {
            "svd_rank_fraction": (svd_fraction),
            "tt_max_rank": (tt_rank),
            "tt_tensor_order": (TT_ORDER),
        },
        "paired_fp32_target": asdict(target),
        "fp32_target_reach": asdict(reference_reach),
        "svd": svd,
        "tt_mps": tt,
        "scientific_control": {
            "same_seed": True,
            "same_dense_initialization_rule": True,
            "same_selected_layers": True,
            "same_optimizer": True,
            "same_scheduler": True,
            "same_training_budget": True,
            "approximately_matched_parameters": True,
            "quantum_hardware_used": False,
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for domain in (
        Domain.AUTONOMOUS_DRIVING,
        Domain.ROBOTICS,
    ):
        payload = _run_domain(domain)

        path = OUTPUT_DIR / (f"{_prefix(domain)}-" "classical-vs-qi-ablation.json")

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        results.append(payload)

        print()
        print("============================================")
        print(domain.value)
        print("============================================")

        print(
            "Layers:",
            len(payload["selected_layer_names"]),
        )

        print(
            "Matched SVD:",
            payload["matched_pair"]["svd"]["configuration"],
        )

        print(
            "Matched TT:",
            payload["matched_pair"]["tt_mps"]["configuration"],
        )

        print(
            "Parameter difference:",
            (f"{payload['matched_pair']['relative_difference_percent']:.2f}%"),
        )

        print(
            "SVD target:",
            payload["svd"]["target_reach"],
        )

        print(
            "TT target:",
            payload["tt_mps"]["target_reach"],
        )

        print(
            "SVD test MSE:",
            payload["svd"]["test_mse"],
        )

        print(
            "TT test MSE:",
            payload["tt_mps"]["test_mse"],
        )

        print(
            "Saved:",
            path,
        )

    summary_path = OUTPUT_DIR / "classical-vs-qi-ablation-summary.json"

    summary_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "domains": results,
                "interpretation": (
                    "Classical SVD and quantum-inspired "
                    "TT/MPS were compared on the same "
                    "selected layers with approximately "
                    "matched trainable parameter budgets. "
                    "Configuration matching used parameter "
                    "counts only, not performance outcomes."
                ),
                "claim_control": {
                    "quantum_advantage": False,
                    "quantum_speedup": False,
                    "quantum_hardware_used": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" CLASSICAL VS QI ABLATION COMPLETE")
    print("============================================")

    print(
        "Saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
