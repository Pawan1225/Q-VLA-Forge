from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.training_validation import (
    MethodValidationSummary,
    build_reach_summary,
    mean_std,
    optional_mean_std,
    robust_ten_percent_efficiency,
)

SEEDS = (
    42,
    123,
    456,
)

ROOT = Path("results") / "training"

VALIDATION_DIR = ROOT / "validation"

OUTPUT = ROOT / "training-validation-summary.json"


def _load(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _structured_files(
    domain: str,
    method: str,
) -> list[dict[str, Any]]:
    matches = []

    for path in sorted(VALIDATION_DIR.glob("*.json")):
        payload = _load(path)

        if payload["domain"] == domain and payload["method"] == method:
            matches.append(payload)

    matches.sort(key=lambda item: item["seed"])

    seeds = tuple(item["seed"] for item in matches)

    if seeds != SEEDS:
        raise RuntimeError(f"{domain}/{method} has seeds {seeds}")

    return matches


def _test_mse(
    item: dict[str, Any],
) -> float:
    return float(item["test_metrics"]["test_mse"])


def _test_mae(
    item: dict[str, Any],
) -> float:
    return float(item["test_metrics"]["test_mae"])


def _summarize(
    domain: str,
    method: str,
) -> MethodValidationSummary:
    runs = _structured_files(
        domain,
        method,
    )

    reaches = [item["summary"]["target_reach"]["reached_target"] for item in runs]

    reach_summary = build_reach_summary(reaches)

    epoch_reduction = optional_mean_std(
        [item["comparison_vs_fp32"]["epoch_reduction_percent"] for item in runs]
    )

    step_reduction = optional_mean_std(
        [item["comparison_vs_fp32"]["step_reduction_percent"] for item in runs]
    )

    sample_reduction = optional_mean_std(
        [item["comparison_vs_fp32"]["sample_reduction_percent"] for item in runs]
    )

    wall_reduction = optional_mean_std(
        [item["comparison_vs_fp32"]["wall_time_reduction_percent"] for item in runs]
    )

    return MethodValidationSummary(
        domain=domain,
        method=method,
        seeds=SEEDS,
        target_reach=(reach_summary),
        trainable_parameters=(
            mean_std([item["summary"]["trainable_parameters"] for item in runs])
        ),
        effective_parameters=(
            mean_std([item["summary"]["effective_parameters"] for item in runs])
        ),
        best_validation_loss=(
            mean_std([item["summary"]["best_validation_loss"] for item in runs])
        ),
        final_validation_loss=(
            mean_std([item["summary"]["final_validation_loss"] for item in runs])
        ),
        test_mse=(mean_std([_test_mse(item) for item in runs])),
        test_mae=(mean_std([_test_mae(item) for item in runs])),
        epoch_to_target=(
            optional_mean_std(
                [item["summary"]["target_reach"]["epoch_to_target"] for item in runs]
            )
        ),
        steps_to_target=(
            optional_mean_std(
                [item["summary"]["target_reach"]["steps_to_target"] for item in runs]
            )
        ),
        samples_to_target=(
            optional_mean_std(
                [item["summary"]["target_reach"]["samples_to_target"] for item in runs]
            )
        ),
        seconds_to_target=(
            optional_mean_std(
                [item["summary"]["target_reach"]["seconds_to_target"] for item in runs]
            )
        ),
        epoch_reduction_percent=(epoch_reduction),
        step_reduction_percent=(step_reduction),
        sample_reduction_percent=(sample_reduction),
        wall_time_reduction_percent=(wall_reduction),
        robust_ten_percent_step_efficiency=(
            robust_ten_percent_efficiency(
                reach_summary=(reach_summary),
                step_reduction=(step_reduction),
            )
        ),
    )


def main() -> None:
    summaries = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "trainable_svd",
            "trainable_tt_mps",
        ):
            summaries.append(
                _summarize(
                    domain,
                    method,
                )
            )

    payload = {
        "seeds": list(SEEDS),
        "statistics": {
            "center": "arithmetic_mean",
            "spread": ("sample_standard_deviation"),
        },
        "primary_efficiency_metric": ("optimizer_steps_to_paired_fp32_target"),
        "wall_clock_interpretation": ("descriptive"),
        "strong_efficiency_rule": {
            "all_three_seeds_must_reach_target": True,
            "mean_step_reduction_percent_at_least": 10.0,
        },
        "methods": [asdict(summary) for summary in summaries],
        "limitations": [
            (
                "The experiments use compact synthetic "
                "driving and robotics proxy tasks."
            ),
            (
                "Target-reaching metrics are absent rather "
                "than censored to the final epoch when a "
                "candidate fails to reach its paired target."
            ),
            (
                "Conditional target-efficiency statistics "
                "include only reached seeds and therefore "
                "must be interpreted together with the "
                "target reach rate."
            ),
            ("CPU wall-clock timing is descriptive."),
            (
                "TT/MPS uses quantum-inspired tensor-network "
                "parameterization without quantum hardware."
            ),
        ],
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" TRAINING-EFFICIENCY THREE-SEED SUMMARY")
    print("============================================")

    for summary in summaries:
        print()
        print(
            summary.domain,
            "/",
            summary.method,
        )

        print(
            "  target reach:",
            (f"{summary.target_reach.reached}" f"/{summary.target_reach.total}"),
        )

        print(
            "  reach rate:",
            (f"{summary.target_reach.rate_percent:.2f}%"),
        )

        if summary.step_reduction_percent is not None:
            print(
                "  step reduction:",
                (
                    f"{summary.step_reduction_percent.mean:.2f}"
                    " ± "
                    f"{summary.step_reduction_percent.std:.2f}%"
                ),
            )

            print(
                "  n reached:",
                (summary.step_reduction_percent.n),
            )
        else:
            print("  step reduction: unavailable")

        print(
            "  test MSE:",
            (f"{summary.test_mse.mean:.8f}" " ± " f"{summary.test_mse.std:.8f}"),
        )

        print(
            "  robust >=10%:",
            (summary.robust_ten_percent_step_efficiency),
        )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
