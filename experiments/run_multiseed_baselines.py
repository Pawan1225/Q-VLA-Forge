from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.driving_baseline import run_driving_baseline
from q_vla_forge.evaluation.multiseed import (
    aggregate_baseline_payloads,
    save_multiseed_summary,
)
from q_vla_forge.evaluation.robotics_baseline import run_robotics_baseline
from q_vla_forge.utils.reproducibility import DEFAULT_SEEDS

RESULTS_DIR = Path("results")


def _format_metric(
    mean: float,
    std: float,
) -> str:
    return f"{mean:.6f} ± {std:.6f}"


def main() -> None:
    driving_payloads: list[dict[str, Any]] = []
    robotics_payloads: list[dict[str, Any]] = []

    print()
    print("===== Q-VLA Forge " "Three-Seed Baseline Validation =====")

    print(
        "Seeds:",
        DEFAULT_SEEDS,
    )

    for seed in DEFAULT_SEEDS:
        print()
        print(f"--- Driving seed {seed} ---")

        driving = run_driving_baseline(
            seed=seed,
            train_size=512,
            validation_size=128,
            test_size=128,
            epochs=20,
            batch_size=32,
            output_path=(RESULTS_DIR / ("driving-baseline-" f"seed-{seed}.json")),
        )

        driving_payloads.append(asdict(driving))

        print(
            "Driving test MSE:",
            f"{driving.test_metrics.test_mse:.6f}",
        )

        print()
        print(f"--- Robotics seed {seed} ---")

        robotics = run_robotics_baseline(
            seed=seed,
            train_size=512,
            validation_size=128,
            test_size=128,
            epochs=20,
            batch_size=32,
            output_path=(RESULTS_DIR / ("robotics-baseline-" f"seed-{seed}.json")),
        )

        robotics_payloads.append(asdict(robotics))

        print(
            "Robotics test MSE:",
            f"{robotics.test_metrics.test_mse:.6f}",
        )

    summary = aggregate_baseline_payloads(
        driving_payloads,
        robotics_payloads,
        DEFAULT_SEEDS,
    )

    summary_path = RESULTS_DIR / "baseline-validation-summary.json"

    save_multiseed_summary(
        summary,
        summary_path,
    )

    print()
    print("===== Aggregate Results =====")

    print()
    print("Driving")

    print(
        "  Test MSE:",
        _format_metric(
            summary.driving.test_mse.mean,
            summary.driving.test_mse.std,
        ),
    )

    print(
        "  Test MAE:",
        _format_metric(
            summary.driving.test_mae.mean,
            summary.driving.test_mae.std,
        ),
    )

    print(
        "  Mean latency:",
        _format_metric(
            summary.driving.mean_latency_ms.mean,
            summary.driving.mean_latency_ms.std,
        ),
        "ms",
    )

    print()
    print("Robotics")

    print(
        "  Test MSE:",
        _format_metric(
            summary.robotics.test_mse.mean,
            summary.robotics.test_mse.std,
        ),
    )

    print(
        "  Test MAE:",
        _format_metric(
            summary.robotics.test_mae.mean,
            summary.robotics.test_mae.std,
        ),
    )

    print(
        "  Mean latency:",
        _format_metric(
            summary.robotics.mean_latency_ms.mean,
            summary.robotics.mean_latency_ms.std,
        ),
        "ms",
    )

    print()
    print(
        "Parameters:",
        summary.driving.parameters,
    )

    print(
        "FP32 model size:",
        f"{summary.driving.fp32_model_size_bytes / (1024**2):.4f}",
        "MB",
    )

    print()
    print(
        "Saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
