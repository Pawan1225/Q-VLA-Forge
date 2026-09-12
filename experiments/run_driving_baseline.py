from __future__ import annotations

from pathlib import Path

from q_vla_forge.evaluation.driving_baseline import (
    run_driving_baseline,
)


def main() -> None:
    result = run_driving_baseline(
        seed=42,
        train_size=512,
        validation_size=128,
        test_size=128,
        epochs=20,
        batch_size=32,
        output_path=Path("results/" "driving-baseline-seed-42.json"),
    )

    print()
    print("===== Q-VLA Forge " "Driving Baseline =====")

    print(f"Experiment: " f"{result.experiment_id}")

    print(f"Seed: " f"{result.seed}")

    print()
    print("Validation MSE")

    print(f"  Initial: " f"{result.initial_validation_mse:.6f}")

    print(f"  Best:    " f"{result.best_validation_mse:.6f}")

    print(f"  Final:   " f"{result.final_validation_mse:.6f}")

    print()
    print("Test Metrics")

    print(f"  MSE:              " f"{result.test_metrics.test_mse:.6f}")

    print(f"  MAE:              " f"{result.test_metrics.test_mae:.6f}")

    print(f"  Steering MAE:     " f"{result.test_metrics.steering_mae:.6f}")

    print(f"  Acceleration MAE: " f"{result.test_metrics.acceleration_mae:.6f}")

    print(f"  Braking MAE:      " f"{result.test_metrics.braking_mae:.6f}")

    print()
    print("Efficiency")

    print(f"  Parameters: " f"{result.parameters}")

    print(f"  FP32 size:  " f"{result.fp32_model_size_bytes / (1024**2):.4f} MB")

    print(f"  Mean latency: " f"{result.latency.mean_ms:.3f} ms")

    print(f"  P95 latency:  " f"{result.latency.p95_ms:.3f} ms")

    print()
    print("Saved:")

    print("  results/" "driving-baseline-seed-42.json")


if __name__ == "__main__":
    main()
