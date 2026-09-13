from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from q_vla_forge.evaluation.training_analysis import (
    EfficiencyQualityPoint,
    mean_curve,
    parameter_reduction_percent,
)

ROOT = Path("results") / "training"

VALIDATION_DIR = ROOT / "validation"

FP32_DIR = ROOT / "fp32"

SUMMARY_PATH = ROOT / "training-validation-summary.json"

ANALYSIS_DIR = ROOT / "analysis"

FIGURE_DIR = Path("figures") / "training"

ANALYSIS_JSON = ANALYSIS_DIR / "training-efficiency-analysis.json"

ANALYSIS_CSV = ANALYSIS_DIR / "training-efficiency-analysis.csv"


SEEDS = (
    42,
    123,
    456,
)


def _load(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required file missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _prefix(
    domain: str,
) -> str:
    if domain == "autonomous_driving":
        return "driving"

    if domain == "robotics":
        return "robotics"

    raise ValueError(f"unsupported domain: {domain}")


def _method_filename(
    method: str,
) -> str:
    if method == "trainable_svd":
        return "svd"

    if method == "trainable_tt_mps":
        return "tt-mps"

    raise ValueError(f"unsupported method: {method}")


def _structured_run(
    domain: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    return _load(
        VALIDATION_DIR
        / (f"{_prefix(domain)}-" f"{_method_filename(method)}-" f"seed-{seed}.json")
    )


def _fp32_run(
    domain: str,
    seed: int,
) -> dict[str, Any]:
    return _load(FP32_DIR / (f"{_prefix(domain)}-" f"fp32-seed-{seed}.json"))


def _summary_item(
    payload: dict[str, Any],
    domain: str,
    method: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in payload["methods"]
        if (item["domain"] == domain and item["method"] == method)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one summary item for " f"{domain}/{method}")

    return matches[0]


def _mean_fp32_parameters(
    domain: str,
) -> float:
    values = [
        float(
            _fp32_run(
                domain,
                seed,
            )["trainable_parameters"]
        )
        for seed in SEEDS
    ]

    return float(statistics.mean(values))


def _mean_validation_curve(
    domain: str,
    method: str,
) -> tuple[float, ...]:
    curves = []

    if method == "shared_vla_fp32":
        for seed in SEEDS:
            run = _fp32_run(
                domain,
                seed,
            )

            curves.append([item["validation_loss"] for item in run["history"]])
    else:
        for seed in SEEDS:
            run = _structured_run(
                domain,
                method,
                seed,
            )

            curves.append(
                [item["validation_loss"] for item in run["summary"]["history"]]
            )

    return mean_curve(curves)


def _build_point(
    summary_payload: dict[str, Any],
    domain: str,
    method: str,
) -> EfficiencyQualityPoint:
    item = _summary_item(
        summary_payload,
        domain,
        method,
    )

    fp32_parameters = _mean_fp32_parameters(domain)

    candidate_parameters = float(item["trainable_parameters"]["mean"])

    reduction = parameter_reduction_percent(
        fp32_parameters,
        candidate_parameters,
    )

    step = item["step_reduction_percent"]

    return EfficiencyQualityPoint(
        domain=domain,
        method=method,
        target_reach_count=(item["target_reach"]["reached"]),
        target_reach_total=(item["target_reach"]["total"]),
        trainable_parameters=(candidate_parameters),
        parameter_reduction_percent=(reduction),
        step_reduction_percent_mean=(None if step is None else float(step["mean"])),
        step_reduction_percent_std=(None if step is None else float(step["std"])),
        test_mse_mean=float(item["test_mse"]["mean"]),
        test_mse_std=float(item["test_mse"]["std"]),
        test_mae_mean=float(item["test_mae"]["mean"]),
        test_mae_std=float(item["test_mae"]["std"]),
        robust_ten_percent_efficiency=bool(item["robust_ten_percent_step_efficiency"]),
    )


def _plot_convergence(
    domain: str,
) -> Path:
    epochs = list(
        range(
            1,
            21,
        )
    )

    fp32 = _mean_validation_curve(
        domain,
        "shared_vla_fp32",
    )

    svd = _mean_validation_curve(
        domain,
        "trainable_svd",
    )

    tt = _mean_validation_curve(
        domain,
        "trainable_tt_mps",
    )

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        fp32,
        label="FP32",
    )

    plt.plot(
        epochs,
        svd,
        label="Trainable SVD",
    )

    plt.plot(
        epochs,
        tt,
        label="Trainable TT/MPS",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Mean validation MSE")

    plt.title(
        ("Driving" if domain == "autonomous_driving" else "Robotics")
        + " — Mean Validation Convergence"
    )

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / (f"{_prefix(domain)}-" "training-convergence.png")

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def _plot_parameter_efficiency(
    points: list[EfficiencyQualityPoint],
) -> Path:
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    for point in points:
        if point.step_reduction_percent_mean is None:
            continue

        label = f"{point.domain} / " f"{point.method}"

        plt.scatter(
            point.parameter_reduction_percent,
            point.step_reduction_percent_mean,
            label=label,
        )

    plt.axhline(
        10.0,
        linestyle="--",
    )

    plt.xlabel("Trainable parameter reduction (%)")

    plt.ylabel("Mean optimizer-step reduction (%)")

    plt.title("Parameter Reduction vs Training Efficiency")

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "parameter-vs-training-efficiency.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def _write_csv(
    points: list[EfficiencyQualityPoint],
) -> None:
    fields = [
        "domain",
        "method",
        "target_reach_count",
        "target_reach_total",
        "target_reach_rate_percent",
        "trainable_parameters",
        "parameter_reduction_percent",
        "step_reduction_percent_mean",
        "step_reduction_percent_std",
        "test_mse_mean",
        "test_mse_std",
        "test_mae_mean",
        "test_mae_std",
        "robust_ten_percent_efficiency",
    ]

    with ANALYSIS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for point in points:
            row = asdict(point)

            row["target_reach_rate_percent"] = point.target_reach_rate_percent

            writer.writerow(row)


def main() -> None:
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_payload = _load(SUMMARY_PATH)

    points = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "trainable_svd",
            "trainable_tt_mps",
        ):
            points.append(
                _build_point(
                    summary_payload,
                    domain,
                    method,
                )
            )

    driving_figure = _plot_convergence("autonomous_driving")

    robotics_figure = _plot_convergence("robotics")

    efficiency_figure = _plot_parameter_efficiency(points)

    payload = {
        "seeds": list(SEEDS),
        "primary_metric": ("optimizer_steps_to_paired_fp32_target"),
        "points": [
            {
                **asdict(point),
                "target_reach_rate_percent": (point.target_reach_rate_percent),
            }
            for point in points
        ],
        "figures": [
            str(driving_figure),
            str(robotics_figure),
            str(efficiency_figure),
        ],
        "claim_rules": {
            "robust_training_efficiency": (
                "all three seeds reach their paired "
                "FP32 targets and mean optimizer-step "
                "reduction is at least 10%"
            ),
            "quantum_advantage_claim": False,
        },
        "limitations": [
            (
                "Results are based on compact synthetic "
                "autonomous-driving and robotics proxies."
            ),
            (
                "The analysis evaluates structured training "
                "efficiency, not full-scale 7B-class VLA models."
            ),
            (
                "Wall-clock timing is descriptive and is not "
                "used as the primary efficiency criterion."
            ),
            (
                "Trainable TT/MPS is quantum-inspired and "
                "does not use quantum hardware."
            ),
        ],
    }

    ANALYSIS_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_csv(points)

    print()
    print("============================================")
    print(" TRAINING EFFICIENCY ANALYSIS")
    print("============================================")

    for point in points:
        print()
        print(
            point.domain,
            "/",
            point.method,
        )

        print(
            "  reach:",
            (f"{point.target_reach_count}/" f"{point.target_reach_total}"),
        )

        print(
            "  reach rate:",
            (f"{point.target_reach_rate_percent:.2f}%"),
        )

        print(
            "  parameter reduction:",
            (f"{point.parameter_reduction_percent:.2f}%"),
        )

        if point.step_reduction_percent_mean is None:
            print("  step reduction: unavailable")
        else:
            print(
                "  step reduction:",
                (
                    f"{point.step_reduction_percent_mean:.2f}"
                    " ± "
                    f"{point.step_reduction_percent_std:.2f}%"
                ),
            )

        print(
            "  test MSE:",
            (f"{point.test_mse_mean:.8f}" " ± " f"{point.test_mse_std:.8f}"),
        )

        print(
            "  robust >=10%:",
            (point.robust_ten_percent_efficiency),
        )

    print()
    print(
        "Saved:",
        ANALYSIS_JSON,
    )

    print(
        "Saved:",
        ANALYSIS_CSV,
    )


if __name__ == "__main__":
    main()
