from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from q_vla_forge.evaluation import (
    ParetoPoint,
    build_domain_pareto_analysis,
    save_domain_pareto_analysis,
)

COMPRESSION_DIR = Path("results") / "compression"

SUMMARY_PATH = COMPRESSION_DIR / "compression-validation-summary.json"

FIGURES_DIR = Path("figures")

GLOBAL_SUMMARY_PATH = COMPRESSION_DIR / "compression-pareto-summary.json"


def _load_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(SUMMARY_PATH)

    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _method_point(
    payload: dict[str, Any],
    *,
    family: str,
) -> ParetoPoint:
    return ParetoPoint(
        method=str(payload["method"]),
        configuration=str(payload["configuration"]),
        family=family,
        compression_ratio_mean=float(payload["compression_ratio"]["mean"]),
        compression_ratio_std=float(payload["compression_ratio"]["std"]),
        mse_change_mean=float(payload["mse_change_percent"]["mean"]),
        mse_change_std=float(payload["mse_change_percent"]["std"]),
        mae_change_mean=float(payload["mae_change_percent"]["mean"]),
        mae_change_std=float(payload["mae_change_percent"]["std"]),
        pilot_feasible_runs=int(payload["pilot_feasible_runs"]),
        all_runs_pilot_feasible=bool(payload["all_runs_pilot_feasible"]),
    )


def _points(
    domain_payload: dict[str, Any],
) -> list[ParetoPoint]:
    return [
        ParetoPoint(
            method="fp32",
            configuration="FP32",
            family="reference",
            compression_ratio_mean=1.0,
            compression_ratio_std=0.0,
            mse_change_mean=0.0,
            mse_change_std=0.0,
            mae_change_mean=0.0,
            mae_change_std=0.0,
            pilot_feasible_runs=0,
            all_runs_pilot_feasible=False,
            is_reference=True,
        ),
        _method_point(
            domain_payload["int8"],
            family=("classical_quantization"),
        ),
        _method_point(
            domain_payload["svd"],
            family=("classical_low_rank"),
        ),
        _method_point(
            domain_payload["tensor_network"],
            family=("quantum_inspired_tensor_network"),
        ),
    ]


def _save_csv(
    *,
    domain: str,
    points: list[ParetoPoint],
    frontier: set[str],
) -> Path:
    path = COMPRESSION_DIR / f"{domain}-compression-pareto.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "method",
                "configuration",
                "family",
                "compression_ratio_mean",
                "compression_ratio_std",
                "mse_change_mean",
                "mse_change_std",
                "mae_change_mean",
                "mae_change_std",
                "pilot_feasible_runs",
                "all_runs_pilot_feasible",
                "pareto_efficient",
                "is_reference",
            ]
        )

        for point in points:
            writer.writerow(
                [
                    point.method,
                    point.configuration,
                    point.family,
                    point.compression_ratio_mean,
                    point.compression_ratio_std,
                    point.mse_change_mean,
                    point.mse_change_std,
                    point.mae_change_mean,
                    point.mae_change_std,
                    point.pilot_feasible_runs,
                    point.all_runs_pilot_feasible,
                    (point.configuration in frontier),
                    point.is_reference,
                ]
            )

    return path


def _plot(
    *,
    domain: str,
    points: list[ParetoPoint],
    frontier: set[str],
) -> Path:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FIGURES_DIR / f"{domain}_compression_pareto.png"

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for point in points:
        marker = "x" if point.is_reference else "o"

        axis.errorbar(
            point.compression_ratio_mean,
            point.mse_change_mean,
            xerr=(point.compression_ratio_std),
            yerr=(point.mse_change_std),
            fmt=marker,
            capsize=4,
        )

        label = point.configuration

        if point.configuration in frontier:
            label += " [Pareto]"

        axis.annotate(
            label,
            (
                point.compression_ratio_mean,
                point.mse_change_mean,
            ),
            xytext=(
                6,
                6,
            ),
            textcoords=("offset points"),
        )

    frontier_points = sorted(
        [point for point in points if (point.configuration in frontier)],
        key=lambda point: (point.compression_ratio_mean),
    )

    if len(frontier_points) >= 2:
        axis.plot(
            [point.compression_ratio_mean for point in frontier_points],
            [point.mse_change_mean for point in frontier_points],
            linestyle="--",
        )

    axis.axvline(
        2.0,
        linestyle=":",
    )

    axis.axhline(
        5.0,
        linestyle=":",
    )

    axis.set_xlabel("Effective Whole-Model Compression Ratio (×)")

    axis.set_ylabel("Relative Test MSE Change (%)")

    title_domain = "Autonomous Driving" if domain == "driving" else "Robotics"

    axis.set_title(f"{title_domain} Compression–Error Pareto Analysis")

    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def _build_domain(
    *,
    domain_name: str,
    payload_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    domain_payload = payload[payload_key]

    points = _points(domain_payload)

    analysis = build_domain_pareto_analysis(
        domain=(domain_payload["domain"]),
        points=points,
        minimum_compression_ratio=float(domain_payload["minimum_compression_ratio"]),
        maximum_mse_increase_percent=float(
            domain_payload["maximum_mse_increase_percent"]
        ),
    )

    json_path = COMPRESSION_DIR / (f"{domain_name}-" "compression-pareto.json")

    save_domain_pareto_analysis(
        analysis,
        json_path,
    )

    frontier = set(analysis.pareto_frontier)

    csv_path = _save_csv(
        domain=domain_name,
        points=points,
        frontier=frontier,
    )

    figure_path = _plot(
        domain=domain_name,
        points=points,
        frontier=frontier,
    )

    print()
    print(domain_name.upper())

    print(
        "Pareto frontier:",
        analysis.pareto_frontier,
    )

    print(
        "Dominated:",
        analysis.dominated,
    )

    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )

    print(
        "Figure:",
        figure_path,
    )

    return {
        "analysis": asdict(analysis),
        "json": str(json_path),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }


def main() -> None:
    payload = _load_summary()

    driving = _build_domain(
        domain_name="driving",
        payload_key="driving",
        payload=payload,
    )

    robotics = _build_domain(
        domain_name="robotics",
        payload_key="robotics",
        payload=payload,
    )

    summary = {
        "source": str(SUMMARY_PATH),
        "driving": driving,
        "robotics": robotics,
        "pareto_objectives": {
            "maximize": ("effective whole-model compression ratio"),
            "minimize": ("relative test MSE increase percent"),
        },
        "pilot_feasibility": {
            "minimum_compression_ratio": 2.0,
            "maximum_mse_increase_percent": 5.0,
        },
    }

    GLOBAL_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== Compression Pareto Analysis Complete =====")

    print(
        "Summary:",
        GLOBAL_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
