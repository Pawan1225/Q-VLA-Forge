from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from q_vla_forge.evaluation.cross_domain_training import (
    CrossDomainMethodComparison,
    DomainMethodEvidence,
    build_cross_domain_comparison,
)

ROOT = Path("results") / "training"

ANALYSIS_FILE = ROOT / "analysis" / "training-efficiency-analysis.json"

ABLATION_FILE = ROOT / "ablation" / "classical-vs-qi-ablation-summary.json"

OUTPUT_DIR = ROOT / "cross-domain"

FIGURE_DIR = Path("figures") / "training" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "cross-domain-training-comparison.json"

OUTPUT_CSV = OUTPUT_DIR / "cross-domain-training-comparison.csv"


METHODS = (
    "trainable_svd",
    "trainable_tt_mps",
)


def _load(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_point(
    analysis: dict[str, Any],
    *,
    domain: str,
    method: str,
) -> dict[str, Any]:
    matches = [
        point
        for point in analysis["points"]
        if (point["domain"] == domain and point["method"] == method)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one analysis point for " f"{domain}/{method}"
        )

    return matches[0]


def _build_evidence(
    analysis: dict[str, Any],
    *,
    domain: str,
    method: str,
) -> DomainMethodEvidence:
    point = _analysis_point(
        analysis,
        domain=domain,
        method=method,
    )

    return DomainMethodEvidence(
        domain=domain,
        method=method,
        target_reached=int(point["target_reach_count"]),
        target_total=int(point["target_reach_total"]),
        parameter_reduction_percent=float(point["parameter_reduction_percent"]),
        step_reduction_percent_mean=(
            None
            if point["step_reduction_percent_mean"] is None
            else float(point["step_reduction_percent_mean"])
        ),
        step_reduction_percent_std=(
            None
            if point["step_reduction_percent_std"] is None
            else float(point["step_reduction_percent_std"])
        ),
        test_mse_mean=float(point["test_mse_mean"]),
        test_mse_std=float(point["test_mse_std"]),
        robust_ten_percent_efficiency=bool(point["robust_ten_percent_efficiency"]),
    )


def _ablation_domain(
    ablation: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    matches = [item for item in ablation["domains"] if item["domain"] == domain]

    if len(matches) != 1:
        raise RuntimeError(f"missing ablation domain: {domain}")

    return matches[0]


def _target_reached(
    payload: dict[str, Any],
    key: str,
) -> bool:
    return bool(payload[key]["target_reach"]["reached_target"])


def _build_ablation_transfer(
    ablation: dict[str, Any],
) -> dict[str, Any]:
    driving = _ablation_domain(
        ablation,
        "autonomous_driving",
    )

    robotics = _ablation_domain(
        ablation,
        "robotics",
    )

    driving_svd = _target_reached(
        driving,
        "svd",
    )

    driving_tt = _target_reached(
        driving,
        "tt_mps",
    )

    robotics_svd = _target_reached(
        robotics,
        "svd",
    )

    robotics_tt = _target_reached(
        robotics,
        "tt_mps",
    )

    return {
        "seed": int(ablation["seed"]),
        "driving": {
            "svd_reached_target": driving_svd,
            "tt_mps_reached_target": driving_tt,
            "matched_parameter_difference_percent": (
                driving["matched_pair"]["relative_difference_percent"]
            ),
        },
        "robotics": {
            "svd_reached_target": robotics_svd,
            "tt_mps_reached_target": robotics_tt,
            "matched_parameter_difference_percent": (
                robotics["matched_pair"]["relative_difference_percent"]
            ),
        },
        "same_qualitative_winner": (
            (driving_tt and not driving_svd and robotics_tt and not robotics_svd)
            or (driving_svd and not driving_tt and robotics_svd and not robotics_tt)
            or (driving_svd == driving_tt == robotics_svd == robotics_tt)
        ),
        "performance_blind_pair_selection": True,
    }


def _plot_cross_domain_efficiency(
    comparisons: list[CrossDomainMethodComparison],
) -> Path:
    labels = []
    driving_values = []
    robotics_values = []

    for comparison in comparisons:
        labels.append("SVD" if comparison.method == "trainable_svd" else "TT/MPS")

        driving_values.append(
            float("nan")
            if comparison.driving.step_reduction_percent_mean is None
            else comparison.driving.step_reduction_percent_mean
        )

        robotics_values.append(
            float("nan")
            if comparison.robotics.step_reduction_percent_mean is None
            else comparison.robotics.step_reduction_percent_mean
        )

    x = list(range(len(labels)))

    width = 0.35

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        [value - width / 2 for value in x],
        driving_values,
        width=width,
        label="Driving",
    )

    plt.bar(
        [value + width / 2 for value in x],
        robotics_values,
        width=width,
        label="Robotics",
    )

    plt.axhline(
        10.0,
        linestyle="--",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Mean optimizer-step reduction (%)")

    plt.title("Cross-Domain Training Efficiency")

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "cross-domain-step-efficiency.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def _plot_parameter_reduction(
    comparisons: list[CrossDomainMethodComparison],
) -> Path:
    labels = []
    driving_values = []
    robotics_values = []

    for comparison in comparisons:
        labels.append("SVD" if comparison.method == "trainable_svd" else "TT/MPS")

        driving_values.append(comparison.driving.parameter_reduction_percent)

        robotics_values.append(comparison.robotics.parameter_reduction_percent)

    x = list(range(len(labels)))

    width = 0.35

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        [value - width / 2 for value in x],
        driving_values,
        width=width,
        label="Driving",
    )

    plt.bar(
        [value + width / 2 for value in x],
        robotics_values,
        width=width,
        label="Robotics",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Trainable parameter reduction (%)")

    plt.title("Cross-Domain Structured Parameter Reduction")

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "cross-domain-parameter-reduction.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def _write_csv(
    comparisons: list[CrossDomainMethodComparison],
) -> None:
    fields = [
        "method",
        "driving_target_reach",
        "robotics_target_reach",
        "target_consistency",
        "driving_parameter_reduction_percent",
        "robotics_parameter_reduction_percent",
        "parameter_reduction_difference_percent",
        "driving_step_reduction_percent",
        "robotics_step_reduction_percent",
        "efficiency_consistency",
        "robust_cross_domain_efficiency",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for item in comparisons:
            writer.writerow(
                {
                    "method": item.method,
                    "driving_target_reach": (
                        f"{item.driving.target_reached}/" f"{item.driving.target_total}"
                    ),
                    "robotics_target_reach": (
                        f"{item.robotics.target_reached}/"
                        f"{item.robotics.target_total}"
                    ),
                    "target_consistency": item.target_consistency,
                    "driving_parameter_reduction_percent": (
                        item.driving.parameter_reduction_percent
                    ),
                    "robotics_parameter_reduction_percent": (
                        item.robotics.parameter_reduction_percent
                    ),
                    "parameter_reduction_difference_percent": (
                        item.parameter_reduction_difference_percent
                    ),
                    "driving_step_reduction_percent": (
                        item.driving.step_reduction_percent_mean
                    ),
                    "robotics_step_reduction_percent": (
                        item.robotics.step_reduction_percent_mean
                    ),
                    "efficiency_consistency": item.efficiency_consistency,
                    "robust_cross_domain_efficiency": (
                        item.robust_cross_domain_efficiency
                    ),
                }
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis = _load(ANALYSIS_FILE)

    ablation = _load(ABLATION_FILE)

    comparisons = []

    for method in METHODS:
        driving = _build_evidence(
            analysis,
            domain="autonomous_driving",
            method=method,
        )

        robotics = _build_evidence(
            analysis,
            domain="robotics",
            method=method,
        )

        comparisons.append(
            build_cross_domain_comparison(
                driving,
                robotics,
            )
        )

    ablation_transfer = _build_ablation_transfer(ablation)

    efficiency_figure = _plot_cross_domain_efficiency(comparisons)

    parameter_figure = _plot_parameter_reduction(comparisons)

    payload = {
        "comparison_type": "shared_structured_training_cross_domain",
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "methods": [asdict(item) for item in comparisons],
        "matched_ablation_transfer": ablation_transfer,
        "architecture_interpretation": {
            "shared_mechanisms": [
                "vision encoder",
                "lightweight trainable text encoder",
                "multimodal fusion",
                "shared latent representation",
                "structured SVD conversion",
                "structured TT/MPS conversion",
                "AdamW training protocol",
                "cosine learning-rate schedule",
                "paired target evaluation",
            ],
            "domain_specific_components": [
                "state adapter input semantics",
                "synthetic task generation",
                "action semantics",
                "selected SVD rank fraction",
                "paired FP32 target values",
            ],
            "universal_trained_model_claim": False,
            "shared_architecture_claim": True,
        },
        "claim_control": {
            "cross_domain_quantum_advantage": False,
            "quantum_speedup": False,
            "quantum_hardware_used": False,
            "native_tt_runtime_acceleration": False,
        },
        "figures": [
            str(efficiency_figure),
            str(parameter_figure),
        ],
        "limitations": [
            (
                "Cross-domain evidence uses compact "
                "synthetic proxy tasks rather than "
                "full-scale production VLA systems."
            ),
            (
                "Shared architecture does not imply one "
                "universally trained model for both domains."
            ),
            (
                "Domain-specific SVD fractions were frozen "
                "from the preceding compression study."
            ),
            ("TT/MPS is quantum-inspired and executes " "classically in this pilot."),
            (
                "The matched classical-vs-QI ablation "
                "uses seed 42 as a controlled isolation "
                "study rather than a three-seed benchmark."
            ),
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_csv(comparisons)

    print()
    print("============================================")
    print(" CROSS-DOMAIN TRAINING COMPARISON")
    print("============================================")

    for item in comparisons:
        print()
        print(item.method)

        print(
            "  Driving reach:",
            f"{item.driving.target_reached}/{item.driving.target_total}",
        )

        print(
            "  Robotics reach:",
            f"{item.robotics.target_reached}/{item.robotics.target_total}",
        )

        print(
            "  Target consistency:",
            item.target_consistency,
        )

        print(
            "  Efficiency consistency:",
            item.efficiency_consistency,
        )

        print(
            "  Parameter reduction difference:",
            f"{item.parameter_reduction_difference_percent:.2f}%",
        )

        print(
            "  Robust cross-domain >=10%:",
            item.robust_cross_domain_efficiency,
        )

    print()
    print(
        "Matched ablation qualitative consistency:",
        ablation_transfer["same_qualitative_winner"],
    )

    print()
    print(
        "Saved:",
        OUTPUT_JSON,
    )

    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
