"""Generate the Sprint 2.2 compression target evidence report."""

from __future__ import annotations

from pathlib import Path

from q_vla_forge.compression import (
    analyze_compression_targets,
    load_frozen_baseline,
    save_compression_target_report,
)
from q_vla_forge.models import SharedVLAModel

RESULTS_DIR = Path("results")

OUTPUT_PATH = RESULTS_DIR / "compression" / "compression-target-analysis.json"


def _yes_no(value: bool) -> str:
    return "YES" if value else "-"


def main() -> None:
    baseline = load_frozen_baseline(RESULTS_DIR)

    model = SharedVLAModel()

    report = analyze_compression_targets(model)

    if report.total_parameters != baseline.parameters:
        raise RuntimeError(
            "current model parameter count does not " "match frozen Sprint 1 baseline"
        )

    if report.trainable_parameters != baseline.parameters:
        raise RuntimeError(
            "trainable parameter count does not " "match frozen Sprint 1 baseline"
        )

    save_compression_target_report(
        report,
        OUTPUT_PATH,
    )

    print()
    print("===== Q-VLA Forge " "Compression Target Analysis =====")

    print()
    print(
        "Frozen baseline:",
        baseline.baseline_name,
    )
    print(
        "Baseline version:",
        baseline.baseline_version,
    )
    print(
        "Total parameters:",
        report.total_parameters,
    )
    print(
        "Candidate weight parameters:",
        report.candidate_weight_parameters,
    )
    print(
        "Other parameters:",
        report.uncategorized_parameters,
    )

    print()
    print("===== Method Coverage =====")

    print(
        "INT8:",
        report.int8_target_parameters,
        f"({report.int8_coverage_percent:.2f}%)",
    )

    print(
        "SVD:",
        report.svd_target_parameters,
        f"({report.svd_coverage_percent:.2f}%)",
    )

    print(
        "TT:",
        report.tensor_train_target_parameters,
        f"({report.tensor_train_coverage_percent:.2f}%)",
    )

    print(
        "MPS:",
        report.mps_target_parameters,
        f"({report.mps_coverage_percent:.2f}%)",
    )

    print()
    print("===== Ranked Layers =====")

    header = (
        f"{'Layer':45}"
        f"{'Type':12}"
        f"{'Shape':18}"
        f"{'Weights':>10}"
        f"{'% Model':>10}"
        f"{'INT8':>8}"
        f"{'SVD':>8}"
        f"{'TT':>8}"
        f"{'Priority':>10}"
    )

    print(header)
    print("-" * len(header))

    for layer in report.layers:
        print(
            f"{layer.name:45}"
            f"{layer.module_type:12}"
            f"{layer.weight_shape!s:18}"
            f"{layer.weight_parameters:>10}"
            f"{layer.model_parameter_percent:>9.2f}%"
            f"{_yes_no(layer.int8_eligible):>8}"
            f"{_yes_no(layer.svd_eligible):>8}"
            f"{_yes_no(layer.tensor_train_eligible):>8}"
            f"{layer.priority:>10}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
