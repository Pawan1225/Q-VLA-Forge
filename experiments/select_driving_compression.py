"""Select seed-42 driving compression candidates for validation."""

from __future__ import annotations

from pathlib import Path

from q_vla_forge.evaluation import (
    build_domain_compression_selection,
    load_compression_candidate,
    save_domain_compression_selection,
)

COMPRESSION_DIR = Path("results") / "compression"

OUTPUT_PATH = COMPRESSION_DIR / "driving-compression-selection.json"

CSV_PATH = COMPRESSION_DIR / "driving-compression-comparison.csv"

DOMAIN = "autonomous_driving"
SEED = 42

MINIMUM_COMPRESSION_RATIO = 2.0
MAXIMUM_MSE_INCREASE_PERCENT = 5.0


def _result_files() -> list[Path]:
    """Return all seven seed-42 driving compression evidence files."""
    files = [
        COMPRESSION_DIR / "driving-int8-seed-42.json",
    ]

    files.extend(sorted(COMPRESSION_DIR.glob("driving-svd-*-seed-42.json")))

    files.extend(sorted(COMPRESSION_DIR.glob("driving-tt-rank-*-seed-42.json")))

    return files


def _write_csv(
    candidates,
) -> None:
    """Write compact driving comparison evidence."""
    lines = [
        (
            "experiment_id,method,configuration,"
            "compression_ratio,storage_reduction_percent,"
            "parameter_reduction_percent,"
            "mse_change_percent,mae_change_percent,"
            "latency_change_percent,pilot_feasible"
        )
    ]

    for candidate in candidates:
        lines.append(
            ",".join(
                [
                    candidate.experiment_id,
                    candidate.method,
                    candidate.configuration_label,
                    f"{candidate.compression_ratio:.8f}",
                    f"{candidate.storage_reduction_percent:.8f}",
                    f"{candidate.parameter_reduction_percent:.8f}",
                    f"{candidate.mse_change_percent:.8f}",
                    f"{candidate.mae_change_percent:.8f}",
                    f"{candidate.latency_change_percent:.8f}",
                    str(candidate.pilot_feasible),
                ]
            )
        )

    CSV_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Normalize and select the driving compression candidates."""
    files = _result_files()

    missing = [path for path in files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "missing driving compression evidence: "
            + ", ".join(str(path) for path in missing)
        )

    candidates = [
        load_compression_candidate(
            path,
            minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
            maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
        )
        for path in files
    ]

    if len(candidates) != 7:
        raise RuntimeError("expected exactly seven driving compression candidates")

    selection = build_domain_compression_selection(
        candidates,
        domain=DOMAIN,
        seed=SEED,
        minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
        maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
    )

    save_domain_compression_selection(
        selection,
        OUTPUT_PATH,
    )

    _write_csv(candidates)

    print()
    print("===== Q-VLA Forge " "Driving Compression Selection =====")

    print()

    print(
        f"{'Method':16}"
        f"{'Config':18}"
        f"{'Ratio':>10}"
        f"{'MSE Δ':>12}"
        f"{'MAE Δ':>12}"
        f"{'Feasible':>12}"
    )

    print("-" * 80)

    for candidate in candidates:
        print(
            f"{candidate.method:16}"
            f"{candidate.configuration_label:18}"
            f"{candidate.compression_ratio:>9.3f}x"
            f"{candidate.mse_change_percent:>11.3f}%"
            f"{candidate.mae_change_percent:>11.3f}%"
            f"{candidate.pilot_feasible!s:>12}"
        )

    print()
    print("===== Selected Configurations =====")

    for name, method in (
        (
            "INT8",
            selection.int8,
        ),
        (
            "SVD",
            selection.svd,
        ),
        (
            "TT/MPS",
            selection.tensor_network,
        ),
    ):
        candidate = method.selected

        print()
        print(
            name,
            "→",
            candidate.configuration_label,
        )

        print(
            "  Experiment:",
            candidate.experiment_id,
        )

        print(
            "  Compression:",
            f"{candidate.compression_ratio:.3f}x",
        )

        print(
            "  MSE change:",
            f"{candidate.mse_change_percent:.3f}%",
        )

        print(
            "  MAE change:",
            f"{candidate.mae_change_percent:.3f}%",
        )

        print(
            "  Pilot feasible:",
            candidate.pilot_feasible,
        )

        print(
            "  Reason:",
            method.selection_reason,
        )

    print()
    print("Selected IDs:")

    for experiment_id in selection.selected_experiment_ids:
        print(
            " -",
            experiment_id,
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "CSV:",
        CSV_PATH,
    )


if __name__ == "__main__":
    main()
