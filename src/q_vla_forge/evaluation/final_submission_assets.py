"""Sprint 7.12 — final proposal figures and tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def build_submission_tables(
    root: Path,
) -> dict[str, Any]:
    """Generate final proposal-ready CSV tables."""

    final_root = root / "results" / "final-validation"

    ablation = load_json(final_root / "ablation" / "final-ablation-matrix.json")

    scorecard = load_json(
        final_root / "scorecard" / "challenge-bottleneck-scorecard.json"
    )

    claims = load_json(final_root / "claims" / "final-claim-registry.json")

    output = final_root / "tables"

    ablation_path = output / "final-ablation-matrix.csv"

    scorecard_path = output / "challenge-bottleneck-scorecard.csv"

    claims_path = output / "final-claim-registry.csv"

    write_csv(
        ablation_path,
        ablation["rows"],
        [
            "area",
            "comparison",
            "finding",
            "status",
        ],
    )

    write_csv(
        scorecard_path,
        scorecard["bottlenecks"],
        [
            "bottleneck",
            "phase1_status",
            "headline",
            "strongest_evidence",
            "boundary",
        ],
    )

    claim_rows = []

    for claim in claims["claims"]:
        claim_rows.append(
            {
                "claim_id": claim["claim_id"],
                "area": claim["area"],
                "status": claim["status"],
                "statement": claim["statement"],
                "limitation": claim.get(
                    "limitation",
                    "",
                ),
                "evidence": claim["evidence"],
            }
        )

    write_csv(
        claims_path,
        claim_rows,
        [
            "claim_id",
            "area",
            "status",
            "statement",
            "limitation",
            "evidence",
        ],
    )

    return {
        "ablation_table": (ablation_path.relative_to(root).as_posix()),
        "scorecard_table": (scorecard_path.relative_to(root).as_posix()),
        "claim_registry_table": (claims_path.relative_to(root).as_posix()),
    }


def build_compression_figure(
    root: Path,
) -> str:
    """Build final compression Pareto figure."""

    source = load_json(
        root
        / "results"
        / "final-validation"
        / "compression"
        / "final-compression-summary.json"
    )

    output = root / "figures" / "final" / "phase1_compression_pareto.png"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = {
        "driving": "o",
        "robotics": "s",
    }

    for domain_key in (
        "driving",
        "robotics",
    ):
        domain = source["domains"][domain_key]

        for method in domain["methods"]:
            if method["is_reference"]:
                continue

            ax.scatter(
                method["compression_ratio_mean"],
                method["mse_change_mean_percent"],
                marker=markers[domain_key],
                s=70,
            )

            ax.annotate(
                (f"{domain_key}: " f"{method['configuration']}"),
                (
                    method["compression_ratio_mean"],
                    method["mse_change_mean_percent"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    criterion = source["criterion"]

    ax.axvline(
        criterion["minimum_compression_ratio"],
        linestyle="--",
    )

    ax.axhline(
        criterion["maximum_relative_mse_increase_percent"],
        linestyle="--",
    )

    ax.set_xlabel("Effective whole-model compression ratio")

    ax.set_ylabel("Relative test MSE change (%)")

    ax.set_title("Q-VLA Forge Phase 1 Compression Comparison")

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=200,
    )

    plt.close(fig)

    return output.relative_to(root).as_posix()


def build_rl_qml_figure(
    root: Path,
) -> str:
    """Build final PPO / matched classical / QML target figure."""

    source = load_json(
        root / "results" / "final-validation" / "rl-qml" / "final-rl-qml-summary.json"
    )

    output = root / "figures" / "final" / "phase1_rl_qml_target_reach.png"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods = [
        "Full PPO",
        "Matched Classical",
        "QML",
    ]

    driving = [
        source["domains"]["autonomous_driving"]["full_ppo"]["target_reach_rate"],
        source["domains"]["autonomous_driving"]["matched_classical"][
            "target_reach_rate"
        ],
        source["domains"]["autonomous_driving"]["qml"]["target_reach_rate"],
    ]

    robotics = [
        source["domains"]["robotics"]["full_ppo"]["target_reach_rate"],
        source["domains"]["robotics"]["matched_classical"]["target_reach_rate"],
        source["domains"]["robotics"]["qml"]["target_reach_rate"],
    ]

    positions = list(range(len(methods)))

    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [value - width / 2 for value in positions],
        driving,
        width=width,
        label="Autonomous driving",
    )

    ax.bar(
        [value + width / 2 for value in positions],
        robotics,
        width=width,
        label="Robotics",
    )

    ax.set_xticks(
        positions,
        methods,
    )

    ax.set_ylim(
        0.0,
        1.1,
    )

    ax.set_ylabel("Target reach rate")

    ax.set_title("Q-VLA Forge Phase 1 RL/QML Target Reach")

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=200,
    )

    plt.close(fig)

    return output.relative_to(root).as_posix()


def build_final_submission_assets(
    root: Path,
) -> dict[str, Any]:
    """Generate all Sprint 7.12 final figures and tables."""

    tables = build_submission_tables(root)

    figures = {
        "compression_pareto": (build_compression_figure(root)),
        "rl_qml_target_reach": (build_rl_qml_figure(root)),
    }

    return {
        "sprint": "7.12",
        "protocol": "final_figures_and_tables",
        "status": "FROZEN",
        "tables": tables,
        "figures": figures,
        "new_training": False,
        "new_experiments": False,
    }
