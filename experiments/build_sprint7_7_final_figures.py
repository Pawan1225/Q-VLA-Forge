"""Build Sprint 7.7 proposal-ready final figures from frozen evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from q_vla_forge.evaluation.final_figures import (
    ABLATION_COMPONENT_ONLY_COUNT,
    ABLATION_DIRECT_COUNT,
    ABLATION_NOT_EVALUATED_COUNT,
    CLAIM_CONTROLS,
    COMPRESSION_MSE_DEGRADATION_THRESHOLD_PERCENT,
    COMPRESSION_RATIO_THRESHOLD,
    FIGURE_CAPTIONS_PATH,
    FIGURE_IDS,
    FIGURE_MANIFEST_PATH,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    OUTPUT_DIR,
    build_all_figure_contracts,
)

ROOT = Path(__file__).resolve().parents[1]

COMPRESSION_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "compression-ablation"
    / "compression-pareto.csv"
)

TRAINING_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "training-efficiency"
    / "final-training-efficiency-summary.json"
)

QML_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "qml-ablation"
    / "qml-ablation-plot-data.csv"
)

SAFETY_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "safety-ablation"
    / "safety-ablation-plot-data.csv"
)

STATISTICS_PATH = (
    ROOT / "results" / "final-validation" / "statistics" / "final-statistics.json"
)

ABLATION_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "full-system-ablation"
    / "full-system-ablation.json"
)

ARCHITECTURE_PATH = (
    ROOT
    / "results"
    / "safety"
    / "cross-domain"
    / "sprint5-cross-domain-architecture.json"
)


def _require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "Required frozen evidence artifact " f"is missing: {path}"
        )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    _require_file(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    _require_file(path)

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _save(
    figure: plt.Figure,
    filename: str,
) -> None:
    path = ROOT / OUTPUT_DIR / filename

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not created: {path}")


def _draw_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False,
    )

    axis.add_patch(patch)

    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
    )


def _draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
            "linestyle": ("--" if dashed else "-"),
        },
    )


def _figure_architecture() -> None:
    payload = _load_json(ARCHITECTURE_PATH)

    reuse = payload["architecture_reuse"]

    if reuse["same_policy_weights"]:
        raise RuntimeError(
            "Architecture evidence unexpectedly "
            "claims shared trained policy weights."
        )

    if reuse["universal_controller_supported"]:
        raise RuntimeError(
            "Architecture evidence unexpectedly " "claims universal controller support."
        )

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.set_xlim(
        0,
        12,
    )
    axis.set_ylim(
        0,
        8,
    )
    axis.axis("off")

    _draw_box(
        axis,
        0.5,
        5.8,
        2.0,
        0.9,
        "Driving\nAdapter",
    )

    _draw_box(
        axis,
        0.5,
        3.2,
        2.0,
        0.9,
        "Robotics\nAdapter",
    )

    _draw_box(
        axis,
        3.4,
        4.5,
        2.2,
        1.0,
        "Shared AI/DL\nCore",
    )

    _draw_box(
        axis,
        6.3,
        4.5,
        2.0,
        1.0,
        "Shared Latent\nRepresentation",
    )

    _draw_box(
        axis,
        9.1,
        6.0,
        2.2,
        0.9,
        "Compression\nINT8 / SVD / TT-MPS",
    )

    _draw_box(
        axis,
        9.1,
        4.4,
        2.2,
        0.9,
        "Policy\nPPO / PQC",
    )

    _draw_box(
        axis,
        9.1,
        2.8,
        2.2,
        0.9,
        "Safety Filter\nClipping / Lyapunov",
    )

    _draw_box(
        axis,
        9.1,
        1.2,
        2.2,
        0.9,
        "Safe Action",
    )

    _draw_arrow(
        axis,
        (2.5, 6.25),
        (3.4, 5.15),
    )

    _draw_arrow(
        axis,
        (2.5, 3.65),
        (3.4, 4.85),
    )

    _draw_arrow(
        axis,
        (5.6, 5.0),
        (6.3, 5.0),
    )

    _draw_arrow(
        axis,
        (8.3, 5.0),
        (9.1, 6.4),
        dashed=True,
    )

    _draw_arrow(
        axis,
        (10.2, 6.0),
        (10.2, 5.3),
        dashed=True,
    )

    _draw_arrow(
        axis,
        (10.2, 4.4),
        (10.2, 3.7),
        dashed=True,
    )

    _draw_arrow(
        axis,
        (10.2, 2.8),
        (10.2, 2.1),
        dashed=True,
    )

    axis.text(
        8.65,
        7.25,
        (
            "Conceptual integration pathway\n"
            "Dashed links were not evaluated "
            "as one matched end-to-end Phase 1 pipeline"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    axis.text(
        5.8,
        0.35,
        (
            "Phase 1 established component, interface, "
            "framework, and protocol evidence.\n"
            "Driving and robotics retained separate trained "
            "policy weights and domain-specific safety semantics."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    axis.set_title(
        "Q-VLA Forge Phase 1 Architecture and Integration Boundary",
        fontsize=15,
        pad=15,
    )

    _save(
        figure,
        "figure-01-architecture.png",
    )


def _compression_value(
    row: dict[str, str],
) -> float:
    if row["method"] == "fp32":
        return 0.0

    value = row["relative_mse_change_mean_percent"]

    if not value:
        raise RuntimeError("Missing compression MSE-change value.")

    return float(value)


def _figure_compression() -> None:
    rows = _load_csv(COMPRESSION_PATH)

    expected = len(rows) == 8

    if not expected:
        raise RuntimeError(
            "Compression Pareto requires exactly " "eight frozen domain/method records."
        )

    figure, axis = plt.subplots(figsize=(11, 7))

    markers = {
        "autonomous_driving": "o",
        "robotics": "s",
    }

    labels_seen: set[str] = set()

    for row in rows:
        domain = row["domain"]

        method = row["method"]

        x_value = float(row["compression_ratio"])

        y_value = _compression_value(row)

        label = domain.replace(
            "_",
            " ",
        ).title()

        scatter_label = label if label not in labels_seen else None

        labels_seen.add(label)

        axis.scatter(
            x_value,
            y_value,
            marker=markers[domain],
            s=75,
            label=scatter_label,
        )

        axis.annotate(
            method.upper(),
            (
                x_value,
                y_value,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.axvline(
        COMPRESSION_RATIO_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
    )

    axis.axhline(
        COMPRESSION_MSE_DEGRADATION_THRESHOLD_PERCENT,
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_yscale(
        "symlog",
        linthresh=5,
    )

    axis.set_xlabel("Compression ratio (×)")

    axis.set_ylabel("Relative MSE change (%)")

    axis.set_title("Compression Pareto Under Frozen Phase 1 Criteria")

    axis.text(
        2.05,
        2.5,
        ("Frozen Phase 1\n" "acceptance region:\n" "compression ≥ 2×\n" "ΔMSE ≤ 5%"),
        fontsize=9,
    )

    axis.legend()

    axis.grid(alpha=0.2)

    _save(
        figure,
        "figure-02-compression-pareto.png",
    )


def _figure_training() -> None:
    payload = _load_json(TRAINING_PATH)

    if payload["robust_ten_percent_efficiency_demonstrated"]:
        raise RuntimeError(
            "Frozen training evidence unexpectedly " "claims robust >=10% efficiency."
        )

    methods = payload["methods"]

    if len(methods) != 4:
        raise RuntimeError("Expected four frozen training " "domain/method records.")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    labels: list[str] = []
    reach_rates: list[float] = []

    for item in methods:
        domain = "Driving" if item["domain"] == "autonomous_driving" else "Robotics"

        method = "SVD" if item["method"] == "trainable_svd" else "TT/MPS"

        labels.append(f"{domain}\n{method}")

        reached = item["target_reach"]["reached"]

        total = item["target_reach"]["total"]

        reach_rates.append(reached / total)

    positions = np.arange(len(labels))

    axes[0].bar(
        positions,
        reach_rates,
    )

    axes[0].set_xticks(
        positions,
        labels,
    )

    axes[0].set_ylim(
        0,
        1.1,
    )

    axes[0].set_ylabel("Target reach fraction")

    axes[0].set_title("Paired FP32 Target Attainment")

    for index, item in enumerate(methods):
        reached = item["target_reach"]["reached"]

        total = item["target_reach"]["total"]

        axes[0].text(
            index,
            reach_rates[index] + 0.04,
            f"{reached}/{total}",
            ha="center",
        )

    axes[1].set_title("Optimizer Steps to Target")

    axes[1].set_ylabel("Optimizer steps")

    axes[1].set_xticks(
        positions,
        labels,
    )

    for index, item in enumerate(methods):
        steps = item["steps_to_target"]

        if steps is None:
            axes[1].text(
                index,
                20,
                "Not reached",
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
            )

            continue

        axes[1].bar(
            index,
            float(steps["mean"]),
            yerr=float(steps["std"]),
            capsize=4,
        )

    axes[1].set_ylim(
        0,
        360,
    )

    figure.suptitle(
        (
            "Training Efficiency and Target Attainment\n"
            "No robust ≥10% optimizer-step improvement "
            "was demonstrated across both domains"
        ),
        fontsize=14,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )

    _save(
        figure,
        "figure-03-training-convergence.png",
    )


def _figure_rl_qml() -> None:
    rows = _load_csv(QML_PATH)

    if len(rows) != 18:
        raise RuntimeError(
            "RL/QML plot data requires " "18 domain-policy-seed records."
        )

    policy_order = (
        "ppo_mlp",
        "matched_classical",
        "ppo_pqc",
    )

    policy_labels = {
        "ppo_mlp": "Classical PPO",
        "matched_classical": ("Matched classical"),
        "ppo_pqc": "PQC/QML",
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    reached_counts = []

    for policy in policy_order:
        policy_rows = [row for row in rows if row["policy"] == policy]

        reached = sum(row["target_reached"].lower() == "true" for row in policy_rows)

        reached_counts.append(reached)

    x_values = np.arange(len(policy_order))

    axes[0].bar(
        x_values,
        reached_counts,
    )

    axes[0].set_xticks(
        x_values,
        [policy_labels[policy] for policy in policy_order],
    )

    axes[0].set_ylim(
        0,
        6.7,
    )

    axes[0].set_ylabel("Target-reaching evaluations (of 6)")

    axes[0].set_title("Sample-Efficiency Target Attainment")

    for index, value in enumerate(reached_counts):
        axes[0].text(
            index,
            value + 0.15,
            f"{value}/6",
            ha="center",
        )

    compactness = {
        "Driving": (
            1318,
            54,
        ),
        "Robotics": (
            1382,
            62,
        ),
    }

    positions = np.arange(len(compactness))

    width = 0.35

    axes[1].bar(
        positions - width / 2,
        [values[0] for values in compactness.values()],
        width,
        label="Classical PPO",
    )

    axes[1].bar(
        positions + width / 2,
        [values[1] for values in compactness.values()],
        width,
        label="PQC actor",
    )

    axes[1].set_xticks(
        positions,
        list(compactness),
    )

    axes[1].set_ylabel("Trainable actor parameters")

    axes[1].set_title("Actor Compactness")

    axes[1].legend()

    axes[1].text(
        0,
        1000,
        "95.90% fewer",
        ha="center",
        fontsize=9,
    )

    axes[1].text(
        1,
        1050,
        "95.51% fewer",
        ha="center",
        fontsize=9,
    )

    figure.suptitle(
        ("RL/QML Ablation — Compact Actor " "≠ Improved Sample Efficiency"),
        fontsize=14,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )

    _save(
        figure,
        "figure-04-rl-sample-efficiency.png",
    )


def _figure_safety() -> None:
    rows = _load_csv(SAFETY_PATH)

    if len(rows) != 6:
        raise RuntimeError(
            "Safety figure requires exactly " "six domain/method records."
        )

    domains = (
        "autonomous_driving",
        "robotics",
    )

    methods = (
        "none",
        "clipping",
        "lyapunov",
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    x_values = np.arange(len(domains))

    width = 0.24

    for method_index, method in enumerate(methods):
        means = []
        stds = []

        for domain in domains:
            matches = [
                row
                for row in rows
                if (row["domain"] == domain and row["method"] == method)
            ]

            if len(matches) != 1:
                raise RuntimeError("Safety dataset is incomplete.")

            row = matches[0]

            means.append(float(row["violation_rate_mean"]))

            stds.append(float(row["violation_rate_sample_std"]))

        offset = (method_index - 1) * width

        axis.bar(
            x_values + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method.upper(),
        )

    axis.set_xticks(
        x_values,
        (
            "Autonomous driving",
            "Robotics",
        ),
    )

    axis.set_ylabel("Observed violation rate")

    axis.set_title("Clean Safety Violations — Mean ± Sample SD")

    axis.legend()

    axis.text(
        0,
        0.53,
        ("CLIPPING and LYAPUNOV:\n" "zero observed clean-evaluation violations"),
        ha="center",
        fontsize=9,
    )

    axis.text(
        1,
        0.055,
        ("Zero observed ≠ guaranteed safe\n" "Lyapunov filter is classical"),
        ha="center",
        fontsize=9,
    )

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    _save(
        figure,
        "figure-05-safety-violations.png",
    )


def _fixed_value(
    payload: dict[str, Any],
    *,
    domain: str,
    experiment: str,
    method: str,
    quantity: str,
) -> Any:
    matches = [
        row
        for row in payload["fixed_values"]
        if (
            row["domain"] == domain
            and row["experiment"] == experiment
            and row["method"] == method
            and row["quantity"] == quantity
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Unable to resolve fixed statistical " "evidence for cross-domain figure."
        )

    return matches[0]["value"]


def _record_mean(
    payload: dict[str, Any],
    *,
    domain: str,
    experiment: str,
    method: str,
    metric: str,
) -> float:
    matches = [
        row
        for row in payload["records"]
        if (
            row["domain"] == domain
            and row["experiment"] == experiment
            and row["method"] == method
            and row["metric"] == metric
            and row.get("condition") is None
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Unable to resolve statistical " "record for cross-domain figure."
        )

    return float(matches[0]["mean"])


def _figure_cross_domain() -> None:
    statistics = _load_json(STATISTICS_PATH)

    domains = (
        "autonomous_driving",
        "robotics",
    )

    matrix = np.zeros(
        (
            7,
            2,
        ),
        dtype=int,
    )

    labels = [
        "INT8 criterion",
        "SVD criterion",
        "TT/MPS criterion",
        "PPO target reach",
        "QML target reach",
        "Clipping violations",
        "Lyapunov violations",
    ]

    text_matrix: list[list[str]] = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    for domain_index, domain in enumerate(domains):
        int8 = bool(
            _fixed_value(
                statistics,
                domain=domain,
                experiment="compression",
                method="INT8",
                quantity="all_runs_pilot_feasible",
            )
        )

        svd_method = "SVD-75%" if domain == "autonomous_driving" else "SVD-50%"

        svd = bool(
            _fixed_value(
                statistics,
                domain=domain,
                experiment="compression",
                method=svd_method,
                quantity="all_runs_pilot_feasible",
            )
        )

        tt = bool(
            _fixed_value(
                statistics,
                domain=domain,
                experiment="compression",
                method="TT-rank-2",
                quantity="all_runs_pilot_feasible",
            )
        )

        ppo_reach = int(
            _fixed_value(
                statistics,
                domain=domain,
                experiment="rl_qml",
                method="full_ppo",
                quantity="target_reach_count",
            )
        )

        qml_reach = int(
            _fixed_value(
                statistics,
                domain=domain,
                experiment="rl_qml",
                method="qml",
                quantity="target_reach_count",
            )
        )

        clipping = _record_mean(
            statistics,
            domain=domain,
            experiment="safety_clean",
            method="clipping",
            metric="violation_step_rate",
        )

        lyapunov = _record_mean(
            statistics,
            domain=domain,
            experiment="safety_clean",
            method="lyapunov",
            metric="violation_step_rate",
        )

        values = (
            int8,
            svd,
            tt,
            ppo_reach == 3,
            qml_reach == 3,
            clipping == 0.0,
            lyapunov == 0.0,
        )

        display = (
            "PASS" if int8 else "FAIL",
            "PASS" if svd else "FAIL",
            "PASS" if tt else "FAIL",
            f"{ppo_reach}/3",
            f"{qml_reach}/3",
            ("0 observed" if clipping == 0.0 else "Nonzero"),
            ("0 observed" if lyapunov == 0.0 else "Nonzero"),
        )

        for row_index, value in enumerate(values):
            matrix[
                row_index,
                domain_index,
            ] = int(value)

            text_matrix[row_index].append(display[row_index])

    figure, axis = plt.subplots(figsize=(9, 7))

    axis.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(
        (
            0,
            1,
        ),
        (
            "Autonomous driving",
            "Robotics",
        ),
    )

    axis.set_yticks(
        np.arange(len(labels)),
        labels,
    )

    for row_index in range(len(labels)):
        for domain_index in range(2):
            axis.text(
                domain_index,
                row_index,
                text_matrix[row_index][domain_index],
                ha="center",
                va="center",
            )

    axis.set_title("Cross-Domain Phase 1 Evidence\n" "No synthetic aggregate score")

    _save(
        figure,
        "figure-06-cross-domain.png",
    )


def _figure_ablation() -> None:
    payload = _load_json(ABLATION_PATH)

    phase1 = payload["phase1"]

    if phase1["direct_count"] != ABLATION_DIRECT_COUNT:
        raise RuntimeError("Sprint 7.6 DIRECT count changed.")

    if phase1["component_only_count"] != ABLATION_COMPONENT_ONLY_COUNT:
        raise RuntimeError("Sprint 7.6 COMPONENT_ONLY count changed.")

    if phase1["not_evaluated_count"] != ABLATION_NOT_EVALUATED_COUNT:
        raise RuntimeError("Sprint 7.6 NOT_EVALUATED count changed.")

    configurations = (
        "baseline",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "full",
    )

    domains = (
        "autonomous_driving",
        "robotics",
    )

    matrix = np.ones(
        (
            len(configurations),
            len(domains),
        )
    )

    records = phase1["matrix"]

    for config_index, configuration in enumerate(configurations):
        for domain_index, domain in enumerate(domains):
            matches = [
                row
                for row in records
                if (row["configuration"] == configuration and row["domain"] == domain)
            ]

            if len(matches) != 1:
                raise RuntimeError("Incomplete full-system evidence matrix.")

            row = matches[0]

            if row["evidence_status"] != "component_only":
                raise RuntimeError("Unexpected direct full-system evidence.")

            if row["can_report_end_to_end_metrics"]:
                raise RuntimeError(
                    "Component-only evidence cannot " "report end-to-end metrics."
                )

    figure, axis = plt.subplots(figsize=(8, 8))

    axis.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=2,
    )

    axis.set_xticks(
        (
            0,
            1,
        ),
        (
            "Autonomous driving",
            "Robotics",
        ),
    )

    axis.set_yticks(
        np.arange(len(configurations)),
        [name.upper() if name != "baseline" else "Baseline" for name in configurations],
    )

    for row_index in range(len(configurations)):
        for column_index in range(len(domains)):
            axis.text(
                column_index,
                row_index,
                "COMPONENT\nONLY",
                ha="center",
                va="center",
                fontsize=8,
            )

    axis.set_title(
        "Full-System Factorial Evidence Status\n"
        "DIRECT 0 | COMPONENT_ONLY 16 | NOT_EVALUATED 0"
    )

    axis.text(
        0.5,
        8.35,
        (
            "No matched Compression × QML × Safety "
            "factorial pipeline was executed in Phase 1.\n"
            "Interaction effects remain a Phase 2 validation objective."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )

    _save(
        figure,
        "figure-07-ablation-evidence.png",
    )


def _captions() -> str:
    return """# Q-VLA Forge — Final Figure Captions

## Figure 1 — Unified Architecture

**Phase 1 architecture and integration boundary.**
Q-VLA Forge reused a common AI/DL architecture, interface conventions,
evaluation contracts, and safety framework across autonomous-driving and
robotics proxy domains while retaining domain-specific trained policy
weights and safety semantics. Dashed integration links represent the
intended system pathway and were not executed as one matched end-to-end
Phase 1 pipeline.

## Figure 2 — Compression Pareto

**Compression Pareto comparison.**
INT8 satisfied the frozen Phase 1 joint criterion of at least 2× storage
compression with no more than 5% relative MSE degradation in both proxy
domains. Evaluated SVD and TT/MPS configurations did not satisfy the joint
criterion. The shaded/threshold region is a frozen Phase 1 acceptance
criterion, not a quantum-advantage region.

## Figure 3 — Training Efficiency

**Training-efficiency evaluation.**
Driving trainable SVD reached its paired FP32 target across all three
locked seeds, but required more optimizer steps on average than the paired
FP32 reference. The evaluated driving TT/MPS, robotics SVD, and robotics
TT/MPS candidates did not reach their paired targets across the locked
three-seed protocol. Phase 1 therefore did not demonstrate a robust
10% or greater optimizer-step efficiency improvement across both domains.
Missing target-reaching quantities remain missing and are not censored to
the final training budget.

## Figure 4 — RL/QML Ablation

**RL/QML target attainment and actor compactness.**
Classical PPO reached the frozen target across all six domain-seed
evaluations, while the evaluated PQC policy reached none and the
parameter-matched classical control reached one. The PQC actor nevertheless
used substantially fewer trainable actor parameters. Compactness is reported
separately from sample efficiency; no quantum advantage or QML
sample-efficiency advantage is claimed.

## Figure 5 — Safety Violations

**Clean safety ablation.**
Clipping and the classical Lyapunov-guided filter produced zero observed
proxy violations under the frozen clean-evaluation contracts in both proxy
domains. These are empirical observations, not formal stability,
certification, production-safety, or real-world safety guarantees. The
safety layer retained privileged true simulator state during perception
perturbation experiments.

## Figure 6 — Cross-Domain Evidence

**Cross-domain Phase 1 comparison.**
The two proxy domains showed the same categorical result for the frozen
INT8, SVD, TT/MPS, PPO target-attainment, QML target-attainment, clipping,
and Lyapunov clean-safety checks. Metrics with different units or semantics
are not normalized into a synthetic overall score.

## Figure 7 — Full-System Ablation Evidence

**Full-system factorial evidence status.**
All sixteen domain/configuration cells in the Compression × QML × Safety
factorial are classified as COMPONENT_ONLY. No exact matched integrated
factorial configuration was directly executed in Phase 1, so no synthetic
full-system performance or interaction effect is reported. Matched
integrated execution remains a Phase 2 validation objective.
"""


def _write_manifest() -> None:
    contracts = build_all_figure_contracts()

    payload = {
        "sprint": "7.7",
        "status": "FROZEN",
        "new_training": (NEW_TRAINING),
        "new_experiments": (NEW_EXPERIMENTS),
        "new_scientific_results": (NEW_SCIENTIFIC_RESULTS),
        "generated_from_frozen_evidence": True,
        "figure_count": len(contracts),
        "required_figure_ids": list(FIGURE_IDS),
        "claim_controls": dict(CLAIM_CONTROLS),
        "figures": [contract.to_dict() for contract in contracts],
    }

    path = ROOT / FIGURE_MANIFEST_PATH

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_captions() -> None:
    path = ROOT / FIGURE_CAPTIONS_PATH

    path.write_text(
        _captions(),
        encoding="utf-8",
    )


def _validate_contract_sources() -> None:
    contracts = build_all_figure_contracts()

    missing = []

    for contract in contracts:
        for source in contract.source_artifacts:
            path = ROOT / source

            if not path.is_file():
                missing.append(source)

    if missing:
        raise FileNotFoundError(
            "Missing canonical figure provenance "
            "artifact(s): " + ", ".join(sorted(set(missing)))
        )


def build() -> None:
    """Build Sprint 7.7 final figure package."""

    output_dir = ROOT / OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_contract_sources()

    _figure_architecture()
    _figure_compression()
    _figure_training()
    _figure_rl_qml()
    _figure_safety()
    _figure_cross_domain()
    _figure_ablation()

    _write_manifest()
    _write_captions()

    print("=" * 68)
    print(" SPRINT 7.7 FINAL FIGURES BUILD")
    print("=" * 68)
    print()
    print("01 Unified architecture              PASS")
    print("02 Compression Pareto                PASS")
    print("03 Training efficiency               PASS")
    print("04 RL/QML sample efficiency          PASS")
    print("05 Safety violations                 PASS")
    print("06 Cross-domain comparison           PASS")
    print("07 Ablation evidence matrix          PASS")
    print()
    print("Figure manifest                     PASS")
    print("Proposal-ready captions             PASS")
    print("Frozen evidence only                PASS")
    print("Synthetic metrics                   BLOCKED")
    print("New experiments                     FALSE")
    print("New scientific results              FALSE")
    print()
    print("SPRINT 7.7 FINAL FIGURES BUILD: PASS")


if __name__ == "__main__":
    build()
