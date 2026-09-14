"""Generate Sprint 4.13 cross-domain RL figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

CROSS_DOMAIN_PATH = (
    Path("results") / "rl" / "cross-domain" / "sprint4-cross-domain.json"
)

FIGURE_DIRECTORY = Path("figures") / "rl"

NORMALIZED_AUC_FIGURE = FIGURE_DIRECTORY / "cross-domain-normalized-auc.png"

TARGET_REACH_FIGURE = FIGURE_DIRECTORY / "cross-domain-target-reach.png"

PARAMETER_COMPACTNESS_FIGURE = (
    FIGURE_DIRECTORY / "cross-domain-parameter-compactness.png"
)

MATCHED_DELTA_FIGURE = FIGURE_DIRECTORY / "cross-domain-matched-delta.png"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "full_ppo",
    "matched_classical",
    "hybrid_qml",
)

DISPLAY_DOMAINS = {
    "autonomous_driving": "Driving",
    "robotics": "Robotics",
}

DISPLAY_METHODS = {
    "full_ppo": "Full PPO",
    "matched_classical": "Matched Classical",
    "hybrid_qml": "Hybrid QML",
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def method_record(
    *,
    payload: dict[str, Any],
    domain: str,
    method: str,
) -> dict[str, Any]:
    """Return exactly one domain/method record."""

    matches = [
        record
        for record in payload["method_records"]
        if (record["domain"] == domain and record["method"] == method)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one record for {domain} {method}")

    return matches[0]


def matched_comparison(
    *,
    payload: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    """Return exactly one matched representation comparison."""

    matches = [
        record
        for record in payload["matched_representation_comparison"]
        if record["domain"] == domain
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one matched comparison for {domain}")

    return matches[0]


def draw_normalized_auc(
    *,
    payload: dict[str, Any],
) -> None:
    """Plot normalized AUC mean ± sample SD across domains."""

    x_positions = [
        0.0,
        1.0,
    ]

    width = 0.22

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1) * width

        means = [
            float(
                method_record(
                    payload=payload,
                    domain=domain,
                    method=method,
                )["normalized_auc_mean"]
            )
            for domain in DOMAINS
        ]

        sds = [
            float(
                method_record(
                    payload=payload,
                    domain=domain,
                    method=method,
                )["normalized_auc_sd"]
            )
            for domain in DOMAINS
        ]

        axis.bar(
            [value + offset for value in x_positions],
            means,
            width=width,
            yerr=sds,
            capsize=4,
            label=(DISPLAY_METHODS[method]),
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xticks(
        x_positions,
        [DISPLAY_DOMAINS[domain] for domain in DOMAINS],
    )

    axis.set_ylabel("Mean normalized learning-curve AUC")

    axis.set_title("Cross-Domain Normalized Learning Performance")

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        NORMALIZED_AUC_FIGURE,
        dpi=200,
    )

    plt.close(figure)


def draw_target_reach(
    *,
    payload: dict[str, Any],
) -> None:
    """Plot target reach count out of three seeds."""

    x_positions = [
        0.0,
        1.0,
    ]

    width = 0.22

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1) * width

        reaches = [
            int(
                method_record(
                    payload=payload,
                    domain=domain,
                    method=method,
                )["target_reach_count"]
            )
            for domain in DOMAINS
        ]

        axis.bar(
            [value + offset for value in x_positions],
            reaches,
            width=width,
            label=(DISPLAY_METHODS[method]),
        )

    axis.set_xticks(
        x_positions,
        [DISPLAY_DOMAINS[domain] for domain in DOMAINS],
    )

    axis.set_ylim(
        0.0,
        3.25,
    )

    axis.set_yticks(
        [
            0,
            1,
            2,
            3,
        ]
    )

    axis.set_ylabel("Frozen target reaches / 3 seeds")

    axis.set_title("Cross-Domain Frozen Target Reach")

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        TARGET_REACH_FIGURE,
        dpi=200,
    )

    plt.close(figure)


def draw_parameter_compactness(
    *,
    payload: dict[str, Any],
) -> None:
    """Plot trainable actor parameter counts."""

    x_positions = [
        0.0,
        1.0,
    ]

    width = 0.22

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1) * width

        parameters = [
            int(
                method_record(
                    payload=payload,
                    domain=domain,
                    method=method,
                )["actor_parameters"]
            )
            for domain in DOMAINS
        ]

        axis.bar(
            [value + offset for value in x_positions],
            parameters,
            width=width,
            label=(DISPLAY_METHODS[method]),
        )

    axis.set_xticks(
        x_positions,
        [DISPLAY_DOMAINS[domain] for domain in DOMAINS],
    )

    axis.set_ylabel("Trainable actor parameters")

    axis.set_title("Cross-Domain Actor Parameter Compactness")

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        PARAMETER_COMPACTNESS_FIGURE,
        dpi=200,
    )

    plt.close(figure)


def draw_matched_delta(
    *,
    payload: dict[str, Any],
) -> None:
    """Plot matched-classical minus QML normalized AUC."""

    domains = list(DOMAINS)

    deltas = [
        float(
            matched_comparison(
                payload=payload,
                domain=domain,
            )["matched_minus_qml_auc"]
        )
        for domain in domains
    ]

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    axis.bar(
        [DISPLAY_DOMAINS[domain] for domain in domains],
        deltas,
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_ylabel("Matched Classical AUC − Hybrid QML AUC")

    axis.set_title("Matched-Budget Representation Effect Across Domains")

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        MATCHED_DELTA_FIGURE,
        dpi=200,
    )

    plt.close(figure)


def main() -> None:
    """Generate Sprint 4.13 cross-domain figures."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = load_json(CROSS_DOMAIN_PATH)

    if payload["sprint"] != "4.13":
        raise RuntimeError("unexpected cross-domain analysis artifact")

    if payload["new_training_performed"] is not False:
        raise RuntimeError("Sprint 4.13 must remain analysis-only")

    draw_normalized_auc(payload=payload)

    draw_target_reach(payload=payload)

    draw_parameter_compactness(payload=payload)

    draw_matched_delta(payload=payload)

    outputs = (
        NORMALIZED_AUC_FIGURE,
        TARGET_REACH_FIGURE,
        PARAMETER_COMPACTNESS_FIGURE,
        MATCHED_DELTA_FIGURE,
    )

    print()
    print("SPRINT 4.13 CROSS-DOMAIN FIGURES")
    print()

    for path in outputs:
        if not path.exists():
            raise RuntimeError(f"missing figure: {path}")

        size = path.stat().st_size

        if size <= 0:
            raise RuntimeError(f"empty figure: {path}")

        print(
            path,
            size,
            "bytes",
        )

    print()

    print(
        "New training performed:",
        False,
    )

    print("SPRINT 4.13 CROSS-DOMAIN FIGURE GENERATION: PASS")


if __name__ == "__main__":
    main()
