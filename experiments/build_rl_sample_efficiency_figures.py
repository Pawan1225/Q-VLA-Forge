"""Generate Sprint 4.11 RL sample-efficiency figures."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ANALYSIS_PATH = Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.json"

FIGURE_DIRECTORY = Path("figures") / "rl"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

POLICIES = (
    "ppo",
    "qml",
)

DISPLAY_NAMES = {
    "autonomous_driving": "Autonomous Driving",
    "robotics": "Robotics",
    "ppo": "Classical PPO",
    "qml": "Hybrid QML",
}

ACTOR_PARAMETERS = {
    "autonomous_driving": {
        "ppo": 1318,
        "qml": 54,
    },
    "robotics": {
        "ppo": 1382,
        "qml": 62,
    },
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd_curves(
    records: list[dict[str, Any]],
    field: str,
) -> tuple[
    list[float],
    list[float],
]:
    """Calculate pointwise mean and sample SD."""

    if len(records) != 3:
        raise ValueError("exactly three seed records required")

    point_count = len(records[0][field])

    means: list[float] = []
    sds: list[float] = []

    for index in range(point_count):
        values = [float(record[field][index]) for record in records]

        means.append(float(statistics.mean(values)))

        sds.append(float(statistics.stdev(values)))

    return means, sds


def records_for(
    *,
    payload: dict[str, Any],
    domain: str,
    policy: str,
) -> list[dict[str, Any]]:
    """Return the three principal seed records."""

    records = payload["domains"][domain]["policies"][policy]["records"]

    if len(records) != 3:
        raise RuntimeError(f"expected three records for {domain}/{policy}")

    return records


def plot_raw_learning_curve(
    *,
    payload: dict[str, Any],
    domain: str,
    output_path: Path,
) -> None:
    """Plot held-out reward versus environment steps."""

    figure = plt.figure(figsize=(9, 6))

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for policy in POLICIES:
        records = records_for(
            payload=payload,
            domain=domain,
            policy=policy,
        )

        steps = [int(step) for step in records[0]["steps"]]

        means, sds = mean_sd_curves(
            records,
            "rewards",
        )

        lower = [
            mean - sd
            for mean, sd in zip(
                means,
                sds,
                strict=True,
            )
        ]

        upper = [
            mean + sd
            for mean, sd in zip(
                means,
                sds,
                strict=True,
            )
        ]

        line = axis.plot(
            steps,
            means,
            marker="o",
            markersize=3,
            label=DISPLAY_NAMES[policy],
        )[0]

        axis.fill_between(
            steps,
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

    axis.set_title(f"{DISPLAY_NAMES[domain]} Learning Curves")

    axis.set_xlabel("Environment Steps")

    axis.set_ylabel("Held-Out Mean Reward")

    axis.legend()

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_normalized_progress(
    *,
    payload: dict[str, Any],
    domain: str,
    output_path: Path,
) -> None:
    """Plot normalized target progress."""

    figure = plt.figure(figsize=(9, 6))

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for policy in POLICIES:
        records = records_for(
            payload=payload,
            domain=domain,
            policy=policy,
        )

        steps = [int(step) for step in records[0]["steps"]]

        means, sds = mean_sd_curves(
            records,
            "normalized_progress",
        )

        lower = [
            mean - sd
            for mean, sd in zip(
                means,
                sds,
                strict=True,
            )
        ]

        upper = [
            mean + sd
            for mean, sd in zip(
                means,
                sds,
                strict=True,
            )
        ]

        line = axis.plot(
            steps,
            means,
            marker="o",
            markersize=3,
            label=DISPLAY_NAMES[policy],
        )[0]

        axis.fill_between(
            steps,
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

    axis.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.0,
        label="Random Reference",
    )

    axis.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
        label="Paired Frozen Target",
    )

    axis.set_title(f"{DISPLAY_NAMES[domain]} Normalized Target Progress")

    axis.set_xlabel("Environment Steps")

    axis.set_ylabel("Normalized Target Progress")

    axis.legend()

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_compactness_tradeoff(
    *,
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot actor parameters versus best normalized progress."""

    figure = plt.figure(figsize=(9, 6))

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for domain in DOMAINS:
        for policy in POLICIES:
            aggregate = payload["domains"][domain]["policies"][policy]["aggregate"]

            actor_parameters = ACTOR_PARAMETERS[domain][policy]

            best_progress = float(aggregate["best_normalized_progress"]["mean"])

            axis.scatter(
                actor_parameters,
                best_progress,
                s=80,
            )

            axis.annotate(
                (f"{DISPLAY_NAMES[domain]}\n" f"{DISPLAY_NAMES[policy]}"),
                (
                    actor_parameters,
                    best_progress,
                ),
                xytext=(
                    8,
                    6,
                ),
                textcoords="offset points",
            )

    axis.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
        label="Paired Frozen Target",
    )

    axis.set_title("Actor Compactness–Performance Trade-Off")

    axis.set_xlabel("Trainable Actor Parameters")

    axis.set_ylabel("Mean Best Normalized Target Progress")

    axis.legend()

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Generate all Sprint 4.11 figures."""

    payload = load_json(ANALYSIS_PATH)

    if payload["sprint"] != "4.11":
        raise RuntimeError("Sprint 4.11 analysis artifact required")

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    driving_raw = FIGURE_DIRECTORY / "driving-learning-curves.png"

    robotics_raw = FIGURE_DIRECTORY / "robotics-learning-curves.png"

    driving_normalized = FIGURE_DIRECTORY / "driving-normalized-progress.png"

    robotics_normalized = FIGURE_DIRECTORY / "robotics-normalized-progress.png"

    compactness = FIGURE_DIRECTORY / "compactness-performance-tradeoff.png"

    plot_raw_learning_curve(
        payload=payload,
        domain="autonomous_driving",
        output_path=driving_raw,
    )

    plot_raw_learning_curve(
        payload=payload,
        domain="robotics",
        output_path=robotics_raw,
    )

    plot_normalized_progress(
        payload=payload,
        domain="autonomous_driving",
        output_path=driving_normalized,
    )

    plot_normalized_progress(
        payload=payload,
        domain="robotics",
        output_path=robotics_normalized,
    )

    plot_compactness_tradeoff(
        payload=payload,
        output_path=compactness,
    )

    figures = (
        driving_raw,
        robotics_raw,
        driving_normalized,
        robotics_normalized,
        compactness,
    )

    print()
    print("===========================================")

    print(" SPRINT 4.11 SAMPLE-EFFICIENCY FIGURES")

    print("===========================================")

    print()

    for path in figures:
        print(
            path,
            "bytes=",
            path.stat().st_size,
        )

    print()

    print(
        "Figures generated:",
        len(figures),
    )

    print(
        "New training:",
        False,
    )

    print()

    print("SPRINT 4.11 FIGURE GENERATION: PASS")


if __name__ == "__main__":
    main()
