"""Generate Sprint 4.12 matched-budget ablation figures."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

SPRINT411_PATH = Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.json"

SPRINT412_PATH = (
    Path("results") / "rl" / "ablation" / "sprint4-classical-vs-qml-ablation.json"
)

ABLATION_DIRECTORY = Path("results") / "rl" / "ablation"

FIGURE_DIRECTORY = Path("figures") / "rl"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

DISPLAY_NAMES = {
    "autonomous_driving": "Autonomous Driving",
    "robotics": "Robotics",
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def matched_run_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return one matched-classical principal run."""

    return ABLATION_DIRECTORY / f"matched-classical-{domain}-seed-{seed}.json"


def mean_sd_curve(
    curves: list[list[float]],
) -> tuple[
    list[float],
    list[float],
]:
    """Return pointwise arithmetic mean and sample SD."""

    if len(curves) != 3:
        raise ValueError("expected exactly three curves")

    point_count = len(curves[0])

    if any(len(curve) != point_count for curve in curves):
        raise ValueError("curve lengths must match")

    means: list[float] = []
    sds: list[float] = []

    for index in range(point_count):
        values = [float(curve[index]) for curve in curves]

        means.append(float(statistics.mean(values)))

        sds.append(float(statistics.stdev(values)))

    return (
        means,
        sds,
    )


def sprint411_records(
    *,
    sprint411: dict[str, Any],
    domain: str,
    policy: str,
) -> list[dict[str, Any]]:
    """Return the frozen Sprint 4.11 run records."""

    records = sprint411["domains"][domain]["policies"][policy]["records"]

    if len(records) != 3:
        raise RuntimeError(f"{domain} {policy} must contain three records")

    records = sorted(
        records,
        key=lambda record: int(record["seed"]),
    )

    expected_seeds = list(SEEDS)

    actual_seeds = [int(record["seed"]) for record in records]

    if actual_seeds != expected_seeds:
        raise RuntimeError(f"{domain} {policy} seed mismatch")

    return records


def sprint411_curve(
    *,
    sprint411: dict[str, Any],
    domain: str,
    policy: str,
) -> tuple[
    list[int],
    list[float],
    list[float],
]:
    """Return Sprint 4.11 mean ± sample-SD normalized progress."""

    records = sprint411_records(
        sprint411=sprint411,
        domain=domain,
        policy=policy,
    )

    steps = [int(value) for value in records[0]["steps"]]

    curves = [
        [float(value) for value in record["normalized_progress"]] for record in records
    ]

    for record in records[1:]:
        record_steps = [int(value) for value in record["steps"]]

        if record_steps != steps:
            raise RuntimeError(f"{domain} {policy} evaluation schedule mismatch")

    mean, sd = mean_sd_curve(curves)

    return (
        steps,
        mean,
        sd,
    )


def matched_normalized_curves(
    *,
    sprint411: dict[str, Any],
    domain: str,
) -> tuple[
    list[int],
    list[list[float]],
]:
    """Return matched-classical normalized-progress trajectories."""

    qml_records = sprint411_records(
        sprint411=sprint411,
        domain=domain,
        policy="qml",
    )

    qml_by_seed = {int(record["seed"]): record for record in qml_records}

    curves: list[list[float]] = []

    common_steps: list[int] | None = None

    for seed in SEEDS:
        matched_run = load_json(
            matched_run_path(
                domain=domain,
                seed=seed,
            )
        )

        if matched_run["domain"] != domain:
            raise RuntimeError(f"{domain} seed {seed} matched domain mismatch")

        if int(matched_run["seed"]) != seed:
            raise RuntimeError(f"{domain} seed {seed} matched seed mismatch")

        reference_record = qml_by_seed[seed]

        random_reference = float(reference_record["random_reference"])

        target_reward = float(reference_record["target_reward"])

        denominator = target_reward - random_reference

        if denominator <= 0.0:
            raise RuntimeError(
                f"{domain} seed {seed} invalid normalization denominator"
            )

        evaluations = matched_run["evaluations"]

        if len(evaluations) != 21:
            raise RuntimeError(f"{domain} seed {seed} must contain 21 evaluations")

        steps = [int(evaluation["environment_steps"]) for evaluation in evaluations]

        rewards = [float(evaluation["mean_reward"]) for evaluation in evaluations]

        curve = [(reward - random_reference) / denominator for reward in rewards]

        if common_steps is None:
            common_steps = steps

        elif common_steps != steps:
            raise RuntimeError(f"{domain} matched evaluation schedules differ")

        curves.append(curve)

    if common_steps is None:
        raise RuntimeError(f"no matched curves found for {domain}")

    return (
        common_steps,
        curves,
    )


def lower_curve(
    *,
    mean: list[float],
    sd: list[float],
) -> list[float]:
    """Return mean minus SD."""

    return [
        mean_value - sd_value
        for (
            mean_value,
            sd_value,
        ) in zip(
            mean,
            sd,
            strict=True,
        )
    ]


def upper_curve(
    *,
    mean: list[float],
    sd: list[float],
) -> list[float]:
    """Return mean plus SD."""

    return [
        mean_value + sd_value
        for (
            mean_value,
            sd_value,
        ) in zip(
            mean,
            sd,
            strict=True,
        )
    ]


def draw_domain_ablation(
    *,
    sprint411: dict[str, Any],
    domain: str,
    output_path: Path,
) -> None:
    """Plot full PPO, matched classical, and QML progress."""

    (
        ppo_steps,
        ppo_mean,
        ppo_sd,
    ) = sprint411_curve(
        sprint411=sprint411,
        domain=domain,
        policy="ppo",
    )

    (
        qml_steps,
        qml_mean,
        qml_sd,
    ) = sprint411_curve(
        sprint411=sprint411,
        domain=domain,
        policy="qml",
    )

    (
        matched_steps,
        matched_curves,
    ) = matched_normalized_curves(
        sprint411=sprint411,
        domain=domain,
    )

    (
        matched_mean,
        matched_sd,
    ) = mean_sd_curve(matched_curves)

    if not (ppo_steps == qml_steps == matched_steps):
        raise RuntimeError(f"{domain} evaluation schedules do not match")

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    axis.plot(
        ppo_steps,
        ppo_mean,
        label="Full PPO",
    )

    axis.fill_between(
        ppo_steps,
        lower_curve(
            mean=ppo_mean,
            sd=ppo_sd,
        ),
        upper_curve(
            mean=ppo_mean,
            sd=ppo_sd,
        ),
        alpha=0.2,
    )

    axis.plot(
        matched_steps,
        matched_mean,
        label="Matched Classical",
    )

    axis.fill_between(
        matched_steps,
        lower_curve(
            mean=matched_mean,
            sd=matched_sd,
        ),
        upper_curve(
            mean=matched_mean,
            sd=matched_sd,
        ),
        alpha=0.2,
    )

    axis.plot(
        qml_steps,
        qml_mean,
        label="Hybrid QML",
    )

    axis.fill_between(
        qml_steps,
        lower_curve(
            mean=qml_mean,
            sd=qml_sd,
        ),
        upper_curve(
            mean=qml_mean,
            sd=qml_sd,
        ),
        alpha=0.2,
    )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Random reference",
    )

    axis.axhline(
        1.0,
        linestyle=":",
        linewidth=1.0,
        label="Frozen target",
    )

    axis.set_title(f"{DISPLAY_NAMES[domain]} — " "Matched-Budget Ablation")

    axis.set_xlabel("Environment steps")

    axis.set_ylabel("Normalized target progress")

    axis.legend()

    axis.grid(alpha=0.2)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def draw_paired_auc(
    *,
    ablation: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot seed-level matched-classical and QML normalized AUC."""

    labels: list[str] = []
    matched_values: list[float] = []
    qml_values: list[float] = []

    for domain in DOMAINS:
        for seed in SEEDS:
            matches = [
                record
                for record in ablation["paired_records"]
                if (record["domain"] == domain and int(record["seed"]) == seed)
            ]

            if len(matches) != 1:
                raise RuntimeError(f"{domain} seed {seed} paired record lookup failed")

            record = matches[0]

            short_domain = "Drive" if domain == "autonomous_driving" else "Robot"

            labels.append(f"{short_domain}-{seed}")

            matched_values.append(float(record["matched_normalized_auc"]))

            qml_values.append(float(record["qml_normalized_auc"]))

    x_values = list(range(len(labels)))

    width = 0.35

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.bar(
        [value - width / 2 for value in x_values],
        matched_values,
        width=width,
        label="Matched Classical",
    )

    axis.bar(
        [value + width / 2 for value in x_values],
        qml_values,
        width=width,
        label="Hybrid QML",
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xticks(
        x_values,
        labels,
        rotation=30,
        ha="right",
    )

    axis.set_ylabel("Normalized learning-curve AUC")

    axis.set_title("Paired Matched-Classical vs Hybrid-QML AUC")

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def draw_budget_performance(
    *,
    sprint411: dict[str, Any],
    ablation: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot actor-parameter budget against mean normalized AUC."""

    points: list[
        tuple[
            str,
            int,
            float,
        ]
    ] = []

    for domain in DOMAINS:
        sprint_domain = sprint411["domains"][domain]["policies"]

        ablation_methods = ablation["domains"][domain]["methods"]

        points.extend(
            [
                (
                    (f"{DISPLAY_NAMES[domain]} " "Full PPO"),
                    int(sprint_domain["ppo"]["aggregate"]["actor_parameters"]),
                    float(sprint_domain["ppo"]["aggregate"]["normalized_auc"]["mean"]),
                ),
                (
                    (f"{DISPLAY_NAMES[domain]} " "Matched"),
                    int(ablation_methods["matched_classical"]["actor_parameters"]),
                    float(
                        ablation_methods["matched_classical"]["normalized_auc"]["mean"]
                    ),
                ),
                (
                    (f"{DISPLAY_NAMES[domain]} " "QML"),
                    int(ablation_methods["qml"]["actor_parameters"]),
                    float(ablation_methods["qml"]["normalized_auc"]["mean"]),
                ),
            ]
        )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for (
        label,
        parameters,
        normalized_auc,
    ) in points:
        axis.scatter(
            parameters,
            normalized_auc,
            s=70,
        )

        axis.annotate(
            label,
            (
                parameters,
                normalized_auc,
            ),
            xytext=(
                6,
                5,
            ),
            textcoords="offset points",
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xlabel("Trainable actor parameters")

    axis.set_ylabel("Mean normalized learning-curve AUC")

    axis.set_title("Performance at Equal Actor Parameter Budgets")

    axis.grid(
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def main() -> None:
    """Generate Sprint 4.12 proposal figures."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    sprint411 = load_json(SPRINT411_PATH)

    ablation = load_json(SPRINT412_PATH)

    outputs = [
        (
            "driving",
            FIGURE_DIRECTORY / "driving-matched-ablation.png",
        ),
        (
            "robotics",
            FIGURE_DIRECTORY / "robotics-matched-ablation.png",
        ),
        (
            "auc",
            FIGURE_DIRECTORY / "matched-ablation-auc.png",
        ),
        (
            "budget",
            FIGURE_DIRECTORY / "matched-budget-performance.png",
        ),
    ]

    draw_domain_ablation(
        sprint411=sprint411,
        domain="autonomous_driving",
        output_path=(outputs[0][1]),
    )

    draw_domain_ablation(
        sprint411=sprint411,
        domain="robotics",
        output_path=(outputs[1][1]),
    )

    draw_paired_auc(
        ablation=ablation,
        output_path=(outputs[2][1]),
    )

    draw_budget_performance(
        sprint411=sprint411,
        ablation=ablation,
        output_path=(outputs[3][1]),
    )

    print()
    print("SPRINT 4.12 FIGURES")
    print()

    for (
        label,
        path,
    ) in outputs:
        if not path.exists():
            raise RuntimeError(f"figure missing: {path}")

        size = path.stat().st_size

        if size <= 0:
            raise RuntimeError(f"empty figure: {path}")

        print(
            label,
            path,
            size,
            "bytes",
        )

    print()
    print(
        "New training performed:",
        False,
    )

    print("SPRINT 4.12 FIGURE GENERATION: PASS")


if __name__ == "__main__":
    main()
