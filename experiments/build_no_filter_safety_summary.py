"""Build Sprint 5.4 no-filter safety summary artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RUN_DIRECTORY = ROOT / "results" / "safety" / "baseline" / "runs"

OUTPUT_DIRECTORY = ROOT / "results" / "safety" / "baseline"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)


def sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def load_run(
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    path = RUN_DIRECTORY / f"{domain}-seed-{seed}.json"

    if not path.exists():
        raise FileNotFoundError(f"missing principal run: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload["domain"] != domain:
        raise ValueError("principal-run domain mismatch")

    if payload["principal_seed"] != seed:
        raise ValueError("principal-run seed mismatch")

    if payload["safety_method"] != "none":
        raise ValueError("principal run is not NONE")

    if payload["robustness_condition"] != "clean":
        raise ValueError("principal run is not CLEAN")

    if payload["new_training_performed"] is not False:
        raise ValueError("principal run reports new training")

    if payload["episode_count"] != 20:
        raise ValueError("principal run does not contain 20 episodes")

    return payload


def aggregate_domain(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [run["summary"] for run in runs]

    violation_rates = [float(summary["violation_step_rate"]) for summary in summaries]

    constraint_rates = [
        float(summary["constraint_violation_rate"]) for summary in summaries
    ]

    critical_rates = [
        float(summary["critical_violation_step_rate"]) for summary in summaries
    ]

    reward_means = [float(summary["mean_reward"]) for summary in summaries]

    success_rates = [float(summary["success_rate"]) for summary in summaries]

    category_names = sorted(
        {
            category
            for summary in summaries
            for category in (summary["category_violation_counts"])
        }
    )

    category_breakdown = {}

    for category in category_names:
        rates = [
            float(
                summary["category_violation_rates"].get(
                    category,
                    0.0,
                )
            )
            for summary in summaries
        ]

        counts = [
            int(
                summary["category_violation_counts"].get(
                    category,
                    0,
                )
            )
            for summary in summaries
        ]

        category_breakdown[category] = {
            "counts_by_seed": {
                str(seed): count
                for seed, count in zip(
                    SEEDS,
                    counts,
                    strict=True,
                )
            },
            "rate_mean": float(statistics.mean(rates)),
            "rate_sample_sd": (sample_sd(rates)),
        }

    return {
        "principal_seed_count": 3,
        "episode_count_per_seed": 20,
        "total_episode_count": 60,
        "violation_step_rate": {
            "mean": float(statistics.mean(violation_rates)),
            "sample_sd": (sample_sd(violation_rates)),
        },
        "constraint_violation_rate": {
            "mean": float(statistics.mean(constraint_rates)),
            "sample_sd": (sample_sd(constraint_rates)),
        },
        "critical_violation_step_rate": {
            "mean": float(statistics.mean(critical_rates)),
            "sample_sd": (sample_sd(critical_rates)),
        },
        "reward": {
            "mean_of_seed_means": float(statistics.mean(reward_means)),
            "sample_sd_of_seed_means": (sample_sd(reward_means)),
        },
        "success": {
            "mean_of_seed_rates": float(statistics.mean(success_rates)),
            "sample_sd_of_seed_rates": (sample_sd(success_rates)),
        },
        "baseline_has_observed_violations": (
            any(run["baseline_has_observed_violations"] for run in runs)
        ),
        "category_breakdown": (category_breakdown),
    }


def build_csv(
    runs: list[dict[str, Any]],
) -> None:
    path = OUTPUT_DIRECTORY / "sprint5-no-filter-safety-summary.csv"

    fieldnames = [
        "domain",
        "principal_seed",
        "episode_count",
        "total_environment_steps",
        "mean_reward",
        "reward_sample_sd",
        "success_rate",
        "mean_episode_length",
        "violation_step_count",
        "violation_step_rate",
        "constraint_violation_count",
        "constraint_violation_rate",
        "critical_violation_step_count",
        "critical_violation_step_rate",
        "intervention_count",
        "intervention_rate",
        "mean_action_correction_l2",
        "p95_action_correction_l2",
        "max_action_correction_l2",
        "policy_action_out_of_bounds_count",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for run in runs:
            summary = run["summary"]

            writer.writerow(
                {
                    field: (
                        run["domain"]
                        if field == "domain"
                        else (
                            run["principal_seed"]
                            if field == "principal_seed"
                            else summary[field]
                        )
                    )
                    for field in fieldnames
                }
            )


def build_markdown(
    *,
    aggregate: dict[
        str,
        Any,
    ],
) -> None:
    lines = [
        "# Sprint 5.4 - No-Filter Safety Baseline",
        "",
        "Method: `none`",
        "",
        "Condition: `clean`",
        "",
        "New training performed: `false`",
        "",
        "Principal seeds: `42, 123, 456`",
        "",
        "Evaluation seeds per cell: `20000..20019`",
        "",
    ]

    for domain in DOMAINS:
        data = aggregate["domains"][domain]

        lines.extend(
            [
                f"## {domain}",
                "",
                (
                    "Violation-step rate: "
                    f"{data['violation_step_rate']['mean']:.6f} "
                    "± "
                    f"{data['violation_step_rate']['sample_sd']:.6f}"
                ),
                "",
                (
                    "Constraint violation rate: "
                    f"{data['constraint_violation_rate']['mean']:.6f} "
                    "± "
                    f"{data['constraint_violation_rate']['sample_sd']:.6f}"
                ),
                "",
                (
                    "Critical violation-step rate: "
                    f"{data['critical_violation_step_rate']['mean']:.6f} "
                    "± "
                    f"{data['critical_violation_step_rate']['sample_sd']:.6f}"
                ),
                "",
                (
                    "Reward: "
                    f"{data['reward']['mean_of_seed_means']:.6f} "
                    "± "
                    f"{data['reward']['sample_sd_of_seed_means']:.6f}"
                ),
                "",
                (
                    "Success rate: "
                    f"{data['success']['mean_of_seed_rates']:.6f} "
                    "± "
                    f"{data['success']['sample_sd_of_seed_rates']:.6f}"
                ),
                "",
                (
                    "Observed clean-condition violations: "
                    f"`{str(data['baseline_has_observed_violations']).lower()}`"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            (
                "This artifact establishes the clean-condition "
                "no-filter control only."
            ),
            "",
            ("Clipping improvement: `NOT_YET_TESTED`"),
            "",
            ("Lyapunov improvement: `NOT_YET_TESTED`"),
            "",
            ("Perturbation robustness: `NOT_YET_TESTED`"),
            "",
            ("No production or formal safety guarantee is claimed."),
            "",
        ]
    )

    path = OUTPUT_DIRECTORY / "sprint5-no-filter-safety-summary.md"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []

    domain_runs = {}

    for domain in DOMAINS:
        current = []

        for seed in SEEDS:
            run = load_run(
                domain=domain,
                seed=seed,
            )

            runs.append(run)
            current.append(run)

        domain_runs[domain] = current

    aggregate = {
        "sprint": "5.4",
        "artifact": ("no-filter-safety-baseline-summary"),
        "method": "none",
        "condition": "clean",
        "new_training_performed": False,
        "principal_seeds": list(SEEDS),
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "domains": {
            domain: aggregate_domain(domain_runs[domain]) for domain in DOMAINS
        },
        "claims": {
            "baseline_measurement_supported": True,
            "clipping_improvement": ("NOT_YET_TESTED"),
            "lyapunov_improvement": ("NOT_YET_TESTED"),
            "robustness_improvement": ("NOT_YET_TESTED"),
            "formal_safety_guarantee": False,
            "production_safety_claim": False,
        },
    }

    json_path = OUTPUT_DIRECTORY / "sprint5-no-filter-safety-summary.json"

    json_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build_csv(runs)

    build_markdown(aggregate=aggregate)

    print("SPRINT 5.4 NO-FILTER SUMMARY BUILT")

    for domain in DOMAINS:
        data = aggregate["domains"][domain]

        print()
        print(domain)
        print(
            "  violation-step rate:",
            f"{data['violation_step_rate']['mean']:.6f}",
            "+/-",
            f"{data['violation_step_rate']['sample_sd']:.6f}",
        )
        print(
            "  reward:",
            f"{data['reward']['mean_of_seed_means']:.6f}",
            "+/-",
            f"{data['reward']['sample_sd_of_seed_means']:.6f}",
        )
        print(
            "  success:",
            f"{data['success']['mean_of_seed_rates']:.6f}",
            "+/-",
            f"{data['success']['sample_sd_of_seed_rates']:.6f}",
        )


if __name__ == "__main__":
    main()
