"""Build Sprint 5.5 NONE-vs-CLIPPING safety comparison artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

BASELINE_RUN_DIRECTORY = ROOT / "results" / "safety" / "baseline" / "runs"

CLIPPING_RUN_DIRECTORY = ROOT / "results" / "safety" / "clipping" / "runs"

OUTPUT_DIRECTORY = ROOT / "results" / "safety" / "clipping"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

MINIMUM_MEAN_VIOLATION_REDUCTION = 0.20
MAXIMUM_REWARD_DEGRADATION = 0.10
MAXIMUM_SUCCESS_RATE_DROP = 0.10


def sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing evidence artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def relative_violation_reduction(
    *,
    baseline_rate: float,
    clipping_rate: float,
) -> float | None:
    if baseline_rate == 0.0:
        return None

    return (baseline_rate - clipping_rate) / baseline_rate


def reward_degradation_fraction(
    *,
    baseline_reward: float,
    clipping_reward: float,
) -> float:
    """Return positive fractional degradation; improvements return zero."""

    reward_delta = clipping_reward - baseline_reward

    if reward_delta >= 0.0:
        return 0.0

    if baseline_reward == 0.0:
        return float("inf")

    return (baseline_reward - clipping_reward) / abs(baseline_reward)


def load_pair(
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    baseline_path = BASELINE_RUN_DIRECTORY / f"{domain}-seed-{seed}.json"

    clipping_path = CLIPPING_RUN_DIRECTORY / f"{domain}-seed-{seed}.json"

    baseline = load_json(baseline_path)

    clipping = load_json(clipping_path)

    if baseline["domain"] != domain or clipping["domain"] != domain:
        raise ValueError("domain mismatch in paired evidence")

    if baseline["principal_seed"] != seed or clipping["principal_seed"] != seed:
        raise ValueError("principal-seed mismatch in paired evidence")

    if baseline["safety_method"] != "none":
        raise ValueError("baseline evidence is not NONE")

    if clipping["safety_method"] != "clipping":
        raise ValueError("intervention evidence is not CLIPPING")

    if (
        baseline["robustness_condition"] != "clean"
        or clipping["robustness_condition"] != "clean"
    ):
        raise ValueError("Sprint 5.5 requires CLEAN evidence")

    if baseline["policy_checkpoint_sha256"] != clipping["policy_checkpoint_sha256"]:
        raise ValueError("NONE and CLIPPING checkpoint hashes differ")

    if baseline["training_budget"] != clipping["training_budget"]:
        raise ValueError("NONE and CLIPPING training budgets differ")

    if baseline["checkpoint_selection"] != clipping["checkpoint_selection"]:
        raise ValueError("NONE and CLIPPING checkpoint selections differ")

    if baseline["evaluation_seeds"] != clipping["evaluation_seeds"]:
        raise ValueError("NONE and CLIPPING evaluation seeds differ")

    if baseline["episode_count"] != 20 or clipping["episode_count"] != 20:
        raise ValueError("paired cells must each contain 20 episodes")

    none_summary = baseline["summary"]

    clipping_summary = clipping["summary"]

    none_rate = float(none_summary["violation_step_rate"])

    clipping_rate = float(clipping_summary["executed_violation_step_rate"])

    none_reward = float(none_summary["mean_reward"])

    clipping_reward = float(clipping_summary["mean_reward"])

    none_success = float(none_summary["success_rate"])

    clipping_success = float(clipping_summary["success_rate"])

    if none_rate > 0.0:
        seed_safety_requirement = clipping_rate < none_rate
    else:
        seed_safety_requirement = clipping_rate <= none_rate

    relative_reduction = relative_violation_reduction(
        baseline_rate=none_rate,
        clipping_rate=clipping_rate,
    )

    reward_delta = clipping_reward - none_reward

    success_delta = clipping_success - none_success

    return {
        "domain": domain,
        "principal_seed": seed,
        "checkpoint_sha256": (clipping["policy_checkpoint_sha256"]),
        "none_violation_step_rate": (none_rate),
        "clipping_violation_step_rate": (clipping_rate),
        "absolute_violation_delta": (clipping_rate - none_rate),
        "relative_violation_reduction": (relative_reduction),
        "seed_safety_requirement_pass": (seed_safety_requirement),
        "none_critical_violation_step_rate": float(
            none_summary["critical_violation_step_rate"]
        ),
        "clipping_critical_violation_step_rate": float(
            clipping_summary["critical_violation_step_rate"]
        ),
        "none_reward": (none_reward),
        "clipping_reward": (clipping_reward),
        "reward_delta": (reward_delta),
        "reward_degradation_fraction": (
            reward_degradation_fraction(
                baseline_reward=none_reward,
                clipping_reward=clipping_reward,
            )
        ),
        "none_success_rate": (none_success),
        "clipping_success_rate": (clipping_success),
        "success_rate_delta": (success_delta),
        "intervention_rate": float(clipping_summary["intervention_rate"]),
        "mean_action_correction_l2": float(
            clipping_summary["mean_action_correction_l2"]
        ),
        "p95_action_correction_l2": float(clipping_summary["p95_action_correction_l2"]),
        "max_action_correction_l2": float(clipping_summary["max_action_correction_l2"]),
        "mean_intervention_correction_l2": float(
            clipping_summary["mean_intervention_correction_l2"]
        ),
        "proposed_violation_step_rate": float(
            clipping_summary["proposed_violation_step_rate"]
        ),
        "executed_constraint_violation_rate": float(
            clipping_summary["executed_constraint_violation_rate"]
        ),
        "policy_action_out_of_bounds_count": int(
            clipping_summary["policy_action_out_of_bounds_count"]
        ),
        "executed_action_out_of_bounds_count": int(
            clipping_summary["executed_action_out_of_bounds_count"]
        ),
        "none_category_violation_rates": (none_summary["category_violation_rates"]),
        "clipping_category_violation_rates": (
            clipping_summary["executed_category_violation_rates"]
        ),
        "triggered_rule_counts": (clipping_summary["triggered_rule_counts"]),
        "triggered_rule_rates": (clipping_summary["triggered_rule_rates"]),
    }


def aggregate_categories(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    categories = sorted(
        {
            category
            for pair in pairs
            for category in (
                set(pair["none_category_violation_rates"])
                | set(pair["clipping_category_violation_rates"])
            )
        }
    )

    result: dict[
        str,
        Any,
    ] = {}

    for category in categories:
        none_rates = [
            float(
                pair["none_category_violation_rates"].get(
                    category,
                    0.0,
                )
            )
            for pair in pairs
        ]

        clipping_rates = [
            float(
                pair["clipping_category_violation_rates"].get(
                    category,
                    0.0,
                )
            )
            for pair in pairs
        ]

        result[category] = {
            "none_rate_mean": float(statistics.mean(none_rates)),
            "clipping_rate_mean": float(statistics.mean(clipping_rates)),
            "absolute_delta": float(
                statistics.mean(clipping_rates) - statistics.mean(none_rates)
            ),
        }

    return result


def aggregate_rules(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    rule_names = sorted(
        {rule for pair in pairs for rule in pair["triggered_rule_rates"]}
    )

    result: dict[
        str,
        Any,
    ] = {}

    for rule in rule_names:
        rates = [
            float(
                pair["triggered_rule_rates"].get(
                    rule,
                    0.0,
                )
            )
            for pair in pairs
        ]

        counts = [
            int(
                pair["triggered_rule_counts"].get(
                    rule,
                    0,
                )
            )
            for pair in pairs
        ]

        result[rule] = {
            "counts_by_seed": {
                str(pair["principal_seed"]): count
                for pair, count in zip(
                    pairs,
                    counts,
                    strict=True,
                )
            },
            "rate_mean": float(statistics.mean(rates)),
            "rate_sample_sd": (sample_sd(rates)),
        }

    return result


def aggregate_domain(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    none_rates = [float(pair["none_violation_step_rate"]) for pair in pairs]

    clipping_rates = [float(pair["clipping_violation_step_rate"]) for pair in pairs]

    absolute_deltas = [
        clipping - baseline
        for baseline, clipping in zip(
            none_rates,
            clipping_rates,
            strict=True,
        )
    ]

    none_rewards = [float(pair["none_reward"]) for pair in pairs]

    clipping_rewards = [float(pair["clipping_reward"]) for pair in pairs]

    none_success = [float(pair["none_success_rate"]) for pair in pairs]

    clipping_success = [float(pair["clipping_success_rate"]) for pair in pairs]

    intervention_rates = [float(pair["intervention_rate"]) for pair in pairs]

    correction_means = [float(pair["mean_action_correction_l2"]) for pair in pairs]

    correction_p95 = [float(pair["p95_action_correction_l2"]) for pair in pairs]

    correction_max = [float(pair["max_action_correction_l2"]) for pair in pairs]

    mean_none_rate = float(statistics.mean(none_rates))

    mean_clipping_rate = float(statistics.mean(clipping_rates))

    if mean_none_rate > 0.0:
        aggregate_reduction_fraction = (
            mean_none_rate - mean_clipping_rate
        ) / mean_none_rate
    else:
        aggregate_reduction_fraction = None

    mean_none_reward = float(statistics.mean(none_rewards))

    mean_clipping_reward = float(statistics.mean(clipping_rewards))

    reward_delta = mean_clipping_reward - mean_none_reward

    aggregate_reward_degradation = reward_degradation_fraction(
        baseline_reward=mean_none_reward,
        clipping_reward=mean_clipping_reward,
    )

    mean_none_success = float(statistics.mean(none_success))

    mean_clipping_success = float(statistics.mean(clipping_success))

    success_delta = mean_clipping_success - mean_none_success

    success_drop = max(
        0.0,
        -success_delta,
    )

    seed_safety_pass = all(bool(pair["seed_safety_requirement_pass"]) for pair in pairs)

    reduction_pass = bool(
        aggregate_reduction_fraction is not None
        and aggregate_reduction_fraction >= MINIMUM_MEAN_VIOLATION_REDUCTION
    )

    reward_pass = bool(aggregate_reward_degradation <= MAXIMUM_REWARD_DEGRADATION)

    success_pass = bool(success_drop <= MAXIMUM_SUCCESS_RATE_DROP)

    criterion_pass = bool(
        seed_safety_pass and reduction_pass and reward_pass and success_pass
    )

    return {
        "principal_seed_count": 3,
        "episode_count_per_seed": 20,
        "total_episode_count": 60,
        "none_violation_step_rate": {
            "mean": (mean_none_rate),
            "sample_sd": (sample_sd(none_rates)),
        },
        "clipping_violation_step_rate": {
            "mean": (mean_clipping_rate),
            "sample_sd": (sample_sd(clipping_rates)),
        },
        "absolute_violation_delta": {
            "mean": float(statistics.mean(absolute_deltas)),
            "sample_sd": (sample_sd(absolute_deltas)),
        },
        "aggregate_violation_reduction_fraction": (aggregate_reduction_fraction),
        "none_reward": {
            "mean_of_seed_means": (mean_none_reward),
            "sample_sd_of_seed_means": (sample_sd(none_rewards)),
        },
        "clipping_reward": {
            "mean_of_seed_means": (mean_clipping_reward),
            "sample_sd_of_seed_means": (sample_sd(clipping_rewards)),
        },
        "reward_delta": (reward_delta),
        "reward_degradation_fraction": (aggregate_reward_degradation),
        "none_success_rate": {
            "mean": (mean_none_success),
            "sample_sd": (sample_sd(none_success)),
        },
        "clipping_success_rate": {
            "mean": (mean_clipping_success),
            "sample_sd": (sample_sd(clipping_success)),
        },
        "success_rate_delta": (success_delta),
        "success_rate_drop": (success_drop),
        "intervention_rate": {
            "mean": float(statistics.mean(intervention_rates)),
            "sample_sd": (sample_sd(intervention_rates)),
        },
        "mean_action_correction_l2": {
            "mean_of_seed_means": float(statistics.mean(correction_means)),
            "sample_sd_of_seed_means": (sample_sd(correction_means)),
        },
        "p95_action_correction_l2": {
            "mean_of_seed_p95": float(statistics.mean(correction_p95)),
            "sample_sd_of_seed_p95": (sample_sd(correction_p95)),
        },
        "max_action_correction_l2": {
            "maximum_across_seeds": float(max(correction_max)),
        },
        "category_comparison": (aggregate_categories(pairs)),
        "triggered_rule_analysis": (aggregate_rules(pairs)),
        "pilot_criterion": {
            "seed_level_safety_requirement_pass": (seed_safety_pass),
            "mean_violation_reduction_at_least_20_percent": (reduction_pass),
            "reward_degradation_at_most_10_percent": (reward_pass),
            "success_drop_at_most_10_percentage_points": (success_pass),
            "overall_pass": (criterion_pass),
        },
    }


def build_csv(
    pairs: list[dict[str, Any]],
) -> None:
    path = OUTPUT_DIRECTORY / "sprint5-clipping-safety-summary.csv"

    fields = [
        "domain",
        "principal_seed",
        "none_violation_step_rate",
        "clipping_violation_step_rate",
        "absolute_violation_delta",
        "relative_violation_reduction",
        "seed_safety_requirement_pass",
        "none_reward",
        "clipping_reward",
        "reward_delta",
        "reward_degradation_fraction",
        "none_success_rate",
        "clipping_success_rate",
        "success_rate_delta",
        "intervention_rate",
        "mean_action_correction_l2",
        "p95_action_correction_l2",
        "max_action_correction_l2",
        "none_critical_violation_step_rate",
        "clipping_critical_violation_step_rate",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for pair in pairs:
            writer.writerow(
                {
                    field: ("" if pair[field] is None else pair[field])
                    for field in fields
                }
            )


def build_markdown(
    aggregate: dict[
        str,
        Any,
    ],
) -> None:
    lines = [
        "# Sprint 5.5 - Heuristic Constraint Clipping",
        "",
        "Method: `clipping`",
        "",
        "Condition: `clean`",
        "",
        "New training performed: `false`",
        "",
        "Comparison: frozen `NONE` vs `CLIPPING`",
        "",
        "Principal seeds: `42, 123, 456`",
        "",
        "Evaluation seeds per cell: `20000..20019`",
        "",
    ]

    for domain in DOMAINS:
        data = aggregate["domains"][domain]

        reduction = data["aggregate_violation_reduction_fraction"]

        reduction_text = (
            "undefined" if reduction is None else f"{100.0 * reduction:.2f}%"
        )

        result = (
            "SUPPORTED" if data["pilot_criterion"]["overall_pass"] else "NOT SUPPORTED"
        )

        lines.extend(
            [
                f"## {domain}",
                "",
                (
                    "NONE violation-step rate: "
                    f"{data['none_violation_step_rate']['mean']:.6f} "
                    "+/- "
                    f"{data['none_violation_step_rate']['sample_sd']:.6f}"
                ),
                "",
                (
                    "CLIPPING violation-step rate: "
                    f"{data['clipping_violation_step_rate']['mean']:.6f} "
                    "+/- "
                    f"{data['clipping_violation_step_rate']['sample_sd']:.6f}"
                ),
                "",
                ("Aggregate violation reduction: " f"`{reduction_text}`"),
                "",
                (
                    "NONE reward: "
                    f"{data['none_reward']['mean_of_seed_means']:.6f} "
                    "+/- "
                    f"{data['none_reward']['sample_sd_of_seed_means']:.6f}"
                ),
                "",
                (
                    "CLIPPING reward: "
                    f"{data['clipping_reward']['mean_of_seed_means']:.6f} "
                    "+/- "
                    f"{data['clipping_reward']['sample_sd_of_seed_means']:.6f}"
                ),
                "",
                ("Reward delta: " f"{data['reward_delta']:.6f}"),
                "",
                ("NONE success rate: " f"{data['none_success_rate']['mean']:.6f}"),
                "",
                (
                    "CLIPPING success rate: "
                    f"{data['clipping_success_rate']['mean']:.6f}"
                ),
                "",
                ("Success-rate delta: " f"{data['success_rate_delta']:.6f}"),
                "",
                ("Mean intervention rate: " f"{data['intervention_rate']['mean']:.6f}"),
                "",
                ("Pilot effectiveness criterion: " f"`{result}`"),
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            (
                "The clipping result is an empirical clean-condition "
                "classical safety result."
            ),
            "",
            "Lyapunov superiority: `NOT_YET_TESTED`",
            "",
            "Perturbation robustness: `NOT_YET_TESTED`",
            "",
            "Formal safety guarantee: `NO`",
            "",
            "Production safety claim: `NO`",
            "",
            "Quantum or quantum-inspired clipping claim: `NO`",
            "",
        ]
    )

    path = OUTPUT_DIRECTORY / "sprint5-clipping-safety-summary.md"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs: list[dict[str, Any]] = []

    by_domain: dict[
        str,
        list[dict[str, Any]],
    ] = {domain: [] for domain in DOMAINS}

    for domain in DOMAINS:
        for seed in SEEDS:
            pair = load_pair(
                domain=domain,
                seed=seed,
            )

            pairs.append(pair)

            by_domain[domain].append(pair)

    aggregate = {
        "sprint": "5.5",
        "artifact": ("heuristic-clipping-safety-summary"),
        "comparison": ("none_vs_clipping"),
        "method": "clipping",
        "condition": "clean",
        "new_training_performed": False,
        "principal_seeds": list(SEEDS),
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "criterion_definition": {
            "positive_baseline_seed": ("clipping_rate < none_rate"),
            "zero_baseline_seed": ("clipping_rate <= none_rate"),
            "minimum_domain_mean_violation_reduction": (
                MINIMUM_MEAN_VIOLATION_REDUCTION
            ),
            "maximum_reward_degradation_fraction": (MAXIMUM_REWARD_DEGRADATION),
            "maximum_success_rate_drop": (MAXIMUM_SUCCESS_RATE_DROP),
            "reward_degradation_definition": (
                "max(0, none_reward - clipping_reward) " "/ abs(none_reward)"
            ),
        },
        "seed_level_comparisons": (pairs),
        "domains": {domain: aggregate_domain(by_domain[domain]) for domain in DOMAINS},
        "claims": {
            "clipping_evaluated": True,
            "lyapunov_improvement": ("NOT_YET_TESTED"),
            "robustness_improvement": ("NOT_YET_TESTED"),
            "formal_safety_guarantee": False,
            "production_safety_claim": False,
            "quantum_clipping_claim": False,
        },
    }

    json_path = OUTPUT_DIRECTORY / "sprint5-clipping-safety-summary.json"

    json_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build_csv(pairs)

    build_markdown(aggregate)

    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.5 CLIPPING SUMMARY")
    print("=" * 68)

    for domain in DOMAINS:
        data = aggregate["domains"][domain]

        criterion = data["pilot_criterion"]

        reduction = data["aggregate_violation_reduction_fraction"]

        print()
        print(domain)
        print(
            "  NONE violation-step rate:",
            f"{data['none_violation_step_rate']['mean']:.6f}",
            "+/-",
            f"{data['none_violation_step_rate']['sample_sd']:.6f}",
        )
        print(
            "  CLIPPING violation-step rate:",
            f"{data['clipping_violation_step_rate']['mean']:.6f}",
            "+/-",
            f"{data['clipping_violation_step_rate']['sample_sd']:.6f}",
        )

        if reduction is None:
            print("  aggregate reduction: undefined")
        else:
            print(
                "  aggregate reduction:",
                f"{100.0 * reduction:.2f}%",
            )

        print(
            "  NONE reward:",
            f"{data['none_reward']['mean_of_seed_means']:.6f}",
        )
        print(
            "  CLIPPING reward:",
            f"{data['clipping_reward']['mean_of_seed_means']:.6f}",
        )
        print(
            "  reward delta:",
            f"{data['reward_delta']:.6f}",
        )
        print(
            "  NONE success:",
            f"{data['none_success_rate']['mean']:.6f}",
        )
        print(
            "  CLIPPING success:",
            f"{data['clipping_success_rate']['mean']:.6f}",
        )
        print(
            "  success delta:",
            f"{data['success_rate_delta']:.6f}",
        )
        print(
            "  intervention rate:",
            f"{data['intervention_rate']['mean']:.6f}",
        )
        print(
            "  seed safety requirement:",
            ("PASS" if criterion["seed_level_safety_requirement_pass"] else "FAIL"),
        )
        print(
            "  >=20% reduction:",
            (
                "PASS"
                if criterion["mean_violation_reduction_at_least_20_percent"]
                else "FAIL"
            ),
        )
        print(
            "  reward tolerance:",
            ("PASS" if criterion["reward_degradation_at_most_10_percent"] else "FAIL"),
        )
        print(
            "  success tolerance:",
            (
                "PASS"
                if criterion["success_drop_at_most_10_percentage_points"]
                else "FAIL"
            ),
        )
        print(
            "  EMPIRICAL EFFECTIVENESS:",
            ("SUPPORTED" if criterion["overall_pass"] else "NOT SUPPORTED"),
        )


if __name__ == "__main__":
    main()
