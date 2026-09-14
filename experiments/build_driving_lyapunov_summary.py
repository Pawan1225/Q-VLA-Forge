"""Build the Sprint 5.8 autonomous-driving Lyapunov safety summary."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.driving_lyapunov_safety import (
    DRIVING_SAFETY_CATEGORIES,
    MAXIMUM_REWARD_DEGRADATION_FRACTION,
    MAXIMUM_SUCCESS_RATE_DROP,
    MINIMUM_MEAN_VIOLATION_REDUCTION,
    domain_mean_violation_reduction,
    empirical_effectiveness_supported,
    relative_violation_reduction,
    reward_degradation_fraction,
    seed_safety_requirement_passes,
)

ROOT = Path(__file__).resolve().parents[1]

LYAPUNOV_RUNS = ROOT / "results" / "safety" / "lyapunov-driving" / "runs"

OUTPUT_ROOT = ROOT / "results" / "safety" / "lyapunov-driving"

NONE_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "baseline" / "sprint5-no-filter-safety-summary.json"
)

CLIPPING_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json"
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def _mean_sd(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": _sample_sd(values),
    }


def _lyapunov_runs() -> dict[int, dict[str, Any]]:
    runs: dict[
        int,
        dict[str, Any],
    ] = {}

    for seed in PRINCIPAL_SEEDS:
        path = LYAPUNOV_RUNS / f"autonomous_driving-seed-{seed}.json"

        payload = _load_json(path)

        if payload["principal_seed"] != seed:
            raise ValueError(f"Lyapunov run seed mismatch for {seed}")

        if payload["domain"] != "autonomous_driving":
            raise ValueError("Lyapunov run domain mismatch")

        if payload["safety_method"] != "lyapunov":
            raise ValueError("Lyapunov run method mismatch")

        if payload["episode_count"] != 20:
            raise ValueError("Lyapunov run must contain 20 episodes")

        runs[seed] = payload

    return runs


def _clipping_rows(
    clipping_summary: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    rows: dict[
        int,
        dict[str, Any],
    ] = {}

    for row in clipping_summary["seed_level_comparisons"]:
        if row["domain"] == "autonomous_driving":
            rows[int(row["principal_seed"])] = row

    if set(rows) != set(PRINCIPAL_SEEDS):
        raise ValueError("clipping summary does not contain all driving seeds")

    return rows


def _none_domain(
    none_summary: dict[str, Any],
) -> dict[str, Any]:
    return none_summary["domains"]["autonomous_driving"]


def main() -> None:
    none_summary = _load_json(NONE_SUMMARY_PATH)

    clipping_summary = _load_json(CLIPPING_SUMMARY_PATH)

    runs = _lyapunov_runs()

    clipping_rows = _clipping_rows(clipping_summary)

    none_domain = _none_domain(none_summary)

    seed_rows: list[dict[str, Any]] = []

    lyapunov_violation_rates: list[float] = []

    lyapunov_rewards: list[float] = []

    lyapunov_success_rates: list[float] = []

    lyapunov_intervention_rates: list[float] = []

    lyapunov_mean_corrections: list[float] = []

    lyapunov_nonincrease_rates: list[float] = []

    lyapunov_strict_decrease_rates: list[float] = []

    lyapunov_fallback_rates: list[float] = []

    lyapunov_mean_latencies: list[float] = []

    lyapunov_p95_latencies: list[float] = []

    seed_safety_passes: list[bool] = []

    for seed in PRINCIPAL_SEEDS:
        lyapunov_run = runs[seed]

        lyapunov = lyapunov_run["seed_summary"]

        clipping = clipping_rows[seed]

        none_violation_rate = float(clipping["none_violation_step_rate"])

        clipping_violation_rate = float(clipping["clipping_violation_step_rate"])

        lyapunov_violation_rate = float(lyapunov["executed_violation_step_rate"])

        safety_pass = seed_safety_requirement_passes(
            none_rate=(none_violation_rate),
            method_rate=(lyapunov_violation_rate),
        )

        seed_safety_passes.append(safety_pass)

        relative_reduction = relative_violation_reduction(
            none_rate=(none_violation_rate),
            method_rate=(lyapunov_violation_rate),
        )

        row = {
            "principal_seed": seed,
            "checkpoint_sha256": (lyapunov_run["policy_checkpoint_sha256"]),
            "none_violation_rate": (none_violation_rate),
            "clipping_violation_rate": (clipping_violation_rate),
            "lyapunov_violation_rate": (lyapunov_violation_rate),
            "lyapunov_relative_reduction_vs_none": (relative_reduction),
            "lyapunov_seed_safety_requirement_pass": (safety_pass),
            "none_reward": float(clipping["none_reward"]),
            "clipping_reward": float(clipping["clipping_reward"]),
            "lyapunov_reward": float(lyapunov["mean_reward"]),
            "none_success_rate": float(clipping["none_success_rate"]),
            "clipping_success_rate": float(clipping["clipping_success_rate"]),
            "lyapunov_success_rate": float(lyapunov["success_rate"]),
            "clipping_intervention_rate": float(clipping["intervention_rate"]),
            "lyapunov_intervention_rate": float(lyapunov["intervention_rate"]),
            "clipping_mean_correction_l2": float(clipping["mean_action_correction_l2"]),
            "lyapunov_mean_correction_l2": float(lyapunov["mean_action_correction_l2"]),
            "lyapunov_nonincrease_rate": float(lyapunov["lyapunov_nonincrease_rate"]),
            "strict_lyapunov_decrease_rate": float(
                lyapunov["strict_lyapunov_decrease_rate"]
            ),
            "selected_lower_than_proposed_rate": float(
                lyapunov["selected_lower_than_proposed_rate"]
            ),
            "emergency_fallback_rate": float(lyapunov["emergency_fallback_rate"]),
            "mean_filter_latency_ms": float(lyapunov["mean_filter_latency_ms"]),
            "p95_filter_latency_ms": float(lyapunov["p95_filter_latency_ms"]),
            "proposed_hard_guard_failure_rate": float(
                lyapunov["proposed_hard_guard_failure_rate"]
            ),
            "selected_hard_guard_failure_rate": float(
                lyapunov["selected_hard_guard_failure_rate"]
            ),
            "intervention_reason_counts": (lyapunov["intervention_reason_counts"]),
            "selected_candidate_source_counts": (
                lyapunov["selected_candidate_source_counts"]
            ),
        }

        seed_rows.append(row)

        lyapunov_violation_rates.append(lyapunov_violation_rate)

        lyapunov_rewards.append(float(lyapunov["mean_reward"]))

        lyapunov_success_rates.append(float(lyapunov["success_rate"]))

        lyapunov_intervention_rates.append(float(lyapunov["intervention_rate"]))

        lyapunov_mean_corrections.append(float(lyapunov["mean_action_correction_l2"]))

        lyapunov_nonincrease_rates.append(float(lyapunov["lyapunov_nonincrease_rate"]))

        lyapunov_strict_decrease_rates.append(
            float(lyapunov["strict_lyapunov_decrease_rate"])
        )

        lyapunov_fallback_rates.append(float(lyapunov["emergency_fallback_rate"]))

        lyapunov_mean_latencies.append(float(lyapunov["mean_filter_latency_ms"]))

        lyapunov_p95_latencies.append(float(lyapunov["p95_filter_latency_ms"]))

    none_mean_violation_rate = float(none_domain["violation_step_rate"]["mean"])

    lyapunov_mean_violation_rate = float(statistics.mean(lyapunov_violation_rates))

    mean_violation_reduction = domain_mean_violation_reduction(
        none_mean_rate=(none_mean_violation_rate),
        method_mean_rate=(lyapunov_mean_violation_rate),
    )

    none_mean_reward = float(none_domain["reward"]["mean_of_seed_means"])

    lyapunov_mean_reward = float(statistics.mean(lyapunov_rewards))

    reward_degradation = reward_degradation_fraction(
        none_reward=(none_mean_reward),
        method_reward=(lyapunov_mean_reward),
    )

    none_mean_success = float(none_domain["success"]["mean_of_seed_rates"])

    lyapunov_mean_success = float(statistics.mean(lyapunov_success_rates))

    success_rate_drop = max(
        0.0,
        none_mean_success - lyapunov_mean_success,
    )

    effectiveness_supported = empirical_effectiveness_supported(
        seed_safety_passes=tuple(seed_safety_passes),
        domain_violation_reduction=(mean_violation_reduction),
        reward_degradation_fraction_value=(reward_degradation),
        success_rate_drop=(success_rate_drop),
    )

    category_table: dict[
        str,
        dict[str, float],
    ] = {}

    for category in DRIVING_SAFETY_CATEGORIES:
        none_rates: list[float] = []

        clipping_rates: list[float] = []

        lyapunov_rates: list[float] = []

        for seed in PRINCIPAL_SEEDS:
            clipping = clipping_rows[seed]

            lyapunov = runs[seed]["seed_summary"]

            none_rates.append(
                float(
                    clipping["none_category_violation_rates"].get(
                        category,
                        0.0,
                    )
                )
            )

            clipping_rates.append(
                float(
                    clipping["clipping_category_violation_rates"].get(
                        category,
                        0.0,
                    )
                )
            )

            lyapunov_rates.append(
                float(
                    lyapunov["executed_category_violation_rates"].get(
                        category,
                        0.0,
                    )
                )
            )

        category_table[category] = {
            "none_rate_mean": float(statistics.mean(none_rates)),
            "clipping_rate_mean": float(statistics.mean(clipping_rates)),
            "lyapunov_rate_mean": float(statistics.mean(lyapunov_rates)),
        }

    clipping_driving_rows = [
        row
        for row in clipping_summary["seed_level_comparisons"]
        if (row["domain"] == "autonomous_driving")
    ]

    clipping_violation_rates = [
        float(row["clipping_violation_step_rate"]) for row in clipping_driving_rows
    ]

    clipping_rewards = [float(row["clipping_reward"]) for row in clipping_driving_rows]

    clipping_success_rates = [
        float(row["clipping_success_rate"]) for row in clipping_driving_rows
    ]

    clipping_intervention_rates = [
        float(row["intervention_rate"]) for row in clipping_driving_rows
    ]

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.8",
        "artifact": ("driving-lyapunov-safety-summary"),
        "domain": ("autonomous_driving"),
        "condition": "clean",
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "new_training_performed": (False),
        "criterion_definition": {
            "positive_baseline_seed": ("lyapunov_rate < none_rate"),
            "zero_baseline_seed": ("lyapunov_rate <= none_rate"),
            "minimum_domain_mean_violation_reduction": (
                MINIMUM_MEAN_VIOLATION_REDUCTION
            ),
            "maximum_reward_degradation_fraction": (
                MAXIMUM_REWARD_DEGRADATION_FRACTION
            ),
            "maximum_success_rate_drop": (MAXIMUM_SUCCESS_RATE_DROP),
            "reward_degradation_definition": (
                "max(0, none_reward - lyapunov_reward) " "/ abs(none_reward)"
            ),
        },
        "seed_level_comparisons": (seed_rows),
        "three_method_summary": {
            "none": {
                "violation_step_rate": (none_domain["violation_step_rate"]),
                "reward": (none_domain["reward"]),
                "success": (none_domain["success"]),
                "intervention_rate": {
                    "mean": 0.0,
                    "sample_sd": 0.0,
                },
            },
            "clipping": {
                "violation_step_rate": (_mean_sd(clipping_violation_rates)),
                "reward": {
                    "mean_of_seed_means": float(statistics.mean(clipping_rewards)),
                    "sample_sd_of_seed_means": (_sample_sd(clipping_rewards)),
                },
                "success": {
                    "mean_of_seed_rates": float(
                        statistics.mean(clipping_success_rates)
                    ),
                    "sample_sd_of_seed_rates": (_sample_sd(clipping_success_rates)),
                },
                "intervention_rate": (_mean_sd(clipping_intervention_rates)),
            },
            "lyapunov": {
                "violation_step_rate": (_mean_sd(lyapunov_violation_rates)),
                "reward": {
                    "mean_of_seed_means": (lyapunov_mean_reward),
                    "sample_sd_of_seed_means": (_sample_sd(lyapunov_rewards)),
                },
                "success": {
                    "mean_of_seed_rates": (lyapunov_mean_success),
                    "sample_sd_of_seed_rates": (_sample_sd(lyapunov_success_rates)),
                },
                "intervention_rate": (_mean_sd(lyapunov_intervention_rates)),
                "mean_action_correction_l2": (_mean_sd(lyapunov_mean_corrections)),
                "lyapunov_nonincrease_rate": (_mean_sd(lyapunov_nonincrease_rates)),
                "strict_lyapunov_decrease_rate": (
                    _mean_sd(lyapunov_strict_decrease_rates)
                ),
                "emergency_fallback_rate": (_mean_sd(lyapunov_fallback_rates)),
                "mean_filter_latency_ms": (_mean_sd(lyapunov_mean_latencies)),
                "p95_filter_latency_ms": (_mean_sd(lyapunov_p95_latencies)),
            },
        },
        "category_comparison": (category_table),
        "effectiveness": {
            "seed_safety_requirements_pass": (all(seed_safety_passes)),
            "seed_safety_results": {
                str(seed): passed
                for seed, passed in zip(
                    PRINCIPAL_SEEDS,
                    seed_safety_passes,
                    strict=True,
                )
            },
            "domain_mean_violation_reduction": (mean_violation_reduction),
            "domain_mean_violation_reduction_pass": (
                mean_violation_reduction >= MINIMUM_MEAN_VIOLATION_REDUCTION
            ),
            "reward_degradation_fraction": (reward_degradation),
            "reward_requirement_pass": (
                reward_degradation <= MAXIMUM_REWARD_DEGRADATION_FRACTION
            ),
            "success_rate_drop": (success_rate_drop),
            "success_requirement_pass": (
                success_rate_drop <= MAXIMUM_SUCCESS_RATE_DROP
            ),
            "driving_lyapunov_empirical_effectiveness_supported": (
                effectiveness_supported
            ),
        },
        "mechanism_interpretation": {
            "principal_lyapunov_decrease_interventions_observed": (
                any(
                    row["intervention_reason_counts"].get(
                        "lyapunov_decrease",
                        0,
                    )
                    > 0
                    for row in seed_rows
                )
            ),
            "all_selected_lyapunov_transitions_nonincreasing": (
                all(rate == 1.0 for rate in (lyapunov_nonincrease_rates))
            ),
            "emergency_fallback_observed": (
                any(rate > 0.0 for rate in (lyapunov_fallback_rates))
            ),
            "interpretation": (
                "Under the frozen clean driving protocol, "
                "the complete Lyapunov-guided filter achieved "
                "the measured safety outcome. Principal action "
                "changes were triggered by domain hard guards; "
                "no additional LYAPUNOV_DECREASE intervention "
                "was observed."
            ),
        },
        "clipping_comparison_scope": ("descriptive_only_no_superiority_threshold"),
        "claims": {
            "formal_stability_proven": False,
            "formal_safety_guarantee": False,
            "collision_free_driving_claim": False,
            "production_safety_claim": False,
            "iso_26262_claim": False,
            "lyapunov_superior_to_clipping_claim": False,
        },
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_ROOT / "sprint5-driving-lyapunov-summary.json"

    csv_path = OUTPUT_ROOT / "sprint5-driving-lyapunov-summary.csv"

    md_path = OUTPUT_ROOT / "sprint5-driving-lyapunov-summary.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_fields = (
        "principal_seed",
        "none_violation_rate",
        "clipping_violation_rate",
        "lyapunov_violation_rate",
        "lyapunov_relative_reduction_vs_none",
        "none_reward",
        "clipping_reward",
        "lyapunov_reward",
        "none_success_rate",
        "clipping_success_rate",
        "lyapunov_success_rate",
        "clipping_intervention_rate",
        "lyapunov_intervention_rate",
        "clipping_mean_correction_l2",
        "lyapunov_mean_correction_l2",
        "lyapunov_nonincrease_rate",
        "strict_lyapunov_decrease_rate",
        "emergency_fallback_rate",
        "mean_filter_latency_ms",
        "p95_filter_latency_ms",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in seed_rows:
            writer.writerow({field: row[field] for field in csv_fields})

    lines = [
        "# Sprint 5.8 â€” Driving Lyapunov Safety Evaluation",
        "",
        "## Scope",
        "",
        "- Domain: autonomous_driving",
        "- Condition: clean",
        "- Principal PPO seeds: 42, 123, 456",
        "- Held-out episodes: 20 per seed",
        "- New training: no",
        "",
        "## Principal result",
        "",
        (
            "- Lyapunov executed violation-step rate: "
            f"{lyapunov_mean_violation_rate:.6f}"
        ),
        ("- Mean violation reduction vs NONE: " f"{mean_violation_reduction:.2%}"),
        ("- Empirical effectiveness supported: " f"{effectiveness_supported}"),
        "",
        "## Mechanism interpretation",
        "",
        (
            "- Lyapunov non-increase rate: "
            f"{statistics.mean(lyapunov_nonincrease_rates):.6f}"
        ),
        (
            "- Strict Lyapunov-decrease intervention observed: "
            f"{payload['mechanism_interpretation']['principal_lyapunov_decrease_interventions_observed']}"
        ),
        (
            "- Emergency fallback observed: "
            f"{payload['mechanism_interpretation']['emergency_fallback_observed']}"
        ),
        "",
        "## Claim boundary",
        "",
        (
            "This is empirical clean-condition proxy evidence. "
            "It is not a formal stability proof, production "
            "safety guarantee, collision-free guarantee, or "
            "ISO 26262 certification result."
        ),
    ]

    md_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Driving Lyapunov seed safety requirements:",
        ("PASS" if all(seed_safety_passes) else "FAIL"),
    )

    print(
        "Mean violation reduction:",
        f"{mean_violation_reduction:.6f}",
    )

    print(
        "Reward degradation:",
        f"{reward_degradation:.6f}",
    )

    print(
        "Success-rate drop:",
        f"{success_rate_drop:.6f}",
    )

    print(
        "Empirical effectiveness:",
        ("SUPPORTED" if effectiveness_supported else "NOT SUPPORTED"),
    )

    print(
        "Principal Lyapunov-decrease interventions:",
        (
            "OBSERVED"
            if payload["mechanism_interpretation"][
                "principal_lyapunov_decrease_interventions_observed"
            ]
            else "NOT OBSERVED"
        ),
    )

    print("SPRINT 5.8 DRIVING LYAPUNOV SUMMARY: BUILT")


if __name__ == "__main__":
    main()
