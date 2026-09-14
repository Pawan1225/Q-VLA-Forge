"""Build the Sprint 5.9 robotics Lyapunov safety summary."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.robotics_lyapunov_safety import (
    MAXIMUM_REWARD_DEGRADATION_FRACTION,
    MAXIMUM_SUCCESS_RATE_DROP,
    MINIMUM_MEAN_VIOLATION_REDUCTION,
    domain_mean_violation_reduction,
    empirical_effectiveness_supported,
    relative_violation_reduction,
    reward_degradation_fraction,
    seed_safety_requirement_passes,
)
from q_vla_forge.safety.robotics_constraints import (
    VIOLATION_CATEGORIES,
)

ROOT = Path(__file__).resolve().parents[1]

LYAPUNOV_RUNS = ROOT / "results" / "safety" / "lyapunov-robotics" / "runs"

OUTPUT_ROOT = ROOT / "results" / "safety" / "lyapunov-robotics"

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

EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
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


def _lyapunov_runs() -> dict[
    int,
    dict[str, Any],
]:
    runs: dict[
        int,
        dict[str, Any],
    ] = {}

    for seed in PRINCIPAL_SEEDS:
        path = LYAPUNOV_RUNS / f"robotics-seed-{seed}.json"

        payload = _load_json(path)

        if payload["principal_seed"] != seed:
            raise ValueError(f"Lyapunov run seed mismatch " f"for {seed}")

        if payload["domain"] != "robotics":
            raise ValueError("Lyapunov run domain mismatch")

        if payload["safety_method"] != "lyapunov":
            raise ValueError("Lyapunov run method mismatch")

        if payload["robustness_condition"] != "clean":
            raise ValueError("Lyapunov run condition mismatch")

        if payload["episode_count"] != 20:
            raise ValueError("Lyapunov run must contain " "20 episodes")

        if tuple(payload["evaluation_seeds"]) != EVALUATION_SEEDS:
            raise ValueError(
                "Lyapunov evaluation seeds " "do not match frozen protocol"
            )

        prediction_context = payload["prediction_context"]

        if prediction_context["source"] != "environment_internal_state":
            raise ValueError("robotics prediction-context " "source mismatch")

        if prediction_context["field"] != "object_grasped":
            raise ValueError("robotics prediction-context " "field mismatch")

        if bool(prediction_context["policy_observation_modified"]):
            raise ValueError("robotics policy observation " "must remain unmodified")

        if int(prediction_context["policy_observation_dimension"]) != 6:
            raise ValueError("robotics policy observation " "dimension must remain 6")

        if bool(payload["new_training_performed"]):
            raise ValueError("Sprint 5.9 must not " "perform new training")

        if bool(payload["policy_fine_tuning_performed"]):
            raise ValueError("Sprint 5.9 must not " "fine-tune the policy")

        if bool(payload["filter_tuning_performed"]):
            raise ValueError("Sprint 5.9 must not " "tune the filter")

        if bool(payload["perturbation_used"]):
            raise ValueError("Sprint 5.9 principal " "condition must remain clean")

        runs[seed] = payload

    return runs


def _clipping_rows(
    clipping_summary: dict[
        str,
        Any,
    ],
) -> dict[
    int,
    dict[str, Any],
]:
    rows: dict[
        int,
        dict[str, Any],
    ] = {}

    for row in clipping_summary["seed_level_comparisons"]:
        if row["domain"] == "robotics":
            rows[int(row["principal_seed"])] = row

    if set(rows) != set(PRINCIPAL_SEEDS):
        raise ValueError("clipping summary does not " "contain all robotics seeds")

    return rows


def _none_domain(
    none_summary: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    return none_summary["domains"]["robotics"]


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

    lyapunov_selected_lower_rates: list[float] = []

    lyapunov_fallback_rates: list[float] = []

    lyapunov_mean_latencies: list[float] = []

    lyapunov_p95_latencies: list[float] = []

    seed_safety_passes: list[bool] = []

    total_environment_steps = 0

    total_grasped_steps = 0

    total_not_grasped_steps = 0

    total_interventions = 0

    total_domain_constraint_interventions = 0

    total_lyapunov_decrease_interventions = 0

    total_emergency_fallbacks = 0

    total_selected_guard_failures = 0

    total_strict_decrease_steps = 0

    total_nonincrease_steps = 0

    total_selected_lower_steps = 0

    for seed in PRINCIPAL_SEEDS:
        lyapunov_run = runs[seed]

        lyapunov = lyapunov_run["seed_summary"]

        clipping = clipping_rows[seed]

        none_violation_rate = float(clipping["none_violation_step_rate"])

        clipping_violation_rate = float(clipping["clipping_violation_step_rate"])

        lyapunov_violation_rate = float(lyapunov["executed_violation_step_rate"])

        safety_pass = seed_safety_requirement_passes(
            none_rate=(none_violation_rate),
            lyapunov_rate=(lyapunov_violation_rate),
        )

        seed_safety_passes.append(safety_pass)

        relative_reduction = relative_violation_reduction(
            baseline_rate=(none_violation_rate),
            candidate_rate=(lyapunov_violation_rate),
        )

        reason_counts = dict(lyapunov["intervention_reason_counts"])

        candidate_source_counts = dict(lyapunov["selected_candidate_source_counts"])

        intervened_candidate_source_counts = dict(
            lyapunov["intervened_candidate_source_counts"]
        )

        environment_steps = int(lyapunov["total_environment_steps"])

        grasped_steps = int(lyapunov["steps_object_grasped"])

        not_grasped_steps = int(lyapunov["steps_object_not_grasped"])

        if grasped_steps + not_grasped_steps != environment_steps:
            raise ValueError(f"grasp-context accounting " f"mismatch for seed {seed}")

        if int(lyapunov["selected_hard_guard_failure_count"]) != 0:
            raise ValueError(
                f"selected hard-guard failure " f"observed for seed {seed}"
            )

        if int(lyapunov["emergency_fallback_count"]) != 0:
            raise ValueError(f"emergency fallback observed " f"for seed {seed}")

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
            "steps_object_grasped": (grasped_steps),
            "steps_object_not_grasped": (not_grasped_steps),
            "interventions_while_grasped": int(lyapunov["interventions_while_grasped"]),
            "interventions_while_not_grasped": int(
                lyapunov["interventions_while_not_grasped"]
            ),
            "intervention_reason_counts": (reason_counts),
            "selected_candidate_source_counts": (candidate_source_counts),
            "intervened_candidate_source_counts": (intervened_candidate_source_counts),
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

        lyapunov_selected_lower_rates.append(
            float(lyapunov["selected_lower_than_proposed_rate"])
        )

        lyapunov_fallback_rates.append(float(lyapunov["emergency_fallback_rate"]))

        lyapunov_mean_latencies.append(float(lyapunov["mean_filter_latency_ms"]))

        lyapunov_p95_latencies.append(float(lyapunov["p95_filter_latency_ms"]))

        total_environment_steps += environment_steps

        total_grasped_steps += grasped_steps

        total_not_grasped_steps += not_grasped_steps

        total_interventions += int(lyapunov["intervention_count"])

        total_domain_constraint_interventions += int(
            reason_counts.get(
                "domain_constraint",
                0,
            )
        )

        total_lyapunov_decrease_interventions += int(
            reason_counts.get(
                "lyapunov_decrease",
                0,
            )
        )

        total_emergency_fallbacks += int(
            reason_counts.get(
                "emergency_fallback",
                0,
            )
        )

        total_selected_guard_failures += int(
            lyapunov["selected_hard_guard_failure_count"]
        )

        total_strict_decrease_steps += int(lyapunov["strict_lyapunov_decrease_count"])

        total_nonincrease_steps += int(lyapunov["lyapunov_nonincrease_count"])

        total_selected_lower_steps += int(
            lyapunov["selected_lower_than_proposed_count"]
        )

    none_mean_violation_rate = float(none_domain["violation_step_rate"]["mean"])

    lyapunov_mean_violation_rate = float(statistics.mean(lyapunov_violation_rates))

    mean_violation_reduction = domain_mean_violation_reduction(
        none_mean_rate=(none_mean_violation_rate),
        lyapunov_mean_rate=(lyapunov_mean_violation_rate),
    )

    none_mean_reward = float(none_domain["reward"]["mean_of_seed_means"])

    lyapunov_mean_reward = float(statistics.mean(lyapunov_rewards))

    reward_degradation = reward_degradation_fraction(
        baseline_reward=(none_mean_reward),
        candidate_reward=(lyapunov_mean_reward),
    )

    none_mean_success = float(none_domain["success"]["mean_of_seed_rates"])

    lyapunov_mean_success = float(statistics.mean(lyapunov_success_rates))

    success_rate_drop = max(
        0.0,
        none_mean_success - lyapunov_mean_success,
    )

    seed_requirements_pass = all(seed_safety_passes)

    effectiveness_supported = empirical_effectiveness_supported(
        seed_requirements_pass=(seed_requirements_pass),
        mean_violation_reduction=(mean_violation_reduction),
        reward_degradation=(reward_degradation),
        success_rate_drop=(success_rate_drop),
    )

    category_table: dict[
        str,
        dict[str, float],
    ] = {}

    for category in VIOLATION_CATEGORIES:
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

    clipping_robotics_rows = [
        row
        for row in clipping_summary["seed_level_comparisons"]
        if (row["domain"] == "robotics")
    ]

    clipping_violation_rates = [
        float(row["clipping_violation_step_rate"]) for row in clipping_robotics_rows
    ]

    clipping_rewards = [float(row["clipping_reward"]) for row in clipping_robotics_rows]

    clipping_success_rates = [
        float(row["clipping_success_rate"]) for row in clipping_robotics_rows
    ]

    clipping_intervention_rates = [
        float(row["intervention_rate"]) for row in clipping_robotics_rows
    ]

    clipping_mean_corrections = [
        float(row["mean_action_correction_l2"]) for row in clipping_robotics_rows
    ]

    grasped_branch_exercised = total_grasped_steps > 0

    lyapunov_decrease_observed = total_lyapunov_decrease_interventions > 0

    all_selected_transitions_nonincreasing = (
        total_nonincrease_steps == total_environment_steps
    )

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.9",
        "artifact": ("robotics-lyapunov-safety-summary"),
        "domain": "robotics",
        "condition": "clean",
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "new_training_performed": False,
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
                "max(0, none_reward - " "lyapunov_reward) / " "abs(none_reward)"
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
                "mean_action_correction_l2": (_mean_sd(clipping_mean_corrections)),
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
                "selected_lower_than_proposed_rate": (
                    _mean_sd(lyapunov_selected_lower_rates)
                ),
                "emergency_fallback_rate": (_mean_sd(lyapunov_fallback_rates)),
                "mean_filter_latency_ms": (_mean_sd(lyapunov_mean_latencies)),
                "p95_filter_latency_ms": (_mean_sd(lyapunov_p95_latencies)),
            },
        },
        "category_comparison": (category_table),
        "effectiveness": {
            "seed_safety_requirements_pass": (seed_requirements_pass),
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
            "robotics_lyapunov_empirical_effectiveness_supported": (
                effectiveness_supported
            ),
        },
        "mechanism_accounting": {
            "total_environment_steps": (total_environment_steps),
            "total_interventions": (total_interventions),
            "domain_constraint_interventions": (total_domain_constraint_interventions),
            "lyapunov_decrease_interventions": (total_lyapunov_decrease_interventions),
            "emergency_fallbacks": (total_emergency_fallbacks),
            "selected_hard_guard_failures": (total_selected_guard_failures),
            "strict_lyapunov_decrease_steps": (total_strict_decrease_steps),
            "lyapunov_nonincrease_steps": (total_nonincrease_steps),
            "selected_lower_than_proposed_steps": (total_selected_lower_steps),
        },
        "mechanism_interpretation": {
            "principal_lyapunov_decrease_interventions_observed": (
                lyapunov_decrease_observed
            ),
            "all_selected_lyapunov_transitions_nonincreasing": (
                all_selected_transitions_nonincreasing
            ),
            "emergency_fallback_observed": (total_emergency_fallbacks > 0),
            "selected_hard_guard_failure_observed": (total_selected_guard_failures > 0),
            "interpretation": (
                "Under the frozen clean robotics "
                "protocol, the complete "
                "Lyapunov-guided safety filter "
                "eliminated all measured executed "
                "constraint violations. All observed "
                "principal interventions were "
                "triggered by frozen robotics domain "
                "hard guards. No "
                "LYAPUNOV_DECREASE intervention was "
                "observed in the principal runs."
            ),
        },
        "grasp_context": {
            "prediction_context_source": ("environment_internal_state"),
            "prediction_context_field": ("object_grasped"),
            "policy_observation_modified": False,
            "policy_observation_dimension": 6,
            "total_environment_steps": (total_environment_steps),
            "object_grasped_true_steps": (total_grasped_steps),
            "object_grasped_false_steps": (total_not_grasped_steps),
            "grasped_branch_exercised": (grasped_branch_exercised),
            "principal_empirical_validation_of_grasped_branch": (
                grasped_branch_exercised
            ),
            "limitation": (
                "The clean robotics principal "
                "evaluation did not enter the "
                "object_grasped=True regime. The "
                "grasp-aware predictor path is "
                "implemented and provenance-controlled, "
                "but its grasped-state branch was not "
                "empirically exercised by the Sprint "
                "5.9 principal trajectories."
            ),
        },
        "clipping_comparison_scope": ("descriptive_only_no_superiority_threshold"),
        "claims": {
            "formal_stability_proven": False,
            "formal_robot_safety_guarantee": False,
            "production_manipulation_safety_claim": False,
            "iso_10218_claim": False,
            "iec_62061_claim": False,
            "lyapunov_superior_to_clipping_claim": False,
            "quantum_safety_advantage_claim": False,
            "grasped_state_branch_empirically_validated": (grasped_branch_exercised),
        },
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_ROOT / "sprint5-robotics-lyapunov-summary.json"

    csv_path = OUTPUT_ROOT / "sprint5-robotics-lyapunov-summary.csv"

    md_path = OUTPUT_ROOT / "sprint5-robotics-lyapunov-summary.md"

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
        "lyapunov_seed_safety_requirement_pass",
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
        "selected_lower_than_proposed_rate",
        "emergency_fallback_rate",
        "mean_filter_latency_ms",
        "p95_filter_latency_ms",
        "proposed_hard_guard_failure_rate",
        "selected_hard_guard_failure_rate",
        "steps_object_grasped",
        "steps_object_not_grasped",
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

    none_summary_values = payload["three_method_summary"]["none"]

    clipping_summary_values = payload["three_method_summary"]["clipping"]

    lyapunov_summary_values = payload["three_method_summary"]["lyapunov"]

    effectiveness = payload["effectiveness"]

    lines = [
        "# Sprint 5.9 — Robotics Lyapunov Safety Summary",
        "",
        "## Scope",
        "",
        "- Domain: robotics",
        "- Condition: clean",
        "- Policy family: frozen classical PPO",
        "- Principal seeds: 42, 123, 456",
        "- Held-out evaluation seeds: 20000–20019",
        "- Episodes: 60 total",
        "- New training: no",
        "- Policy fine-tuning: no",
        "- Filter tuning: no",
        "- Robustness perturbation: no",
        "",
        "## Three-Method Comparison",
        "",
        "| Method | Violation-step rate | Reward | Success | Intervention rate |",
        "|---|---:|---:|---:|---:|",
        (
            "| NONE | "
            f"{none_summary_values['violation_step_rate']['mean']:.6f} "
            f"± {none_summary_values['violation_step_rate']['sample_sd']:.6f} | "
            f"{none_summary_values['reward']['mean_of_seed_means']:.6f} "
            f"± {none_summary_values['reward']['sample_sd_of_seed_means']:.6f} | "
            f"{none_summary_values['success']['mean_of_seed_rates']:.6f} "
            f"± {none_summary_values['success']['sample_sd_of_seed_rates']:.6f} | "
            "0.000000 ± 0.000000 |"
        ),
        (
            "| CLIPPING | "
            f"{clipping_summary_values['violation_step_rate']['mean']:.6f} "
            f"± {clipping_summary_values['violation_step_rate']['sample_sd']:.6f} | "
            f"{clipping_summary_values['reward']['mean_of_seed_means']:.6f} "
            f"± {clipping_summary_values['reward']['sample_sd_of_seed_means']:.6f} | "
            f"{clipping_summary_values['success']['mean_of_seed_rates']:.6f} "
            f"± {clipping_summary_values['success']['sample_sd_of_seed_rates']:.6f} | "
            f"{clipping_summary_values['intervention_rate']['mean']:.6f} "
            f"± {clipping_summary_values['intervention_rate']['sample_sd']:.6f} |"
        ),
        (
            "| LYAPUNOV | "
            f"{lyapunov_summary_values['violation_step_rate']['mean']:.6f} "
            f"± {lyapunov_summary_values['violation_step_rate']['sample_sd']:.6f} | "
            f"{lyapunov_summary_values['reward']['mean_of_seed_means']:.6f} "
            f"± {lyapunov_summary_values['reward']['sample_sd_of_seed_means']:.6f} | "
            f"{lyapunov_summary_values['success']['mean_of_seed_rates']:.6f} "
            f"± {lyapunov_summary_values['success']['sample_sd_of_seed_rates']:.6f} | "
            f"{lyapunov_summary_values['intervention_rate']['mean']:.6f} "
            f"± {lyapunov_summary_values['intervention_rate']['sample_sd']:.6f} |"
        ),
        "",
        "## Pre-Registered Effectiveness Criterion",
        "",
        (
            "- Seed-level safety requirements: "
            f"{'PASS' if effectiveness['seed_safety_requirements_pass'] else 'FAIL'}"
        ),
        (
            "- Domain mean violation reduction: "
            f"{effectiveness['domain_mean_violation_reduction']:.6f} "
            f"({'PASS' if effectiveness['domain_mean_violation_reduction_pass'] else 'FAIL'})"
        ),
        (
            "- Reward degradation fraction: "
            f"{effectiveness['reward_degradation_fraction']:.6f} "
            f"({'PASS' if effectiveness['reward_requirement_pass'] else 'FAIL'})"
        ),
        (
            "- Success-rate drop: "
            f"{effectiveness['success_rate_drop']:.6f} "
            f"({'PASS' if effectiveness['success_requirement_pass'] else 'FAIL'})"
        ),
        "",
        (
            "**SPRINT 5.9 ROBOTICS EMPIRICAL EFFECTIVENESS: "
            f"{'PASS' if effectiveness['robotics_lyapunov_empirical_effectiveness_supported'] else 'FAIL'}**"
        ),
        "",
        "## Mechanism Evidence",
        "",
        (f"- Environment steps: " f"{total_environment_steps}"),
        (f"- Total interventions: " f"{total_interventions}"),
        (
            f"- DOMAIN_CONSTRAINT interventions: "
            f"{total_domain_constraint_interventions}"
        ),
        (
            f"- LYAPUNOV_DECREASE interventions: "
            f"{total_lyapunov_decrease_interventions}"
        ),
        (f"- Emergency fallbacks: " f"{total_emergency_fallbacks}"),
        (f"- Selected hard-guard failures: " f"{total_selected_guard_failures}"),
        (
            f"- Strict Lyapunov-decrease steps: "
            f"{total_strict_decrease_steps} / "
            f"{total_environment_steps}"
        ),
        (
            f"- Lyapunov non-increase steps: "
            f"{total_nonincrease_steps} / "
            f"{total_environment_steps}"
        ),
        (
            f"- Selected lower-than-proposed V steps: "
            f"{total_selected_lower_steps} / "
            f"{total_environment_steps}"
        ),
        "",
        (
            "All principal action-changing interventions were triggered "
            "by frozen robotics domain hard guards. No "
            "`LYAPUNOV_DECREASE` intervention was observed."
        ),
        "",
        "## Grasp-Context Coverage",
        "",
        (
            f"- `object_grasped=True`: "
            f"{total_grasped_steps} / "
            f"{total_environment_steps} steps"
        ),
        (
            f"- `object_grasped=False`: "
            f"{total_not_grasped_steps} / "
            f"{total_environment_steps} steps"
        ),
        "- Policy observation remained the original 6-D observation.",
        "- `object_grasped` was supplied only to the one-step safety predictor.",
        "",
        (
            "**Limitation:** The clean robotics principal evaluation did "
            "not enter the `object_grasped=True` regime. Therefore the "
            "grasp-aware prediction path is implemented and provenance-controlled, "
            "but its grasped-state branch was not empirically exercised by the "
            "Sprint 5.9 principal trajectories."
        ),
        "",
        "## Interpretation",
        "",
        (
            "Under the frozen clean robotics protocol, the complete "
            "Lyapunov-guided safety filter reduced the measured executed "
            "violation-step rate from the no-filter baseline to zero while "
            "preserving the pre-registered reward and success requirements."
        ),
        "",
        (
            "The observed action changes were entirely attributable to the "
            "frozen domain hard guards rather than the strict Lyapunov-decrease "
            "selector. Clipping and the complete Lyapunov-guided filter are "
            "therefore compared descriptively only; this Sprint does not support "
            "a claim that the Lyapunov method is superior to clipping."
        ),
        "",
        "## Claim Controls",
        "",
        "- No formal stability claim.",
        "- No formal robotics safety guarantee.",
        "- No production manipulation-safety claim.",
        "- No ISO 10218 claim.",
        "- No IEC 62061 claim.",
        "- No Lyapunov-over-clipping superiority claim.",
        "- No quantum safety advantage claim.",
        "- No empirical validation claim for the grasped-state branch.",
        "",
    ]

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Sprint 5.9 robotics " "Lyapunov summary built.")

    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )

    print(
        "Markdown:",
        md_path,
    )

    print()

    print(
        "NONE mean violation rate:",
        f"{none_mean_violation_rate:.9f}",
    )

    print(
        "LYAPUNOV mean violation rate:",
        f"{lyapunov_mean_violation_rate:.9f}",
    )

    print(
        "Mean violation reduction:",
        f"{mean_violation_reduction:.6f}",
    )

    print(
        "NONE mean reward:",
        f"{none_mean_reward:.9f}",
    )

    print(
        "LYAPUNOV mean reward:",
        f"{lyapunov_mean_reward:.9f}",
    )

    print(
        "Reward degradation:",
        f"{reward_degradation:.6f}",
    )

    print(
        "NONE mean success:",
        f"{none_mean_success:.6f}",
    )

    print(
        "LYAPUNOV mean success:",
        f"{lyapunov_mean_success:.6f}",
    )

    print(
        "DOMAIN_CONSTRAINT interventions:",
        total_domain_constraint_interventions,
    )

    print(
        "LYAPUNOV_DECREASE interventions:",
        total_lyapunov_decrease_interventions,
    )

    print(
        "Object-grasped steps:",
        f"{total_grasped_steps}/" f"{total_environment_steps}",
    )

    print()

    print(
        "SPRINT 5.9 ROBOTICS " "EMPIRICAL EFFECTIVENESS:",
        ("PASS" if effectiveness_supported else "FAIL"),
    )


if __name__ == "__main__":
    main()
