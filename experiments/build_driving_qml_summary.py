"""Build the Sprint 4.8 autonomous-driving QML summary artifact."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

SEEDS = (
    42,
    123,
    456,
)

INPUT_DIRECTORY = Path("results") / "rl" / "qml"

OUTPUT_PATH = INPUT_DIRECTORY / "sprint4-driving-qml-summary.json"

EXPECTED_DOMAIN = "autonomous_driving"

CLASSICAL_ACTOR_PARAMETERS = 1318
HYBRID_ACTOR_PARAMETERS = 54
HYBRID_QUANTUM_PARAMETERS = 16

ROBUST_IMPROVEMENT_THRESHOLD_PERCENT = 10.0


def load_seed_artifact(
    seed: int,
) -> dict[str, Any]:
    """Load one principal autonomous-driving QML artifact."""
    path = INPUT_DIRECTORY / f"autonomous_driving-seed-{seed}.json"

    if not path.exists():
        raise FileNotFoundError(f"missing QML artifact: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("domain") != EXPECTED_DOMAIN:
        raise RuntimeError(f"unexpected domain in {path}: " f"{data.get('domain')!r}")

    if data.get("seed") != seed:
        raise RuntimeError(f"unexpected seed in {path}: " f"{data.get('seed')!r}")

    return data


def mean_or_none(
    values: list[float],
) -> float | None:
    """Return the arithmetic mean, preserving missing groups as None."""
    if not values:
        return None

    return float(statistics.mean(values))


def sample_sd_or_none(
    values: list[float],
) -> float | None:
    """Return sample standard deviation when at least two values exist."""
    if len(values) < 2:
        return None

    return float(statistics.stdev(values))


def finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    """Convert one required numeric value and require finiteness."""
    numeric = float(value)

    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} must be finite")

    return numeric


def build_summary() -> dict[str, Any]:
    """Build the locked three-seed autonomous-driving summary."""
    seed_records: list[dict[str, Any]] = []

    best_rewards: list[float] = []

    final_rewards: list[float] = []

    final_success_rates: list[float] = []

    valid_qml_steps: list[float] = []

    valid_improvements: list[float] = []

    target_reach_count = 0

    for seed in SEEDS:
        data = load_seed_artifact(seed)

        evaluations = data["evaluations"]

        if not evaluations:
            raise RuntimeError(f"seed {seed} has no evaluations")

        best_reward = max(
            finite_float(
                evaluation["mean_reward"],
                name=(f"seed {seed} " "evaluation mean_reward"),
            )
            for evaluation in evaluations
        )

        final = evaluations[-1]

        final_reward = finite_float(
            final["mean_reward"],
            name=(f"seed {seed} " "final mean_reward"),
        )

        final_success = finite_float(
            final["success_rate"],
            name=(f"seed {seed} " "final success_rate"),
        )

        if not (0.0 <= final_success <= 1.0):
            raise RuntimeError(f"seed {seed} final success rate " "must be in [0, 1]")

        comparison = data["comparison"]

        target_reward = finite_float(
            comparison["target_evaluation_reward"],
            name=(f"seed {seed} " "target_evaluation_reward"),
        )

        classical_steps = comparison["classical_ppo_environment_steps_to_target"]

        qml_steps = comparison["qml_environment_steps_to_target"]

        qml_episodes = comparison["qml_episodes_to_target"]

        improvement = comparison["sample_efficiency_improvement_percent"]

        target_reached = qml_steps is not None

        if target_reached:
            target_reach_count += 1

            qml_steps_value = finite_float(
                qml_steps,
                name=(f"seed {seed} " "qml_environment_steps_to_target"),
            )

            valid_qml_steps.append(qml_steps_value)

            if improvement is None:
                raise RuntimeError(
                    f"seed {seed} reached target " "but improvement is None"
                )

            improvement_value = finite_float(
                improvement,
                name=(f"seed {seed} " "sample_efficiency_improvement_percent"),
            )

            valid_improvements.append(improvement_value)

        else:
            if qml_episodes is not None:
                raise RuntimeError(
                    f"seed {seed} did not reach target "
                    "but qml_episodes_to_target is not None"
                )

            if improvement is not None:
                raise RuntimeError(
                    f"seed {seed} did not reach target " "but improvement is not None"
                )

        best_rewards.append(best_reward)

        final_rewards.append(final_reward)

        final_success_rates.append(final_success)

        seed_records.append(
            {
                "seed": seed,
                "target_evaluation_reward": (target_reward),
                "classical_ppo_environment_steps_to_target": (classical_steps),
                "qml_environment_steps_to_target": (qml_steps),
                "qml_episodes_to_target": (qml_episodes),
                "sample_efficiency_improvement_percent": (improvement),
                "target_reached": (target_reached),
                "best_qml_reward": (best_reward),
                "final_qml_reward": (final_reward),
                "final_success_rate": (final_success),
                "total_environment_steps": (data["total_environment_steps"]),
                "completed_training_episodes": (data["completed_training_episodes"]),
                "training_seconds": (data["training_seconds"]),
            }
        )

    all_seeds_reached_target = target_reach_count == len(SEEDS)

    mean_valid_improvement = mean_or_none(valid_improvements)

    robust_improvement_criterion_met = (
        all_seeds_reached_target
        and mean_valid_improvement is not None
        and mean_valid_improvement >= ROBUST_IMPROVEMENT_THRESHOLD_PERCENT
    )

    if target_reach_count == 0:
        outcome = (
            "No hybrid QML seed reached its paired "
            "frozen PPO reward target within the "
            "20,000-step training budget."
        )

    elif not all_seeds_reached_target:
        outcome = (
            "At least one hybrid QML seed reached its paired "
            "frozen PPO reward target, but target reach was "
            "not consistent across all three seeds."
        )

    elif (
        mean_valid_improvement is not None
        and mean_valid_improvement >= ROBUST_IMPROVEMENT_THRESHOLD_PERCENT
    ):
        outcome = (
            "All three hybrid QML seeds reached their paired "
            "frozen PPO targets and the mean sample-efficiency "
            "improvement satisfied the pre-specified 10 percent "
            "robustness threshold."
        )

    else:
        outcome = (
            "All three hybrid QML seeds reached their paired "
            "frozen PPO targets, but the mean sample-efficiency "
            "improvement did not satisfy the pre-specified "
            "10 percent robustness threshold."
        )

    return {
        "sprint": "4.8",
        "domain": EXPECTED_DOMAIN,
        "experiment": ("classical_ppo_vs_hybrid_qml_ppo"),
        "seeds": list(SEEDS),
        "training_budget_environment_steps_per_seed": (20_000),
        "evaluation_frequency_environment_steps": (1_000),
        "principal_seed_results": (seed_records),
        "aggregate": {
            "best_qml_reward": {
                "mean": mean_or_none(best_rewards),
                "sample_standard_deviation": (sample_sd_or_none(best_rewards)),
            },
            "final_qml_reward": {
                "mean": mean_or_none(final_rewards),
                "sample_standard_deviation": (sample_sd_or_none(final_rewards)),
            },
            "final_success_rate": {
                "mean": mean_or_none(final_success_rates),
                "sample_standard_deviation": (sample_sd_or_none(final_success_rates)),
            },
            "qml_environment_steps_to_target": {
                "mean_over_reached_seeds": (mean_or_none(valid_qml_steps)),
                "sample_standard_deviation_over_reached_seeds": (
                    sample_sd_or_none(valid_qml_steps)
                ),
                "missing_value_rule": (
                    "Seeds that did not reach the target remain "
                    "None and are not censored to the training budget."
                ),
            },
            "sample_efficiency_improvement_percent": {
                "mean_over_reached_seeds": (mean_valid_improvement),
                "sample_standard_deviation_over_reached_seeds": (
                    sample_sd_or_none(valid_improvements)
                ),
                "missing_value_rule": (
                    "Improvement remains None when the paired "
                    "QML target was not reached."
                ),
            },
            "target_reach_count": (target_reach_count),
            "target_reach_total": (len(SEEDS)),
            "all_seeds_reached_target": (all_seeds_reached_target),
        },
        "robust_sample_efficiency_criterion": {
            "required_seed_reach_count": (len(SEEDS)),
            "required_mean_improvement_percent": (ROBUST_IMPROVEMENT_THRESHOLD_PERCENT),
            "criterion": (
                "All three seeds must reach their paired frozen "
                "PPO reward target and the mean valid "
                "sample-efficiency improvement must be at least "
                "10 percent."
            ),
            "met": (robust_improvement_criterion_met),
        },
        "parameter_accounting": {
            "classical_ppo_actor_parameters": (CLASSICAL_ACTOR_PARAMETERS),
            "hybrid_qml_actor_parameters": (HYBRID_ACTOR_PARAMETERS),
            "hybrid_quantum_parameters": (HYBRID_QUANTUM_PARAMETERS),
            "hybrid_actor_parameter_reduction_percent": (
                100.0
                * (CLASSICAL_ACTOR_PARAMETERS - HYBRID_ACTOR_PARAMETERS)
                / CLASSICAL_ACTOR_PARAMETERS
            ),
        },
        "outcome": (outcome),
        "scientific_boundaries": {
            "quantum_hardware_used": False,
            "quantum_speedup_claimed": False,
            "wall_clock_speedup_claimed": False,
            "robotics_result_included": False,
            "cross_domain_claim_included": False,
            "production_autonomous_driving_claimed": False,
            "runtime_interpretation": (
                "Wall-clock runtime is simulator and machine dependent "
                "and is not used as evidence of quantum speedup or "
                "sample-efficiency advantage."
            ),
        },
    }


def main() -> None:
    """Write and print the three-seed Sprint 4.8 summary."""
    summary = build_summary()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    aggregate = summary["aggregate"]

    robust = summary["robust_sample_efficiency_criterion"]

    print("==============================================")
    print(" SPRINT 4.8 DRIVING QML SUMMARY")
    print("==============================================")

    print(
        "Seeds:",
        summary["seeds"],
    )

    print(
        "Target reach:",
        f"{aggregate['target_reach_count']}" f"/{aggregate['target_reach_total']}",
    )

    print(
        "Best reward mean:",
        aggregate["best_qml_reward"]["mean"],
    )

    print(
        "Best reward SD:",
        aggregate["best_qml_reward"]["sample_standard_deviation"],
    )

    print(
        "Final reward mean:",
        aggregate["final_qml_reward"]["mean"],
    )

    print(
        "Final reward SD:",
        aggregate["final_qml_reward"]["sample_standard_deviation"],
    )

    print(
        "Final success mean:",
        aggregate["final_success_rate"]["mean"],
    )

    print(
        "QML mean steps to target:",
        aggregate["qml_environment_steps_to_target"]["mean_over_reached_seeds"],
    )

    print(
        "Mean sample-efficiency improvement:",
        aggregate["sample_efficiency_improvement_percent"]["mean_over_reached_seeds"],
    )

    print(
        "Robust 10% criterion met:",
        robust["met"],
    )

    print(
        "Outcome:",
        summary["outcome"],
    )

    print(
        "Artifact:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
