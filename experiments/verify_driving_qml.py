"""Independently verify Sprint 4.8 autonomous-driving QML evidence."""

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

QML_DIRECTORY = Path("results") / "rl" / "qml"

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

TARGET_PATH = PPO_DIRECTORY / "sprint4-ppo-targets.json"

SUMMARY_PATH = QML_DIRECTORY / "sprint4-driving-qml-summary.json"

EXPECTED_DOMAIN = "autonomous_driving"

EXPECTED_EVALUATION_STEPS = tuple(
    range(
        0,
        20_001,
        1_000,
    )
)

EXPECTED_PROJECTION_PARAMETERS = 20
EXPECTED_QUANTUM_PARAMETERS = 16
EXPECTED_ACTION_HEAD_PARAMETERS = 15
EXPECTED_ACTOR_PARAMETERS = 54
EXPECTED_CRITIC_PARAMETERS = 1249
EXPECTED_TOTAL_PARAMETERS = 1303

ROBUST_THRESHOLD_PERCENT = 10.0


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def sample_sd_or_none(
    values: list[float],
) -> float | None:
    """Return sample standard deviation or None."""
    if len(values) < 2:
        return None

    return float(statistics.stdev(values))


def mean_or_none(
    values: list[float],
) -> float | None:
    """Return arithmetic mean or None."""
    if not values:
        return None

    return float(statistics.mean(values))


def reconstruct_first_target_reach(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> tuple[int | None, int | None]:
    """Reconstruct first held-out target crossing."""
    for evaluation in evaluations:
        reward = float(evaluation["mean_reward"])

        if reward >= target_reward:
            return (
                int(evaluation["environment_steps"]),
                int(evaluation["completed_training_episodes"]),
            )

    return (
        None,
        None,
    )


def reconstruct_improvement(
    *,
    classical_steps: int,
    qml_steps: int | None,
) -> float | None:
    """Reconstruct paired sample-efficiency improvement."""
    if qml_steps is None:
        return None

    return 100.0 * (classical_steps - qml_steps) / classical_steps


def assert_optional_float_equal(
    actual: float | None,
    expected: float | None,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Compare optional floating-point values."""
    if expected is None:
        assert actual is None
        return

    assert actual is not None

    assert math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def main() -> None:
    """Run the independent Sprint 4.8 verification."""
    targets = load_json(TARGET_PATH)

    summary = load_json(SUMMARY_PATH)

    assert targets["qml_results_seen"] is False

    assert targets["domains"][EXPECTED_DOMAIN]["targets_frozen_before_qml"] is True

    target_records = targets["domains"][EXPECTED_DOMAIN]["seeds"]

    frozen_by_seed = {int(record["seed"]): record for record in target_records}

    assert set(frozen_by_seed) == set(SEEDS)

    best_rewards: list[float] = []
    final_rewards: list[float] = []
    final_success_rates: list[float] = []

    reached_qml_steps: list[float] = []
    valid_improvements: list[float] = []

    target_reach_count = 0

    print("==============================================")
    print(" SPRINT 4.8 INDEPENDENT DRIVING QML VERIFIER")
    print("==============================================")

    for seed in SEEDS:
        qml_path = QML_DIRECTORY / f"autonomous_driving-seed-{seed}.json"

        qml = load_json(qml_path)

        assert qml["domain"] == EXPECTED_DOMAIN

        assert qml["seed"] == seed

        assert qml["total_environment_steps"] == 20_000

        assert qml["projection_parameters"] == EXPECTED_PROJECTION_PARAMETERS

        assert qml["quantum_parameters"] == EXPECTED_QUANTUM_PARAMETERS

        assert qml["action_head_parameters"] == EXPECTED_ACTION_HEAD_PARAMETERS

        assert qml["actor_parameters"] == EXPECTED_ACTOR_PARAMETERS

        assert qml["critic_parameters"] == EXPECTED_CRITIC_PARAMETERS

        assert qml["total_parameters"] == EXPECTED_TOTAL_PARAMETERS

        evaluations = qml["evaluations"]

        assert len(evaluations) == len(EXPECTED_EVALUATION_STEPS)

        actual_steps = tuple(
            int(evaluation["environment_steps"]) for evaluation in evaluations
        )

        assert actual_steps == EXPECTED_EVALUATION_STEPS

        for evaluation in evaluations:
            reward = float(evaluation["mean_reward"])

            reward_sd = float(evaluation["reward_standard_deviation"])

            success_rate = float(evaluation["success_rate"])

            assert math.isfinite(reward)

            assert math.isfinite(reward_sd)

            assert 0.0 <= success_rate <= 1.0

        frozen = frozen_by_seed[seed]

        target_reward = float(frozen["target_evaluation_reward"])

        classical_steps = frozen["ppo_environment_steps_to_target"]

        assert classical_steps is not None

        classical_steps = int(classical_steps)

        comparison = qml["comparison"]

        assert math.isclose(
            float(comparison["target_evaluation_reward"]),
            target_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert (
            comparison["classical_ppo_environment_steps_to_target"] == classical_steps
        )

        reconstructed_steps, reconstructed_episodes = reconstruct_first_target_reach(
            evaluations=evaluations,
            target_reward=target_reward,
        )

        stored_qml_steps = comparison["qml_environment_steps_to_target"]

        stored_qml_episodes = comparison["qml_episodes_to_target"]

        assert stored_qml_steps == reconstructed_steps

        assert stored_qml_episodes == reconstructed_episodes

        reconstructed_improvement = reconstruct_improvement(
            classical_steps=classical_steps,
            qml_steps=reconstructed_steps,
        )

        stored_improvement = comparison["sample_efficiency_improvement_percent"]

        assert_optional_float_equal(
            stored_improvement,
            reconstructed_improvement,
        )

        best_reward = max(
            float(evaluation["mean_reward"]) for evaluation in evaluations
        )

        final_reward = float(evaluations[-1]["mean_reward"])

        final_success = float(evaluations[-1]["success_rate"])

        best_rewards.append(best_reward)

        final_rewards.append(final_reward)

        final_success_rates.append(final_success)

        if reconstructed_steps is not None:
            target_reach_count += 1

            reached_qml_steps.append(float(reconstructed_steps))

            assert reconstructed_improvement is not None

            valid_improvements.append(reconstructed_improvement)

        print()
        print(
            "seed=",
            seed,
        )
        print(
            "target=",
            target_reward,
        )
        print(
            "classical_steps=",
            classical_steps,
        )
        print(
            "reconstructed_qml_steps=",
            reconstructed_steps,
        )
        print(
            "reconstructed_improvement=",
            reconstructed_improvement,
        )
        print(
            "best_reward=",
            best_reward,
        )
        print(
            "final_reward=",
            final_reward,
        )

    reconstructed_best_mean = mean_or_none(best_rewards)

    reconstructed_best_sd = sample_sd_or_none(best_rewards)

    reconstructed_final_mean = mean_or_none(final_rewards)

    reconstructed_final_sd = sample_sd_or_none(final_rewards)

    reconstructed_success_mean = mean_or_none(final_success_rates)

    reconstructed_success_sd = sample_sd_or_none(final_success_rates)

    reconstructed_qml_steps_mean = mean_or_none(reached_qml_steps)

    reconstructed_qml_steps_sd = sample_sd_or_none(reached_qml_steps)

    reconstructed_improvement_mean = mean_or_none(valid_improvements)

    reconstructed_improvement_sd = sample_sd_or_none(valid_improvements)

    all_reached = target_reach_count == len(SEEDS)

    robust_criterion = (
        all_reached
        and reconstructed_improvement_mean is not None
        and reconstructed_improvement_mean >= ROBUST_THRESHOLD_PERCENT
    )

    aggregate = summary["aggregate"]

    assert_optional_float_equal(
        aggregate["best_qml_reward"]["mean"],
        reconstructed_best_mean,
    )

    assert_optional_float_equal(
        aggregate["best_qml_reward"]["sample_standard_deviation"],
        reconstructed_best_sd,
    )

    assert_optional_float_equal(
        aggregate["final_qml_reward"]["mean"],
        reconstructed_final_mean,
    )

    assert_optional_float_equal(
        aggregate["final_qml_reward"]["sample_standard_deviation"],
        reconstructed_final_sd,
    )

    assert_optional_float_equal(
        aggregate["final_success_rate"]["mean"],
        reconstructed_success_mean,
    )

    assert_optional_float_equal(
        aggregate["final_success_rate"]["sample_standard_deviation"],
        reconstructed_success_sd,
    )

    assert_optional_float_equal(
        aggregate["qml_environment_steps_to_target"]["mean_over_reached_seeds"],
        reconstructed_qml_steps_mean,
    )

    assert_optional_float_equal(
        aggregate["qml_environment_steps_to_target"][
            "sample_standard_deviation_over_reached_seeds"
        ],
        reconstructed_qml_steps_sd,
    )

    assert_optional_float_equal(
        aggregate["sample_efficiency_improvement_percent"]["mean_over_reached_seeds"],
        reconstructed_improvement_mean,
    )

    assert_optional_float_equal(
        aggregate["sample_efficiency_improvement_percent"][
            "sample_standard_deviation_over_reached_seeds"
        ],
        reconstructed_improvement_sd,
    )

    assert aggregate["target_reach_count"] == target_reach_count

    assert aggregate["target_reach_total"] == len(SEEDS)

    assert aggregate["all_seeds_reached_target"] is all_reached

    assert summary["robust_sample_efficiency_criterion"]["met"] is robust_criterion

    assert summary["parameter_accounting"]["classical_ppo_actor_parameters"] == 1318

    assert summary["parameter_accounting"]["hybrid_qml_actor_parameters"] == 54

    assert summary["parameter_accounting"]["hybrid_quantum_parameters"] == 16

    boundaries = summary["scientific_boundaries"]

    assert boundaries["quantum_hardware_used"] is False

    assert boundaries["quantum_speedup_claimed"] is False

    assert boundaries["wall_clock_speedup_claimed"] is False

    assert boundaries["robotics_result_included"] is False

    assert boundaries["cross_domain_claim_included"] is False

    print()
    print(
        "target_reach_count=",
        target_reach_count,
    )

    print(
        "reconstructed_best_mean=",
        reconstructed_best_mean,
    )

    print(
        "reconstructed_final_mean=",
        reconstructed_final_mean,
    )

    print(
        "reconstructed_improvement_mean=",
        reconstructed_improvement_mean,
    )

    print(
        "robust_criterion=",
        robust_criterion,
    )

    print()
    print("SPRINT 4.8.24 INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
