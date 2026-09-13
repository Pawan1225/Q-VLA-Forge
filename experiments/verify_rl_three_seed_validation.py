"""Independent verification for Sprint 4.10."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

QML_DIRECTORY = Path("results") / "rl" / "qml"

VALIDATION_PATH = (
    Path("results") / "rl" / "validation" / "sprint4-three-seed-validation.json"
)

SEEDS = (
    42,
    123,
    456,
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

EXPECTED_STEPS = list(
    range(
        0,
        20_001,
        1_000,
    )
)


def load(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def target_record(
    *,
    targets: dict[str, Any],
    domain: str,
    seed: int,
) -> dict[str, Any]:
    """Return one frozen target record."""

    matches = [
        record
        for record in targets["domains"][domain]["seeds"]
        if int(record["seed"]) == seed
    ]

    if len(matches) != 1:
        raise RuntimeError("target lookup failed")

    return matches[0]


def first_crossing(
    *,
    evaluations: list[dict[str, Any]],
    target: float,
) -> int | None:
    """Return first environment-step target crossing."""

    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target:
            return int(evaluation["environment_steps"])

    return None


def assert_run_shape(
    run: dict[str, Any],
) -> None:
    """Validate the frozen principal-run structure."""

    assert int(run["total_environment_steps"]) == 20_000

    evaluations = run["evaluations"]

    assert len(evaluations) == 21

    assert [
        int(evaluation["environment_steps"]) for evaluation in evaluations
    ] == EXPECTED_STEPS

    for evaluation in evaluations:
        assert math.isfinite(float(evaluation["mean_reward"]))

        assert math.isfinite(float(evaluation["reward_standard_deviation"]))

        assert float(evaluation["reward_standard_deviation"]) >= 0.0

        success = float(evaluation["success_rate"])

        assert 0.0 <= success <= 1.0


def verify_reproduction(
    *,
    principal_path: Path,
    reproduction_path: Path,
) -> None:
    """Verify exact seed-42 held-out trajectory reproduction."""

    principal = load(principal_path)

    reproduction = load(reproduction_path)

    principal_evaluations = principal["evaluations"]

    reproduction_evaluations = reproduction["evaluations"]

    assert len(principal_evaluations) == 21

    assert len(reproduction_evaluations) == 21

    for first, second in zip(
        principal_evaluations,
        reproduction_evaluations,
        strict=True,
    ):
        assert first["environment_steps"] == second["environment_steps"]

        assert (
            first["completed_training_episodes"]
            == second["completed_training_episodes"]
        )

        assert float(first["mean_reward"]) == float(second["mean_reward"])

        assert float(first["reward_standard_deviation"]) == float(
            second["reward_standard_deviation"]
        )

        assert float(first["success_rate"]) == float(second["success_rate"])


def main() -> None:
    """Independently verify Sprint 4.10 evidence."""

    targets = load(PPO_DIRECTORY / "sprint4-ppo-targets.json")

    validation = load(VALIDATION_PATH)

    assert validation["sprint"] == "4.10"

    assert validation["principal_training_performed"] is False

    assert validation["principal_seeds"] == list(SEEDS)

    reconstructed_qml_reaches = 0

    for domain in DOMAINS:
        validation_records = {
            int(record["seed"]): record
            for record in validation["domains"][domain]["records"]
        }

        assert set(validation_records) == set(SEEDS)

        qml_best_rewards: list[float] = []

        qml_gaps: list[float] = []

        qml_final_rewards: list[float] = []

        qml_final_success: list[float] = []

        classical_reach_count = 0
        qml_reach_count = 0

        for seed in SEEDS:
            classical = load(PPO_DIRECTORY / f"{domain}-seed-{seed}.json")

            qml = load(QML_DIRECTORY / f"{domain}-seed-{seed}.json")

            assert_run_shape(classical)

            assert_run_shape(qml)

            assert classical["domain"] == domain

            assert qml["domain"] == domain

            assert int(classical["seed"]) == seed

            assert int(qml["seed"]) == seed

            frozen = target_record(
                targets=targets,
                domain=domain,
                seed=seed,
            )

            target = float(frozen["target_evaluation_reward"])

            classical_crossing = first_crossing(
                evaluations=classical["evaluations"],
                target=target,
            )

            qml_crossing = first_crossing(
                evaluations=qml["evaluations"],
                target=target,
            )

            assert classical_crossing is not None

            assert classical_crossing == int(frozen["ppo_environment_steps_to_target"])

            classical_reach_count += 1

            qml_best = max(
                float(evaluation["mean_reward"]) for evaluation in qml["evaluations"]
            )

            qml_final = float(qml["evaluations"][-1]["mean_reward"])

            qml_success = float(qml["evaluations"][-1]["success_rate"])

            gap = qml_best - target

            record = validation_records[seed]

            assert math.isclose(
                float(record["target_reward"]),
                target,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

            assert math.isclose(
                float(record["qml_best_reward"]),
                qml_best,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

            assert math.isclose(
                float(record["qml_final_reward"]),
                qml_final,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

            assert math.isclose(
                float(record["qml_final_success_rate"]),
                qml_success,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

            assert math.isclose(
                float(record["best_reward_target_gap"]),
                gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

            assert record["qml_steps_to_target"] == qml_crossing

            assert record["qml_target_reached"] is (qml_crossing is not None)

            if qml_crossing is None:
                assert record["sample_efficiency_improvement_percent"] is None
            else:
                qml_reach_count += 1
                reconstructed_qml_reaches += 1

                expected_improvement = (
                    100.0
                    * (int(classical_crossing) - qml_crossing)
                    / int(classical_crossing)
                )

                assert math.isclose(
                    float(record["sample_efficiency_improvement_percent"]),
                    expected_improvement,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )

            qml_best_rewards.append(qml_best)

            qml_gaps.append(gap)

            qml_final_rewards.append(qml_final)

            qml_final_success.append(qml_success)

        summary = validation["domains"][domain]["summary"]

        assert summary["classical_target_reach_count"] == classical_reach_count == 3

        assert summary["qml_target_reach_count"] == qml_reach_count

        assert math.isclose(
            float(summary["qml_best_reward_mean"]),
            statistics.mean(qml_best_rewards),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_best_reward_sd"]),
            statistics.stdev(qml_best_rewards),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_final_reward_mean"]),
            statistics.mean(qml_final_rewards),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_final_reward_sd"]),
            statistics.stdev(qml_final_rewards),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_final_success_mean"]),
            statistics.mean(qml_final_success),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_final_success_sd"]),
            statistics.stdev(qml_final_success),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_best_reward_target_gap_mean"]),
            statistics.mean(qml_gaps),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            float(summary["qml_best_reward_target_gap_sd"]),
            statistics.stdev(qml_gaps),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert summary["robust_10_percent_criterion"] is False

    verify_reproduction(
        principal_path=(QML_DIRECTORY / "autonomous_driving-seed-42.json"),
        reproduction_path=(QML_DIRECTORY / "autonomous_driving-seed-42-repro.json"),
    )

    verify_reproduction(
        principal_path=(QML_DIRECTORY / "robotics-seed-42.json"),
        reproduction_path=(QML_DIRECTORY / "robotics-seed-42-repro.json"),
    )

    global_validation = validation["global_validation"]

    assert global_validation["principal_classical_runs"] == 6

    assert global_validation["principal_qml_runs"] == 6

    assert global_validation["principal_runs_total"] == 12

    assert global_validation["qml_targets_reached"] == reconstructed_qml_reaches

    assert global_validation["qml_targets_available"] == 6

    assert global_validation["driving_seed42_reproduction_exists"] is True

    assert global_validation["robotics_seed42_reproduction_exists"] is True

    boundary = validation["scientific_boundary"]

    assert boundary["new_training_performed"] is False

    assert boundary["hyperparameter_tuning_performed"] is False

    assert boundary["targets_modified"] is False

    assert boundary["environments_modified"] is False

    assert boundary["target_gap_is_descriptive_only"] is True

    assert boundary["cross_domain_reward_scale_comparison_allowed"] is False

    assert boundary["quantum_speedup_claimed"] is False

    assert boundary["quantum_hardware_advantage_claimed"] is False

    print()
    print("==============================================")
    print(" SPRINT 4.10 THREE-SEED VALIDATION VERIFIER")
    print("==============================================")

    print(
        "Principal classical runs:",
        6,
    )

    print(
        "Principal QML runs:",
        6,
    )

    print(
        "QML target reaches:",
        reconstructed_qml_reaches,
        "/ 6",
    )

    print(
        "Driving seed-42 reproduction:",
        "PASS",
    )

    print(
        "Robotics seed-42 reproduction:",
        "PASS",
    )

    print(
        "No new training:",
        "PASS",
    )

    print()

    print("SPRINT 4.10 THREE-SEED VALIDATION: PASS")


if __name__ == "__main__":
    main()
