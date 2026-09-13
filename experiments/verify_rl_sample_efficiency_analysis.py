"""Independent verification for Sprint 4.11 sample-efficiency analysis."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ANALYSIS_PATH = Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.json"

CSV_PATH = Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.csv"

MARKDOWN_PATH = Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.md"

ENVIRONMENT_AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

TARGETS_PATH = Path("results") / "rl" / "ppo" / "sprint4-ppo-targets.json"

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

QML_DIRECTORY = Path("results") / "rl" / "qml"

FIGURES = (
    Path("figures/rl/driving-learning-curves.png"),
    Path("figures/rl/robotics-learning-curves.png"),
    Path("figures/rl/driving-normalized-progress.png"),
    Path("figures/rl/robotics-normalized-progress.png"),
    Path("figures/rl/compactness-performance-tradeoff.png"),
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

POLICIES = (
    "ppo",
    "qml",
)

RANDOM_KEYS = {
    "autonomous_driving": "driving_random",
    "robotics": "robotics_random",
}

EXPECTED_ACTOR_PARAMETERS = {
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
    return json.loads(path.read_text(encoding="utf-8"))


def close(
    observed: float,
    expected: float,
) -> bool:
    return math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def target_for_seed(
    *,
    targets: dict[str, Any],
    domain: str,
    seed: int,
) -> float:
    records = targets["domains"][domain]["seeds"]

    match = [record for record in records if int(record["seed"]) == seed]

    if len(match) != 1:
        raise AssertionError(f"target lookup failed for {domain} seed {seed}")

    return float(match[0]["target_evaluation_reward"])


def run_path(
    *,
    domain: str,
    seed: int,
    policy: str,
) -> Path:
    directory = PPO_DIRECTORY if policy == "ppo" else QML_DIRECTORY

    return directory / f"{domain}-seed-{seed}.json"


def normalized_progress(
    *,
    reward: float,
    random_reference: float,
    target_reward: float,
) -> float:
    denominator = target_reward - random_reference

    assert denominator > 0.0

    return float((reward - random_reference) / denominator)


def normalized_auc(
    *,
    steps: list[int],
    values: list[float],
) -> float:
    assert len(steps) == 21
    assert len(values) == 21

    area = 0.0

    for index in range(
        1,
        len(steps),
    ):
        width = steps[index] - steps[index - 1]

        assert width == 1000

        area += width * (values[index - 1] + values[index]) / 2.0

    return float(area / 20_000.0)


def mean_sd(
    values: list[float],
) -> tuple[float, float]:
    return (
        float(statistics.mean(values)),
        float(statistics.stdev(values)),
    )


def first_crossing(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> int | None:
    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target_reward:
            return int(evaluation["environment_steps"])

    return None


def main() -> None:
    analysis = load_json(ANALYSIS_PATH)

    environment_audit = load_json(ENVIRONMENT_AUDIT_PATH)

    targets = load_json(TARGETS_PATH)

    assert analysis["sprint"] == "4.11"
    assert analysis["new_training_performed"] is False

    total_ppo_reaches = 0
    total_qml_reaches = 0

    for domain in DOMAINS:
        random_reference = float(
            environment_audit["summaries"][RANDOM_KEYS[domain]]["mean_reward"]
        )

        observed_random = float(analysis["domains"][domain]["random_reference"])

        assert close(
            observed_random,
            random_reference,
        )

        for policy in POLICIES:
            reconstructed_records = []

            for seed in SEEDS:
                target_reward = target_for_seed(
                    targets=targets,
                    domain=domain,
                    seed=seed,
                )

                run = load_json(
                    run_path(
                        domain=domain,
                        seed=seed,
                        policy=policy,
                    )
                )

                evaluations = run["evaluations"]

                assert len(evaluations) == 21

                steps = [
                    int(evaluation["environment_steps"]) for evaluation in evaluations
                ]

                assert steps == list(
                    range(
                        0,
                        20_001,
                        1000,
                    )
                )

                rewards = [
                    float(evaluation["mean_reward"]) for evaluation in evaluations
                ]

                progress = [
                    normalized_progress(
                        reward=reward,
                        random_reference=random_reference,
                        target_reward=target_reward,
                    )
                    for reward in rewards
                ]

                auc = normalized_auc(
                    steps=steps,
                    values=progress,
                )

                crossing = first_crossing(
                    evaluations=evaluations,
                    target_reward=target_reward,
                )

                reconstructed_records.append(
                    {
                        "seed": seed,
                        "auc": auc,
                        "best_progress": max(progress),
                        "final_progress": progress[-1],
                        "best_reward": max(rewards),
                        "final_reward": rewards[-1],
                        "crossing": crossing,
                    }
                )

            stored_records = analysis["domains"][domain]["policies"][policy]["records"]

            assert len(stored_records) == 3

            for expected in reconstructed_records:
                stored = next(
                    record
                    for record in stored_records
                    if int(record["seed"]) == expected["seed"]
                )

                assert close(
                    float(stored["normalized_auc"]),
                    expected["auc"],
                )

                assert close(
                    float(stored["best_normalized_progress"]),
                    expected["best_progress"],
                )

                assert close(
                    float(stored["final_normalized_progress"]),
                    expected["final_progress"],
                )

                assert close(
                    float(stored["best_reward"]),
                    expected["best_reward"],
                )

                assert close(
                    float(stored["final_reward"]),
                    expected["final_reward"],
                )

                assert stored["steps_to_target"] == expected["crossing"]

                assert (
                    int(stored["actor_parameters"])
                    == EXPECTED_ACTOR_PARAMETERS[domain][policy]
                )

            aggregate = analysis["domains"][domain]["policies"][policy]["aggregate"]

            auc_mean, auc_sd = mean_sd(
                [record["auc"] for record in reconstructed_records]
            )

            best_mean, best_sd = mean_sd(
                [record["best_progress"] for record in reconstructed_records]
            )

            final_mean, final_sd = mean_sd(
                [record["final_progress"] for record in reconstructed_records]
            )

            assert close(
                float(aggregate["normalized_auc"]["mean"]),
                auc_mean,
            )

            assert close(
                float(aggregate["normalized_auc"]["sample_standard_deviation"]),
                auc_sd,
            )

            assert close(
                float(aggregate["best_normalized_progress"]["mean"]),
                best_mean,
            )

            assert close(
                float(
                    aggregate["best_normalized_progress"]["sample_standard_deviation"]
                ),
                best_sd,
            )

            assert close(
                float(aggregate["final_normalized_progress"]["mean"]),
                final_mean,
            )

            assert close(
                float(
                    aggregate["final_normalized_progress"]["sample_standard_deviation"]
                ),
                final_sd,
            )

            reach_count = sum(
                record["crossing"] is not None for record in reconstructed_records
            )

            assert aggregate["target_reach_count"] == reach_count

            if policy == "ppo":
                total_ppo_reaches += reach_count
            else:
                total_qml_reaches += reach_count

    assert total_ppo_reaches == 6
    assert total_qml_reaches == 0

    assert analysis["primary_metric"]["classical_target_reaches"] == 6

    assert analysis["primary_metric"]["qml_target_reaches"] == 0

    assert analysis["primary_metric"]["qml_mean_steps_to_target"] is None

    assert analysis["primary_metric"]["failed_qml_targets_censored_to_budget"] is False

    compactness = analysis["compactness"]

    assert close(
        float(compactness["autonomous_driving"]["reduction_percent"]),
        95.90288315629742,
    )

    assert close(
        float(compactness["robotics"]["reduction_percent"]),
        95.5137481910275,
    )

    boundary = analysis["scientific_boundary"]

    assert boundary["primary_metric_preserved"] is True

    assert boundary["secondary_metrics_are_descriptive"] is True

    assert boundary["new_training_performed"] is False

    assert boundary["hyperparameter_tuning_performed"] is False

    assert boundary["targets_modified"] is False

    assert boundary["environments_modified"] is False

    assert boundary["policies_modified"] is False

    assert boundary["pqc_failure_cause_isolated"] is False

    assert boundary["quantum_speedup_claimed"] is False

    assert boundary["quantum_hardware_advantage_claimed"] is False

    assert CSV_PATH.exists()
    assert CSV_PATH.stat().st_size > 0

    assert MARKDOWN_PATH.exists()
    assert MARKDOWN_PATH.stat().st_size > 0

    for figure in FIGURES:
        assert figure.exists()
        assert figure.stat().st_size > 0

    print()
    print("==============================================")
    print(" SPRINT 4.11 SAMPLE-EFFICIENCY VERIFIER")
    print("==============================================")
    print()
    print("Principal PPO runs: 6")
    print("Principal QML runs: 6")
    print("PPO target reaches: 6 / 6")
    print("QML target reaches: 0 / 6")
    print("Figures verified: 5")
    print("New training: False")
    print()
    print("SPRINT 4.11 SAMPLE-EFFICIENCY ANALYSIS: PASS")


if __name__ == "__main__":
    main()
