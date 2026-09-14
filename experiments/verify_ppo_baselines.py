"""Independent verification for Sprint 4.5 PPO baselines."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

TARGET_PATH = PPO_DIRECTORY / "sprint4-ppo-targets.json"

SUMMARY_PATH = PPO_DIRECTORY / "sprint4-ppo-summary.json"

RESULTS_ROOT = Path("results")

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

EXPECTED_EVALUATION_STEPS = list(
    range(
        0,
        20_001,
        1_000,
    )
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Assert that two floating-point values are equal within tolerance."""
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        f"value mismatch: " f"actual={actual}, " f"expected={expected}"
    )


def verify_ppo_baselines() -> None:
    """Independently verify the frozen Sprint 4.5 PPO evidence."""
    audit = load_json(AUDIT_PATH)

    targets = load_json(TARGET_PATH)

    summary = load_json(SUMMARY_PATH)

    assert targets["qml_results_seen"] is False

    assert targets["target_fraction"] == 0.95

    assert summary["seeds"] == list(SEEDS)

    random_references = {
        "autonomous_driving": (audit["summaries"]["driving_random"]["mean_reward"]),
        "robotics": (audit["summaries"]["robotics_random"]["mean_reward"]),
    }

    for domain in DOMAINS:
        print()
        print("========================================")
        print(
            "VERIFY",
            domain,
        )
        print("========================================")

        target_domain = targets["domains"][domain]

        summary_domain = summary["domains"][domain]

        assert target_domain["targets_frozen_before_qml"] is True

        target_records = {record["seed"]: record for record in target_domain["seeds"]}

        summary_runs = {run["seed"]: run for run in summary_domain["runs"]}

        assert set(target_records) == set(SEEDS)

        assert set(summary_runs) == set(SEEDS)

        best_rewards: list[float] = []
        final_rewards: list[float] = []
        final_success_rates: list[float] = []
        steps_to_target: list[float] = []
        episodes_to_target: list[float] = []

        for seed in SEEDS:
            path = PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

            result = load_json(path)

            evaluations = result["evaluations"]

            assert result["domain"] == domain

            assert result["seed"] == seed

            assert result["total_environment_steps"] == 20_000

            assert len(evaluations) == 21

            actual_steps = [
                evaluation["environment_steps"] for evaluation in evaluations
            ]

            assert actual_steps == EXPECTED_EVALUATION_STEPS

            assert all(
                math.isfinite(evaluation["mean_reward"]) for evaluation in evaluations
            )

            assert all(
                math.isfinite(evaluation["reward_standard_deviation"])
                for evaluation in evaluations
            )

            assert all(
                0.0 <= evaluation["success_rate"] <= 1.0 for evaluation in evaluations
            )

            assert result["actor_parameters"] > 0

            assert result["critic_parameters"] > 0

            assert (
                result["actor_parameters"] + result["critic_parameters"]
                == result["total_parameters"]
            )

            best_reward = max(evaluation["mean_reward"] for evaluation in evaluations)

            final = evaluations[-1]

            random_reference = random_references[domain]

            assert best_reward > random_reference

            expected_target = random_reference + 0.95 * (best_reward - random_reference)

            target_record = target_records[seed]

            assert_close(
                target_record["random_reference_reward"],
                random_reference,
            )

            assert_close(
                target_record["ppo_best_evaluation_reward"],
                best_reward,
            )

            assert_close(
                target_record["target_evaluation_reward"],
                expected_target,
            )

            first_reach = next(
                (
                    evaluation
                    for evaluation in evaluations
                    if evaluation["mean_reward"] >= expected_target
                ),
                None,
            )

            assert first_reach is not None

            assert (
                target_record["ppo_environment_steps_to_target"]
                == first_reach["environment_steps"]
            )

            assert (
                target_record["ppo_episodes_to_target"]
                == first_reach["completed_training_episodes"]
            )

            summary_run = summary_runs[seed]

            assert_close(
                summary_run["best_evaluation_reward"],
                best_reward,
            )

            assert_close(
                summary_run["final_evaluation_reward"],
                final["mean_reward"],
            )

            assert_close(
                summary_run["final_success_rate"],
                final["success_rate"],
            )

            assert (
                summary_run["environment_steps_to_target"]
                == first_reach["environment_steps"]
            )

            assert (
                summary_run["episodes_to_target"]
                == first_reach["completed_training_episodes"]
            )

            best_rewards.append(best_reward)

            final_rewards.append(final["mean_reward"])

            final_success_rates.append(final["success_rate"])

            steps_to_target.append(float(first_reach["environment_steps"]))

            episodes_to_target.append(float(first_reach["completed_training_episodes"]))

            print(
                "seed",
                seed,
                "PASS",
            )

        assert_close(
            summary_domain["best_evaluation_reward"]["mean"],
            statistics.mean(best_rewards),
        )

        assert_close(
            summary_domain["best_evaluation_reward"]["sample_standard_deviation"],
            statistics.stdev(best_rewards),
        )

        assert_close(
            summary_domain["final_evaluation_reward"]["mean"],
            statistics.mean(final_rewards),
        )

        assert_close(
            summary_domain["final_evaluation_reward"]["sample_standard_deviation"],
            statistics.stdev(final_rewards),
        )

        assert_close(
            summary_domain["final_success_rate"]["mean"],
            statistics.mean(final_success_rates),
        )

        assert_close(
            summary_domain["final_success_rate"]["sample_standard_deviation"],
            statistics.stdev(final_success_rates),
        )

        assert_close(
            summary_domain["environment_steps_to_target"]["mean"],
            statistics.mean(steps_to_target),
        )

        assert_close(
            summary_domain["environment_steps_to_target"]["sample_standard_deviation"],
            statistics.stdev(steps_to_target),
        )

        assert_close(
            summary_domain["episodes_to_target"]["mean"],
            statistics.mean(episodes_to_target),
        )

        assert_close(
            summary_domain["episodes_to_target"]["sample_standard_deviation"],
            statistics.stdev(episodes_to_target),
        )

        assert summary_domain["target_reach_count"] == 3

        print(
            domain,
            "SUMMARY VERIFICATION: PASS",
        )

    print()

    print(
        "QML contamination check:",
        targets["qml_results_seen"] is False,
    )

    print("SPRINT 4.5.24 INDEPENDENT PPO VERIFIER: PASS")


def verify_claim_boundary() -> None:
    """Verify the historical Sprint 4.5 PPO/QML claim boundary."""
    targets = load_json(TARGET_PATH)

    print()
    print("========================================")
    print(" SPRINT 4.5 CLAIM / QML CONTAMINATION AUDIT")
    print("========================================")

    assert targets["qml_results_seen"] is False

    assert all(
        domain["targets_frozen_before_qml"] is True
        for domain in targets["domains"].values()
    )

    print("Frozen targets before QML: PASS")

    print(
        "qml_results_seen:",
        targets["qml_results_seen"],
    )

    expected_ppo_files = {
        "autonomous_driving-seed-42.json",
        "autonomous_driving-seed-123.json",
        "autonomous_driving-seed-456.json",
        "robotics-seed-42.json",
        "robotics-seed-123.json",
        "robotics-seed-456.json",
        "sprint4-ppo-targets.json",
        "sprint4-ppo-summary.json",
    }

    actual_ppo_files = {path.name for path in PPO_DIRECTORY.glob("*.json")}

    missing = expected_ppo_files - actual_ppo_files

    assert not missing, "missing PPO artifacts: " f"{sorted(missing)}"

    print("Classical PPO artifact inventory: PASS")

    qml_result_candidates = []

    for path in RESULTS_ROOT.rglob("*"):
        if not path.is_file():
            continue

        lowered = str(path).lower()

        if (
            "qml" in lowered
            or "quantum_policy" in lowered
            or "hybrid_policy" in lowered
            or "pqc_result" in lowered
        ):
            qml_result_candidates.append(str(path))

    print(
        "Post-freeze QML-like result artifacts found:",
        len(qml_result_candidates),
    )

    for candidate in qml_result_candidates:
        print(
            " ",
            candidate,
        )

    # Sprint 4.5's scientific contamination boundary is temporal.
    #
    # Classical PPO results and their paired reward targets were
    # established and frozen before principal QML experimentation.
    #
    # Later Sprint 4.6+ work is expected to create QML, PQC, and
    # hybrid-policy artifacts. Their presence after the freeze does
    # not invalidate the historical Sprint 4.5 baseline boundary.
    #
    # The frozen target artifact remains the source of truth for
    # whether QML results had been observed when those targets were
    # created.
    assert targets["qml_results_seen"] is False

    assert all(
        domain["targets_frozen_before_qml"] is True
        for domain in targets["domains"].values()
    )

    if qml_result_candidates:
        print("Post-freeze QML artifacts present: " "EXPECTED after Sprint 4.5")
    else:
        print("Post-freeze QML artifacts present: NONE")

    print("Historical PPO/QML contamination boundary: PASS")

    claim_files = [
        path for path in RESULTS_ROOT.rglob("*.json") if "claim" in path.name.lower()
    ]

    control_artifacts = []

    post_freeze_claim_artifacts = []

    result_like_claim_artifacts = []

    for path in claim_files:
        lowered_name = path.name.lower()

        lowered_path = str(path).lower()

        lowered_parts = {part.lower() for part in path.parts}

        if "registry" in lowered_name or "readiness" in lowered_path:
            control_artifacts.append(path)

        elif "evidence" in lowered_parts:
            # Sprint 4.14 proposal evidence is intentionally
            # generated after the historical Sprint 4.5 freeze.
            #
            # Its wording may describe later QML findings and
            # limitations, so it must not be interpreted as a
            # claim that existed when the PPO targets were frozen.
            post_freeze_claim_artifacts.append(path)

        else:
            result_like_claim_artifacts.append(path)

    for path in result_like_claim_artifacts:
        text = path.read_text(encoding="utf-8").lower()

        forbidden_promotions = (
            "quantum advantage",
            "quantum speedup",
            "qml advantage",
            "sample-efficiency advantage",
            "sample efficiency advantage",
        )

        for phrase in forbidden_promotions:
            assert phrase not in text, (
                "premature claim phrase " f"{phrase!r} found in {path}"
            )

    print(
        "Claim-control artifacts:",
        len(control_artifacts),
    )

    for path in control_artifacts:
        print(
            " controlled:",
            path,
        )

    print(
        "Post-freeze claim artifacts:",
        len(post_freeze_claim_artifacts),
    )

    for path in post_freeze_claim_artifacts:
        print(
            " post-freeze:",
            path,
        )

    print(
        "Result-like claim artifacts inspected:",
        len(result_like_claim_artifacts),
    )

    print("No premature quantum/sample-efficiency claim: PASS")

    print()
    print("Allowed Sprint 4.5 claim:")

    print("Classical PPO baselines and paired " "reward targets have been established.")

    print()
    print("Historical Sprint 4.5 boundary:")

    print(
        "No principal QML result had been seen when "
        "the paired classical PPO targets were frozen."
    )

    print()
    print("Later Sprint 4.6+ QML artifacts:")

    print("Allowed after the historical Sprint 4.5 freeze.")

    print()
    print("Not established by Sprint 4.5:")

    print(
        "QML sample-efficiency improvement, " "quantum advantage, or quantum speedup."
    )

    print()

    print("SPRINT 4.5.25 CLAIM / QML " "CONTAMINATION AUDIT: PASS")


def main() -> None:
    """Run the complete Sprint 4.5 independent verification."""
    verify_ppo_baselines()
    verify_claim_boundary()


if __name__ == "__main__":
    main()
