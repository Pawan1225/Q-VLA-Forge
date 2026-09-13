"""Independently verify Sprint 4.9 robotics QML evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path("results") / "rl"

QML_DIRECTORY = ROOT / "qml"

TARGET_PATH = ROOT / "ppo" / "sprint4-ppo-targets.json"

SUMMARY_PATH = QML_DIRECTORY / "sprint4-robotics-qml-summary.json"

SEEDS = (
    42,
    123,
    456,
)

ROBUST_THRESHOLD_PERCENT = 10.0


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def sample_mean(
    values: list[float],
) -> float:
    """Return arithmetic mean."""
    return float(statistics.mean(values))


def sample_sd(
    values: list[float],
) -> float:
    """Return sample standard deviation."""
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def find_target_record(
    *,
    targets: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Return exactly one frozen robotics target record."""
    records = targets["domains"]["robotics"]["seeds"]

    matches = [record for record in records if record["seed"] == seed]

    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one frozen robotics target " f"for seed {seed}"
        )

    return matches[0]


def first_target_reach(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> tuple[int | None, int | None]:
    """Reconstruct first held-out target reach."""
    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target_reward:
            return (
                int(evaluation["environment_steps"]),
                int(evaluation["completed_training_episodes"]),
            )

    return (
        None,
        None,
    )


def improvement_percent(
    *,
    classical_steps: int,
    qml_steps: int | None,
) -> float | None:
    """Reconstruct paired environment-step improvement."""
    if qml_steps is None:
        return None

    return float(100.0 * (classical_steps - qml_steps) / classical_steps)


def main() -> None:
    """Verify all Sprint 4.9 robotics evidence independently."""
    targets = load_json(TARGET_PATH)

    summary = load_json(SUMMARY_PATH)

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("historical PPO freeze marker is invalid")

    if targets["domains"]["robotics"]["targets_frozen_before_qml"] is not True:
        raise RuntimeError("robotics targets were not frozen before QML")

    reconstructed_best_rewards: list[float] = []
    reconstructed_final_rewards: list[float] = []
    reconstructed_final_success: list[float] = []
    reconstructed_improvements: list[float] = []

    target_reach_count = 0

    print("==============================================")
    print(" SPRINT 4.9 INDEPENDENT ROBOTICS QML VERIFIER")
    print("==============================================")

    for seed in SEEDS:
        run_path = QML_DIRECTORY / f"robotics-seed-{seed}.json"

        run = load_json(run_path)

        if run["domain"] != "robotics":
            raise RuntimeError(f"wrong domain in seed {seed}")

        if int(run["seed"]) != seed:
            raise RuntimeError(f"wrong seed in artifact {seed}")

        if int(run["total_environment_steps"]) != 20_000:
            raise RuntimeError(f"wrong environment-step budget for seed {seed}")

        evaluations = run["evaluations"]

        if len(evaluations) != 21:
            raise RuntimeError(f"expected 21 evaluations for seed {seed}")

        reference = find_target_record(
            targets=targets,
            seed=seed,
        )

        target_reward = float(reference["target_evaluation_reward"])

        classical_steps_raw = reference["ppo_environment_steps_to_target"]

        if classical_steps_raw is None:
            raise RuntimeError(f"classical PPO target missing for seed {seed}")

        classical_steps = int(classical_steps_raw)

        (
            reconstructed_qml_steps,
            reconstructed_qml_episodes,
        ) = first_target_reach(
            evaluations=evaluations,
            target_reward=target_reward,
        )

        reconstructed_improvement = improvement_percent(
            classical_steps=classical_steps,
            qml_steps=reconstructed_qml_steps,
        )

        stored_comparison = run["comparison"]

        if (
            stored_comparison["qml_environment_steps_to_target"]
            != reconstructed_qml_steps
        ):
            raise RuntimeError(f"QML target-step mismatch for seed {seed}")

        if stored_comparison["qml_episodes_to_target"] != reconstructed_qml_episodes:
            raise RuntimeError(f"QML target-episode mismatch for seed {seed}")

        if (
            stored_comparison["sample_efficiency_improvement_percent"]
            != reconstructed_improvement
        ):
            raise RuntimeError(f"improvement mismatch for seed {seed}")

        best_reward = max(
            float(evaluation["mean_reward"]) for evaluation in evaluations
        )

        final_reward = float(evaluations[-1]["mean_reward"])

        final_success = float(evaluations[-1]["success_rate"])

        reconstructed_best_rewards.append(best_reward)

        reconstructed_final_rewards.append(final_reward)

        reconstructed_final_success.append(final_success)

        if reconstructed_qml_steps is not None:
            target_reach_count += 1

        if reconstructed_improvement is not None:
            reconstructed_improvements.append(reconstructed_improvement)

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
            reconstructed_qml_steps,
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

    reconstructed_best_mean = sample_mean(reconstructed_best_rewards)

    reconstructed_best_sd = sample_sd(reconstructed_best_rewards)

    reconstructed_final_mean = sample_mean(reconstructed_final_rewards)

    reconstructed_final_sd = sample_sd(reconstructed_final_rewards)

    reconstructed_success_mean = sample_mean(reconstructed_final_success)

    reconstructed_success_sd = sample_sd(reconstructed_final_success)

    reconstructed_improvement_mean = (
        sample_mean(reconstructed_improvements) if reconstructed_improvements else None
    )

    robust_criterion = bool(
        target_reach_count == 3
        and len(reconstructed_improvements) == 3
        and reconstructed_improvement_mean is not None
        and reconstructed_improvement_mean >= ROBUST_THRESHOLD_PERCENT
    )

    if summary["qml_target_reach_count"] != target_reach_count:
        raise RuntimeError("summary target-reach count mismatch")

    summary_best = summary["best_reward"]

    if summary_best["mean"] != reconstructed_best_mean:
        raise RuntimeError("summary best-reward mean mismatch")

    if summary_best["sample_standard_deviation"] != reconstructed_best_sd:
        raise RuntimeError("summary best-reward SD mismatch")

    summary_final = summary["final_reward"]

    if summary_final["mean"] != reconstructed_final_mean:
        raise RuntimeError("summary final-reward mean mismatch")

    if summary_final["sample_standard_deviation"] != reconstructed_final_sd:
        raise RuntimeError("summary final-reward SD mismatch")

    summary_success = summary["final_success_rate"]

    if summary_success["mean"] != reconstructed_success_mean:
        raise RuntimeError("summary success mean mismatch")

    if summary_success["sample_standard_deviation"] != reconstructed_success_sd:
        raise RuntimeError("summary success SD mismatch")

    if reconstructed_improvements:
        summary_improvement = summary["sample_efficiency_improvement_percent"]

        if summary_improvement is None:
            raise RuntimeError("summary unexpectedly omits improvements")

        if summary_improvement["mean"] != reconstructed_improvement_mean:
            raise RuntimeError("summary improvement mean mismatch")
    else:
        if summary["sample_efficiency_improvement_percent"] is not None:
            raise RuntimeError("failed targets must preserve missing improvement")

        if summary["qml_steps_to_target"] is not None:
            raise RuntimeError("failed targets must preserve missing QML steps")

    if summary["robotics_robust_10_percent_criterion"] is not robust_criterion:
        raise RuntimeError("summary robust criterion mismatch")

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
        "reconstructed_best_sd=",
        reconstructed_best_sd,
    )

    print(
        "reconstructed_final_mean=",
        reconstructed_final_mean,
    )

    print(
        "reconstructed_final_sd=",
        reconstructed_final_sd,
    )

    print(
        "reconstructed_success_mean=",
        reconstructed_success_mean,
    )

    print(
        "reconstructed_success_sd=",
        reconstructed_success_sd,
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
    print("SPRINT 4.9.20 INDEPENDENT ROBOTICS QML VERIFICATION: PASS")


if __name__ == "__main__":
    main()
