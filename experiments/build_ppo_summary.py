"""Build Sprint 4.5 three-seed PPO summary evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

TARGET_PATH = PPO_DIRECTORY / "sprint4-ppo-targets.json"

OUTPUT_PATH = PPO_DIRECTORY / "sprint4-ppo-summary.json"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_and_sample_sd(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def main() -> None:
    targets = load_json(TARGET_PATH)

    summary: dict[
        str,
        Any,
    ] = {
        "sprint": "4.5",
        "method": "classical_ppo",
        "seeds": list(SEEDS),
        "domains": {},
    }

    for domain in DOMAINS:
        runs = []

        for seed in SEEDS:
            path = PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

            result = load_json(path)

            evaluations = result["evaluations"]

            best_reward = max(evaluation["mean_reward"] for evaluation in evaluations)

            final = evaluations[-1]

            target_record = next(
                record
                for record in targets["domains"][domain]["seeds"]
                if record["seed"] == seed
            )

            runs.append(
                {
                    "seed": seed,
                    "best_evaluation_reward": (best_reward),
                    "final_evaluation_reward": (final["mean_reward"]),
                    "final_success_rate": (final["success_rate"]),
                    "environment_steps_to_target": (
                        target_record["ppo_environment_steps_to_target"]
                    ),
                    "episodes_to_target": (target_record["ppo_episodes_to_target"]),
                    "completed_training_episodes": (
                        result["completed_training_episodes"]
                    ),
                    "actor_parameters": (result["actor_parameters"]),
                    "critic_parameters": (result["critic_parameters"]),
                    "total_parameters": (result["total_parameters"]),
                    "training_seconds": (result["training_seconds"]),
                }
            )

        best_rewards = [float(run["best_evaluation_reward"]) for run in runs]

        final_rewards = [float(run["final_evaluation_reward"]) for run in runs]

        final_success_rates = [float(run["final_success_rate"]) for run in runs]

        steps_to_target = [
            float(run["environment_steps_to_target"])
            for run in runs
            if run["environment_steps_to_target"] is not None
        ]

        episodes_to_target = [
            float(run["episodes_to_target"])
            for run in runs
            if run["episodes_to_target"] is not None
        ]

        training_seconds = [float(run["training_seconds"]) for run in runs]

        summary["domains"][domain] = {
            "runs": runs,
            "best_evaluation_reward": (mean_and_sample_sd(best_rewards)),
            "final_evaluation_reward": (mean_and_sample_sd(final_rewards)),
            "final_success_rate": (mean_and_sample_sd(final_success_rates)),
            "environment_steps_to_target": (
                mean_and_sample_sd(steps_to_target) if steps_to_target else None
            ),
            "episodes_to_target": (
                mean_and_sample_sd(episodes_to_target) if episodes_to_target else None
            ),
            "training_seconds": (mean_and_sample_sd(training_seconds)),
            "target_reach_count": len(steps_to_target),
            "parameter_count": {
                "actor": runs[0]["actor_parameters"],
                "critic": runs[0]["critic_parameters"],
                "total": runs[0]["total_parameters"],
            },
        }

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print(" SPRINT 4.5 THREE-SEED PPO SUMMARY")
    print("========================================")

    for domain in DOMAINS:
        data = summary["domains"][domain]

        print()
        print(domain)

        print(
            " best reward mean ± SD =",
            data["best_evaluation_reward"]["mean"],
            "±",
            data["best_evaluation_reward"]["sample_standard_deviation"],
        )

        print(
            " final reward mean ± SD =",
            data["final_evaluation_reward"]["mean"],
            "±",
            data["final_evaluation_reward"]["sample_standard_deviation"],
        )

        print(
            " final success mean ± SD =",
            data["final_success_rate"]["mean"],
            "±",
            data["final_success_rate"]["sample_standard_deviation"],
        )

        print(
            " steps to target mean ± SD =",
            data["environment_steps_to_target"]["mean"],
            "±",
            data["environment_steps_to_target"]["sample_standard_deviation"],
        )

        print(
            " target reach count =",
            data["target_reach_count"],
        )

    print()

    print(
        "Artifact:",
        OUTPUT_PATH,
    )

    print("SPRINT 4.5.23 THREE-SEED PPO SUMMARY: PASS")


if __name__ == "__main__":
    main()
