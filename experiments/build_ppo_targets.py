"""Build frozen paired reward targets from PPO references."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.ppo_baseline import (
    derive_reward_target,
)

AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

OUTPUT = PPO_DIRECTORY / "sprint4-ppo-targets.json"

SEEDS = (
    42,
    123,
    456,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Build paired PPO targets before any QML experiment."""
    audit = load_json(AUDIT_PATH)

    random_references = {
        "autonomous_driving": (audit["summaries"]["driving_random"]["mean_reward"]),
        "robotics": (audit["summaries"]["robotics_random"]["mean_reward"]),
    }

    domains: dict[
        str,
        dict[str, Any],
    ] = {}

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        seed_records: list[dict[str, Any]] = []

        for seed in SEEDS:
            path = PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

            result = load_json(path)

            evaluations = result["evaluations"]

            best_reward = max(evaluation["mean_reward"] for evaluation in evaluations)

            target = derive_reward_target(
                random_reference_reward=(random_references[domain]),
                ppo_best_reward=best_reward,
                target_fraction=0.95,
            )

            first_reach = next(
                (
                    evaluation
                    for evaluation in evaluations
                    if evaluation["mean_reward"] >= target
                ),
                None,
            )

            if first_reach is None:
                steps_to_target = None
                episodes_to_target = None

            else:
                steps_to_target = first_reach["environment_steps"]

                episodes_to_target = first_reach["completed_training_episodes"]

            seed_records.append(
                {
                    "seed": seed,
                    "random_reference_reward": (random_references[domain]),
                    "ppo_best_evaluation_reward": (best_reward),
                    "target_fraction": 0.95,
                    "target_evaluation_reward": (target),
                    "ppo_environment_steps_to_target": (steps_to_target),
                    "ppo_episodes_to_target": (episodes_to_target),
                }
            )

        valid_steps = [
            int(record["ppo_environment_steps_to_target"])
            for record in seed_records
            if record["ppo_environment_steps_to_target"] is not None
        ]

        valid_episodes = [
            int(record["ppo_episodes_to_target"])
            for record in seed_records
            if record["ppo_episodes_to_target"] is not None
        ]

        domains[domain] = {
            "random_reference_reward": (random_references[domain]),
            "seeds": seed_records,
            "ppo_target_reach_count": len(valid_steps),
            "mean_ppo_steps_to_target": (
                statistics.mean(valid_steps) if valid_steps else None
            ),
            "sample_sd_ppo_steps_to_target": (
                statistics.stdev(valid_steps)
                if len(valid_steps) > 1
                else 0.0 if valid_steps else None
            ),
            "mean_ppo_episodes_to_target": (
                statistics.mean(valid_episodes) if valid_episodes else None
            ),
            "sample_sd_ppo_episodes_to_target": (
                statistics.stdev(valid_episodes)
                if len(valid_episodes) > 1
                else 0.0 if valid_episodes else None
            ),
            "targets_frozen_before_qml": True,
        }

    payload = {
        "sprint": "4.5",
        "reference_method": "classical_ppo",
        "target_rule": ("random + 0.95 * " "(ppo_best - random)"),
        "target_fraction": 0.95,
        "qml_results_seen": False,
        "domains": domains,
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print(" SPRINT 4.5 PPO TARGETS")
    print("========================================")

    for domain, data in domains.items():
        print()
        print(domain)

        for record in data["seeds"]:
            print(
                " seed",
                record["seed"],
                "| best=",
                round(
                    record["ppo_best_evaluation_reward"],
                    6,
                ),
                "| target=",
                round(
                    record["target_evaluation_reward"],
                    6,
                ),
                "| steps=",
                record["ppo_environment_steps_to_target"],
                "| episodes=",
                record["ppo_episodes_to_target"],
            )

        print(
            " mean steps to target=",
            data["mean_ppo_steps_to_target"],
        )

        print(
            " sample SD steps=",
            data["sample_sd_ppo_steps_to_target"],
        )

    print()

    print("Targets frozen before QML: PASS")

    print(
        "QML results seen:",
        payload["qml_results_seen"],
    )

    print(
        "Artifact:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
