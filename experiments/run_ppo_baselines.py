"""Run frozen Sprint 4.5 classical PPO baselines."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.ppo_baseline import (
    PPO_EVALUATION_SEEDS,
    training_result_to_dict,
)
from q_vla_forge.rl.driving_env import DrivingRLEnv
from q_vla_forge.rl.ppo import DEFAULT_PPO_CONFIG, train_ppo
from q_vla_forge.rl.robotics_env import RoboticsRLEnv

SEEDS = (
    123,
    456,
)

OUTPUT_DIRECTORY = Path("results") / "rl" / "ppo"


def run_domain(
    *,
    domain: str,
) -> None:
    if domain == "autonomous_driving":
        factory = DrivingRLEnv

    elif domain == "robotics":
        factory = RoboticsRLEnv

    else:
        raise ValueError(f"unsupported domain: {domain}")

    for seed in SEEDS:
        print()
        print("========================================")

        print(
            "PPO",
            domain,
            "seed",
            seed,
        )

        print("========================================")

        result = train_ppo(
            domain=domain,
            seed=seed,
            environment_factory=factory,
            evaluation_seeds=PPO_EVALUATION_SEEDS,
            config=DEFAULT_PPO_CONFIG,
        )

        payload = training_result_to_dict(result)

        path = OUTPUT_DIRECTORY / f"{domain}-seed-{seed}.json"

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        final = result.evaluations[-1]

        best_reward = max(evaluation.mean_reward for evaluation in result.evaluations)

        print(
            "Best evaluation reward:",
            best_reward,
        )

        print(
            "Final evaluation reward:",
            final.mean_reward,
        )

        print(
            "Final success rate:",
            final.success_rate,
        )

        print(
            "Training seconds:",
            result.training_seconds,
        )

        print(
            "Artifact:",
            path,
        )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_domain(domain="autonomous_driving")

    run_domain(domain="robotics")


if __name__ == "__main__":
    main()
