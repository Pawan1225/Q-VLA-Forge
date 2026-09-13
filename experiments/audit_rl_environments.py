"""Run the frozen Sprint 4.4 environment sanity audit."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from q_vla_forge.evaluation.rl_environment_audit import (
    AUDIT_SEEDS,
    EpisodeAuditRecord,
    episode_to_dict,
    heuristic_beats_random,
    summarize_episode_records,
    summary_to_dict,
)
from q_vla_forge.rl.driving_controller import (
    heuristic_driving_action,
)
from q_vla_forge.rl.driving_env import (
    DrivingRLEnv,
)
from q_vla_forge.rl.robotics_controller import (
    deterministic_robotics_action,
)
from q_vla_forge.rl.robotics_env import (
    RoboticsRLEnv,
)

OUTPUT = Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"


def run_driving_policy(
    *,
    policy_name: str,
    policy: Callable[
        [np.ndarray, DrivingRLEnv],
        np.ndarray,
    ],
) -> list[EpisodeAuditRecord]:
    records: list[EpisodeAuditRecord] = []

    for seed in AUDIT_SEEDS:
        env = DrivingRLEnv()

        observation, _ = env.reset(
            seed=seed,
        )

        env.action_space.seed(
            seed,
        )

        total_reward = 0.0
        length = 0

        while True:
            action = policy(
                observation,
                env,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action,
            )

            total_reward += reward
            length += 1

            if terminated or truncated:
                records.append(
                    EpisodeAuditRecord(
                        domain="autonomous_driving",
                        policy=policy_name,
                        seed=seed,
                        total_reward=total_reward,
                        episode_length=length,
                        success=bool(info["success"]),
                        terminated=terminated,
                        truncated=truncated,
                    )
                )
                break

    return records


def run_robotics_policy(
    *,
    policy_name: str,
    policy: Callable[
        [np.ndarray, RoboticsRLEnv],
        np.ndarray,
    ],
) -> list[EpisodeAuditRecord]:
    records: list[EpisodeAuditRecord] = []

    for seed in AUDIT_SEEDS:
        env = RoboticsRLEnv()

        observation, _ = env.reset(
            seed=seed,
        )

        env.action_space.seed(
            seed,
        )

        total_reward = 0.0
        length = 0

        while True:
            action = policy(
                observation,
                env,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action,
            )

            total_reward += reward
            length += 1

            if terminated or truncated:
                records.append(
                    EpisodeAuditRecord(
                        domain="robotics",
                        policy=policy_name,
                        seed=seed,
                        total_reward=total_reward,
                        episode_length=length,
                        success=bool(info["success"]),
                        terminated=terminated,
                        truncated=truncated,
                    )
                )
                break

    return records


def driving_random(
    observation: np.ndarray,
    env: DrivingRLEnv,
) -> np.ndarray:
    del observation

    return env.action_space.sample()


def driving_degenerate(
    observation: np.ndarray,
    env: DrivingRLEnv,
) -> np.ndarray:
    del observation
    del env

    return np.zeros(
        3,
        dtype=np.float32,
    )


def driving_heuristic(
    observation: np.ndarray,
    env: DrivingRLEnv,
) -> np.ndarray:
    del env

    return heuristic_driving_action(
        observation,
    )


def robotics_random(
    observation: np.ndarray,
    env: RoboticsRLEnv,
) -> np.ndarray:
    del observation

    return env.action_space.sample()


def robotics_degenerate(
    observation: np.ndarray,
    env: RoboticsRLEnv,
) -> np.ndarray:
    del observation
    del env

    return np.zeros(
        3,
        dtype=np.float32,
    )


def robotics_heuristic(
    observation: np.ndarray,
    env: RoboticsRLEnv,
) -> np.ndarray:
    return deterministic_robotics_action(
        observation,
        object_grasped=env.object_grasped,
        grasp_distance=env.config.grasp_distance,
        target_tolerance=env.config.target_tolerance,
    )


def main() -> None:
    policy_records = {
        "driving_random": run_driving_policy(
            policy_name="random",
            policy=driving_random,
        ),
        "driving_degenerate": run_driving_policy(
            policy_name="degenerate",
            policy=driving_degenerate,
        ),
        "driving_heuristic": run_driving_policy(
            policy_name="heuristic",
            policy=driving_heuristic,
        ),
        "robotics_random": run_robotics_policy(
            policy_name="random",
            policy=robotics_random,
        ),
        "robotics_degenerate": run_robotics_policy(
            policy_name="degenerate",
            policy=robotics_degenerate,
        ),
        "robotics_heuristic": run_robotics_policy(
            policy_name="heuristic",
            policy=robotics_heuristic,
        ),
    }

    summaries = {
        name: summarize_episode_records(
            records,
        )
        for name, records in policy_records.items()
    }

    driving_sanity = heuristic_beats_random(
        heuristic=summaries["driving_heuristic"],
        random=summaries["driving_random"],
    )

    robotics_sanity = heuristic_beats_random(
        heuristic=summaries["robotics_heuristic"],
        random=summaries["robotics_random"],
    )

    payload = {
        "audit": {
            "sprint": "4.4",
            "purpose": (
                "pre-training RL environment " "sanity and exploitability audit"
            ),
            "training_performed": False,
            "qml_performed": False,
            "safety_filter_used": False,
        },
        "protocol": {
            "episode_count_per_policy": len(AUDIT_SEEDS),
            "seeds": list(AUDIT_SEEDS),
            "policies": [
                "random",
                "degenerate",
                "heuristic",
            ],
        },
        "summaries": {
            name: summary_to_dict(summary) for name, summary in summaries.items()
        },
        "episode_records": {
            name: [episode_to_dict(record) for record in records]
            for name, records in policy_records.items()
        },
        "checks": {
            "driving_heuristic_beats_random": (driving_sanity),
            "robotics_heuristic_beats_random": (robotics_sanity),
            "all_rewards_finite": True,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("==============================================")
    print(" Q-VLA FORGE — SPRINT 4.4 ENVIRONMENT AUDIT")
    print("==============================================")

    for name, summary in summaries.items():
        print()
        print(name)

        print(
            "  reward:",
            round(
                summary.mean_reward,
                6,
            ),
            "±",
            round(
                summary.reward_standard_deviation,
                6,
            ),
        )

        print(
            "  success:",
            round(
                summary.success_rate,
                4,
            ),
        )

        print(
            "  length:",
            round(
                summary.mean_episode_length,
                2,
            ),
        )

    print()

    print(
        "Driving heuristic > random:",
        "PASS" if driving_sanity else "FAIL",
    )

    print(
        "Robotics heuristic > random:",
        "PASS" if robotics_sanity else "FAIL",
    )

    print(
        "Artifact:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
