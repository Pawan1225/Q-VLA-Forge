"""Run Sprint 4.12 matched-classical PPO ablation experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.rl.driving_env import DrivingRLEnv
from q_vla_forge.rl.matched_classical_ppo import (
    MatchedClassicalPPOTrainingResult,
    train_matched_classical_ppo,
)
from q_vla_forge.rl.ppo import DEFAULT_PPO_CONFIG
from q_vla_forge.rl.robotics_env import RoboticsRLEnv

RESULTS_DIRECTORY = Path("results/rl/ablation")

EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)

VALID_DOMAINS = (
    "autonomous_driving",
    "robotics",
)

VALID_SEEDS = (
    42,
    123,
    456,
)


def environment_factory_for_domain(
    domain: str,
) -> Any:
    """Return the frozen environment factory for one domain."""

    if domain == "autonomous_driving":
        return DrivingRLEnv

    if domain == "robotics":
        return RoboticsRLEnv

    raise ValueError(f"unsupported domain: {domain}")


def result_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return the principal artifact path."""

    return RESULTS_DIRECTORY / (f"matched-classical-{domain}-seed-{seed}.json")


def serialize_training_result(
    result: MatchedClassicalPPOTrainingResult,
) -> dict[str, Any]:
    """Convert one training result to proposal evidence."""

    evaluations = [asdict(evaluation) for evaluation in result.evaluations]

    best_evaluation = max(
        result.evaluations,
        key=lambda evaluation: evaluation.mean_reward,
    )

    final_evaluation = result.evaluations[-1]

    return {
        "sprint": "4.12",
        "experiment": ("matched_classical_parameter_ablation"),
        "policy": "matched_classical",
        "domain": result.domain,
        "seed": result.seed,
        "protocol": {
            "total_environment_steps": (DEFAULT_PPO_CONFIG.total_environment_steps),
            "evaluation_frequency_steps": (
                DEFAULT_PPO_CONFIG.evaluation_frequency_steps
            ),
            "evaluation_episode_count": len(EVALUATION_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "gamma": DEFAULT_PPO_CONFIG.gamma,
            "gae_lambda": (DEFAULT_PPO_CONFIG.gae_lambda),
            "clip_coefficient": (DEFAULT_PPO_CONFIG.clip_coefficient),
            "learning_rate": (DEFAULT_PPO_CONFIG.learning_rate),
            "rollout_steps": (DEFAULT_PPO_CONFIG.rollout_steps),
            "update_epochs": (DEFAULT_PPO_CONFIG.update_epochs),
            "minibatch_size": (DEFAULT_PPO_CONFIG.minibatch_size),
            "value_coefficient": (DEFAULT_PPO_CONFIG.value_coefficient),
            "entropy_coefficient": (DEFAULT_PPO_CONFIG.entropy_coefficient),
            "max_gradient_norm": (DEFAULT_PPO_CONFIG.max_gradient_norm),
        },
        "parameter_counts": {
            "projection_parameters": (result.projection_parameters),
            "core_parameters": (result.core_parameters),
            "action_head_parameters": (result.action_head_parameters),
            "actor_parameters": (result.actor_parameters),
            "critic_parameters": (result.critic_parameters),
            "total_parameters": (result.total_parameters),
        },
        "evaluations": evaluations,
        "summary": {
            "evaluation_count": len(result.evaluations),
            "total_environment_steps": (result.total_environment_steps),
            "completed_training_episodes": (result.completed_training_episodes),
            "best_mean_reward": (best_evaluation.mean_reward),
            "best_reward_standard_deviation": (
                best_evaluation.reward_standard_deviation
            ),
            "best_success_rate": (best_evaluation.success_rate),
            "best_environment_steps": (best_evaluation.environment_steps),
            "final_mean_reward": (final_evaluation.mean_reward),
            "final_reward_standard_deviation": (
                final_evaluation.reward_standard_deviation
            ),
            "final_success_rate": (final_evaluation.success_rate),
        },
        "scientific_boundary": {
            "new_training_performed": True,
            "new_qml_training_performed": False,
            "matched_classical_training_performed": True,
            "qml_architecture_modified": False,
            "ppo_protocol_modified": False,
            "targets_modified": False,
            "environments_modified": False,
            "parameter_matched_ablation": True,
            "statistical_significance_claimed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_advantage_claimed": False,
        },
        "training_seconds": (result.training_seconds),
    }


def run_experiment(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Run one principal matched-classical ablation experiment."""

    if domain not in VALID_DOMAINS:
        raise ValueError(f"domain must be one of {VALID_DOMAINS}")

    if seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {VALID_SEEDS}")

    environment_factory = environment_factory_for_domain(domain)

    print()
    print("=" * 58)
    print(f"SPRINT 4.12 MATCHED CLASSICAL — {domain} — seed {seed}")
    print("=" * 58)

    def progress_callback(
        evaluation: Any,
    ) -> None:
        print(
            "steps=",
            evaluation.environment_steps,
            "reward=",
            evaluation.mean_reward,
            "sd=",
            evaluation.reward_standard_deviation,
            "success=",
            evaluation.success_rate,
        )

    result = train_matched_classical_ppo(
        domain=domain,
        seed=seed,
        environment_factory=environment_factory,
        evaluation_seeds=EVALUATION_SEEDS,
        progress_callback=progress_callback,
    )

    payload = serialize_training_result(result)

    output_path = result_path(
        domain=domain,
        seed=seed,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Actor parameters:",
        result.actor_parameters,
    )
    print(
        "Critic parameters:",
        result.critic_parameters,
    )
    print(
        "Total parameters:",
        result.total_parameters,
    )
    print(
        "Evaluations:",
        len(result.evaluations),
    )
    print(
        "Environment steps:",
        result.total_environment_steps,
    )
    print(
        "Best reward:",
        payload["summary"]["best_mean_reward"],
    )
    print(
        "Final reward:",
        payload["summary"]["final_mean_reward"],
    )
    print(
        "Final success:",
        payload["summary"]["final_success_rate"],
    )
    print(
        "Artifact:",
        output_path,
    )

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse experiment CLI arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=VALID_SEEDS,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()

    run_experiment(
        domain=args.domain,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
