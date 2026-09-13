"""Reproduce Sprint 4.9 robotics hybrid-QML PPO seed 42."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.ppo_baseline import PPO_EVALUATION_SEEDS
from q_vla_forge.evaluation.robotics_qml import (
    first_qml_target_reach,
    sample_efficiency_improvement_percent,
)
from q_vla_forge.rl.hybrid_ppo import (
    HybridPPOTrainingResult,
    train_hybrid_ppo,
)
from q_vla_forge.rl.ppo import DEFAULT_PPO_CONFIG, PPOEvaluation
from q_vla_forge.rl.robotics_env import RoboticsRLEnv

SEED = 42

TARGET_PATH = Path("results") / "rl" / "ppo" / "sprint4-ppo-targets.json"

OUTPUT_DIRECTORY = Path("results") / "rl" / "qml"


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def result_to_dict(
    result: HybridPPOTrainingResult,
) -> dict[str, Any]:
    """Serialize one hybrid PPO training result."""
    return {
        "domain": result.domain,
        "seed": result.seed,
        "total_environment_steps": (result.total_environment_steps),
        "completed_training_episodes": (result.completed_training_episodes),
        "projection_parameters": (result.projection_parameters),
        "quantum_parameters": (result.quantum_parameters),
        "action_head_parameters": (result.action_head_parameters),
        "actor_parameters": (result.actor_parameters),
        "critic_parameters": (result.critic_parameters),
        "total_parameters": (result.total_parameters),
        "training_seconds": (result.training_seconds),
        "evaluations": [
            {
                "environment_steps": (evaluation.environment_steps),
                "completed_training_episodes": (evaluation.completed_training_episodes),
                "mean_reward": (evaluation.mean_reward),
                "reward_standard_deviation": (evaluation.reward_standard_deviation),
                "success_rate": (evaluation.success_rate),
            }
            for evaluation in result.evaluations
        ],
    }


def paired_target(
    *,
    targets: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Return the frozen paired robotics PPO target for one seed."""
    records = targets["domains"]["robotics"]["seeds"]

    matches = [record for record in records if record["seed"] == seed]

    if len(matches) != 1:
        raise RuntimeError(f"expected one robotics target for seed {seed}")

    return matches[0]


def print_progress(
    evaluation: PPOEvaluation,
) -> None:
    """Print held-out robotics evaluation progress."""
    print(
        f"[step {evaluation.environment_steps:>5}/20000] "
        f"episodes={evaluation.completed_training_episodes:<4} "
        f"reward={evaluation.mean_reward:>10.6f} "
        f"sd={evaluation.reward_standard_deviation:>9.6f} "
        f"success={evaluation.success_rate:.3f}",
        flush=True,
    )


def main() -> None:
    """Reproduce the frozen robotics seed-42 QML run."""
    targets = load_json(TARGET_PATH)

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("historical PPO target artifact is invalid")

    robotics_targets = targets["domains"]["robotics"]

    if robotics_targets["targets_frozen_before_qml"] is not True:
        raise RuntimeError("robotics PPO targets were not frozen before QML")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = paired_target(
        targets=targets,
        seed=SEED,
    )

    target_reward = float(reference["target_evaluation_reward"])

    classical_steps = reference["ppo_environment_steps_to_target"]

    if classical_steps is None:
        raise RuntimeError("classical PPO target was not reached")

    print()
    print("=====================================================")
    print(
        " ROBOTICS HYBRID QML REPRODUCIBILITY — SEED",
        SEED,
    )
    print("=====================================================")

    print(
        "Frozen target:",
        target_reward,
    )

    print(
        "Classical PPO steps to target:",
        classical_steps,
    )

    result = train_hybrid_ppo(
        domain="robotics",
        seed=SEED,
        environment_factory=RoboticsRLEnv,
        evaluation_seeds=PPO_EVALUATION_SEEDS,
        config=DEFAULT_PPO_CONFIG,
        progress_callback=print_progress,
    )

    (
        qml_steps,
        qml_episodes,
    ) = first_qml_target_reach(
        result=result,
        target_reward=target_reward,
    )

    improvement = sample_efficiency_improvement_percent(
        classical_steps=int(classical_steps),
        qml_steps=qml_steps,
    )

    payload = result_to_dict(result)

    payload["comparison"] = {
        "target_source": ("frozen_sprint4_ppo_target"),
        "target_evaluation_reward": (target_reward),
        "classical_ppo_environment_steps_to_target": (classical_steps),
        "qml_environment_steps_to_target": (qml_steps),
        "qml_episodes_to_target": (qml_episodes),
        "sample_efficiency_improvement_percent": (improvement),
        "positive_means": ("QML required fewer environment steps"),
    }

    payload["scientific_boundary"] = {
        "quantum_hardware_used": False,
        "simulator": ("PennyLane default.qubit"),
        "shots": None,
        "quantum_speedup_claimed": False,
        "wall_clock_speedup_claimed": False,
    }

    payload["reproducibility_run"] = True

    payload["reproduces_seed"] = SEED

    output_path = OUTPUT_DIRECTORY / f"robotics-seed-{SEED}-repro.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_reward = max(evaluation.mean_reward for evaluation in result.evaluations)

    final = result.evaluations[-1]

    print()
    print(
        "===== SEED",
        SEED,
        "REPRO RESULT =====",
    )

    print(
        "Best QML reward:",
        best_reward,
    )

    print(
        "Final QML reward:",
        final.mean_reward,
    )

    print(
        "Final QML success:",
        final.success_rate,
    )

    print(
        "QML steps to target:",
        qml_steps,
    )

    print(
        "QML episodes to target:",
        qml_episodes,
    )

    print(
        "Sample-efficiency improvement:",
        improvement,
    )

    print(
        "Training seconds:",
        result.training_seconds,
    )

    print(
        "Artifact:",
        output_path,
    )


if __name__ == "__main__":
    main()
