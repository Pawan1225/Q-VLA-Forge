"""Reproduce Sprint 4.8 autonomous-driving hybrid QML PPO seed 42."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.driving_qml import (
    first_qml_target_reach,
    hybrid_training_result_to_dict,
    sample_efficiency_improvement_percent,
)
from q_vla_forge.evaluation.ppo_baseline import (
    PPO_EVALUATION_SEEDS,
)
from q_vla_forge.rl.driving_env import (
    DrivingRLEnv,
)
from q_vla_forge.rl.hybrid_ppo import (
    train_hybrid_ppo,
)
from q_vla_forge.rl.ppo import (
    DEFAULT_PPO_CONFIG,
    PPOEvaluation,
)

SEEDS = (42,)

PPO_TARGET_PATH = Path("results") / "rl" / "ppo" / "sprint4-ppo-targets.json"

OUTPUT_DIRECTORY = Path("results") / "rl" / "qml"


def load_targets() -> dict:
    """Load the frozen Sprint 4.5 target artifact."""
    return json.loads(PPO_TARGET_PATH.read_text(encoding="utf-8"))


def paired_target_record(
    *,
    targets: dict,
    seed: int,
) -> dict:
    """Return the frozen paired driving target for one seed."""
    records = targets["domains"]["autonomous_driving"]["seeds"]

    matches = [record for record in records if record["seed"] == seed]

    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen target for seed {seed}")

    return matches[0]


def print_progress(
    evaluation: PPOEvaluation,
) -> None:
    """Print held-out evaluation progress without changing training."""
    print(
        f"[step {evaluation.environment_steps:>5}/20000] "
        f"episodes={evaluation.completed_training_episodes:<4} "
        f"reward={evaluation.mean_reward:>10.6f} "
        f"sd={evaluation.reward_standard_deviation:>9.6f} "
        f"success={evaluation.success_rate:.3f}",
        flush=True,
    )


def main() -> None:
    """Reproduce the frozen autonomous-driving QML seed-42 run."""
    targets = load_targets()

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("frozen target artifact indicates " "QML contamination")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in SEEDS:
        target_record = paired_target_record(
            targets=targets,
            seed=seed,
        )

        target_reward = float(target_record["target_evaluation_reward"])

        classical_steps = target_record["ppo_environment_steps_to_target"]

        if classical_steps is None:
            raise RuntimeError("paired classical PPO target was not reached")

        print()
        print("=====================================================")
        print(
            " HYBRID QML DRIVING REPRODUCIBILITY — SEED",
            seed,
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
            domain="autonomous_driving",
            seed=seed,
            environment_factory=DrivingRLEnv,
            evaluation_seeds=PPO_EVALUATION_SEEDS,
            config=DEFAULT_PPO_CONFIG,
            progress_callback=print_progress,
        )

        qml_steps, qml_episodes = first_qml_target_reach(
            result=result,
            target_reward=target_reward,
        )

        improvement = sample_efficiency_improvement_percent(
            classical_steps=int(classical_steps),
            qml_steps=qml_steps,
        )

        payload = hybrid_training_result_to_dict(result)

        payload["comparison"] = {
            "target_source": ("frozen_sprint4_ppo_target"),
            "target_evaluation_reward": (target_reward),
            "classical_ppo_environment_steps_to_target": (classical_steps),
            "qml_environment_steps_to_target": (qml_steps),
            "qml_episodes_to_target": (qml_episodes),
            "sample_efficiency_improvement_percent": (improvement),
            "positive_means": ("QML required fewer environment steps"),
        }

        payload["reproducibility_run"] = True

        payload["reproduces_seed"] = seed

        path = OUTPUT_DIRECTORY / (f"autonomous_driving-seed-" f"{seed}-repro.json")

        path.write_text(
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
            seed,
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
            "Final success:",
            final.success_rate,
        )

        print(
            "QML steps to target:",
            qml_steps,
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
            path,
        )


if __name__ == "__main__":
    main()
