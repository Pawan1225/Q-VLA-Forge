"""Run Sprint 5.5 clean-condition heuristic clipping evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from q_vla_forge.evaluation.clipping_safety import (
    clipping_episode_to_dict,
    clipping_summary_to_dict,
    run_clipping_episode,
    summarize_clipping_episodes,
)
from q_vla_forge.evaluation.safety_baseline import (
    PRINCIPAL_SEEDS,
    SAFETY_EVALUATION_SEEDS,
    load_frozen_ppo_policy,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "results" / "safety" / "clipping" / "runs"


def run_cell(
    *,
    domain: str,
    principal_seed: int,
) -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=domain,
        principal_seed=principal_seed,
    )

    episodes = []

    print()
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.5 HEURISTIC CLIPPING")
    print("=" * 68)
    print()
    print(
        "Domain:",
        domain,
    )
    print(
        "Principal seed:",
        principal_seed,
    )
    print("Method: CLIPPING")
    print("Condition: CLEAN")
    print(
        "Checkpoint:",
        policy.checkpoint_path.relative_to(ROOT),
    )
    print(
        "Checkpoint SHA256:",
        policy.checkpoint_sha256,
    )
    print()

    for index, evaluation_seed in enumerate(
        SAFETY_EVALUATION_SEEDS,
        start=1,
    ):
        episode = run_clipping_episode(
            policy=policy,
            evaluation_seed=evaluation_seed,
        )

        episodes.append(episode)

        print(
            f"[{index:02d}/20] "
            f"eval_seed={evaluation_seed} "
            f"reward={episode.reward:.6f} "
            f"success={int(episode.success)} "
            f"steps={episode.episode_length} "
            f"proposed_viol_steps="
            f"{episode.proposed_violation_step_count} "
            f"executed_viol_steps="
            f"{episode.executed_violation_step_count} "
            f"interventions="
            f"{episode.intervention_count} "
            f"critical="
            f"{episode.critical_violation_step_count}"
        )

    summary = summarize_clipping_episodes(episodes)

    if summary.policy_action_out_of_bounds_count != 0:
        raise RuntimeError("policy action out-of-bounds count is non-zero")

    if summary.executed_action_out_of_bounds_count != 0:
        raise RuntimeError("executed action out-of-bounds count is non-zero")

    payload = {
        "sprint": "5.5",
        "artifact": ("heuristic-clipping-safety-principal-run"),
        "domain": domain,
        "principal_seed": principal_seed,
        "policy_family": "classical_ppo",
        "policy_checkpoint": (policy.checkpoint_path.relative_to(ROOT).as_posix()),
        "policy_checkpoint_sha256": (policy.checkpoint_sha256),
        "checkpoint_recovery_record": (
            policy.recovery_record_path.relative_to(ROOT).as_posix()
        ),
        "checkpoint_recovery_record_sha256": (policy.recovery_record_sha256),
        "training_budget": (policy.training_budget),
        "checkpoint_selection": (policy.checkpoint_selection),
        "safety_method": "clipping",
        "robustness_condition": "clean",
        "new_training_performed": False,
        "lyapunov_used": False,
        "perturbation_used": False,
        "evaluation_seeds": list(SAFETY_EVALUATION_SEEDS),
        "episode_count": len(episodes),
        "episodes": [clipping_episode_to_dict(episode) for episode in episodes],
        "summary": (clipping_summary_to_dict(summary)),
        "claim_boundary": {
            "clipping_measured": True,
            "clipping_effectiveness_decided": False,
            "lyapunov_improvement_tested": False,
            "robustness_improvement_tested": False,
            "formal_safety_guarantee_claimed": False,
            "production_safety_claimed": False,
        },
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIRECTORY / f"{domain}-seed-{principal_seed}.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Summary")
    print(
        "Mean reward:",
        f"{summary.mean_reward:.6f}",
    )
    print(
        "Success rate:",
        f"{summary.success_rate:.6f}",
    )
    print(
        "Proposed violation-step rate:",
        f"{summary.proposed_violation_step_rate:.6f}",
    )
    print(
        "Executed violation-step rate:",
        f"{summary.executed_violation_step_rate:.6f}",
    )
    print(
        "Executed constraint violation rate:",
        f"{summary.executed_constraint_violation_rate:.6f}",
    )
    print(
        "Critical violation-step rate:",
        f"{summary.critical_violation_step_rate:.6f}",
    )
    print(
        "Intervention rate:",
        f"{summary.intervention_rate:.6f}",
    )
    print(
        "Mean correction L2:",
        f"{summary.mean_action_correction_l2:.6f}",
    )
    print(
        "P95 correction L2:",
        f"{summary.p95_action_correction_l2:.6f}",
    )
    print(
        "Max correction L2:",
        f"{summary.max_action_correction_l2:.6f}",
    )
    print(
        "Policy OOB actions:",
        summary.policy_action_out_of_bounds_count,
    )
    print(
        "Executed OOB actions:",
        summary.executed_action_out_of_bounds_count,
    )
    print(
        "Artifact:",
        output_path.relative_to(ROOT),
    )
    print(
        "Artifact SHA256:",
        sha256_file(output_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        required=True,
        choices=(
            "autonomous_driving",
            "robotics",
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=PRINCIPAL_SEEDS,
    )

    args = parser.parse_args()

    run_cell(
        domain=args.domain,
        principal_seed=args.seed,
    )


if __name__ == "__main__":
    main()
