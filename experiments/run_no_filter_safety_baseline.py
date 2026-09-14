"""Run Sprint 5.4 clean-condition no-filter safety baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from q_vla_forge.evaluation.safety_baseline import (
    PRINCIPAL_SEEDS,
    SAFETY_EVALUATION_SEEDS,
    episode_evidence_to_dict,
    load_frozen_ppo_policy,
    run_no_filter_episode,
    seed_summary_to_dict,
    sha256_file,
    summarize_episode_evidence,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "results" / "safety" / "baseline" / "runs"


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
    print("=" * 64)
    print(" Q-VLA FORGE - SPRINT 5.4 NO-FILTER BASELINE")
    print("=" * 64)
    print()
    print("Domain:", domain)
    print("Principal seed:", principal_seed)
    print("Method: NONE")
    print("Condition: CLEAN")
    print("Checkpoint:", policy.checkpoint_path.relative_to(ROOT))
    print("Checkpoint SHA256:", policy.checkpoint_sha256)
    print()

    for index, evaluation_seed in enumerate(
        SAFETY_EVALUATION_SEEDS,
        start=1,
    ):
        episode = run_no_filter_episode(
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
            f"violation_steps={episode.violation_step_count} "
            f"constraints={episode.constraint_violation_count} "
            f"critical={episode.critical_violation_step_count}"
        )

    summary = summarize_episode_evidence(episodes)

    if summary.policy_action_out_of_bounds_count != 0:
        raise RuntimeError("policy action out-of-bounds count is non-zero")

    if summary.intervention_count != 0:
        raise RuntimeError("NONE baseline unexpectedly intervened")

    payload = {
        "sprint": "5.4",
        "artifact": ("no-filter-safety-baseline-principal-run"),
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
        "safety_method": "none",
        "robustness_condition": "clean",
        "new_training_performed": False,
        "safety_filter_applied": False,
        "evaluation_seeds": list(SAFETY_EVALUATION_SEEDS),
        "episode_count": len(episodes),
        "episodes": [episode_evidence_to_dict(episode) for episode in episodes],
        "summary": (seed_summary_to_dict(summary)),
        "baseline_has_observed_violations": (summary.violation_step_count > 0),
        "claim_boundary": {
            "baseline_measured": True,
            "clipping_improvement_tested": False,
            "lyapunov_improvement_tested": False,
            "robustness_improvement_tested": False,
            "production_safety_claimed": False,
            "formal_safety_guarantee_claimed": False,
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
        "Violation-step rate:",
        f"{summary.violation_step_rate:.6f}",
    )
    print(
        "Constraint violation rate:",
        f"{summary.constraint_violation_rate:.6f}",
    )
    print(
        "Critical violation-step rate:",
        f"{summary.critical_violation_step_rate:.6f}",
    )
    print(
        "Out-of-bounds actions:",
        summary.policy_action_out_of_bounds_count,
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
