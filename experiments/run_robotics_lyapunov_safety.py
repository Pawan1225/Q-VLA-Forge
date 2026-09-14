"""Run Sprint 5.9 robotics Lyapunov safety evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.robotics_lyapunov_safety import (
    ROBOTICS_DOMAIN,
    robotics_lyapunov_audit_to_dict,
    robotics_lyapunov_episode_to_dict,
    robotics_lyapunov_summary_to_dict,
    run_robotics_lyapunov_episode,
    summarize_robotics_lyapunov_episodes,
)
from q_vla_forge.evaluation.safety_baseline import (
    PRINCIPAL_SEEDS,
    SAFETY_EVALUATION_SEEDS,
    load_frozen_ppo_policy,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "results" / "safety" / "lyapunov-robotics"

RUNS_DIR = OUTPUT_ROOT / "runs"

ROBOTICS_CONTRACT_PATH = (
    ROOT / "src" / "q_vla_forge" / "safety" / "robotics_constraints.py"
)

LYAPUNOV_FOUNDATION_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-foundation"
    / "sprint5-lyapunov-foundation.json"
)

LYAPUNOV_FILTER_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-filter"
    / "sprint5-lyapunov-filter-mechanism.json"
)

LYAPUNOV_FILTER_SOURCE_PATH = (
    ROOT / "src" / "q_vla_forge" / "safety" / "lyapunov_filter.py"
)

PREDICTOR_SOURCE_PATH = ROOT / "src" / "q_vla_forge" / "safety" / "predictors.py"

ROBOTICS_ENV_SOURCE_PATH = ROOT / "src" / "q_vla_forge" / "rl" / "robotics_env.py"


def _relative(
    path: Path,
) -> str:
    return str(path.relative_to(ROOT)).replace(
        "\\",
        "/",
    )


def _run_seed(
    *,
    principal_seed: int,
    evaluation_seeds: tuple[int, ...],
    output_path: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=ROBOTICS_DOMAIN,
        principal_seed=principal_seed,
    )

    episodes = []

    audit_snapshots: list[dict[str, object]] = []

    grasped_audit_snapshots: list[dict[str, object]] = []

    for evaluation_seed in evaluation_seeds:
        run = run_robotics_lyapunov_episode(
            policy=policy,
            evaluation_seed=evaluation_seed,
        )

        episodes.append(run.episode)

        audit_snapshots.extend(
            robotics_lyapunov_audit_to_dict(snapshot)
            for snapshot in run.intervened_audits
        )

        audit_snapshots.extend(
            robotics_lyapunov_audit_to_dict(snapshot)
            for snapshot in run.nonintervened_audits
        )

        grasped_audit_snapshots.extend(
            robotics_lyapunov_audit_to_dict(snapshot) for snapshot in run.grasped_audits
        )

        print(
            f"seed={principal_seed} "
            f"eval={evaluation_seed} "
            f"reward={run.episode.reward:.6f} "
            f"violations="
            f"{run.episode.executed_violation_step_count} "
            f"interventions="
            f"{run.episode.intervention_count} "
            f"grasped_steps="
            f"{run.episode.steps_object_grasped}"
        )

    summary = summarize_robotics_lyapunov_episodes(episodes)

    payload: dict[str, Any] = {
        "sprint": "5.9",
        "artifact": (
            "robotics-lyapunov-safety-smoke"
            if smoke
            else "robotics-lyapunov-safety-principal-run"
        ),
        "domain": ROBOTICS_DOMAIN,
        "principal_seed": principal_seed,
        "policy_family": "classical_ppo",
        "policy_checkpoint": (_relative(policy.checkpoint_path)),
        "policy_checkpoint_sha256": (policy.checkpoint_sha256),
        "checkpoint_recovery_record": (_relative(policy.recovery_record_path)),
        "checkpoint_recovery_record_sha256": (policy.recovery_record_sha256),
        "training_budget": (policy.training_budget),
        "checkpoint_selection": (policy.checkpoint_selection),
        "safety_method": "lyapunov",
        "robustness_condition": "clean",
        "new_training_performed": False,
        "policy_fine_tuning_performed": False,
        "filter_tuning_performed": False,
        "perturbation_used": False,
        "evaluation_seeds": list(evaluation_seeds),
        "episode_count": len(episodes),
        "prediction_context": {
            "source": ("environment_internal_state"),
            "field": "object_grasped",
            "environment_api": ("env.object_grasped"),
            "policy_observation_modified": False,
            "policy_observation_dimension": 6,
            "context_class": ("RoboticsPredictionContext"),
            "context_used_for": ("one_step_safety_prediction"),
        },
        "protocol_controls": {
            "domain": "robotics",
            "condition": "clean",
            "observed_state_equals_true_state": True,
            "new_training": False,
            "policy_fine_tuning": False,
            "qml_evaluation": False,
            "lyapunov_function_modified": False,
            "candidate_set_modified": False,
            "selector_modified": False,
            "safety_thresholds_modified": False,
            "robustness_perturbation": False,
            "parameter_tuning": False,
            "policy_hidden_state_added": False,
        },
        "provenance": {
            "robotics_safety_contract": {
                "path": _relative(ROBOTICS_CONTRACT_PATH),
                "sha256": sha256_file(ROBOTICS_CONTRACT_PATH),
            },
            "lyapunov_foundation": {
                "path": _relative(LYAPUNOV_FOUNDATION_PATH),
                "sha256": sha256_file(LYAPUNOV_FOUNDATION_PATH),
            },
            "lyapunov_filter_mechanism": {
                "path": _relative(LYAPUNOV_FILTER_PATH),
                "sha256": sha256_file(LYAPUNOV_FILTER_PATH),
            },
            "lyapunov_filter_source": {
                "path": _relative(LYAPUNOV_FILTER_SOURCE_PATH),
                "sha256": sha256_file(LYAPUNOV_FILTER_SOURCE_PATH),
            },
            "robotics_predictor_source": {
                "path": _relative(PREDICTOR_SOURCE_PATH),
                "sha256": sha256_file(PREDICTOR_SOURCE_PATH),
            },
            "robotics_environment_source": {
                "path": _relative(ROBOTICS_ENV_SOURCE_PATH),
                "sha256": sha256_file(ROBOTICS_ENV_SOURCE_PATH),
            },
        },
        "episodes": [
            robotics_lyapunov_episode_to_dict(episode) for episode in episodes
        ],
        "seed_summary": (robotics_lyapunov_summary_to_dict(summary)),
        "audit_snapshots": (audit_snapshots),
        "grasped_state_audit_snapshots": (grasped_audit_snapshots),
        "claim_controls": {
            "formal_stability_proven": False,
            "formal_robot_safety_guarantee": False,
            "production_manipulation_safety_claim": False,
            "iso_10218_claim": False,
            "iec_62061_claim": False,
            "lyapunov_superior_to_clipping_claim": False,
            "quantum_safety_advantage_claim": False,
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--principal-seed",
        type=int,
        choices=PRINCIPAL_SEEDS,
        default=None,
    )

    parser.add_argument(
        "--evaluation-limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.evaluation_limit is not None:
        if args.evaluation_limit <= 0:
            raise ValueError("evaluation limit must be positive")

        if args.evaluation_limit > len(SAFETY_EVALUATION_SEEDS):
            raise ValueError("evaluation limit exceeds " "frozen evaluation seed set")

    evaluation_seeds = (
        SAFETY_EVALUATION_SEEDS[: args.evaluation_limit]
        if (args.evaluation_limit is not None)
        else SAFETY_EVALUATION_SEEDS
    )

    principal_seeds = (
        (args.principal_seed,) if (args.principal_seed is not None) else PRINCIPAL_SEEDS
    )

    if args.smoke and (args.principal_seed is None or args.evaluation_limit != 1):
        raise ValueError(
            "smoke mode requires " "--principal-seed and " "--evaluation-limit 1"
        )

    if not args.smoke and (
        args.principal_seed is not None or args.evaluation_limit is not None
    ):
        raise ValueError("partial evaluation is allowed " "only in --smoke mode")

    for principal_seed in principal_seeds:
        if args.smoke:
            output_path = (
                OUTPUT_ROOT
                / "smoke"
                / ("robotics" f"-seed-{principal_seed}" "-smoke.json")
            )

        else:
            output_path = RUNS_DIR / ("robotics" f"-seed-{principal_seed}.json")

        _run_seed(
            principal_seed=principal_seed,
            evaluation_seeds=tuple(evaluation_seeds),
            output_path=output_path,
            smoke=bool(args.smoke),
        )

    print()

    print("SPRINT 5.9 ROBOTICS " "LYAPUNOV RUNNER: PASS")


if __name__ == "__main__":
    main()
