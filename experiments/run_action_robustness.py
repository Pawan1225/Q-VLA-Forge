"""Run Sprint 5.12 structured action-perturbation robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from q_vla_forge.evaluation.action_robustness import (
    ActionRobustnessEpisodeEvidence,
    environment_effective_action,
    gripper_semantic,
    summarize_action_robustness_seed,
)
from q_vla_forge.evaluation.safety_baseline import (
    deterministic_policy_action,
    environment_for_domain,
    evaluate_domain_violations,
    load_frozen_ppo_policy,
)
from q_vla_forge.safety.clipping import (
    apply_clipping_safety_filter,
)
from q_vla_forge.safety.lyapunov_filter import (
    RoboticsPredictionContext,
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.perturbations import (
    StructuredActionPerturbation,
    action_perturbation_by_name,
    action_perturbations_for_domain,
    apply_structured_action_perturbation,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "results" / "safety" / "action-robustness"

RUNS_ROOT = OUTPUT_ROOT / "runs"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"

CONFIG_PATH = ROOT / "configs" / "action_robustness.yaml"

ENVIRONMENT_AUDIT_PATH = OUTPUT_ROOT / "environment-action-handling.json"

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)

AUDIT_LIMIT = 5


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _slug(
    value: str,
) -> str:
    return value.replace(
        "_",
        "-",
    )


def _clean_reference_path(
    *,
    method: str,
    domain: str,
) -> Path:
    if method == "none":
        return (
            ROOT
            / "results"
            / "safety"
            / "baseline"
            / "sprint5-no-filter-safety-summary.json"
        )

    if method == "clipping":
        return (
            ROOT
            / "results"
            / "safety"
            / "clipping"
            / "sprint5-clipping-safety-summary.json"
        )

    if method == "lyapunov":
        folder = (
            "lyapunov-driving"
            if domain == "autonomous_driving"
            else "lyapunov-robotics"
        )

        filename = (
            "sprint5-driving-lyapunov-summary.json"
            if domain == "autonomous_driving"
            else "sprint5-robotics-lyapunov-summary.json"
        )

        return ROOT / "results" / "safety" / folder / filename

    raise ValueError(f"unsupported method: {method}")


def _bounds_for_domain(
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    if domain == "autonomous_driving":
        return (
            np.asarray(
                [-1.0, -1.0, 0.0],
                dtype=np.float32,
            ),
            np.asarray(
                [1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
        )

    if domain == "robotics":
        return (
            np.full(
                3,
                -1.0,
                dtype=np.float32,
            ),
            np.full(
                3,
                1.0,
                dtype=np.float32,
            ),
        )

    raise ValueError(f"unsupported domain: {domain}")


def _method_decision(
    *,
    domain: str,
    method: str,
    true_state: np.ndarray,
    perturbed_action: np.ndarray,
    env: Any,
) -> tuple[
    np.ndarray,
    bool,
    str,
    float,
    dict[str, Any],
]:
    """Apply one frozen explicit safety method after action perturbation."""

    if method == "none":
        return (
            perturbed_action.copy(),
            False,
            "none",
            0.0,
            {},
        )

    if method == "clipping":
        decision = apply_clipping_safety_filter(
            domain=domain,
            true_state=true_state,
            proposed_action=perturbed_action,
        )

        reason = (
            decision.intervention_reason.value
            if hasattr(
                decision.intervention_reason,
                "value",
            )
            else str(decision.intervention_reason)
        )

        return (
            np.asarray(
                decision.executed_action,
                dtype=np.float32,
            ),
            bool(decision.intervened),
            str(reason),
            float(decision.correction_l2),
            {
                "triggered_rules": list(
                    decision.metadata.get(
                        "triggered_rules",
                        [],
                    )
                ),
            },
        )

    if method == "lyapunov":
        prediction_context = None

        if domain == "robotics":
            prediction_context = RoboticsPredictionContext(
                object_grasped=bool(env.object_grasped)
            )

        decision = apply_lyapunov_safety_filter(
            domain=domain,
            true_state=true_state,
            proposed_action=perturbed_action,
            robotics_context=prediction_context,
        )

        reason = (
            decision.intervention_reason.value
            if hasattr(
                decision.intervention_reason,
                "value",
            )
            else str(decision.intervention_reason)
        )

        metadata = decision.metadata

        candidate_scores = list(
            metadata.get(
                "candidate_scores",
                [],
            )
        )

        proposed_score = candidate_scores[0] if candidate_scores else {}

        current_v = float(
            metadata.get(
                "current_v",
                proposed_score.get(
                    "current_v",
                    0.0,
                ),
            )
        )

        perturbed_next_v = float(
            proposed_score.get(
                "next_v",
                current_v,
            )
        )

        perturbed_delta_v = float(
            proposed_score.get(
                "delta_v",
                (perturbed_next_v - current_v),
            )
        )

        selected_next_v = float(
            metadata.get(
                "selected_next_v",
                perturbed_next_v,
            )
        )

        selected_delta_v = float(
            metadata.get(
                "selected_delta_v",
                (selected_next_v - current_v),
            )
        )

        tolerance = float(
            metadata.get(
                "lyapunov_improvement_tolerance",
                1.0e-12,
            )
        )

        strict_lyapunov_decrease = bool(selected_delta_v < -tolerance)

        lyapunov_nonincrease = bool(selected_delta_v <= tolerance)

        selected_lower_than_perturbed = bool(
            selected_next_v < (perturbed_next_v - tolerance)
        )

        mechanism = {
            "candidate_count": int(
                metadata.get(
                    "candidate_count",
                    len(candidate_scores),
                )
            ),
            "current_v": current_v,
            "perturbed_next_v": (perturbed_next_v),
            "perturbed_delta_v": (perturbed_delta_v),
            "selected_next_v": (selected_next_v),
            "selected_delta_v": (selected_delta_v),
            "selected_candidate_source": (metadata.get("selected_candidate_source")),
            "strict_lyapunov_decrease": (strict_lyapunov_decrease),
            "lyapunov_nonincrease": (lyapunov_nonincrease),
            "selected_lower_than_perturbed": (selected_lower_than_perturbed),
            "emergency_fallback": bool(
                metadata.get(
                    "emergency_fallback",
                    False,
                )
            ),
        }

        return (
            np.asarray(
                decision.executed_action,
                dtype=np.float32,
            ),
            bool(decision.intervened),
            str(reason),
            float(decision.correction_l2),
            mechanism,
        )

    raise ValueError(f"unsupported method: {method}")


def _violation_categories(
    result: Any,
) -> set[str]:
    """Return names of currently violated frozen constraints."""

    return {str(record.name) for record in result if bool(record.violated)}


def run_action_episode(
    *,
    domain: str,
    perturbation: StructuredActionPerturbation,
    method: str,
    principal_seed: int,
    evaluation_seed: int,
) -> tuple[
    ActionRobustnessEpisodeEvidence,
    list[dict[str, Any]],
    str,
]:
    env = environment_for_domain(domain)

    observation, _ = env.reset(seed=evaluation_seed)

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=domain,
        principal_seed=principal_seed,
    )

    checkpoint_sha = str(policy.checkpoint_sha256)

    lower_bounds, upper_bounds = _bounds_for_domain(domain)

    total_reward = 0.0
    success = False
    step_count = 0

    perturbed_violation_steps = 0
    executed_violation_steps = 0

    perturbed_constraint_violations = 0
    executed_constraint_violations = 0

    critical_violation_steps = 0

    unsafe_perturbed_steps = 0
    recovered_unsafe_steps = 0
    unresolved_unsafe_steps = 0

    intervention_count = 0

    perturbation_l2_sum = 0.0
    perturbation_linf_max = 0.0

    safety_correction_l2_sum = 0.0
    safety_correction_l2_values: list[float] = []
    safety_correction_l2_max = 0.0

    total_action_displacement_l2_sum = 0.0

    interface_adjustment_count = 0
    interface_adjustment_l2_sum = 0.0
    interface_adjustment_l2_max = 0.0

    perturbed_category_counts: Counter[str] = Counter()

    executed_category_counts: Counter[str] = Counter()

    recovered_category_counts: Counter[str] = Counter()

    intervention_reason_counts: Counter[str] = Counter()

    candidate_source_counts: Counter[str] = Counter()

    strict_lyapunov_decrease_count = 0
    lyapunov_nonincrease_count = 0
    selected_lower_count = 0
    emergency_fallback_count = 0

    steps_object_grasped = 0
    episode_had_grasp = False
    perturbed_unsafe_while_grasped = 0
    interventions_while_grasped = 0

    proposed_to_perturbed_gripper_changes = 0
    perturbed_to_executed_gripper_changes = 0

    audit_snapshots: list[dict[str, Any]] = []

    while True:
        true_state = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()

        proposed_action = np.asarray(
            deterministic_policy_action(
                policy=policy,
                observation=true_state,
            ),
            dtype=np.float32,
        )

        (
            perturbed_action,
            perturbation_vector,
        ) = apply_structured_action_perturbation(
            proposed_action=proposed_action,
            perturbation=perturbation,
        )

        perturbed_violations = evaluate_domain_violations(
            domain=domain,
            state=true_state,
            action=perturbed_action,
        )

        perturbed_categories = _violation_categories(perturbed_violations)

        perturbed_has_violation = bool(perturbed_categories)

        if perturbed_has_violation:
            perturbed_violation_steps += 1
            unsafe_perturbed_steps += 1

        perturbed_constraint_violations += len(perturbed_categories)

        perturbed_category_counts.update(perturbed_categories)

        object_grasped = False

        if domain == "robotics":
            object_grasped = bool(env.object_grasped)

            if object_grasped:
                steps_object_grasped += 1
                episode_had_grasp = True

                if perturbed_has_violation:
                    perturbed_unsafe_while_grasped += 1

            proposed_semantic = gripper_semantic(float(proposed_action[2]))

            perturbed_semantic = gripper_semantic(float(perturbed_action[2]))

            if proposed_semantic != perturbed_semantic:
                proposed_to_perturbed_gripper_changes += 1

        (
            executed_action,
            intervened,
            reason,
            correction_l2,
            mechanism,
        ) = _method_decision(
            domain=domain,
            method=method,
            true_state=true_state,
            perturbed_action=perturbed_action,
            env=env,
        )

        executed_violations = evaluate_domain_violations(
            domain=domain,
            state=true_state,
            action=executed_action,
        )

        executed_categories = _violation_categories(executed_violations)

        executed_violated_records = tuple(
            record for record in executed_violations if bool(record.violated)
        )

        if any(
            (
                record.severity.value
                if hasattr(
                    record.severity,
                    "value",
                )
                else str(record.severity)
            )
            == "critical"
            for record in executed_violated_records
        ):
            critical_violation_steps += 1

        executed_has_violation = bool(executed_categories)

        if executed_has_violation:
            executed_violation_steps += 1

        executed_constraint_violations += len(executed_categories)

        executed_category_counts.update(executed_categories)

        if perturbed_has_violation and not executed_has_violation:
            recovered_unsafe_steps += 1

        if perturbed_has_violation and executed_has_violation:
            unresolved_unsafe_steps += 1

        for category in perturbed_categories - executed_categories:
            recovered_category_counts[category] += 1

        if intervened:
            intervention_count += 1

            intervention_reason_counts[reason] += 1

            if domain == "robotics" and object_grasped:
                interventions_while_grasped += 1

        else:
            intervention_reason_counts["none"] += 1

        if domain == "robotics":
            executed_semantic = gripper_semantic(float(executed_action[2]))

            perturbed_semantic = gripper_semantic(float(perturbed_action[2]))

            if perturbed_semantic != executed_semantic:
                perturbed_to_executed_gripper_changes += 1

        perturbation_l2 = float(np.linalg.norm(perturbation_vector))

        perturbation_linf = float(np.max(np.abs(perturbation_vector)))

        safety_correction = float(np.linalg.norm(executed_action - perturbed_action))

        total_displacement = float(np.linalg.norm(executed_action - proposed_action))

        perturbation_l2_sum += perturbation_l2

        perturbation_linf_max = max(
            perturbation_linf_max,
            perturbation_linf,
        )

        safety_correction_l2_sum += safety_correction

        safety_correction_l2_values.append(safety_correction)

        safety_correction_l2_max = max(
            safety_correction_l2_max,
            safety_correction,
        )

        total_action_displacement_l2_sum += total_displacement

        environment_action = environment_effective_action(
            executed_action=executed_action,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        interface_adjustment = float(
            np.linalg.norm(environment_action - executed_action)
        )

        if interface_adjustment > 1.0e-8:
            interface_adjustment_count += 1

        interface_adjustment_l2_sum += interface_adjustment

        interface_adjustment_l2_max = max(
            interface_adjustment_l2_max,
            interface_adjustment,
        )

        if method == "lyapunov":
            if mechanism.get(
                "strict_lyapunov_decrease",
                False,
            ):
                strict_lyapunov_decrease_count += 1

            if mechanism.get(
                "lyapunov_nonincrease",
                False,
            ):
                lyapunov_nonincrease_count += 1

            if mechanism.get(
                "selected_lower_than_perturbed",
                False,
            ):
                selected_lower_count += 1

            if mechanism.get(
                "emergency_fallback",
                False,
            ):
                emergency_fallback_count += 1

            source = mechanism.get("selected_candidate_source")

            if source is not None:
                candidate_source_counts[str(source)] += 1

        should_audit = len(audit_snapshots) < AUDIT_LIMIT and (
            perturbed_has_violation
            or intervened
            or interface_adjustment > 1.0e-8
            or mechanism.get(
                "strict_lyapunov_decrease",
                False,
            )
        )

        if should_audit:
            audit_snapshots.append(
                {
                    "step": step_count,
                    "true_state": (true_state.tolist()),
                    "proposed_action": (proposed_action.tolist()),
                    "action_perturbation_vector": (perturbation_vector.tolist()),
                    "perturbed_action": (perturbed_action.tolist()),
                    "executed_action": (executed_action.tolist()),
                    "environment_effective_action": (environment_action.tolist()),
                    "perturbed_violation_categories": sorted(perturbed_categories),
                    "executed_violation_categories": sorted(executed_categories),
                    "intervened": intervened,
                    "intervention_reason": reason,
                    "safety_correction_l2": (correction_l2),
                    "environment_interface_adjustment_l2": (interface_adjustment),
                    "object_grasped": (object_grasped),
                    **mechanism,
                }
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(executed_action.copy())

        total_reward += float(reward)

        step_count += 1

        success = bool(
            info.get(
                "success",
                False,
            )
        )

        if terminated or truncated:
            break

    evidence = ActionRobustnessEpisodeEvidence(
        domain=domain,
        method=method,
        perturbation_name=(perturbation.name),
        perturbation_family=(perturbation.family),
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        reward=total_reward,
        success=success,
        episode_length=step_count,
        perturbed_violation_step_count=(perturbed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        perturbed_constraint_violation_count=(perturbed_constraint_violations),
        executed_constraint_violation_count=(executed_constraint_violations),
        critical_violation_step_count=(critical_violation_steps),
        unsafe_perturbed_steps=(unsafe_perturbed_steps),
        recovered_unsafe_steps=(recovered_unsafe_steps),
        unresolved_unsafe_steps=(unresolved_unsafe_steps),
        intervention_count=(intervention_count),
        action_perturbation_l2_sum=(perturbation_l2_sum),
        action_perturbation_linf_max=(perturbation_linf_max),
        safety_correction_l2_sum=(safety_correction_l2_sum),
        safety_correction_l2_values=tuple(safety_correction_l2_values),
        safety_correction_l2_max=(safety_correction_l2_max),
        total_action_displacement_l2_sum=(total_action_displacement_l2_sum),
        environment_interface_adjustment_count=(interface_adjustment_count),
        environment_interface_adjustment_l2_sum=(interface_adjustment_l2_sum),
        environment_interface_adjustment_l2_max=(interface_adjustment_l2_max),
        perturbed_category_violation_counts=dict(perturbed_category_counts),
        executed_category_violation_counts=dict(executed_category_counts),
        recovered_category_counts=dict(recovered_category_counts),
        intervention_reason_counts=dict(intervention_reason_counts),
        selected_candidate_source_counts=dict(candidate_source_counts),
        strict_lyapunov_decrease_count=(strict_lyapunov_decrease_count),
        lyapunov_nonincrease_count=(lyapunov_nonincrease_count),
        selected_lower_than_perturbed_count=(selected_lower_count),
        emergency_fallback_count=(emergency_fallback_count),
        steps_object_grasped=(steps_object_grasped),
        episodes_with_object_grasped=int(episode_had_grasp),
        perturbed_unsafe_steps_while_grasped=(perturbed_unsafe_while_grasped),
        interventions_while_grasped=(interventions_while_grasped),
        proposed_to_perturbed_gripper_semantic_changes=(
            proposed_to_perturbed_gripper_changes
        ),
        perturbed_to_executed_gripper_semantic_changes=(
            perturbed_to_executed_gripper_changes
        ),
    )

    return (
        evidence,
        audit_snapshots,
        checkpoint_sha,
    )


def _run_cell(
    *,
    domain: str,
    perturbation: StructuredActionPerturbation,
    method: str,
    principal_seed: int,
    evaluation_seeds: tuple[int, ...],
    smoke: bool,
) -> Path:
    episodes: list[ActionRobustnessEpisodeEvidence] = []

    audits: list[list[dict[str, Any]]] = []

    checkpoint_sha = None

    for evaluation_seed in evaluation_seeds:
        (
            evidence,
            audit,
            current_checkpoint_sha,
        ) = run_action_episode(
            domain=domain,
            perturbation=perturbation,
            method=method,
            principal_seed=principal_seed,
            evaluation_seed=evaluation_seed,
        )

        if checkpoint_sha is None:
            checkpoint_sha = current_checkpoint_sha
        elif checkpoint_sha != current_checkpoint_sha:
            raise RuntimeError("checkpoint changed within cell")

        episodes.append(evidence)

        audits.append(audit)

    if checkpoint_sha is None:
        raise RuntimeError("empty action robustness cell")

    seed_summary = summarize_action_robustness_seed(episodes)

    clean_path = _clean_reference_path(
        method=method,
        domain=domain,
    )

    output_root = SMOKE_ROOT if smoke else RUNS_ROOT

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{_slug(domain)}-"
        f"{_slug(perturbation.name)}-"
        f"{method}-"
        f"seed-{principal_seed}.json"
    )

    output_path = output_root / filename

    payload = {
        "sprint": "5.12",
        "condition": "action_perturbation",
        "artifact_type": ("smoke" if smoke else "principal"),
        "domain": domain,
        "method": method,
        "perturbation_name": (perturbation.name),
        "perturbation_family": (perturbation.family),
        "perturbation_updates": [
            {
                "index": index,
                "delta": delta,
            }
            for index, delta in perturbation.updates
        ],
        "principal_seed": (principal_seed),
        "evaluation_seeds": list(evaluation_seeds),
        "episode_count": len(episodes),
        "checkpoint_sha256": (checkpoint_sha),
        "clean_reference_path": str(clean_path.relative_to(ROOT)),
        "clean_reference_sha256": (_sha256_file(clean_path)),
        "action_config_sha256": (_sha256_file(CONFIG_PATH)),
        "environment_handling_sha256": (_sha256_file(ENVIRONMENT_AUDIT_PATH)),
        "new_training_performed": False,
        "policy_fine_tuning_performed": False,
        "filter_tuning_performed": False,
        "observation_perturbation_applied": False,
        "gaussian_state_noise_applied": False,
        "structured_state_perturbation_applied": False,
        "action_perturbation_after_policy": True,
        "action_perturbation_before_safety_filter": True,
        "pre_filter_action_clipping": False,
        "none_executed_action_equals_perturbed_action": True,
        "environment_internal_action_clipping": True,
        "environment_interface_adjustment_not_filter_recovery": True,
        "object_grasped_added_to_policy_observation": False,
        "object_grasped_perturbed": False,
        "seed_summary": asdict(seed_summary),
        "episodes": [
            {
                **asdict(episode),
                "audit_snapshots": audit,
            }
            for episode, audit in zip(
                episodes,
                audits,
                strict=True,
            )
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=(
            "autonomous_driving",
            "robotics",
        ),
    )

    parser.add_argument(
        "--perturbation",
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
    )

    parser.add_argument(
        "--principal-seed",
        type=int,
        choices=PRINCIPAL_SEEDS,
    )

    parser.add_argument(
        "--evaluation-limit",
        type=int,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.evaluation_limit is not None and not args.smoke:
        raise ValueError("--evaluation-limit is smoke-only")

    domains = (
        (args.domain,)
        if args.domain
        else (
            "autonomous_driving",
            "robotics",
        )
    )

    methods = (args.method,) if args.method else METHODS

    seeds = (
        (args.principal_seed,) if args.principal_seed is not None else PRINCIPAL_SEEDS
    )

    evaluation_seeds = EVALUATION_SEEDS

    if args.evaluation_limit is not None:
        if args.evaluation_limit <= 0:
            raise ValueError("evaluation limit must be positive")

        evaluation_seeds = EVALUATION_SEEDS[: args.evaluation_limit]

    completed = 0

    for domain in domains:
        perturbations: tuple[
            StructuredActionPerturbation,
            ...,
        ]

        if args.perturbation is not None:
            perturbations = (
                action_perturbation_by_name(
                    domain=domain,
                    name=args.perturbation,
                ),
            )
        else:
            perturbations = action_perturbations_for_domain(domain)

        for perturbation in perturbations:
            for method in methods:
                for principal_seed in seeds:
                    _run_cell(
                        domain=domain,
                        perturbation=perturbation,
                        method=method,
                        principal_seed=principal_seed,
                        evaluation_seeds=(evaluation_seeds),
                        smoke=args.smoke,
                    )

                    completed += 1

                    print(
                        "[PASS]",
                        domain,
                        perturbation.name,
                        method,
                        f"seed={principal_seed}",
                        f"episodes={len(evaluation_seeds)}",
                    )

    print()
    print(
        "Completed cells:",
        completed,
    )

    if args.smoke:
        print("SPRINT 5.12 ACTION ROBUSTNESS SMOKE: PASS")
    elif (
        args.domain is None
        and args.perturbation is None
        and args.method is None
        and args.principal_seed is None
    ):
        if completed != 216:
            raise RuntimeError(f"expected 216 principal cells, got {completed}")

        print("SPRINT 5.12 ACTION ROBUSTNESS PRINCIPAL MATRIX: COMPLETE")


if __name__ == "__main__":
    main()
