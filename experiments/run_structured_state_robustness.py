"""Run Sprint 5.11 structured-state robustness evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    deterministic_policy_action,
    environment_for_domain,
    evaluate_domain_violations,
    load_frozen_ppo_policy,
)
from q_vla_forge.evaluation.structured_state_robustness import (
    StructuredStateEpisodeEvidence,
    summarize_structured_state_seed,
)
from q_vla_forge.safety.clipping import (
    apply_clipping_safety_filter,
)
from q_vla_forge.safety.lyapunov_filter import (
    RoboticsPredictionContext,
    apply_lyapunov_safety_filter,
)
from q_vla_forge.safety.perturbations import (
    StructuredStatePerturbation,
    apply_structured_state_perturbation,
    structured_perturbation_by_name,
    structured_perturbations_for_domain,
)

ROOT = Path(__file__).resolve().parents[1]

SPRINT = "5.11"
CONDITION = "state_perturbation"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

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

AUDIT_STEP_LIMIT = 5


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


def _clean_reference_path(
    *,
    domain: str,
    method: str,
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

    if method == "lyapunov" and domain == "autonomous_driving":
        return (
            ROOT
            / "results"
            / "safety"
            / "lyapunov-driving"
            / "sprint5-driving-lyapunov-summary.json"
        )

    if method == "lyapunov" and domain == "robotics":
        return (
            ROOT
            / "results"
            / "safety"
            / "lyapunov-robotics"
            / "sprint5-robotics-lyapunov-summary.json"
        )

    raise ValueError(f"unsupported clean reference: " f"{domain} {method}")


def _enum_value(
    value: object,
) -> str:
    raw = getattr(
        value,
        "value",
        value,
    )

    return str(raw)


def _violated(
    violations: tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        item
        for item in violations
        if bool(
            getattr(
                item,
                "violated",
                False,
            )
        )
    )


def _is_critical(
    violation: object,
) -> bool:
    severity = getattr(
        violation,
        "severity",
        "",
    )

    raw = getattr(
        severity,
        "value",
        severity,
    )

    return str(raw).lower() == "critical"


def _category_counts(
    violations: tuple[Any, ...],
) -> Counter[str]:
    result: Counter[str] = Counter()

    for violation in _violated(violations):
        result[
            str(
                getattr(
                    violation,
                    "name",
                    "unknown",
                )
            )
        ] += 1

    return result


def _metadata_float(
    metadata: dict[str, Any],
    *names: str,
) -> float | None:
    for name in names:
        if name not in metadata:
            continue

        value = metadata[name]

        try:
            converted = float(
                cast(
                    Any,
                    value,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(converted):
            return converted

    return None


def _candidate_next_v(
    metadata: dict[str, Any],
    *,
    source: str,
) -> float | None:
    scores = metadata.get("candidate_scores")

    if not isinstance(
        scores,
        list,
    ):
        return None

    for score in scores:
        if not isinstance(
            score,
            dict,
        ):
            continue

        if score.get("source") != source:
            continue

        value = score.get("next_v")

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            converted = float(value)

            if math.isfinite(converted):
                return converted

    return None


def _method_decision(
    *,
    domain: str,
    method: str,
    true_state: np.ndarray,
    proposed_action: np.ndarray,
    env: Any,
) -> tuple[
    np.ndarray,
    bool,
    str,
    float,
    dict[str, Any],
]:
    if method == "none":
        return (
            proposed_action.copy(),
            False,
            "none",
            0.0,
            {},
        )

    if method == "clipping":
        decision = apply_clipping_safety_filter(
            domain=domain,
            true_state=true_state,
            proposed_action=proposed_action,
        )

    elif method == "lyapunov":
        context = None

        if domain == "robotics":
            context = RoboticsPredictionContext(object_grasped=bool(env.object_grasped))

        decision = apply_lyapunov_safety_filter(
            domain=domain,
            true_state=true_state,
            proposed_action=proposed_action,
            robotics_context=context,
        )

    else:
        raise ValueError(f"unsupported method: {method}")

    return (
        np.asarray(
            decision.executed_action,
            dtype=np.float32,
        ).copy(),
        bool(decision.intervened),
        _enum_value(decision.intervention_reason),
        float(decision.correction_l2),
        dict(decision.metadata),
    )


def run_structured_episode(
    *,
    domain: str,
    perturbation: StructuredStatePerturbation,
    method: str,
    principal_seed: int,
    evaluation_seed: int,
) -> tuple[
    StructuredStateEpisodeEvidence,
    list[dict[str, Any]],
    str,
]:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=domain,
        principal_seed=principal_seed,
    )

    env = environment_for_domain(domain)

    env.reset(seed=evaluation_seed)

    total_reward = 0.0
    episode_length = 0

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraint_count = 0
    executed_constraint_count = 0

    critical_violation_steps = 0

    intervention_count = 0

    correction_l2_sum = 0.0
    correction_l2_max = 0.0

    perturbation_l1_sum = 0.0
    perturbation_l2_sum = 0.0
    perturbation_linf_max = 0.0

    category_counts: Counter[str] = Counter()

    intervention_reasons: Counter[str] = Counter()

    selected_candidate_sources: Counter[str] = Counter()

    intervened_candidate_sources: Counter[str] = Counter()

    strict_lyapunov_decrease_count = 0
    lyapunov_nonincrease_count = 0
    selected_lower_than_proposed_count = 0
    emergency_fallback_count = 0

    steps_object_grasped = 0
    steps_object_not_grasped = 0
    interventions_while_grasped = 0

    audit_snapshots: list[dict[str, Any]] = []

    while True:
        true_state = np.asarray(
            env.state,
            dtype=np.float32,
        ).copy()

        (
            observed_state,
            perturbation_vector,
        ) = apply_structured_state_perturbation(
            true_state=true_state,
            perturbation=perturbation,
        )

        if not np.array_equal(
            observed_state,
            true_state + perturbation_vector,
        ):
            raise AssertionError("structured observation relation failed")

        proposed_action = deterministic_policy_action(
            policy=policy,
            observation=observed_state,
        )

        proposed_action = np.asarray(
            proposed_action,
            dtype=np.float32,
        ).copy()

        proposed_violations = evaluate_domain_violations(
            domain=domain,
            state=true_state,
            action=proposed_action,
        )

        (
            executed_action,
            intervened,
            reason,
            correction_l2,
            metadata,
        ) = _method_decision(
            domain=domain,
            method=method,
            true_state=true_state,
            proposed_action=proposed_action,
            env=env,
        )

        executed_violations = evaluate_domain_violations(
            domain=domain,
            state=true_state,
            action=executed_action,
        )

        proposed_active = _violated(proposed_violations)

        executed_active = _violated(executed_violations)

        if proposed_active:
            proposed_violation_steps += 1

        if executed_active:
            executed_violation_steps += 1

        proposed_constraint_count += len(proposed_active)

        executed_constraint_count += len(executed_active)

        if any(_is_critical(violation) for violation in executed_active):
            critical_violation_steps += 1

        category_counts.update(_category_counts(executed_violations))

        if intervened:
            intervention_count += 1

        intervention_reasons[reason] += 1

        correction_l2_sum += correction_l2

        correction_l2_max = max(
            correction_l2_max,
            correction_l2,
        )

        perturbation_l1 = float(
            np.linalg.norm(
                perturbation_vector,
                ord=1,
            )
        )

        perturbation_l2 = float(
            np.linalg.norm(
                perturbation_vector,
            )
        )

        perturbation_linf = float(
            np.linalg.norm(
                perturbation_vector,
                ord=np.inf,
            )
        )

        perturbation_l1_sum += perturbation_l1

        perturbation_l2_sum += perturbation_l2

        perturbation_linf_max = max(
            perturbation_linf_max,
            perturbation_linf,
        )

        object_grasped = False

        if domain == "robotics":
            object_grasped = bool(env.object_grasped)

            if object_grasped:
                steps_object_grasped += 1

                if intervened:
                    interventions_while_grasped += 1

            else:
                steps_object_not_grasped += 1

        if method == "lyapunov":
            selected_source = str(
                metadata.get(
                    "selected_candidate_source",
                    "unknown",
                )
            )

            selected_candidate_sources[selected_source] += 1

            if intervened:
                intervened_candidate_sources[selected_source] += 1

            selected_delta_v = _metadata_float(
                metadata,
                "selected_delta_v",
            )

            current_v = _metadata_float(
                metadata,
                "current_v",
            )

            selected_v = _metadata_float(
                metadata,
                "selected_next_v",
            )

            proposed_v = _candidate_next_v(
                metadata,
                source="bounded_proposed",
            )

            if selected_delta_v is not None and selected_delta_v < -1.0e-12:
                strict_lyapunov_decrease_count += 1

            if (selected_delta_v is not None and selected_delta_v <= 1.0e-12) or (
                current_v is not None
                and selected_v is not None
                and selected_v <= current_v + 1.0e-12
            ):
                lyapunov_nonincrease_count += 1

            if (
                proposed_v is not None
                and selected_v is not None
                and selected_v < proposed_v - 1.0e-12
            ):
                selected_lower_than_proposed_count += 1

            if bool(
                metadata.get(
                    "emergency_fallback",
                    False,
                )
            ):
                emergency_fallback_count += 1

        if len(audit_snapshots) < AUDIT_STEP_LIMIT:
            audit_snapshots.append(
                {
                    "step": episode_length,
                    "true_state": (true_state.tolist()),
                    "perturbation_vector": (perturbation_vector.tolist()),
                    "observed_state": (observed_state.tolist()),
                    "proposed_action": (proposed_action.tolist()),
                    "executed_action": (executed_action.tolist()),
                    "intervened": (intervened),
                    "intervention_reason": (reason),
                    "correction_l2": (correction_l2),
                    "object_grasped": (object_grasped),
                    "current_v": (
                        metadata.get("current_v") if method == "lyapunov" else None
                    ),
                    "proposed_next_v": (
                        _candidate_next_v(
                            metadata,
                            source=("bounded_proposed"),
                        )
                        if method == "lyapunov"
                        else None
                    ),
                    "selected_next_v": (
                        metadata.get("selected_next_v")
                        if method == "lyapunov"
                        else None
                    ),
                    "selected_delta_v": (
                        metadata.get("selected_delta_v")
                        if method == "lyapunov"
                        else None
                    ),
                    "selected_candidate_source": (
                        metadata.get("selected_candidate_source")
                        if method == "lyapunov"
                        else None
                    ),
                }
            )

        (
            _,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(executed_action)

        total_reward += float(reward)

        episode_length += 1

        if bool(terminated) or bool(truncated):
            break

    evidence = StructuredStateEpisodeEvidence(
        domain=domain,
        method=method,
        perturbation_name=(perturbation.name),
        perturbation_family=(perturbation.family),
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        reward=total_reward,
        success=bool(env.last_success),
        episode_length=episode_length,
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_constraint_violation_count=(proposed_constraint_count),
        executed_constraint_violation_count=(executed_constraint_count),
        critical_violation_step_count=(critical_violation_steps),
        intervention_count=(intervention_count),
        perturbation_l1_sum=(perturbation_l1_sum),
        perturbation_l2_sum=(perturbation_l2_sum),
        perturbation_linf_max=(perturbation_linf_max),
        action_correction_l2_sum=(correction_l2_sum),
        action_correction_l2_max=(correction_l2_max),
        category_violation_counts=dict(sorted(category_counts.items())),
        intervention_reason_counts=dict(sorted(intervention_reasons.items())),
        selected_candidate_source_counts=dict(
            sorted(selected_candidate_sources.items())
        ),
        intervened_candidate_source_counts=dict(
            sorted(intervened_candidate_sources.items())
        ),
        strict_lyapunov_decrease_count=(strict_lyapunov_decrease_count),
        lyapunov_nonincrease_count=(lyapunov_nonincrease_count),
        selected_lower_than_proposed_count=(selected_lower_than_proposed_count),
        emergency_fallback_count=(emergency_fallback_count),
        steps_object_grasped=(steps_object_grasped),
        steps_object_not_grasped=(steps_object_not_grasped),
        interventions_while_grasped=(interventions_while_grasped),
    )

    return (
        evidence,
        audit_snapshots,
        policy.checkpoint_sha256,
    )


def _domain_slug(
    domain: str,
) -> str:
    return domain.replace(
        "_",
        "-",
    )


def _artifact_path(
    *,
    output_dir: Path,
    domain: str,
    perturbation: StructuredStatePerturbation,
    method: str,
    principal_seed: int,
) -> Path:
    return output_dir / (
        f"{_domain_slug(domain)}-"
        f"{perturbation.name.replace('_', '-')}-"
        f"{method}-"
        f"seed-{principal_seed}.json"
    )


def run_cell(
    *,
    domain: str,
    perturbation: StructuredStatePerturbation,
    method: str,
    principal_seed: int,
    evaluation_seeds: tuple[int, ...],
    output_dir: Path,
    smoke: bool,
) -> Path:
    episodes: list[StructuredStateEpisodeEvidence] = []

    episode_payloads: list[dict[str, Any]] = []

    checkpoint_sha256: str | None = None

    for evaluation_seed in evaluation_seeds:
        (
            evidence,
            audits,
            checkpoint_sha,
        ) = run_structured_episode(
            domain=domain,
            perturbation=perturbation,
            method=method,
            principal_seed=principal_seed,
            evaluation_seed=evaluation_seed,
        )

        if checkpoint_sha256 is None:
            checkpoint_sha256 = checkpoint_sha

        elif checkpoint_sha256 != checkpoint_sha:
            raise AssertionError("checkpoint changed within cell")

        episodes.append(evidence)

        payload = asdict(evidence)

        payload["audit_snapshots"] = audits

        episode_payloads.append(payload)

    summary = summarize_structured_state_seed(episodes)

    clean_reference = _clean_reference_path(
        domain=domain,
        method=method,
    )

    payload = {
        "sprint": SPRINT,
        "condition": CONDITION,
        "artifact_type": ("smoke" if smoke else "principal"),
        "domain": domain,
        "perturbation_name": (perturbation.name),
        "perturbation_family": (perturbation.family),
        "perturbation_updates": [
            {
                "index": index,
                "delta": delta,
            }
            for index, delta in perturbation.updates
        ],
        "method": method,
        "principal_seed": (principal_seed),
        "evaluation_seeds": list(evaluation_seeds),
        "episode_count": len(evaluation_seeds),
        "checkpoint_sha256": (checkpoint_sha256),
        "clean_reference_path": str(clean_reference.relative_to(ROOT)),
        "clean_reference_sha256": (_sha256_file(clean_reference)),
        "new_training_performed": False,
        "policy_fine_tuning_performed": False,
        "filter_tuning_performed": False,
        "gaussian_noise_applied": False,
        "action_perturbation_applied": False,
        "environment_dynamics_perturbation": False,
        "policy_uses_perturbed_observation": True,
        "safety_uses_true_state": True,
        "environment_uses_true_state": True,
        "observation_clipped_after_perturbation": False,
        "object_grasped_added_to_policy_observation": False,
        "object_grasped_perturbed": False,
        "episodes": episode_payloads,
        "seed_summary": asdict(summary),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = _artifact_path(
        output_dir=output_dir,
        domain=domain,
        perturbation=perturbation,
        method=method,
        principal_seed=principal_seed,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[PASS] {domain} "
        f"{perturbation.name} "
        f"{method} "
        f"seed={principal_seed} "
        f"episodes={len(evaluation_seeds)}"
    )

    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=DOMAINS,
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

    domains = (args.domain,) if args.domain else DOMAINS

    methods = (args.method,) if args.method else METHODS

    principal_seeds = (
        (args.principal_seed,) if args.principal_seed is not None else PRINCIPAL_SEEDS
    )

    evaluation_seeds = EVALUATION_SEEDS

    if args.evaluation_limit is not None:
        if not args.smoke:
            raise ValueError("evaluation-limit is smoke-only")

        if args.evaluation_limit <= 0:
            raise ValueError("evaluation-limit must be positive")

        evaluation_seeds = evaluation_seeds[: args.evaluation_limit]

    output_dir = (
        ROOT
        / "results"
        / "safety"
        / "structured-state-robustness"
        / ("smoke" if args.smoke else "runs")
    )

    cell_count = 0

    for domain in domains:
        if args.perturbation is not None:
            perturbations: tuple[
                StructuredStatePerturbation,
                ...,
            ]

            perturbations = (
                structured_perturbation_by_name(
                    domain=domain,
                    name=args.perturbation,
                ),
            )
        else:
            perturbations = structured_perturbations_for_domain(domain)

        for perturbation in perturbations:
            for method in methods:
                for principal_seed in principal_seeds:
                    run_cell(
                        domain=domain,
                        perturbation=perturbation,
                        method=method,
                        principal_seed=principal_seed,
                        evaluation_seeds=tuple(evaluation_seeds),
                        output_dir=output_dir,
                        smoke=args.smoke,
                    )

                    cell_count += 1

    print()
    print(f"Completed cells: {cell_count}")

    if args.smoke:
        print("SPRINT 5.11 STRUCTURED " "STATE ROBUSTNESS SMOKE: PASS")
    else:
        print("SPRINT 5.11 STRUCTURED " "STATE PRINCIPAL MATRIX: COMPLETE")


if __name__ == "__main__":
    main()
