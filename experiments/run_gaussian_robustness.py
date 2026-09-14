"""Run Sprint 5.10 Gaussian observation-noise robustness evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from q_vla_forge.evaluation.gaussian_robustness import (
    GaussianEpisodeEvidence,
    summarize_gaussian_seed,
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
    GAUSSIAN_PRINCIPAL_NOISY_LEVELS,
    apply_gaussian_observation_noise,
    make_gaussian_rng,
)

ROOT = Path(__file__).resolve().parents[1]

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

CONDITION = "gaussian_state_perturbation"
SPRINT = "5.10"


def _enum_value(
    value: Any,
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
        violation
        for violation in violations
        if bool(
            getattr(
                violation,
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

    value = getattr(
        severity,
        "value",
        severity,
    )

    return str(value).lower() == "critical"


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


def _safe_float(
    value: object,
    *,
    default: float = 0.0,
) -> float:
    try:
        converted = float(cast(Any, value))
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(converted):
        return default

    return converted


def _metadata_float(
    metadata: dict[str, Any],
    *names: str,
) -> float | None:
    for name in names:
        if name not in metadata:
            continue

        value = metadata[name]

        try:
            converted = float(cast(Any, value))
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
    """Read next-V from a named frozen candidate."""

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
            "NONE",
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
        robotics_context = None

        if domain == "robotics":
            robotics_context = RoboticsPredictionContext(
                object_grasped=bool(env.object_grasped)
            )

        decision = apply_lyapunov_safety_filter(
            domain=domain,
            true_state=true_state,
            proposed_action=proposed_action,
            robotics_context=robotics_context,
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


def run_gaussian_episode(
    *,
    domain: str,
    method: str,
    principal_seed: int,
    evaluation_seed: int,
    sigma: float,
) -> tuple[
    GaussianEpisodeEvidence,
    list[dict[str, Any]],
    str,
]:
    """Run one frozen-policy Gaussian robustness episode."""

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=domain,
        principal_seed=principal_seed,
    )

    env = environment_for_domain(domain)

    env.reset(seed=evaluation_seed)

    rng, noise_seed = make_gaussian_rng(
        domain=domain,
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        sigma=sigma,
    )

    total_reward = 0.0
    episode_length = 0

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraint_count = 0
    executed_constraint_count = 0

    critical_violation_steps = 0

    intervention_count = 0

    noise_l2_sum = 0.0
    noise_l2_max = 0.0

    correction_l2_sum = 0.0
    correction_l2_max = 0.0

    intervention_reasons: Counter[str] = Counter()

    category_counts: Counter[str] = Counter()

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

        observed_state, noise_vector = apply_gaussian_observation_noise(
            true_state=true_state,
            sigma=sigma,
            rng=rng,
        )

        if not np.array_equal(
            observed_state,
            true_state + noise_vector,
        ):
            raise AssertionError(
                "observed state does not equal " "true state plus Gaussian noise"
            )

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

        executed_action, intervened, reason, correction_l2, metadata = _method_decision(
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

        noise_l2 = float(np.linalg.norm(noise_vector))

        noise_l2_sum += noise_l2

        noise_l2_max = max(
            noise_l2_max,
            noise_l2,
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
                    "observed_state": (observed_state.tolist()),
                    "noise_vector": (noise_vector.tolist()),
                    "proposed_action": (proposed_action.tolist()),
                    "executed_action": (executed_action.tolist()),
                    "intervened": (intervened),
                    "intervention_reason": (reason),
                    "correction_l2": (correction_l2),
                    "object_grasped": (object_grasped),
                }
            )

        step_result = env.step(executed_action)

        _, reward, terminated, truncated, _ = step_result

        total_reward += float(reward)

        episode_length += 1

        if bool(terminated) or bool(truncated):
            break

    success = bool(env.last_success)

    evidence = GaussianEpisodeEvidence(
        domain=domain,
        method=method,
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        sigma=sigma,
        noise_seed=noise_seed,
        reward=total_reward,
        success=success,
        episode_length=episode_length,
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_constraint_violation_count=(proposed_constraint_count),
        executed_constraint_violation_count=(executed_constraint_count),
        critical_violation_step_count=(critical_violation_steps),
        intervention_count=(intervention_count),
        noise_sample_count=(episode_length),
        noise_l2_sum=(noise_l2_sum),
        noise_l2_max=(noise_l2_max),
        action_correction_l2_sum=(correction_l2_sum),
        action_correction_l2_max=(correction_l2_max),
        intervention_reason_counts=dict(sorted(intervention_reasons.items())),
        category_violation_counts=dict(sorted(category_counts.items())),
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


def _sigma_slug(
    sigma: float,
) -> str:
    return f"{sigma:.2f}".replace(
        ".",
        "p",
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
    method: str,
    principal_seed: int,
    sigma: float,
) -> Path:
    name = (
        f"{_domain_slug(domain)}-"
        f"{method}-"
        f"seed-{principal_seed}-"
        f"sigma-{_sigma_slug(sigma)}.json"
    )

    return output_dir / name


def run_cell(
    *,
    domain: str,
    method: str,
    principal_seed: int,
    sigma: float,
    evaluation_seeds: tuple[int, ...],
    output_dir: Path,
    smoke: bool,
) -> Path:
    episodes: list[GaussianEpisodeEvidence] = []

    episode_payloads: list[dict[str, Any]] = []

    checkpoint_sha256: str | None = None

    for evaluation_seed in evaluation_seeds:
        evidence, audits, checkpoint_sha = run_gaussian_episode(
            domain=domain,
            method=method,
            principal_seed=principal_seed,
            evaluation_seed=evaluation_seed,
            sigma=sigma,
        )

        if checkpoint_sha256 is None:
            checkpoint_sha256 = checkpoint_sha
        elif checkpoint_sha256 != checkpoint_sha:
            raise AssertionError("checkpoint changed within cell")

        episodes.append(evidence)

        episode_payload = asdict(evidence)

        episode_payload["audit_snapshots"] = audits

        episode_payloads.append(episode_payload)

    summary = summarize_gaussian_seed(episodes)

    payload = {
        "sprint": SPRINT,
        "condition": CONDITION,
        "artifact_type": ("smoke" if smoke else "principal"),
        "domain": domain,
        "method": method,
        "principal_seed": (principal_seed),
        "sigma": sigma,
        "evaluation_seeds": list(evaluation_seeds),
        "episode_count": len(evaluation_seeds),
        "checkpoint_sha256": (checkpoint_sha256),
        "new_training_performed": False,
        "policy_fine_tuning_performed": False,
        "filter_tuning_performed": False,
        "action_perturbation": False,
        "structured_state_perturbation": False,
        "environment_dynamics_noise": False,
        "policy_uses_observed_state": True,
        "safety_uses_true_state": True,
        "observation_clipped_after_noise": False,
        "noise_seed_independent_of_method": True,
        "episodes": episode_payloads,
        "seed_summary": asdict(summary),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = _artifact_path(
        output_dir=output_dir,
        domain=domain,
        method=method,
        principal_seed=principal_seed,
        sigma=sigma,
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

    print(
        f"[PASS] {domain} "
        f"{method} "
        f"seed={principal_seed} "
        f"sigma={sigma:.2f} "
        f"episodes={len(evaluation_seeds)}"
    )

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=DOMAINS,
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
        "--sigma",
        type=float,
        choices=(*GAUSSIAN_PRINCIPAL_NOISY_LEVELS,),
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

    sigmas = (
        (args.sigma,) if args.sigma is not None else GAUSSIAN_PRINCIPAL_NOISY_LEVELS
    )

    evaluation_seeds = EVALUATION_SEEDS

    if args.evaluation_limit is not None:
        if args.evaluation_limit <= 0:
            raise ValueError("evaluation-limit must be positive")

        evaluation_seeds = evaluation_seeds[: args.evaluation_limit]

    if args.smoke:
        output_dir = ROOT / "results" / "safety" / "gaussian-robustness" / "smoke"
    else:
        if args.evaluation_limit is not None:
            raise ValueError("evaluation-limit is allowed " "only with --smoke")

        output_dir = ROOT / "results" / "safety" / "gaussian-robustness" / "runs"

    cell_count = 0

    for domain in domains:
        for method in methods:
            for principal_seed in principal_seeds:
                for sigma in sigmas:
                    run_cell(
                        domain=domain,
                        method=method,
                        principal_seed=principal_seed,
                        sigma=float(sigma),
                        evaluation_seeds=tuple(evaluation_seeds),
                        output_dir=output_dir,
                        smoke=args.smoke,
                    )

                    cell_count += 1

    print()
    print(f"Completed cells: {cell_count}")

    if args.smoke:
        print("SPRINT 5.10 GAUSSIAN " "ROBUSTNESS SMOKE: PASS")
    else:
        print("SPRINT 5.10 GAUSSIAN " "PRINCIPAL MATRIX: COMPLETE")


if __name__ == "__main__":
    main()
