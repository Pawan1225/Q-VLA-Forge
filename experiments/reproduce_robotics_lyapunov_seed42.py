"""Reproduce the frozen Sprint 5.9 robotics Lyapunov seed-42 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from q_vla_forge.evaluation.robotics_lyapunov_safety import (
    ROBOTICS_DOMAIN,
    robotics_lyapunov_summary_to_dict,
    run_robotics_lyapunov_episode,
    summarize_robotics_lyapunov_episodes,
)
from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    load_frozen_ppo_policy,
)

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    ROOT / "results" / "safety" / "lyapunov-robotics" / "runs" / "robotics-seed-42.json"
)

FLOAT_ATOL = 1.0e-10


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def _close(
    actual: float,
    expected: float,
    *,
    name: str,
) -> None:
    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=FLOAT_ATOL,
    ):
        raise AssertionError(
            f"{name} mismatch: " f"actual={actual!r}, " f"expected={expected!r}"
        )


def _compare_mapping(
    actual: dict[str, int],
    expected: dict[str, Any],
    *,
    name: str,
) -> None:
    normalized_expected = {str(key): int(value) for key, value in expected.items()}

    if actual != normalized_expected:
        raise AssertionError(
            f"{name} mismatch: "
            f"actual={actual!r}, "
            f"expected={normalized_expected!r}"
        )


def main() -> None:
    print("=" * 64)
    print(" Q-VLA FORGE - " "SPRINT 5.9 ROBOTICS SEED-42 REPRODUCTION")
    print("=" * 64)
    print()

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    _require(
        reference["principal_seed"] == 42,
        "reference principal seed is 42",
    )

    _require(
        reference["domain"] == ROBOTICS_DOMAIN,
        "reference domain is robotics",
    )

    _require(
        reference["safety_method"] == "lyapunov",
        "reference method is Lyapunov",
    )

    _require(
        reference["robustness_condition"] == "clean",
        "reference condition is clean",
    )

    _require(
        reference["evaluation_seeds"] == list(SAFETY_EVALUATION_SEEDS),
        "reference evaluation seeds match frozen set",
    )

    prediction_context = reference["prediction_context"]

    _require(
        prediction_context["source"] == "environment_internal_state",
        "prediction context uses environment internal state",
    )

    _require(
        prediction_context["field"] == "object_grasped",
        "prediction context field is object_grasped",
    )

    _require(
        prediction_context["policy_observation_modified"] is False,
        "policy observation remains unmodified",
    )

    _require(
        int(prediction_context["policy_observation_dimension"]) == 6,
        "policy observation remains 6-D",
    )

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=ROBOTICS_DOMAIN,
        principal_seed=42,
    )

    _require(
        policy.checkpoint_sha256 == reference["policy_checkpoint_sha256"],
        "checkpoint SHA matches reference",
    )

    reproduced_episodes = []

    for index, evaluation_seed in enumerate(SAFETY_EVALUATION_SEEDS):
        run = run_robotics_lyapunov_episode(
            policy=policy,
            evaluation_seed=evaluation_seed,
        )

        actual = run.episode

        expected = reference["episodes"][index]

        _require(
            actual.evaluation_seed == expected["evaluation_seed"],
            (f"episode {evaluation_seed} " "evaluation seed"),
        )

        _close(
            actual.reward,
            float(expected["reward"]),
            name=(f"episode {evaluation_seed} " "reward"),
        )

        _require(
            actual.success == bool(expected["success"]),
            (f"episode {evaluation_seed} " "success"),
        )

        _require(
            actual.episode_length == int(expected["episode_length"]),
            (f"episode {evaluation_seed} " "length"),
        )

        discrete_fields = (
            "proposed_violation_step_count",
            "executed_violation_step_count",
            "proposed_constraint_violation_count",
            "executed_constraint_violation_count",
            "critical_violation_step_count",
            "intervention_count",
            "policy_action_out_of_bounds_count",
            "executed_action_out_of_bounds_count",
            "proposed_hard_guard_failure_count",
            "selected_hard_guard_failure_count",
            "emergency_fallback_count",
            "strict_lyapunov_decrease_count",
            "lyapunov_nonincrease_count",
            "selected_lower_than_proposed_count",
            "zero_risk_state_step_count",
            "zero_risk_preserved_step_count",
            "zero_risk_recovery_count",
            "steps_object_grasped",
            "steps_object_not_grasped",
            "interventions_while_grasped",
            "interventions_while_not_grasped",
        )

        for field in discrete_fields:
            _require(
                getattr(
                    actual,
                    field,
                )
                == int(expected[field]),
                (f"episode {evaluation_seed} " f"{field}"),
            )

        _compare_mapping(
            actual.proposed_category_violation_counts,
            expected["proposed_category_violation_counts"],
            name=(f"episode {evaluation_seed} " "proposed categories"),
        )

        _compare_mapping(
            actual.executed_category_violation_counts,
            expected["executed_category_violation_counts"],
            name=(f"episode {evaluation_seed} " "executed categories"),
        )

        _compare_mapping(
            actual.intervention_reason_counts,
            expected["intervention_reason_counts"],
            name=(f"episode {evaluation_seed} " "intervention reasons"),
        )

        _compare_mapping(
            actual.selected_candidate_source_counts,
            expected["selected_candidate_source_counts"],
            name=(f"episode {evaluation_seed} " "candidate sources"),
        )

        _compare_mapping(
            actual.intervened_candidate_source_counts,
            expected["intervened_candidate_source_counts"],
            name=(f"episode {evaluation_seed} " "intervened candidate sources"),
        )

        _require(
            tuple(actual.candidate_counts) == tuple(expected["candidate_counts"]),
            (f"episode {evaluation_seed} " "candidate counts"),
        )

        _require(
            tuple(actual.eligible_candidate_counts)
            == tuple(expected["eligible_candidate_counts"]),
            (f"episode {evaluation_seed} " "eligible candidate counts"),
        )

        for field in (
            "action_corrections_l2",
            "current_v_values",
            "proposed_next_v_values",
            "selected_next_v_values",
            "proposed_delta_v_values",
            "selected_delta_v_values",
            "selection_v_delta_values",
        ):
            actual_values = np.asarray(
                getattr(
                    actual,
                    field,
                ),
                dtype=np.float64,
            )

            expected_values = np.asarray(
                expected[field],
                dtype=np.float64,
            )

            _require(
                actual_values.shape == expected_values.shape,
                (f"episode {evaluation_seed} " f"{field} shape"),
            )

            if not np.allclose(
                actual_values,
                expected_values,
                rtol=0.0,
                atol=FLOAT_ATOL,
            ):
                raise AssertionError(
                    f"episode " f"{evaluation_seed} " f"{field} mismatch"
                )

        _require(
            tuple(actual.zero_risk_recovery_steps)
            == tuple(expected["zero_risk_recovery_steps"]),
            (f"episode {evaluation_seed} " "recovery steps"),
        )

        _require(
            (actual.steps_object_grasped + actual.steps_object_not_grasped)
            == actual.episode_length,
            (f"episode {evaluation_seed} " "grasp-context accounting"),
        )

        reproduced_episodes.append(actual)

        print(f"[PASS] reproduced episode " f"{evaluation_seed}")

    reproduced_summary = summarize_robotics_lyapunov_episodes(reproduced_episodes)

    reproduced_payload = robotics_lyapunov_summary_to_dict(reproduced_summary)

    reference_summary = reference["seed_summary"]

    discrete_summary_fields = (
        "episode_count",
        "total_environment_steps",
        "proposed_violation_step_count",
        "executed_violation_step_count",
        "proposed_constraint_violation_count",
        "executed_constraint_violation_count",
        "critical_violation_step_count",
        "intervention_count",
        "proposed_hard_guard_failure_count",
        "selected_hard_guard_failure_count",
        "emergency_fallback_count",
        "strict_lyapunov_decrease_count",
        "lyapunov_nonincrease_count",
        "selected_lower_than_proposed_count",
        "zero_risk_state_step_count",
        "zero_risk_preserved_step_count",
        "zero_risk_recovery_count",
        "steps_object_grasped",
        "steps_object_not_grasped",
        "interventions_while_grasped",
        "interventions_while_not_grasped",
    )

    for field in discrete_summary_fields:
        reproduced_value = cast(
            int,
            reproduced_payload[field],
        )

        reference_value = int(reference_summary[field])

        _require(
            reproduced_value == reference_value,
            f"seed summary {field}",
        )

    floating_summary_fields = (
        "mean_reward",
        "reward_sample_sd",
        "success_rate",
        "mean_episode_length",
        "proposed_violation_step_rate",
        "executed_violation_step_rate",
        "proposed_constraint_violation_rate",
        "executed_constraint_violation_rate",
        "critical_violation_step_rate",
        "intervention_rate",
        "mean_action_correction_l2",
        "p95_action_correction_l2",
        "max_action_correction_l2",
        "mean_intervention_correction_l2",
        "proposed_hard_guard_failure_rate",
        "selected_hard_guard_failure_rate",
        "emergency_fallback_rate",
        "mean_current_v",
        "mean_proposed_next_v",
        "mean_selected_next_v",
        "mean_proposed_delta_v",
        "mean_selected_delta_v",
        "mean_selection_v_delta",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "selected_lower_than_proposed_rate",
        "zero_risk_preservation_rate",
        "mean_steps_to_zero_risk_recovery",
        "intervention_rate_while_grasped",
        "intervention_rate_while_not_grasped",
    )

    for field in floating_summary_fields:
        reproduced_float_value = cast(
            float,
            reproduced_payload[field],
        )

        reference_float_value = float(reference_summary[field])

        _close(
            reproduced_float_value,
            reference_float_value,
            name=(f"seed summary {field}"),
        )

    _compare_mapping(
        reproduced_summary.intervention_reason_counts,
        reference_summary["intervention_reason_counts"],
        name=("seed summary " "intervention reasons"),
    )

    _compare_mapping(
        reproduced_summary.selected_candidate_source_counts,
        reference_summary["selected_candidate_source_counts"],
        name=("seed summary " "candidate sources"),
    )

    _compare_mapping(
        reproduced_summary.intervened_candidate_source_counts,
        reference_summary["intervened_candidate_source_counts"],
        name=("seed summary " "intervened candidate sources"),
    )

    # Wall-clock latency is intentionally
    # excluded from deterministic equality.
    print("[PASS] latency excluded " "from deterministic equality")

    print()
    print("20 / 20 episodes reproduced")
    print("Safety accounting: PASS")
    print("Candidate behavior: PASS")
    print("Lyapunov accounting: PASS")
    print("Fallback accounting: PASS")
    print("Grasp-context accounting: PASS")
    print()
    print("SPRINT 5.9 ROBOTICS " "SEED-42 REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
