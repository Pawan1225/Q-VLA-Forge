"""Independent verifier for Sprint 5.9 robotics Lyapunov safety evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RUNS_DIR = ROOT / "results" / "safety" / "lyapunov-robotics" / "runs"

SUMMARY_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-robotics"
    / "sprint5-robotics-lyapunov-summary.json"
)

NONE_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "baseline" / "sprint5-no-filter-safety-summary.json"
)

CLIPPING_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json"
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

EXPECTED_CHECKPOINT_HASHES = {
    42: ("ce4703c3080b4485d8788ed7e7225a61" "f821cfdbd8d473a3f99c33511df1cd63"),
    123: ("54694c3691fe8cd255b4c85b1433ceef" "65dfd04aca9b28c3f466a7a9447fdb13"),
    456: ("ae48eebf0b3e0dae09414d2fd895ab64" "4178296800c13ffd293b3b812087ab13"),
}

FLOAT_ATOL = 1.0e-10


def _load(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65_536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
    message: str,
) -> None:
    _require(
        bool(
            np.isclose(
                actual,
                expected,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
        ),
        message,
    )


def _p95(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            95,
        )
    )


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - " "SPRINT 5.9 ROBOTICS " "LYAPUNOV SAFETY VERIFICATION")
    print("=" * 68)
    print()

    summary = _load(SUMMARY_PATH)

    none_summary = _load(NONE_SUMMARY_PATH)

    clipping_summary = _load(CLIPPING_SUMMARY_PATH)

    _require(
        summary["sprint"] == "5.9",
        "summary sprint is 5.9",
    )

    _require(
        summary["domain"] == "robotics",
        "summary domain is robotics",
    )

    _require(
        summary["condition"] == "clean",
        "summary condition is clean",
    )

    _require(
        summary["principal_seeds"] == list(PRINCIPAL_SEEDS),
        "principal seeds exactly 42,123,456",
    )

    _require(
        summary["evaluation_seeds"] == list(EVALUATION_SEEDS),
        ("evaluation seeds exactly " "20000..20019"),
    )

    _require(
        summary["new_training_performed"] is False,
        "summary records no new training",
    )

    seed_violation_rates: list[float] = []

    seed_rewards: list[float] = []

    seed_success_rates: list[float] = []

    seed_intervention_rates: list[float] = []

    total_environment_steps = 0
    total_grasped_steps = 0
    total_not_grasped_steps = 0
    total_domain_constraint_interventions = 0
    total_lyapunov_decrease_interventions = 0
    total_fallbacks = 0
    total_selected_guard_failures = 0
    total_strict_decreases = 0
    total_nonincreases = 0
    total_selected_lower = 0

    for seed in PRINCIPAL_SEEDS:
        path = RUNS_DIR / f"robotics-seed-{seed}.json"

        _require(
            path.exists(),
            ("principal run exists " f"seed {seed}"),
        )

        payload = _load(path)

        _require(
            payload["sprint"] == "5.9",
            f"run sprint valid seed {seed}",
        )

        _require(
            payload["artifact"] == ("robotics-lyapunov-" "safety-principal-run"),
            f"artifact identity seed {seed}",
        )

        _require(
            payload["domain"] == "robotics",
            f"domain valid seed {seed}",
        )

        _require(
            payload["principal_seed"] == seed,
            ("principal seed valid " f"seed {seed}"),
        )

        _require(
            payload["safety_method"] == "lyapunov",
            f"method valid seed {seed}",
        )

        _require(
            payload["robustness_condition"] == "clean",
            f"condition valid seed {seed}",
        )

        _require(
            payload["new_training_performed"] is False,
            f"no training seed {seed}",
        )

        _require(
            payload["policy_fine_tuning_performed"] is False,
            ("no policy fine-tuning " f"seed {seed}"),
        )

        _require(
            payload["filter_tuning_performed"] is False,
            ("no filter tuning " f"seed {seed}"),
        )

        _require(
            payload["perturbation_used"] is False,
            ("no perturbation " f"seed {seed}"),
        )

        _require(
            payload["evaluation_seeds"] == list(EVALUATION_SEEDS),
            ("evaluation seeds valid " f"seed {seed}"),
        )

        _require(
            payload["episode_count"] == 20,
            ("20 episodes " f"seed {seed}"),
        )

        _require(
            len(payload["episodes"]) == 20,
            ("episode payload count " f"seed {seed}"),
        )

        _require(
            payload["policy_checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASHES[seed],
            ("checkpoint SHA " f"seed {seed}"),
        )

        checkpoint_path = ROOT / payload["policy_checkpoint"]

        _require(
            _sha256(checkpoint_path) == payload["policy_checkpoint_sha256"],
            ("current checkpoint provenance " f"seed {seed}"),
        )

        recovery_path = ROOT / payload["checkpoint_recovery_record"]

        _require(
            _sha256(recovery_path) == payload["checkpoint_recovery_record_sha256"],
            ("current recovery provenance " f"seed {seed}"),
        )

        prediction_context = payload["prediction_context"]

        _require(
            prediction_context["source"] == "environment_internal_state",
            ("prediction context source " f"seed {seed}"),
        )

        _require(
            prediction_context["field"] == "object_grasped",
            ("prediction context field " f"seed {seed}"),
        )

        _require(
            prediction_context["policy_observation_modified"] is False,
            ("policy observation unmodified " f"seed {seed}"),
        )

        _require(
            int(prediction_context["policy_observation_dimension"]) == 6,
            ("policy observation dimension " f"seed {seed}"),
        )

        for (
            provenance_name,
            provenance,
        ) in payload["provenance"].items():
            provenance_path = ROOT / provenance["path"]

            _require(
                provenance_path.exists(),
                ("provenance exists " f"{provenance_name} " f"seed {seed}"),
            )

            _require(
                _sha256(provenance_path) == provenance["sha256"],
                ("provenance SHA " f"{provenance_name} " f"seed {seed}"),
            )

        episodes = payload["episodes"]

        total_steps = sum(int(episode["episode_length"]) for episode in episodes)

        _require(
            total_steps == int(payload["seed_summary"]["total_environment_steps"]),
            ("total step arithmetic " f"seed {seed}"),
        )

        _require(
            total_steps == 2000,
            ("2000 environment steps " f"seed {seed}"),
        )

        executed_violation_steps = sum(
            int(episode["executed_violation_step_count"]) for episode in episodes
        )

        reconstructed_violation_rate = executed_violation_steps / total_steps

        _close(
            reconstructed_violation_rate,
            float(payload["seed_summary"]["executed_violation_step_rate"]),
            message=("violation rate arithmetic " f"seed {seed}"),
        )

        executed_constraints = sum(
            int(episode["executed_constraint_violation_count"]) for episode in episodes
        )

        _close(
            executed_constraints / total_steps,
            float(payload["seed_summary"]["executed_constraint_violation_rate"]),
            message=("constraint rate arithmetic " f"seed {seed}"),
        )

        critical_steps = sum(
            int(episode["critical_violation_step_count"]) for episode in episodes
        )

        _close(
            critical_steps / total_steps,
            float(payload["seed_summary"]["critical_violation_step_rate"]),
            message=("critical rate arithmetic " f"seed {seed}"),
        )

        interventions = sum(int(episode["intervention_count"]) for episode in episodes)

        reconstructed_intervention_rate = interventions / total_steps

        _close(
            reconstructed_intervention_rate,
            float(payload["seed_summary"]["intervention_rate"]),
            message=("intervention rate arithmetic " f"seed {seed}"),
        )

        corrections = [
            float(value)
            for episode in episodes
            for value in episode["action_corrections_l2"]
        ]

        _close(
            float(statistics.mean(corrections)),
            float(payload["seed_summary"]["mean_action_correction_l2"]),
            message=("mean correction arithmetic " f"seed {seed}"),
        )

        _close(
            _p95(corrections),
            float(payload["seed_summary"]["p95_action_correction_l2"]),
            message=("P95 correction arithmetic " f"seed {seed}"),
        )

        _close(
            max(
                corrections,
                default=0.0,
            ),
            float(payload["seed_summary"]["max_action_correction_l2"]),
            message=("max correction arithmetic " f"seed {seed}"),
        )

        reason_counts: dict[
            str,
            int,
        ] = {}

        candidate_counts: dict[
            str,
            int,
        ] = {}

        intervened_candidate_counts: dict[
            str,
            int,
        ] = {}

        strict_decreases = 0
        nonincreases = 0
        selected_lower = 0
        fallback_count = 0
        grasped_steps = 0
        not_grasped_steps = 0

        for episode in episodes:
            for (
                reason,
                count,
            ) in episode["intervention_reason_counts"].items():
                reason_counts[reason] = reason_counts.get(
                    reason,
                    0,
                ) + int(count)

            for (
                source,
                count,
            ) in episode["selected_candidate_source_counts"].items():
                candidate_counts[source] = candidate_counts.get(
                    source,
                    0,
                ) + int(count)

            for (
                source,
                count,
            ) in episode["intervened_candidate_source_counts"].items():
                intervened_candidate_counts[source] = intervened_candidate_counts.get(
                    source,
                    0,
                ) + int(count)

            strict_decreases += int(episode["strict_lyapunov_decrease_count"])

            nonincreases += int(episode["lyapunov_nonincrease_count"])

            selected_lower += int(episode["selected_lower_than_proposed_count"])

            fallback_count += int(episode["emergency_fallback_count"])

            grasped_steps += int(episode["steps_object_grasped"])

            not_grasped_steps += int(episode["steps_object_not_grasped"])

            _require(
                max(episode["candidate_counts"]) <= 9,
                ("candidate ceiling " f"seed {seed}"),
            )

            _require(
                (
                    int(episode["steps_object_grasped"])
                    + int(episode["steps_object_not_grasped"])
                )
                == int(episode["episode_length"]),
                ("episode grasp-context " f"accounting seed {seed}"),
            )

        summary_reason_counts = {
            str(key): int(value)
            for (
                key,
                value,
            ) in payload[
                "seed_summary"
            ]["intervention_reason_counts"].items()
        }

        normalized_reason_counts = {
            reason: int(
                reason_counts.get(
                    reason,
                    0,
                )
            )
            for reason in summary_reason_counts
        }

        _require(
            normalized_reason_counts == summary_reason_counts,
            ("reason accounting " f"seed {seed}"),
        )

        summary_candidate_counts = {
            str(key): int(value)
            for (
                key,
                value,
            ) in payload[
                "seed_summary"
            ]["selected_candidate_source_counts"].items()
        }

        normalized_candidate_counts = {
            source: int(
                candidate_counts.get(
                    source,
                    0,
                )
            )
            for source in summary_candidate_counts
        }

        _require(
            normalized_candidate_counts == summary_candidate_counts,
            ("candidate-source accounting " f"seed {seed}"),
        )

        summary_intervened_candidate_counts = {
            str(key): int(value)
            for (
                key,
                value,
            ) in payload[
                "seed_summary"
            ]["intervened_candidate_source_counts"].items()
        }

        normalized_intervened_candidate_counts = {
            source: int(
                intervened_candidate_counts.get(
                    source,
                    0,
                )
            )
            for source in summary_intervened_candidate_counts
        }

        _require(
            normalized_intervened_candidate_counts
            == summary_intervened_candidate_counts,
            ("intervened candidate-source " f"accounting seed {seed}"),
        )

        _require(
            sum(normalized_intervened_candidate_counts.values()) == interventions,
            ("intervened candidate total " f"seed {seed}"),
        )

        _close(
            strict_decreases / total_steps,
            float(payload["seed_summary"]["strict_lyapunov_decrease_rate"]),
            message=("strict decrease rate " f"seed {seed}"),
        )

        _close(
            nonincreases / total_steps,
            float(payload["seed_summary"]["lyapunov_nonincrease_rate"]),
            message=("nonincrease rate " f"seed {seed}"),
        )

        _close(
            selected_lower / total_steps,
            float(payload["seed_summary"]["selected_lower_than_proposed_rate"]),
            message=("selected-lower rate " f"seed {seed}"),
        )

        _close(
            fallback_count / total_steps,
            float(payload["seed_summary"]["emergency_fallback_rate"]),
            message=("fallback rate " f"seed {seed}"),
        )

        _require(
            grasped_steps == int(payload["seed_summary"]["steps_object_grasped"]),
            ("grasped-step accounting " f"seed {seed}"),
        )

        _require(
            not_grasped_steps
            == int(payload["seed_summary"]["steps_object_not_grasped"]),
            ("not-grasped-step accounting " f"seed {seed}"),
        )

        _require(
            (grasped_steps + not_grasped_steps) == total_steps,
            ("grasp-context total " f"seed {seed}"),
        )

        _require(
            grasped_steps == 0,
            ("principal grasped-state branch " f"unexercised seed {seed}"),
        )

        _require(
            int(payload["seed_summary"]["selected_hard_guard_failure_count"]) == 0,
            ("selected hard guards valid " f"seed {seed}"),
        )

        executed_action_out_of_bounds_count = sum(
            int(episode["executed_action_out_of_bounds_count"]) for episode in episodes
        )

        _require(
            executed_action_out_of_bounds_count == 0,
            ("executed actions in bounds " f"seed {seed}"),
        )

        total_environment_steps += total_steps

        total_grasped_steps += grasped_steps

        total_not_grasped_steps += not_grasped_steps

        total_domain_constraint_interventions += int(
            reason_counts.get(
                "domain_constraint",
                0,
            )
        )

        total_lyapunov_decrease_interventions += int(
            reason_counts.get(
                "lyapunov_decrease",
                0,
            )
        )

        total_fallbacks += fallback_count

        total_selected_guard_failures += int(
            payload["seed_summary"]["selected_hard_guard_failure_count"]
        )

        total_strict_decreases += strict_decreases

        total_nonincreases += nonincreases

        total_selected_lower += selected_lower

        seed_violation_rates.append(reconstructed_violation_rate)

        rewards = [float(episode["reward"]) for episode in episodes]

        successes = [float(episode["success"]) for episode in episodes]

        seed_rewards.append(float(statistics.mean(rewards)))

        seed_success_rates.append(float(statistics.mean(successes)))

        seed_intervention_rates.append(reconstructed_intervention_rate)

    _require(
        total_environment_steps == 6000,
        ("principal evaluation contains " "6000 environment steps"),
    )

    _require(
        total_grasped_steps == 0,
        ("object_grasped=True " "coverage is 0/6000"),
    )

    _require(
        total_not_grasped_steps == 6000,
        ("object_grasped=False " "coverage is 6000/6000"),
    )

    none_domain = none_summary["domains"]["robotics"]

    none_mean_violation = float(none_domain["violation_step_rate"]["mean"])

    lyapunov_mean_violation = float(statistics.mean(seed_violation_rates))

    reconstructed_reduction = (
        none_mean_violation - lyapunov_mean_violation
    ) / none_mean_violation

    _close(
        reconstructed_reduction,
        float(summary["effectiveness"]["domain_mean_violation_reduction"]),
        message=("domain mean violation reduction"),
    )

    clipping_rows = {
        int(row["principal_seed"]): row
        for row in clipping_summary["seed_level_comparisons"]
        if (row["domain"] == "robotics")
    }

    _require(
        set(clipping_rows) == set(PRINCIPAL_SEEDS),
        ("clipping evidence contains " "all robotics seeds"),
    )

    for seed in PRINCIPAL_SEEDS:
        none_rate = float(clipping_rows[seed]["none_violation_step_rate"])

        lyapunov_rate = seed_violation_rates[PRINCIPAL_SEEDS.index(seed)]

        if none_rate == 0.0:
            passes = lyapunov_rate <= none_rate
        else:
            passes = lyapunov_rate < none_rate

        _require(
            passes,
            ("frozen seed safety " f"criterion seed {seed}"),
        )

    none_reward = float(none_domain["reward"]["mean_of_seed_means"])

    lyapunov_reward = float(statistics.mean(seed_rewards))

    reward_degradation = max(
        0.0,
        none_reward - lyapunov_reward,
    ) / abs(none_reward)

    _close(
        reward_degradation,
        float(summary["effectiveness"]["reward_degradation_fraction"]),
        message=("reward degradation reconstruction"),
    )

    none_success = float(none_domain["success"]["mean_of_seed_rates"])

    lyapunov_success = float(statistics.mean(seed_success_rates))

    success_drop = max(
        0.0,
        none_success - lyapunov_success,
    )

    _close(
        success_drop,
        float(summary["effectiveness"]["success_rate_drop"]),
        message=("success drop reconstruction"),
    )

    _require(
        reconstructed_reduction >= 0.20,
        "mean violation reduction >= 20%",
    )

    _require(
        reward_degradation <= 0.10,
        "reward degradation <= 10%",
    )

    _require(
        success_drop <= 0.10,
        ("success-rate drop <= " "10 percentage points"),
    )

    _require(
        summary["effectiveness"]["robotics_lyapunov_empirical_effectiveness_supported"]
        is True,
        ("mechanical effectiveness " "is supported"),
    )

    clipping_mean = float(
        summary["three_method_summary"]["clipping"]["violation_step_rate"]["mean"]
    )

    lyapunov_mean = float(
        summary["three_method_summary"]["lyapunov"]["violation_step_rate"]["mean"]
    )

    _require(
        clipping_mean == 0.0 and lyapunov_mean == 0.0,
        ("clipping and Lyapunov both " "have zero clean violation-step rate"),
    )

    mechanism = summary["mechanism_accounting"]

    _require(
        int(mechanism["total_environment_steps"]) == total_environment_steps,
        ("summary mechanism total " "environment steps"),
    )

    _require(
        int(mechanism["domain_constraint_interventions"])
        == total_domain_constraint_interventions,
        ("domain-constraint intervention " "accounting"),
    )

    _require(
        total_domain_constraint_interventions == 88,
        ("principal domain-constraint " "interventions equal 88"),
    )

    _require(
        int(mechanism["lyapunov_decrease_interventions"])
        == total_lyapunov_decrease_interventions,
        ("Lyapunov-decrease intervention " "accounting"),
    )

    _require(
        total_lyapunov_decrease_interventions == 0,
        ("no principal " "LYAPUNOV_DECREASE intervention"),
    )

    _require(
        int(mechanism["emergency_fallbacks"]) == total_fallbacks == 0,
        "no emergency fallback observed",
    )

    _require(
        int(mechanism["selected_hard_guard_failures"])
        == total_selected_guard_failures
        == 0,
        ("no selected hard-guard " "failure observed"),
    )

    _require(
        int(mechanism["strict_lyapunov_decrease_steps"]) == total_strict_decreases == 0,
        ("strict Lyapunov decrease " "count is zero"),
    )

    _require(
        int(mechanism["lyapunov_nonincrease_steps"]) == total_nonincreases == 6000,
        ("all 6000 selected transitions " "are Lyapunov non-increasing"),
    )

    _require(
        int(mechanism["selected_lower_than_proposed_steps"])
        == total_selected_lower
        == 34,
        ("selected-lower-than-proposed " "count equals 34"),
    )

    interpretation = summary["mechanism_interpretation"]

    _require(
        interpretation["principal_lyapunov_decrease_interventions_observed"] is False,
        ("no principal LYAPUNOV_DECREASE " "intervention observed"),
    )

    _require(
        interpretation["all_selected_lyapunov_transitions_nonincreasing"] is True,
        ("selected transitions are " "Lyapunov non-increasing"),
    )

    _require(
        interpretation["emergency_fallback_observed"] is False,
        "no emergency fallback observed",
    )

    _require(
        interpretation["selected_hard_guard_failure_observed"] is False,
        ("no selected hard-guard " "failure observed"),
    )

    grasp_context = summary["grasp_context"]

    _require(
        int(grasp_context["object_grasped_true_steps"]) == 0,
        ("summary records zero " "grasped-state steps"),
    )

    _require(
        int(grasp_context["object_grasped_false_steps"]) == 6000,
        ("summary records 6000 " "not-grasped steps"),
    )

    _require(
        grasp_context["grasped_branch_exercised"] is False,
        ("grasped branch correctly " "marked unexercised"),
    )

    _require(
        grasp_context["principal_empirical_validation_of_grasped_branch"] is False,
        ("no grasped-branch empirical " "validation claim"),
    )

    claims = summary["claims"]

    for key in (
        "formal_stability_proven",
        "formal_robot_safety_guarantee",
        "production_manipulation_safety_claim",
        "iso_10218_claim",
        "iec_62061_claim",
        "lyapunov_superior_to_clipping_claim",
        "quantum_safety_advantage_claim",
        "grasped_state_branch_empirically_validated",
    ):
        _require(
            claims[key] is False,
            ("claim remains controlled: " f"{key}"),
        )

    _require(
        math.isfinite(
            float(
                summary["three_method_summary"]["lyapunov"]["mean_filter_latency_ms"][
                    "mean"
                ]
            )
        ),
        ("CPU proxy filter " "latency is finite"),
    )

    print()
    print("Protocol: PASS")
    print("Provenance: PASS")
    print("Policy fairness: PASS")
    print("Episode pairing: PASS")
    print("Safety accounting: PASS")
    print("Correction accounting: PASS")
    print("Candidate accounting: PASS")
    print("Lyapunov accounting: PASS")
    print("Fallback accounting: PASS")
    print("Grasp-context accounting: PASS")
    print("NONE comparison: PASS")
    print("CLIPPING comparison: PASS")
    print("Claim controls: PASS")
    print()
    print("Robotics empirical " "effectiveness: SUPPORTED")
    print()
    print("SPRINT 5.9 ROBOTICS " "LYAPUNOV SAFETY: PASS")


if __name__ == "__main__":
    main()
