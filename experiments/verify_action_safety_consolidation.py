"""Independently verify Sprint 5.13E action robustness consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "action-robustness"
    / "sprint5-action-robustness-summary.json"
)

SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-action-three-seed-summary.json"
)

SEEDS = (
    42,
    123,
    456,
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

SUMMARY_TO_SOURCE_METRIC = {
    "perturbed_violation_step_rate": "perturbed_violation_step_rate",
    "executed_violation_step_rate": "executed_violation_step_rate",
    "executed_constraint_violation_rate": "executed_constraint_violation_rate",
    "critical_violation_step_rate": "critical_violation_step_rate",
    "mean_reward": "mean_reward",
    "success_rate": "success_rate",
    "intervention_rate": "intervention_rate",
    "mean_safety_correction_l2": "mean_safety_correction_l2",
    "p95_safety_correction_l2": "p95_safety_correction_l2",
    "executed_violation_delta_from_clean": "executed_violation_delta_from_clean",
    "reward_delta_from_clean": "reward_delta_from_clean",
    "success_delta_from_clean": "success_delta_from_clean",
    "selected_lower_than_perturbed_rate": "selected_lower_than_perturbed_rate",
    "strict_lyapunov_decrease_rate": "strict_lyapunov_decrease_rate",
    "lyapunov_nonincrease_rate": "lyapunov_nonincrease_rate",
    "environment_interface_adjustment_rate": "environment_interface_adjustment_rate",
    "proposed_to_perturbed_gripper_semantic_change_rate": "proposed_to_perturbed_gripper_semantic_change_rate",
    "perturbed_to_executed_gripper_semantic_change_rate": "perturbed_to_executed_gripper_semantic_change_rate",
}

NULLABLE_METRICS = (
    "recovery_rate",
    "within_filter_violation_reduction",
)


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"expected dict: {path}")

    return payload


def _check(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def _close(
    actual: float,
    expected: float,
    *,
    tol: float = 1e-12,
) -> bool:
    return abs(actual - expected) <= tol


def _summary(
    values: list[float],
) -> tuple[float, float]:
    return (
        float(mean(values)),
        float(stdev(values)),
    )


def main() -> None:
    print("=" * 88)
    print(" Q-VLA FORGE - SPRINT 5.13E ACTION ROBUSTNESS VERIFICATION")
    print("=" * 88)
    print()

    source = _load(SOURCE)

    summary = _load(SUMMARY)

    _check(
        summary["analysis_only"] is True,
        "analysis-only scope",
    )

    _check(
        summary["new_training"] is False,
        "no new training",
    )

    _check(
        summary["new_principal_runs"] is False,
        "no new principal runs",
    )

    source_cells = source["cells"]

    summary_rows = summary["seed_rows"]

    _check(
        isinstance(
            source_cells,
            list,
        )
        and len(source_cells) == 216,
        "216 frozen source cells",
    )

    _check(
        isinstance(
            summary_rows,
            list,
        )
        and len(summary_rows) == 216,
        "216 consolidated seed cells",
    )

    source_map: dict[
        tuple[
            str,
            str,
            str,
            int,
        ],
        dict[str, Any],
    ] = {}

    for raw in source_cells:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("source cell must be dict")

        source_key = (
            str(raw["domain"]),
            str(raw["method"]),
            str(raw["perturbation_name"]),
            int(raw["principal_seed"]),
        )

        source_map[source_key] = raw

    _check(
        len(source_map) == 216,
        "216 unique source keys",
    )

    undefined_recovery = 0
    undefined_reduction = 0

    for raw in summary_rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("summary row must be dict")

        source_key = (
            str(raw["domain"]),
            str(raw["method"]),
            str(raw["perturbation_name"]),
            int(raw["principal_seed"]),
        )

        source_row = source_map[source_key]

        _check(
            str(raw["perturbation_family"]) == str(source_row["perturbation_family"]),
            (
                f"{source_key[0]}/"
                f"{source_key[1]}/"
                f"{source_key[2]}/"
                f"seed-{source_key[3]} family"
            ),
        )

        for (
            summary_metric,
            source_metric,
        ) in SUMMARY_TO_SOURCE_METRIC.items():
            _check(
                _close(
                    float(raw[summary_metric]),
                    float(source_row[source_metric]),
                ),
                (
                    f"{source_key[0]}/"
                    f"{source_key[1]}/"
                    f"{source_key[2]}/"
                    f"seed-{source_key[3]} "
                    f"{summary_metric}"
                ),
            )

        for metric in NULLABLE_METRICS:
            source_value = source_row[metric]

            summary_value = raw[metric]

            if source_value is None:
                _check(
                    summary_value is None,
                    (
                        f"{source_key[0]}/"
                        f"{source_key[1]}/"
                        f"{source_key[2]}/"
                        f"seed-{source_key[3]} "
                        f"{metric} null preserved"
                    ),
                )

                if metric == "recovery_rate":
                    undefined_recovery += 1
                else:
                    undefined_reduction += 1

            else:
                _check(
                    summary_value is not None
                    and _close(
                        float(summary_value),
                        float(source_value),
                    ),
                    (
                        f"{source_key[0]}/"
                        f"{source_key[1]}/"
                        f"{source_key[2]}/"
                        f"seed-{source_key[3]} "
                        f"{metric}"
                    ),
                )

        for count_metric in (
            "unsafe_perturbed_steps",
            "recovered_unsafe_steps",
            "unresolved_unsafe_steps",
            "steps_object_grasped",
            "interventions_while_grasped",
            "perturbed_unsafe_steps_while_grasped",
        ):
            _check(
                int(raw[count_metric]) == int(source_row[count_metric]),
                (
                    f"{source_key[0]}/"
                    f"{source_key[1]}/"
                    f"{source_key[2]}/"
                    f"seed-{source_key[3]} "
                    f"{count_metric}"
                ),
            )

    _check(
        undefined_recovery == 46,
        "46 undefined recovery-rate cells preserved",
    )

    _check(
        undefined_reduction == 46,
        "46 undefined violation-reduction cells preserved",
    )

    aggregates = summary["three_seed_summary"]

    _check(
        isinstance(
            aggregates,
            list,
        )
        and len(aggregates) == 72,
        "72 three-seed aggregate rows",
    )

    aggregate_map: dict[
        tuple[
            str,
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    for raw in aggregates:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("aggregate row must be dict")

        aggregate_key = (
            str(raw["domain"]),
            str(raw["method"]),
            str(raw["perturbation_name"]),
        )

        aggregate_map[aggregate_key] = raw

    _check(
        len(aggregate_map) == 72,
        "72 unique aggregate keys",
    )

    identities = sorted(
        {
            (
                str(raw["domain"]),
                str(raw["perturbation_name"]),
            )
            for raw in source_cells
            if isinstance(
                raw,
                dict,
            )
        }
    )

    _check(
        len(identities) == 24,
        "24 semantic action perturbations",
    )

    for (
        domain,
        perturbation_name,
    ) in identities:
        for method in METHODS:
            aggregate = aggregate_map[
                (
                    domain,
                    method,
                    perturbation_name,
                )
            ]

            for (
                summary_metric,
                source_metric,
            ) in SUMMARY_TO_SOURCE_METRIC.items():
                values = [
                    float(
                        source_map[
                            (
                                domain,
                                method,
                                perturbation_name,
                                seed,
                            )
                        ][source_metric]
                    )
                    for seed in SEEDS
                ]

                expected_mean, expected_sd = _summary(values)

                reported = aggregate[summary_metric]

                if not isinstance(
                    reported,
                    dict,
                ):
                    raise TypeError(f"{summary_metric} aggregate must be dict")

                _check(
                    _close(
                        float(reported["mean"]),
                        expected_mean,
                    ),
                    (
                        f"{domain}/{method}/"
                        f"{perturbation_name}/"
                        f"{summary_metric} mean"
                    ),
                )

                _check(
                    _close(
                        float(reported["sample_sd"]),
                        expected_sd,
                    ),
                    (
                        f"{domain}/{method}/"
                        f"{perturbation_name}/"
                        f"{summary_metric} SD"
                    ),
                )

            for metric in NULLABLE_METRICS:
                values_nullable = [
                    source_map[
                        (
                            domain,
                            method,
                            perturbation_name,
                            seed,
                        )
                    ][metric]
                    for seed in SEEDS
                ]

                defined = [
                    float(value) for value in values_nullable if value is not None
                ]

                reported = aggregate[metric]

                if not isinstance(
                    reported,
                    dict,
                ):
                    raise TypeError(f"{metric} aggregate must be dict")

                _check(
                    int(reported["defined_seed_count"]) == len(defined),
                    (
                        f"{domain}/{method}/"
                        f"{perturbation_name}/"
                        f"{metric} defined seed count"
                    ),
                )

                if not defined:
                    _check(
                        reported["mean"] is None and reported["sample_sd"] is None,
                        (
                            f"{domain}/{method}/"
                            f"{perturbation_name}/"
                            f"{metric} undefined aggregate"
                        ),
                    )

                else:
                    expected_mean = float(mean(defined))

                    expected_sd = 0.0 if len(defined) == 1 else float(stdev(defined))

                    _check(
                        _close(
                            float(reported["mean"]),
                            expected_mean,
                        ),
                        (
                            f"{domain}/{method}/"
                            f"{perturbation_name}/"
                            f"{metric} mean"
                        ),
                    )

                    _check(
                        _close(
                            float(reported["sample_sd"]),
                            expected_sd,
                        ),
                        (f"{domain}/{method}/" f"{perturbation_name}/" f"{metric} SD"),
                    )

    _check(
        int(summary["principal_cells"]) == 216,
        "216 principal cells",
    )

    _check(
        int(summary["principal_episodes"]) == 4320,
        "4320 principal episodes",
    )

    _check(
        int(summary["total_perturbation_count"]) == 24,
        "24 action perturbations",
    )

    mechanism = summary["mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("mechanism must be dict")

    _check(
        int(mechanism["unsafe_perturbed_steps"]) == 111341,
        "111341 unsafe perturbed steps",
    )

    _check(
        int(mechanism["recovered_unsafe_steps"]) == 69367,
        "69367 recovered unsafe steps",
    )

    _check(
        int(mechanism["unresolved_unsafe_steps"]) == 41974,
        "41974 unresolved unsafe steps",
    )

    _check(
        int(mechanism["recovered_unsafe_steps"])
        + int(mechanism["unresolved_unsafe_steps"])
        == int(mechanism["unsafe_perturbed_steps"]),
        "recovery accounting closes exactly",
    )

    expected_fraction = int(mechanism["recovered_unsafe_steps"]) / int(
        mechanism["unsafe_perturbed_steps"]
    )

    _check(
        _close(
            float(mechanism["overall_explicit_filter_recovery_fraction"]),
            expected_fraction,
        ),
        "overall recovery fraction",
    )

    reasons = mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        reasons,
        dict,
    ):
        raise TypeError("Lyapunov reasons must be dict")

    _check(
        int(reasons["action_bound"]) == 1493,
        "1493 action-bound reasons",
    )

    _check(
        int(reasons["domain_constraint"]) == 35729,
        "35729 domain-constraint reasons",
    )

    _check(
        int(reasons["lyapunov_decrease"]) == 1584,
        "1584 Lyapunov-decrease reasons",
    )

    _check(
        int(reasons["none"]) == 105194,
        "105194 no-intervention reasons",
    )

    _check(
        int(mechanism["strict_lyapunov_decrease_steps"]) == 70,
        "70 strict Lyapunov decreases",
    )

    _check(
        mechanism["strict_lyapunov_decrease_observed"] is True,
        "strict Lyapunov decrease observed",
    )

    _check(
        mechanism["lyapunov_decrease_interventions_observed"] is True,
        "Lyapunov-decrease intervention observed",
    )

    _check(
        int(mechanism["selected_lower_than_perturbed_steps"]) == 4887,
        "4887 selected-lower steps",
    )

    _check(
        int(mechanism["emergency_fallback_steps"]) == 0,
        "zero emergency fallback steps",
    )

    _check(
        int(mechanism["environment_interface_adjustment_steps"]) == 1579,
        "1579 environment-interface adjustments",
    )

    _check(
        int(mechanism["lyapunov_robotics_grasped_steps"]) == 4908,
        "4908 Lyapunov robotics grasped steps",
    )

    _check(
        int(mechanism["lyapunov_robotics_unsafe_while_grasped"]) == 1145,
        "1145 unsafe grasped Lyapunov steps",
    )

    _check(
        int(mechanism["lyapunov_robotics_interventions_while_grasped"]) == 1145,
        "1145 grasped-step interventions",
    )

    _check(
        int(mechanism["robotics_proposed_to_perturbed_gripper_semantic_changes"])
        == 26053,
        "26053 proposed-to-perturbed gripper changes",
    )

    _check(
        int(mechanism["robotics_perturbed_to_executed_gripper_semantic_changes"])
        == 6146,
        "6146 perturbed-to-executed gripper changes",
    )

    controls = summary["claim_controls"]

    if not isinstance(
        controls,
        dict,
    ):
        raise TypeError("claim controls must be dict")

    for key in (
        "actuator_fault_certification",
        "formal_lyapunov_stability",
        "formal_worst_case_robustness",
        "iso_safety_compliance",
        "lyapunov_superiority",
        "production_robustness",
        "quantum_safety_advantage",
    ):
        _check(
            controls[key] is False,
            f"claim control disabled: {key}",
        )

    print()
    print("Seed reconstruction: PASS")
    print("Nullable recovery semantics: PASS")
    print("Three-seed aggregation: PASS")
    print("Action perturbation identity: PASS")
    print("Recovery accounting: PASS")
    print("Lyapunov mechanism attribution: PASS")
    print("Robotics gripper accounting: PASS")
    print("Claim controls: PASS")
    print("No scientific reruns: PASS")

    print()
    print("SPRINT 5.13E ACTION ROBUSTNESS VERIFICATION: PASS")


if __name__ == "__main__":
    main()
