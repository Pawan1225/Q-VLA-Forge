"""Independently verify Sprint 5.13D structured-state consolidation."""

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
    / "structured-state-robustness"
    / "sprint5-structured-state-robustness-summary.json"
)

SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-structured-state-three-seed-summary.json"
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
    "violation_step_rate": "violation_step_rate",
    "critical_violation_step_rate": "critical_violation_step_rate",
    "constraint_violation_rate": "constraint_violation_rate",
    "reward": "reward",
    "success_rate": "success_rate",
    "intervention_rate": "intervention_rate",
    "mean_correction_l2": "mean_action_correction_l2",
    "violation_delta_from_clean": "violation_delta_from_clean",
    "critical_delta_from_clean": "critical_delta_from_clean",
    "constraint_delta_from_clean": "constraint_delta_from_clean",
    "reward_delta_from_clean": "reward_delta_from_clean",
    "success_delta_from_clean": "success_delta_from_clean",
    "strict_lyapunov_decrease_rate": "strict_lyapunov_decrease_rate",
    "lyapunov_nonincrease_rate": "lyapunov_nonincrease_rate",
}


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
    print("=" * 86)
    print(" Q-VLA FORGE - SPRINT 5.13D STRUCTURED-STATE VERIFICATION")
    print("=" * 86)
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
        and len(source_cells) == 198,
        "198 frozen source cells",
    )

    _check(
        isinstance(
            summary_rows,
            list,
        )
        and len(summary_rows) == 198,
        "198 consolidated seed cells",
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
        len(source_map) == 198,
        "198 unique source keys",
    )

    for raw in summary_rows:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("summary seed row must be dict")

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

        _check(
            int(raw["steps_object_grasped"]) == int(source_row["steps_object_grasped"]),
            (
                f"{source_key[0]}/"
                f"{source_key[1]}/"
                f"{source_key[2]}/"
                f"seed-{source_key[3]} grasp steps"
            ),
        )

    aggregates = summary["three_seed_summary"]

    _check(
        isinstance(
            aggregates,
            list,
        )
        and len(aggregates) == 66,
        "66 three-seed aggregate rows",
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
        len(aggregate_map) == 66,
        "66 unique aggregate keys",
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
        len(identities) == 22,
        "22 semantic perturbations",
    )

    for (
        domain,
        perturbation_name,
    ) in identities:
        for method in METHODS:
            aggregate_key = (
                domain,
                method,
                perturbation_name,
            )

            aggregate = aggregate_map[aggregate_key]

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

    _check(
        int(summary["driving_perturbation_count"]) == 10,
        "10 driving perturbations",
    )

    _check(
        int(summary["robotics_perturbation_count"]) == 12,
        "12 robotics perturbations",
    )

    _check(
        int(summary["total_perturbation_count"]) == 22,
        "22 total perturbations",
    )

    _check(
        int(summary["principal_cells"]) == 198,
        "198 principal cells",
    )

    _check(
        int(summary["principal_episodes"]) == 3960,
        "3960 principal episodes",
    )

    _check(
        summary["policy_uses_perturbed_observation"] is True,
        "policy uses perturbed observation",
    )

    _check(
        summary["safety_layer_uses_true_state"] is True,
        "safety layer uses true state",
    )

    _check(
        summary["environment_uses_true_state"] is True,
        "environment uses true state",
    )

    mechanism = summary["lyapunov_mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("mechanism must be dict")

    reasons = mechanism["intervention_reason_counts"]

    if not isinstance(
        reasons,
        dict,
    ):
        raise TypeError("reason counts must be dict")

    _check(
        int(reasons["domain_constraint"]) == 25992,
        "25992 domain-constraint reasons",
    )

    _check(
        int(reasons["none"]) == 106008,
        "106008 no-intervention reasons",
    )

    _check(
        int(mechanism["strict_decrease_steps"]) == 0,
        "zero strict Lyapunov decreases",
    )

    _check(
        mechanism["lyapunov_decrease_interventions_observed"] is False,
        "Lyapunov-specific pathway inactive",
    )

    _check(
        int(mechanism["robotics_grasped_steps"]) == 0,
        "zero robotics grasped steps",
    )

    print()
    print("Seed reconstruction: PASS")
    print("Three-seed aggregation: PASS")
    print("Semantic perturbation identity: PASS")
    print("Corpus accounting: PASS")
    print("True-state separation: PASS")
    print("Lyapunov mechanism accounting: PASS")
    print("No scientific reruns: PASS")

    print()
    print("SPRINT 5.13D STRUCTURED-STATE VERIFICATION: PASS")


if __name__ == "__main__":
    main()
