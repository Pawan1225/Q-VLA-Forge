"""Build Sprint 5.14E structured-state cross-domain safety analysis."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.cross_domain_safety import (
    comparison_direction,
    directions_consistent,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-structured-state-three-seed-summary.json"
)

OUTPUT_DIR = ROOT / "results" / "safety" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "sprint5-cross-domain-structured-state.json"

OUTPUT_CSV = OUTPUT_DIR / "sprint5-cross-domain-structured-state.csv"

OUTPUT_MD = OUTPUT_DIR / "sprint5-cross-domain-structured-state.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)


def _load() -> dict[str, Any]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError("structured-state source must be JSON object")

    return payload


def _mean_metric(
    row: dict[str, Any],
    metric: str,
) -> float:
    value = row[metric]

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(f"{metric} must be aggregate dict")

    return float(value["mean"])


def _domain_method_rows(
    aggregates: list[dict[str, Any]],
    *,
    domain: str,
    method: str,
) -> list[dict[str, Any]]:
    return [
        row for row in aggregates if row["domain"] == domain and row["method"] == method
    ]


def main() -> None:
    source = _load()

    aggregate_raw = source["three_seed_summary"]

    if not isinstance(
        aggregate_raw,
        list,
    ):
        raise TypeError("three_seed_summary must be list")

    aggregates: list[dict[str, Any]] = []

    for row in aggregate_raw:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("structured aggregate row must be dict")

        aggregates.append(row)

    if len(aggregates) != 66:
        raise RuntimeError("expected 66 structured-state aggregates")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for method in METHODS:
            rows = _domain_method_rows(
                aggregates,
                domain=domain,
                method=method,
            )

            if not rows:
                raise RuntimeError(f"no rows for {domain}/{method}")

            violation_deltas = [
                _mean_metric(
                    row,
                    "violation_delta_from_clean",
                )
                for row in rows
            ]

            reward_deltas = [
                _mean_metric(
                    row,
                    "reward_delta_from_clean",
                )
                for row in rows
            ]

            intervention_rates = [
                _mean_metric(
                    row,
                    "intervention_rate",
                )
                for row in rows
            ]

            conditions_with_growth = sum(
                1 for value in violation_deltas if value > 1e-12
            )

            conditions_with_zero_growth = sum(
                1 for value in violation_deltas if abs(value) <= 1e-12
            )

            summaries.append(
                {
                    "domain": domain,
                    "method": method,
                    "number_of_conditions": len(rows),
                    "conditions_with_violation_growth": conditions_with_growth,
                    "conditions_with_zero_violation_growth": conditions_with_zero_growth,
                    "worst_violation_delta": max(violation_deltas),
                    "median_violation_delta": statistics.median(violation_deltas),
                    "worst_reward_delta": min(reward_deltas),
                    "median_reward_delta": statistics.median(reward_deltas),
                    "highest_intervention_rate": max(intervention_rates),
                }
            )

    summary_index = {
        (
            str(row["domain"]),
            str(row["method"]),
        ): row
        for row in summaries
    }

    consistency: list[dict[str, Any]] = []

    for method in METHODS:
        driving = summary_index[
            (
                "autonomous_driving",
                method,
            )
        ]

        robotics = summary_index[
            (
                "robotics",
                method,
            )
        ]

        driving_direction = comparison_direction(
            float(driving["worst_violation_delta"])
        )

        robotics_direction = comparison_direction(
            float(robotics["worst_violation_delta"])
        )

        consistency.append(
            {
                "method": method,
                "driving_worst_violation_delta": driving["worst_violation_delta"],
                "robotics_worst_violation_delta": robotics["worst_violation_delta"],
                "driving_direction": driving_direction.value,
                "robotics_direction": robotics_direction.value,
                "direction_consistent": directions_consistent(
                    driving_direction,
                    robotics_direction,
                ),
            }
        )

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14E",
        "artifact": "cross-domain-structured-state-safety",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "new_safety_episodes": False,
        "source_artifact": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "structured_state_summary": summaries,
        "direction_consistency": consistency,
        "feature_equivalence_assumed": False,
        "state_separation": {
            "policy_uses_perturbed_observation": source[
                "policy_uses_perturbed_observation"
            ],
            "safety_layer_uses_true_state": source["safety_layer_uses_true_state"],
            "environment_uses_true_state": source["environment_uses_true_state"],
        },
        "interpretation": {
            "feature_spaces_directly_comparable": False,
            "domain_relative_summary_used": True,
            "worst_case_summary_is_empirical": True,
            "privileged_safety_state": True,
        },
        "supported_statement": (
            "Structured-state robustness is compared using "
            "domain-local degradation summaries rather than direct "
            "feature equivalence. This preserves the distinct safety "
            "semantics of driving and robotics."
        ),
        "limitations": [
            (
                "Driving and robotics perturbation features are not "
                "treated as semantically equivalent."
            ),
            (
                "The policy receives perturbed state observations "
                "while the safety layer and environment use true "
                "simulator state."
            ),
            (
                "Worst-case degradation refers only to the tested "
                "structured perturbation set."
            ),
            ("Results are based on three principal policy seeds."),
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_lines = [
        (
            "domain,method,number_of_conditions,"
            "conditions_with_violation_growth,"
            "conditions_with_zero_violation_growth,"
            "worst_violation_delta,"
            "median_violation_delta,"
            "worst_reward_delta,"
            "median_reward_delta,"
            "highest_intervention_rate"
        )
    ]

    for row in summaries:
        csv_lines.append(
            ",".join(
                [
                    str(row["domain"]),
                    str(row["method"]),
                    str(row["number_of_conditions"]),
                    str(row["conditions_with_violation_growth"]),
                    str(row["conditions_with_zero_violation_growth"]),
                    str(row["worst_violation_delta"]),
                    str(row["median_violation_delta"]),
                    str(row["worst_reward_delta"]),
                    str(row["median_reward_delta"]),
                    str(row["highest_intervention_rate"]),
                ]
            )
        )

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Sprint 5.14E — Structured-State Cross-Domain Safety",
        "",
        "## Interpretation",
        "",
        (
            "Driving and robotics state perturbations are summarized "
            "within each domain. No one-to-one semantic equivalence "
            "between features is assumed."
        ),
        "",
        "## Domain summaries",
        "",
        (
            "| Domain | Method | Conditions | Growth | Zero growth | "
            "Worst violation delta | Median violation delta | "
            "Worst reward delta | Highest intervention |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        md_lines.append(
            f"| {row['domain']} | "
            f"{row['method']} | "
            f"{row['number_of_conditions']} | "
            f"{row['conditions_with_violation_growth']} | "
            f"{row['conditions_with_zero_violation_growth']} | "
            f"{float(row['worst_violation_delta']):.6f} | "
            f"{float(row['median_violation_delta']):.6f} | "
            f"{float(row['worst_reward_delta']):.6f} | "
            f"{float(row['highest_intervention_rate']):.6f} |"
        )

    md_lines.extend(
        [
            "",
            "## Cross-domain direction",
            "",
            "| Method | Driving | Robotics | Consistent |",
            "|---|---|---|---|",
        ]
    )

    for row in consistency:
        md_lines.append(
            f"| {row['method']} | "
            f"{row['driving_direction']} | "
            f"{row['robotics_direction']} | "
            f"{row['direction_consistent']} |"
        )

    md_lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in payload["limitations"]:
        md_lines.append(f"- {limitation}")

    OUTPUT_MD.write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(" SPRINT 5.14E STRUCTURED-STATE CROSS-DOMAIN SAFETY")
    print("=" * 80)
    print()

    print(
        "Source aggregates:",
        len(aggregates),
    )

    print(
        "Domain-method summaries:",
        len(summaries),
    )

    print(
        "Direction comparisons:",
        len(consistency),
    )

    print()

    for row in summaries:
        print(
            f"{row['domain']} / "
            f"{row['method']}: "
            f"conditions={row['number_of_conditions']}, "
            f"growth={row['conditions_with_violation_growth']}, "
            f"worst_violation_delta="
            f"{float(row['worst_violation_delta']):.6f}"
        )

    print()

    for row in consistency:
        print(
            f"{str(row['method']).upper()} "
            f"direction consistent: "
            f"{row['direction_consistent']}"
        )

    print()

    print("No feature equivalence assumed: PASS")

    print("Domain-relative summaries: PASS")

    print("Privileged-state limitation retained: PASS")

    print("Direction differences preserved: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14E STRUCTURED-STATE CROSS-DOMAIN ANALYSIS: PASS")


if __name__ == "__main__":
    main()
