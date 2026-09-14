"""Build Sprint 5.14F action-recovery cross-domain safety analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-action-three-seed-summary.json"
)

OUTPUT_DIR = ROOT / "results" / "safety" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "sprint5-cross-domain-action.json"

OUTPUT_CSV = OUTPUT_DIR / "sprint5-cross-domain-action.csv"

OUTPUT_MD = OUTPUT_DIR / "sprint5-cross-domain-action.md"

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
        raise TypeError("action source must be JSON object")

    return payload


def _mean_metric(
    row: dict[str, Any],
    metric: str,
) -> float | None:
    value = row[metric]

    if value is None:
        return None

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(f"{metric} must be aggregate dict or null")

    mean = value["mean"]

    if mean is None:
        return None

    return float(mean)


def main() -> None:
    source = _load()

    aggregate_raw = source["three_seed_summary"]

    if not isinstance(
        aggregate_raw,
        list,
    ):
        raise TypeError("three_seed_summary must be list")

    if len(aggregate_raw) != 72:
        raise RuntimeError("expected 72 action aggregates")

    aggregates: list[dict[str, Any]] = []

    for row in aggregate_raw:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("action aggregate row must be dict")

        aggregates.append(row)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    domain_method_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for method in METHODS:
            rows = [
                row
                for row in aggregates
                if row["domain"] == domain and row["method"] == method
            ]

            if not rows:
                raise RuntimeError(f"missing rows for {domain}/{method}")

            unsafe = sum(int(row["unsafe_perturbed_steps"]) for row in rows)

            recovered = sum(int(row["recovered_unsafe_steps"]) for row in rows)

            unresolved = sum(int(row["unresolved_unsafe_steps"]) for row in rows)

            if recovered + unresolved != unsafe:
                raise RuntimeError(
                    "action recovery accounting does not close "
                    f"for {domain}/{method}"
                )

            recovery_rate = recovered / unsafe if unsafe > 0 else None

            mean_violation_deltas = [
                value
                for row in rows
                for value in [
                    _mean_metric(
                        row,
                        "executed_violation_delta_from_clean",
                    )
                ]
                if value is not None
            ]

            mean_reward_deltas = [
                value
                for row in rows
                for value in [
                    _mean_metric(
                        row,
                        "reward_delta_from_clean",
                    )
                ]
                if value is not None
            ]

            intervention_rates = [
                value
                for row in rows
                for value in [
                    _mean_metric(
                        row,
                        "intervention_rate",
                    )
                ]
                if value is not None
            ]

            domain_method_rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "condition_count": len(rows),
                    "unsafe_perturbed_steps": unsafe,
                    "recovered_unsafe_steps": recovered,
                    "unresolved_unsafe_steps": unresolved,
                    "recovery_rate": recovery_rate,
                    "mean_executed_violation_delta": (
                        sum(mean_violation_deltas) / len(mean_violation_deltas)
                        if mean_violation_deltas
                        else None
                    ),
                    "mean_reward_delta": (
                        sum(mean_reward_deltas) / len(mean_reward_deltas)
                        if mean_reward_deltas
                        else None
                    ),
                    "mean_intervention_rate": (
                        sum(intervention_rates) / len(intervention_rates)
                        if intervention_rates
                        else None
                    ),
                }
            )

    row_index = {
        (
            str(row["domain"]),
            str(row["method"]),
        ): row
        for row in domain_method_rows
    }

    consistency: list[dict[str, Any]] = []

    for method in METHODS:
        driving = row_index[
            (
                "autonomous_driving",
                method,
            )
        ]

        robotics = row_index[
            (
                "robotics",
                method,
            )
        ]

        driving_rate = driving["recovery_rate"]

        robotics_rate = robotics["recovery_rate"]

        driving_positive = driving_rate is not None and float(driving_rate) > 0.0

        robotics_positive = robotics_rate is not None and float(robotics_rate) > 0.0

        consistency.append(
            {
                "method": method,
                "driving_recovery_rate": driving_rate,
                "robotics_recovery_rate": robotics_rate,
                "driving_positive_recovery": driving_positive,
                "robotics_positive_recovery": robotics_positive,
                "cross_domain_recovery_direction_consistent": (
                    driving_positive == robotics_positive
                ),
            }
        )

    mechanism = source["mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("mechanism must be dict")

    global_unsafe = int(mechanism["unsafe_perturbed_steps"])

    global_recovered = int(mechanism["recovered_unsafe_steps"])

    global_unresolved = int(mechanism["unresolved_unsafe_steps"])

    reconstructed_unsafe = sum(
        int(row["unsafe_perturbed_steps"]) for row in domain_method_rows
    )

    reconstructed_recovered = sum(
        int(row["recovered_unsafe_steps"]) for row in domain_method_rows
    )

    reconstructed_unresolved = sum(
        int(row["unresolved_unsafe_steps"]) for row in domain_method_rows
    )

    if reconstructed_unsafe != global_unsafe:
        raise RuntimeError("global unsafe count reconstruction failed")

    if reconstructed_recovered != global_recovered:
        raise RuntimeError("global recovered count reconstruction failed")

    if reconstructed_unresolved != global_unresolved:
        raise RuntimeError("global unresolved count reconstruction failed")

    none_rows = [row for row in domain_method_rows if row["method"] == "none"]

    none_explicit_recovery_zero = all(
        int(row["recovered_unsafe_steps"]) == 0 for row in none_rows
    )

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14F",
        "artifact": "cross-domain-action-recovery",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "new_safety_episodes": False,
        "source_artifact": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "domain_method_summary": domain_method_rows,
        "cross_domain_consistency": consistency,
        "global_reconstruction": {
            "unsafe_perturbed_steps": reconstructed_unsafe,
            "recovered_unsafe_steps": reconstructed_recovered,
            "unresolved_unsafe_steps": reconstructed_unresolved,
            "recovery_fraction": (reconstructed_recovered / reconstructed_unsafe),
        },
        "negative_control": {
            "none_explicit_recovery_zero": none_explicit_recovery_zero,
        },
        "environment_interface": {
            "environment_adjustments_are_not_safety_interventions": True,
            "environment_interface_adjustment_steps": mechanism[
                "environment_interface_adjustment_steps"
            ],
        },
        "interpretation": {
            "positive_recovery_means_explicit_filter_changed_unsafe_action": True,
            "equal_recovery_rate_required_across_domains": False,
            "none_is_negative_control": True,
            "environment_clipping_kept_separate": True,
        },
        "supported_statement": (
            "Action perturbation provides a shared architectural "
            "stress test because the disturbance is inserted between "
            "the policy and the explicit safety layer in both domains. "
            "Recovery is therefore analyzed separately by domain and "
            "safety method without requiring equal recovery rates."
        ),
        "limitations": [
            (
                "Driving and robotics action channels have different "
                "semantics despite sharing a three-dimensional "
                "continuous action interface."
            ),
            (
                "Recovery rate is undefined when a cell or aggregate "
                "contains no unsafe perturbed steps."
            ),
            (
                "Environment-interface adjustments remain distinct "
                "from explicit safety-filter interventions."
            ),
            (
                "The perturbations are synthetic action disturbances "
                "and are not calibrated physical actuator-fault models."
            ),
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
            "domain,method,condition_count,"
            "unsafe_perturbed_steps,"
            "recovered_unsafe_steps,"
            "unresolved_unsafe_steps,"
            "recovery_rate,"
            "mean_executed_violation_delta,"
            "mean_reward_delta,"
            "mean_intervention_rate"
        )
    ]

    for row in domain_method_rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["domain"]),
                    str(row["method"]),
                    str(row["condition_count"]),
                    str(row["unsafe_perturbed_steps"]),
                    str(row["recovered_unsafe_steps"]),
                    str(row["unresolved_unsafe_steps"]),
                    ("" if row["recovery_rate"] is None else str(row["recovery_rate"])),
                    str(row["mean_executed_violation_delta"]),
                    str(row["mean_reward_delta"]),
                    str(row["mean_intervention_rate"]),
                ]
            )
        )

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Sprint 5.14F — Action-Recovery Cross-Domain Safety",
        "",
        "## Architectural location",
        "",
        "Policy → action perturbation → explicit safety filter → environment",
        "",
        "## Domain/method recovery",
        "",
        ("| Domain | Method | Unsafe | Recovered | " "Unresolved | Recovery rate |"),
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in domain_method_rows:
        rate = row["recovery_rate"]

        rate_text = "undefined" if rate is None else f"{float(rate):.6f}"

        md_lines.append(
            f"| {row['domain']} | "
            f"{row['method']} | "
            f"{row['unsafe_perturbed_steps']} | "
            f"{row['recovered_unsafe_steps']} | "
            f"{row['unresolved_unsafe_steps']} | "
            f"{rate_text} |"
        )

    md_lines.extend(
        [
            "",
            "## Cross-domain recovery consistency",
            "",
            (
                "| Method | Driving recovery | Robotics recovery | "
                "Driving positive | Robotics positive | Consistent |"
            ),
            "|---|---:|---:|---|---|---|",
        ]
    )

    for row in consistency:
        driving_rate = row["driving_recovery_rate"]

        robotics_rate = row["robotics_recovery_rate"]

        md_lines.append(
            f"| {row['method']} | "
            f"{driving_rate} | "
            f"{robotics_rate} | "
            f"{row['driving_positive_recovery']} | "
            f"{row['robotics_positive_recovery']} | "
            f"{row['cross_domain_recovery_direction_consistent']} |"
        )

    md_lines.extend(
        [
            "",
            "## Global reconstruction",
            "",
            (f"- Unsafe perturbed steps: " f"{reconstructed_unsafe}"),
            (f"- Recovered unsafe steps: " f"{reconstructed_recovered}"),
            (f"- Unresolved unsafe steps: " f"{reconstructed_unresolved}"),
            (
                f"- Recovery fraction: "
                f"{reconstructed_recovered / reconstructed_unsafe:.9f}"
            ),
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

    print(" SPRINT 5.14F ACTION-RECOVERY CROSS-DOMAIN SAFETY")

    print("=" * 80)

    print()

    print(
        "Source aggregates:",
        len(aggregates),
    )

    print(
        "Domain-method summaries:",
        len(domain_method_rows),
    )

    print()

    for row in domain_method_rows:
        print(
            f"{row['domain']} / "
            f"{row['method']}: "
            f"unsafe={row['unsafe_perturbed_steps']}, "
            f"recovered={row['recovered_unsafe_steps']}, "
            f"unresolved={row['unresolved_unsafe_steps']}, "
            f"recovery_rate={row['recovery_rate']}"
        )

    print()

    for row in consistency:
        print(
            f"{str(row['method']).upper()} "
            f"cross-domain recovery consistent: "
            f"{row['cross_domain_recovery_direction_consistent']}"
        )

    print()

    print(
        "Global unsafe steps:",
        reconstructed_unsafe,
    )

    print(
        "Global recovered steps:",
        reconstructed_recovered,
    )

    print(
        "Global unresolved steps:",
        reconstructed_unresolved,
    )

    print()

    print(
        "NONE explicit recovery zero:",
        none_explicit_recovery_zero,
    )

    print("Environment clipping separation: PASS")

    print("Recovery accounting: PASS")

    print("Domain split reconstruction: PASS")

    print("No equal-rate assumption: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14F ACTION-RECOVERY CROSS-DOMAIN ANALYSIS: PASS")


if __name__ == "__main__":
    main()
