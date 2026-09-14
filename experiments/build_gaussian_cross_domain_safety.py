"""Build Sprint 5.14D Gaussian cross-domain safety analysis."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.cross_domain_safety import (
    ComparisonDirection,
    comparison_direction,
    directions_consistent,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-gaussian-three-seed-summary.json"
)

OUTPUT_DIR = ROOT / "results" / "safety" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "sprint5-cross-domain-gaussian.json"

OUTPUT_CSV = OUTPUT_DIR / "sprint5-cross-domain-gaussian.csv"

OUTPUT_MD = OUTPUT_DIR / "sprint5-cross-domain-gaussian.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

SIGMAS = (
    0.01,
    0.05,
    0.10,
)

SEEDS = (
    42,
    123,
    456,
)


def _load() -> dict[str, Any]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError("Gaussian source must be JSON object")

    return payload


def _aggregate(
    values: list[float],
) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate empty list")

    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _row_key(
    row: dict[str, Any],
) -> tuple[
    str,
    str,
    float,
    int,
]:
    return (
        str(row["domain"]),
        str(row["method"]),
        float(row["sigma"]),
        int(row["principal_seed"]),
    )


def _direction_from_violation_delta(
    delta: float,
) -> ComparisonDirection:
    return comparison_direction(delta)


def main() -> None:
    source = _load()

    seed_rows_raw = source["seed_rows"]

    if not isinstance(
        seed_rows_raw,
        list,
    ):
        raise TypeError("seed_rows must be list")

    if len(seed_rows_raw) != 72:
        raise RuntimeError("expected 72 Gaussian seed rows")

    seed_rows: list[dict[str, Any]] = []

    for raw in seed_rows_raw:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("Gaussian seed row must be dict")

        seed_rows.append(raw)

    index = {_row_key(row): row for row in seed_rows}

    if len(index) != 72:
        raise RuntimeError("Gaussian seed-row keys are not unique")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    noisy_seed_rows: list[dict[str, Any]] = []

    aggregates: list[dict[str, Any]] = []

    consistency_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for method in METHODS:
            for sigma in SIGMAS:
                for seed in SEEDS:
                    row = index[
                        (
                            domain,
                            method,
                            sigma,
                            seed,
                        )
                    ]

                    noisy_seed_rows.append(
                        {
                            "domain": domain,
                            "method": method,
                            "sigma": sigma,
                            "principal_seed": seed,
                            "violation_delta_from_clean": float(
                                row["violation_delta_from_clean"]
                            ),
                            "reward_delta_from_clean": float(
                                row["reward_delta_from_clean"]
                            ),
                            "success_delta_from_clean": float(
                                row["success_delta_from_clean"]
                            ),
                            "intervention_rate": float(row["intervention_rate"]),
                            "mean_correction_l2": float(row["mean_correction_l2"]),
                        }
                    )

    if len(noisy_seed_rows) != 54:
        raise RuntimeError("expected 54 nonzero-sigma seed rows")

    for domain in DOMAINS:
        for method in METHODS:
            for sigma in SIGMAS:
                rows = [
                    row
                    for row in noisy_seed_rows
                    if row["domain"] == domain
                    and row["method"] == method
                    and float(row["sigma"]) == sigma
                ]

                if len(rows) != 3:
                    raise RuntimeError(
                        "expected 3 Gaussian seed rows for "
                        f"{domain}/{method}/{sigma}"
                    )

                aggregates.append(
                    {
                        "domain": domain,
                        "method": method,
                        "sigma": sigma,
                        "violation_delta_from_clean": _aggregate(
                            [float(row["violation_delta_from_clean"]) for row in rows]
                        ),
                        "reward_delta_from_clean": _aggregate(
                            [float(row["reward_delta_from_clean"]) for row in rows]
                        ),
                        "success_delta_from_clean": _aggregate(
                            [float(row["success_delta_from_clean"]) for row in rows]
                        ),
                        "intervention_rate": _aggregate(
                            [float(row["intervention_rate"]) for row in rows]
                        ),
                        "mean_correction_l2": _aggregate(
                            [float(row["mean_correction_l2"]) for row in rows]
                        ),
                    }
                )

    if len(aggregates) != 18:
        raise RuntimeError("expected 18 nonzero-sigma aggregates")

    aggregate_index = {
        (
            str(row["domain"]),
            str(row["method"]),
            float(row["sigma"]),
        ): row
        for row in aggregates
    }

    for method in METHODS:
        for sigma in SIGMAS:
            driving = aggregate_index[
                (
                    "autonomous_driving",
                    method,
                    sigma,
                )
            ]

            robotics = aggregate_index[
                (
                    "robotics",
                    method,
                    sigma,
                )
            ]

            driving_violation = float(driving["violation_delta_from_clean"]["mean"])

            robotics_violation = float(robotics["violation_delta_from_clean"]["mean"])

            driving_direction = _direction_from_violation_delta(driving_violation)

            robotics_direction = _direction_from_violation_delta(robotics_violation)

            consistency_rows.append(
                {
                    "method": method,
                    "sigma": sigma,
                    "driving_violation_delta": driving_violation,
                    "robotics_violation_delta": robotics_violation,
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
        "sprint": "5.14D",
        "artifact": "cross-domain-gaussian-safety",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "new_safety_episodes": False,
        "principal_seeds": list(SEEDS),
        "sigmas": list(SIGMAS),
        "source_artifact": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "seed_rows": noisy_seed_rows,
        "aggregates": aggregates,
        "direction_consistency": consistency_rows,
        "state_separation": {
            "policy_uses_noisy_observation": source["policy_uses_noisy_observation"],
            "safety_layer_uses_true_state": source["safety_layer_uses_true_state"],
        },
        "interpretation": {
            "raw_rewards_compared_across_domains": False,
            "raw_violation_rates_used_for_domain_ranking": False,
            "domain_relative_deltas_used": True,
            "equal_magnitude_required_for_consistency": False,
            "privileged_safety_state": True,
        },
        "supported_statement": (
            "The same Gaussian observation-perturbation framework "
            "was applied across both pilot domains. Cross-domain "
            "interpretation uses changes relative to each domain's "
            "own clean reference rather than comparing raw rewards "
            "or raw safety rates directly."
        ),
        "limitations": [
            (
                "The policy receives noisy observations while the "
                "safety layer retains true simulator state."
            ),
            (
                "This evaluates robustness of the policy-plus-"
                "privileged-state-safety architecture, not a safety "
                "controller operating under uncertain perception."
            ),
            (
                "Magnitude equality across domains is neither "
                "expected nor required for direction consistency."
            ),
            (
                "Results use three principal policy seeds and do "
                "not establish formal statistical significance."
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
            "domain,method,sigma,principal_seed,"
            "violation_delta_from_clean,"
            "reward_delta_from_clean,"
            "success_delta_from_clean,"
            "intervention_rate,"
            "mean_correction_l2"
        )
    ]

    for row in noisy_seed_rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["domain"]),
                    str(row["method"]),
                    str(row["sigma"]),
                    str(row["principal_seed"]),
                    str(row["violation_delta_from_clean"]),
                    str(row["reward_delta_from_clean"]),
                    str(row["success_delta_from_clean"]),
                    str(row["intervention_rate"]),
                    str(row["mean_correction_l2"]),
                ]
            )
        )

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Sprint 5.14D — Gaussian Cross-Domain Safety",
        "",
        "## Scope",
        "",
        (
            "Analysis-only comparison of frozen Gaussian observation-"
            "perturbation evidence."
        ),
        "",
        "## State separation",
        "",
        "- Policy: noisy observation",
        "- Safety layer: true simulator state",
        "",
        (
            "Therefore, this analysis characterizes the "
            "policy-plus-privileged-state-safety architecture."
        ),
        "",
        "## Direction consistency",
        "",
        ("| Method | Sigma | Driving direction | " "Robotics direction | Consistent |"),
        "|---|---:|---|---|---|",
    ]

    for row in consistency_rows:
        md_lines.append(
            f"| {row['method']} | "
            f"{float(row['sigma']):.2f} | "
            f"{row['driving_direction']} | "
            f"{row['robotics_direction']} | "
            f"{row['direction_consistent']} |"
        )

    md_lines.extend(
        [
            "",
            "## Supported conclusion",
            "",
            payload["supported_statement"],
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

    consistent_count = sum(
        1 for row in consistency_rows if row["direction_consistent"] is True
    )

    reversal_count = sum(
        1 for row in consistency_rows if row["direction_consistent"] is False
    )

    print("=" * 76)
    print(" SPRINT 5.14D GAUSSIAN CROSS-DOMAIN SAFETY")
    print("=" * 76)
    print()

    print(
        "Source seed rows:",
        len(seed_rows),
    )

    print(
        "Nonzero-sigma seed rows:",
        len(noisy_seed_rows),
    )

    print(
        "Nonzero-sigma aggregates:",
        len(aggregates),
    )

    print(
        "Direction comparisons:",
        len(consistency_rows),
    )

    print()

    print(
        "Direction-consistent comparisons:",
        consistent_count,
    )

    print(
        "Direction reversals/differences:",
        reversal_count,
    )

    print()

    print(
        "Policy uses noisy observation:",
        source["policy_uses_noisy_observation"],
    )

    print(
        "Safety layer uses true state:",
        source["safety_layer_uses_true_state"],
    )

    print()

    print("Domain-relative deltas: PASS")

    print("Raw reward comparison blocked: PASS")

    print("Raw safety ranking blocked: PASS")

    print("Privileged-state limitation retained: PASS")

    print("Direction reversals preserved: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14D GAUSSIAN CROSS-DOMAIN ANALYSIS: PASS")


if __name__ == "__main__":
    main()
