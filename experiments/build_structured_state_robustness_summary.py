"""Build Sprint 5.11 structured-state robustness summary."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "safety" / "structured-state-robustness" / "runs"

OUTPUT_ROOT = ROOT / "results" / "safety" / "structured-state-robustness"

CLEAN_SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "gaussian-robustness"
    / "sprint5-gaussian-robustness-summary.json"
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

SEEDS = (
    42,
    123,
    456,
)


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _load_clean_lookup() -> dict[
    tuple[str, str, int],
    dict[str, Any],
]:
    payload = json.loads(CLEAN_SUMMARY.read_text(encoding="utf-8"))

    lookup: dict[
        tuple[str, str, int],
        dict[str, Any],
    ] = {}

    for cell in payload["cells"]:
        if float(cell["sigma"]) != 0.0:
            continue

        key = (
            str(cell["domain"]),
            str(cell["method"]),
            int(cell["principal_seed"]),
        )

        lookup[key] = cell

    if len(lookup) != 18:
        raise RuntimeError(f"expected 18 frozen clean cells, " f"found {len(lookup)}")

    return lookup


def _load_structured_cells() -> tuple[
    list[dict[str, Any]],
    Counter[str],
    int,
    int,
]:
    files = sorted(RUNS.glob("*.json"))

    if len(files) != 198:
        raise RuntimeError(f"expected 198 principal files, " f"found {len(files)}")

    cells: list[dict[str, Any]] = []

    lyapunov_reasons: Counter[str] = Counter()

    strict_decrease_steps = 0
    grasped_steps = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        summary = payload["seed_summary"]

        cell = {
            "domain": payload["domain"],
            "perturbation_name": payload["perturbation_name"],
            "perturbation_family": payload["perturbation_family"],
            "perturbation_updates": payload["perturbation_updates"],
            "method": payload["method"],
            "principal_seed": payload["principal_seed"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "clean_reference_path": payload["clean_reference_path"],
            "clean_reference_sha256": payload["clean_reference_sha256"],
            "violation_step_rate": summary["executed_violation_step_rate"],
            "critical_violation_step_rate": summary["critical_violation_step_rate"],
            "constraint_violation_rate": summary["executed_constraint_violation_rate"],
            "reward": summary["mean_reward"],
            "reward_sample_sd": summary["reward_sample_sd"],
            "success_rate": summary["success_rate"],
            "mean_episode_length": summary["mean_episode_length"],
            "intervention_rate": summary["intervention_rate"],
            "mean_action_correction_l2": summary["mean_action_correction_l2"],
            "max_action_correction_l2": summary["max_action_correction_l2"],
            "mean_perturbation_l1": summary["mean_perturbation_l1"],
            "mean_perturbation_l2": summary["mean_perturbation_l2"],
            "max_perturbation_linf": summary["max_perturbation_linf"],
            "strict_lyapunov_decrease_rate": summary["strict_lyapunov_decrease_rate"],
            "lyapunov_nonincrease_rate": summary["lyapunov_nonincrease_rate"],
            "selected_lower_than_proposed_rate": summary[
                "selected_lower_than_proposed_rate"
            ],
            "emergency_fallback_rate": summary["emergency_fallback_rate"],
            "steps_object_grasped": summary["steps_object_grasped"],
            "steps_object_not_grasped": summary["steps_object_not_grasped"],
            "interventions_while_grasped": summary["interventions_while_grasped"],
            "category_violation_rates": summary["category_violation_rates"],
            "intervention_reason_rates": summary["intervention_reason_rates"],
            "selected_candidate_source_rates": summary[
                "selected_candidate_source_rates"
            ],
            "intervened_candidate_source_rates": summary[
                "intervened_candidate_source_rates"
            ],
            "source": str(path.relative_to(ROOT)),
        }

        cells.append(cell)

        if payload["method"] == "lyapunov":
            for episode in payload["episodes"]:
                lyapunov_reasons.update(
                    {
                        str(key): int(value)
                        for key, value in episode["intervention_reason_counts"].items()
                    }
                )

                strict_decrease_steps += int(episode["strict_lyapunov_decrease_count"])

                grasped_steps += int(episode["steps_object_grasped"])

    return (
        cells,
        lyapunov_reasons,
        strict_decrease_steps,
        grasped_steps,
    )


def _attach_clean_deltas(
    cells: list[dict[str, Any]],
    clean_lookup: dict[
        tuple[str, str, int],
        dict[str, Any],
    ],
) -> None:
    for cell in cells:
        key = (
            str(cell["domain"]),
            str(cell["method"]),
            int(cell["principal_seed"]),
        )

        clean = clean_lookup[key]

        cell["clean_violation_step_rate"] = clean["violation_step_rate"]

        cell["clean_critical_violation_step_rate"] = clean[
            "critical_violation_step_rate"
        ]

        cell["clean_constraint_violation_rate"] = clean["constraint_violation_rate"]

        cell["clean_reward"] = clean["reward"]

        cell["clean_success_rate"] = clean["success_rate"]

        cell["clean_mean_episode_length"] = clean["mean_episode_length"]

        cell["violation_delta_from_clean"] = (
            cell["violation_step_rate"] - clean["violation_step_rate"]
        )

        cell["critical_delta_from_clean"] = (
            cell["critical_violation_step_rate"] - clean["critical_violation_step_rate"]
        )

        cell["constraint_delta_from_clean"] = (
            cell["constraint_violation_rate"] - clean["constraint_violation_rate"]
        )

        cell["reward_delta_from_clean"] = cell["reward"] - clean["reward"]

        cell["success_delta_from_clean"] = cell["success_rate"] - clean["success_rate"]

        cell["episode_length_delta_from_clean"] = (
            cell["mean_episode_length"] - clean["mean_episode_length"]
        )

        clean_violation = float(clean["violation_step_rate"])

        if clean_violation > 0.0:
            cell["relative_violation_degradation"] = (
                cell["violation_delta_from_clean"] / clean_violation
            )
        else:
            cell["relative_violation_degradation"] = None


def _aggregate_cells(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for cell in cells:
        groups[
            (
                str(cell["domain"]),
                str(cell["perturbation_family"]),
                str(cell["perturbation_name"]),
                str(cell["method"]),
            )
        ].append(cell)

    metrics = (
        "violation_step_rate",
        "critical_violation_step_rate",
        "constraint_violation_rate",
        "reward",
        "success_rate",
        "mean_episode_length",
        "intervention_rate",
        "mean_action_correction_l2",
        "violation_delta_from_clean",
        "critical_delta_from_clean",
        "constraint_delta_from_clean",
        "reward_delta_from_clean",
        "success_delta_from_clean",
        "episode_length_delta_from_clean",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "selected_lower_than_proposed_rate",
        "emergency_fallback_rate",
    )

    output: list[dict[str, Any]] = []

    for key, group in sorted(groups.items()):
        if len(group) != 3:
            raise RuntimeError("aggregate does not contain " "three principal seeds")

        (
            domain,
            family,
            perturbation,
            method,
        ) = key

        row: dict[str, Any] = {
            "domain": domain,
            "perturbation_family": family,
            "perturbation_name": perturbation,
            "method": method,
            "principal_seed_count": 3,
        }

        for metric in metrics:
            values = [float(cell[metric]) for cell in group]

            row[metric] = {
                "mean": float(mean(values)),
                "sample_sd": (_sample_sd(values)),
            }

        row["max_action_correction_l2"] = max(
            float(cell["max_action_correction_l2"]) for cell in group
        )

        row["steps_object_grasped"] = sum(
            int(cell["steps_object_grasped"]) for cell in group
        )

        row["interventions_while_grasped"] = sum(
            int(cell["interventions_while_grasped"]) for cell in group
        )

        output.append(row)

    if len(output) != 66:
        raise RuntimeError(f"expected 66 aggregate rows, " f"found {len(output)}")

    return output


def _family_aggregates(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in aggregates:
        groups[
            (
                str(row["domain"]),
                str(row["perturbation_family"]),
                str(row["method"]),
            )
        ].append(row)

    output: list[dict[str, Any]] = []

    for (
        domain,
        family,
        method,
    ), rows in sorted(groups.items()):
        output.append(
            {
                "domain": domain,
                "perturbation_family": family,
                "method": method,
                "condition_count": len(rows),
                "mean_violation_delta": float(
                    mean(
                        [
                            float(row["violation_delta_from_clean"]["mean"])
                            for row in rows
                        ]
                    )
                ),
                "mean_reward_delta": float(
                    mean(
                        [float(row["reward_delta_from_clean"]["mean"]) for row in rows]
                    )
                ),
                "mean_success_delta": float(
                    mean(
                        [float(row["success_delta_from_clean"]["mean"]) for row in rows]
                    )
                ),
                "mean_intervention_rate": float(
                    mean([float(row["intervention_rate"]["mean"]) for row in rows])
                ),
            }
        )

    return output


def _sensitivity(
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[
        str,
        Any,
    ] = {}

    for domain in DOMAINS:
        result[domain] = {}

        for method in METHODS:
            rows = [
                row
                for row in aggregates
                if (row["domain"] == domain and row["method"] == method)
            ]

            worst_violation = max(
                rows,
                key=lambda row: float(row["violation_delta_from_clean"]["mean"]),
            )

            worst_reward = min(
                rows,
                key=lambda row: float(row["reward_delta_from_clean"]["mean"]),
            )

            worst_success = min(
                rows,
                key=lambda row: float(row["success_delta_from_clean"]["mean"]),
            )

            highest_intervention = max(
                rows,
                key=lambda row: float(row["intervention_rate"]["mean"]),
            )

            result[domain][method] = {
                "worst_violation": {
                    "perturbation": (worst_violation["perturbation_name"]),
                    "mean_delta": (
                        worst_violation["violation_delta_from_clean"]["mean"]
                    ),
                },
                "worst_reward": {
                    "perturbation": (worst_reward["perturbation_name"]),
                    "mean_delta": (worst_reward["reward_delta_from_clean"]["mean"]),
                },
                "worst_success": {
                    "perturbation": (worst_success["perturbation_name"]),
                    "mean_delta": (worst_success["success_delta_from_clean"]["mean"]),
                },
                "highest_intervention": {
                    "perturbation": (highest_intervention["perturbation_name"]),
                    "mean_rate": (highest_intervention["intervention_rate"]["mean"]),
                },
            }

    return result


def _directional_asymmetry(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (
            row["domain"],
            row["method"],
            row["perturbation_name"],
        ): row
        for row in aggregates
    }

    pairs = (
        (
            "autonomous_driving",
            "lane_offset_plus_0p05",
            "lane_offset_minus_0p05",
        ),
        (
            "autonomous_driving",
            "lane_offset_plus_0p10",
            "lane_offset_minus_0p10",
        ),
        (
            "autonomous_driving",
            "speed_plus_0p05",
            "speed_minus_0p05",
        ),
        (
            "autonomous_driving",
            "heading_error_plus_0p05",
            "heading_error_minus_0p05",
        ),
        (
            "robotics",
            "robot_x_plus_0p05",
            "robot_x_minus_0p05",
        ),
        (
            "robotics",
            "robot_y_plus_0p05",
            "robot_y_minus_0p05",
        ),
        (
            "robotics",
            "object_x_plus_0p05",
            "object_x_minus_0p05",
        ),
        (
            "robotics",
            "object_y_plus_0p05",
            "object_y_minus_0p05",
        ),
        (
            "robotics",
            "target_x_plus_0p05",
            "target_x_minus_0p05",
        ),
        (
            "robotics",
            "target_y_plus_0p05",
            "target_y_minus_0p05",
        ),
    )

    output: list[dict[str, Any]] = []

    for (
        domain,
        plus_name,
        minus_name,
    ) in pairs:
        for method in METHODS:
            plus = rows_by_key[
                (
                    domain,
                    method,
                    plus_name,
                )
            ]

            minus = rows_by_key[
                (
                    domain,
                    method,
                    minus_name,
                )
            ]

            output.append(
                {
                    "domain": domain,
                    "method": method,
                    "plus_condition": plus_name,
                    "minus_condition": minus_name,
                    "violation_delta_asymmetry_plus_minus": (
                        float(plus["violation_delta_from_clean"]["mean"])
                        - float(minus["violation_delta_from_clean"]["mean"])
                    ),
                    "reward_delta_asymmetry_plus_minus": (
                        float(plus["reward_delta_from_clean"]["mean"])
                        - float(minus["reward_delta_from_clean"]["mean"])
                    ),
                }
            )

    return output


def _write_csv(
    cells: list[dict[str, Any]],
    path: Path,
) -> None:
    columns = (
        "domain",
        "perturbation_family",
        "perturbation_name",
        "method",
        "principal_seed",
        "violation_step_rate",
        "clean_violation_step_rate",
        "violation_delta_from_clean",
        "critical_violation_step_rate",
        "constraint_violation_rate",
        "reward",
        "clean_reward",
        "reward_delta_from_clean",
        "success_rate",
        "clean_success_rate",
        "success_delta_from_clean",
        "intervention_rate",
        "mean_action_correction_l2",
        "max_action_correction_l2",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "selected_lower_than_proposed_rate",
        "emergency_fallback_rate",
        "steps_object_grasped",
        "interventions_while_grasped",
        "source",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for cell in cells:
            writer.writerow({key: cell.get(key) for key in columns})


def _write_markdown(
    *,
    sensitivity: dict[str, Any],
    lyapunov_reason_counts: Counter[str],
    strict_decrease_steps: int,
    grasped_steps: int,
    path: Path,
) -> None:
    lines = [
        "# Sprint 5.11 — Structured State Robustness",
        "",
        "## Protocol",
        "",
        "- 10 autonomous-driving structured perturbations.",
        "- 12 robotics structured perturbations.",
        "- NONE, CLIPPING, and LYAPUNOV.",
        "- Principal seeds 42, 123, and 456.",
        "- 20 held-out evaluation episodes per cell.",
        "- 198 principal cells and 3960 principal episodes.",
        "- Policy receives perturbed observation.",
        "- Safety layer and environment receive true simulator state.",
        "- No Gaussian noise, action perturbation, training, or filter tuning.",
        "",
        "## Feature sensitivity",
        "",
    ]

    for domain in DOMAINS:
        lines.append(f"### {domain}")
        lines.append("")

        for method in METHODS:
            item = sensitivity[domain][method]

            lines.append(
                f"- **{method}** worst violation: "
                f"`{item['worst_violation']['perturbation']}` "
                f"({item['worst_violation']['mean_delta']:+.6f}); "
                f"worst reward: "
                f"`{item['worst_reward']['perturbation']}` "
                f"({item['worst_reward']['mean_delta']:+.6f}); "
                f"worst success: "
                f"`{item['worst_success']['perturbation']}` "
                f"({item['worst_success']['mean_delta']:+.6f})."
            )

        lines.append("")

    lines.extend(
        [
            "## Lyapunov mechanism",
            "",
            (
                "- Intervention reason counts: "
                f"`{dict(sorted(lyapunov_reason_counts.items()))}`"
            ),
            ("- Strict Lyapunov-decrease steps: " f"{strict_decrease_steps}"),
            ("- Robotics grasped-state steps in Lyapunov runs: " f"{grasped_steps}"),
            "",
            "## Limitations",
            "",
            "- Structured perturbations are synthetic fixed coordinate offsets.",
            "- They are not calibrated production perception or sensor-fusion faults.",
            "- The safety layer retains privileged access to true simulator state.",
            "- No formal, certified, production, adversarial, ISO, or quantum robustness claim is made.",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    clean_lookup = _load_clean_lookup()

    (
        cells,
        lyapunov_reasons,
        strict_decrease_steps,
        grasped_steps,
    ) = _load_structured_cells()

    if len(cells) != 198:
        raise RuntimeError("expected 198 structured seed cells")

    _attach_clean_deltas(
        cells,
        clean_lookup,
    )

    aggregates = _aggregate_cells(cells)

    family_aggregates = _family_aggregates(aggregates)

    sensitivity = _sensitivity(aggregates)

    directional_asymmetry = _directional_asymmetry(aggregates)

    findings = {
        "lyapunov_intervention_reason_counts": dict(sorted(lyapunov_reasons.items())),
        "strict_lyapunov_decrease_steps": (strict_decrease_steps),
        "robotics_grasped_steps_in_lyapunov_runs": (grasped_steps),
        "lyapunov_decrease_interventions_observed": (
            lyapunov_reasons.get(
                "lyapunov_decrease",
                0,
            )
            + lyapunov_reasons.get(
                "LYAPUNOV_DECREASE",
                0,
            )
            > 0
        ),
    }

    payload = {
        "artifact": ("sprint5-structured-state-robustness-summary"),
        "sprint": "5.11",
        "condition": ("state_perturbation"),
        "principal_seeds": list(SEEDS),
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "driving_perturbation_count": 10,
        "robotics_perturbation_count": 12,
        "total_perturbation_count": 22,
        "principal_cells": 198,
        "principal_episodes": 3960,
        "clean_reference_source": str(CLEAN_SUMMARY.relative_to(ROOT)),
        "policy_uses_perturbed_observation": True,
        "safety_layer_uses_true_state": True,
        "environment_uses_true_state": True,
        "cells": cells,
        "aggregates": aggregates,
        "family_aggregates": family_aggregates,
        "directional_asymmetry": directional_asymmetry,
        "sensitivity": sensitivity,
        "findings": findings,
        "claims": {
            "real_sensor_fault_tolerance": False,
            "formal_adversarial_robustness": False,
            "worst_case_certified_robustness": False,
            "production_estimator_resilience": False,
            "iso_fault_tolerance": False,
            "quantum_robustness_advantage": False,
            "formal_safety_guarantee": False,
        },
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_ROOT / "sprint5-structured-state-robustness-summary.json"

    csv_path = OUTPUT_ROOT / "sprint5-structured-state-robustness-summary.csv"

    md_path = OUTPUT_ROOT / "sprint5-structured-state-robustness-summary.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(
        cells,
        csv_path,
    )

    _write_markdown(
        sensitivity=sensitivity,
        lyapunov_reason_counts=(lyapunov_reasons),
        strict_decrease_steps=(strict_decrease_steps),
        grasped_steps=grasped_steps,
        path=md_path,
    )

    print(
        "Structured seed cells:",
        len(cells),
    )

    print(
        "Aggregate perturbation/method rows:",
        len(aggregates),
    )

    print(
        "Family aggregate rows:",
        len(family_aggregates),
    )

    print()
    print(
        "Lyapunov reason counts:",
        findings["lyapunov_intervention_reason_counts"],
    )

    print(
        "Strict Lyapunov-decrease steps:",
        strict_decrease_steps,
    )

    print(
        "Lyapunov robotics grasped steps:",
        grasped_steps,
    )

    print()
    print("SPRINT 5.11 STRUCTURED " "STATE SUMMARY: PASS")


if __name__ == "__main__":
    main()
