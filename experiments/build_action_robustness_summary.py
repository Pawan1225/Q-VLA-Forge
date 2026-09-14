"""Build Sprint 5.12 action-robustness summary artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = ROOT / "results" / "safety" / "action-robustness"

RUN_ROOT = RESULT_ROOT / "runs"

GAUSSIAN_SUMMARY = (
    ROOT
    / "results"
    / "safety"
    / "gaussian-robustness"
    / "sprint5-gaussian-robustness-summary.json"
)

JSON_OUT = RESULT_ROOT / "sprint5-action-robustness-summary.json"

CSV_OUT = RESULT_ROOT / "sprint5-action-robustness-summary.csv"

MD_OUT = RESULT_ROOT / "sprint5-action-robustness-summary.md"


def _sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(stdev(values))


def _metric(
    mapping: dict[str, Any],
    *names: str,
) -> float:
    for name in names:
        if name in mapping:
            return float(mapping[name])

    raise KeyError(f"none of metric names found: {names}")


def _clean_lookup() -> dict[
    tuple[str, str, int],
    dict[str, float],
]:
    payload = json.loads(GAUSSIAN_SUMMARY.read_text(encoding="utf-8"))

    cells = payload["cells"]

    lookup: dict[
        tuple[str, str, int],
        dict[str, float],
    ] = {}

    for cell in cells:
        sigma = float(
            cell.get(
                "sigma",
                cell.get(
                    "noise_std",
                    -1.0,
                ),
            )
        )

        if sigma != 0.0:
            continue

        key = (
            str(cell["domain"]),
            str(cell["method"]),
            int(cell["principal_seed"]),
        )

        lookup[key] = {
            "violation": _metric(
                cell,
                "executed_violation_step_rate",
                "violation_step_rate",
            ),
            "critical": _metric(
                cell,
                "critical_violation_step_rate",
            ),
            "constraint": _metric(
                cell,
                "executed_constraint_violation_rate",
                "constraint_violation_rate",
            ),
            "reward": _metric(
                cell,
                "mean_reward",
                "reward",
            ),
            "success": _metric(
                cell,
                "success_rate",
            ),
            "episode_length": _metric(
                cell,
                "mean_episode_length",
            ),
        }

    if len(lookup) != 18:
        raise RuntimeError(
            "expected 18 normalized clean seed cells, " f"found {len(lookup)}"
        )

    return lookup


def _aggregate_metric(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float]:
    values = [float(row[field]) for row in rows]

    return {
        "mean": float(mean(values)),
        "sample_sd": _sample_sd(values),
    }


def _optional_aggregate(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row[field] is not None]

    if not values:
        return {
            "mean": None,
            "sample_sd": None,
        }

    return {
        "mean": float(mean(values)),
        "sample_sd": _sample_sd(values),
    }


def main() -> None:
    files = sorted(RUN_ROOT.glob("*.json"))

    if len(files) != 216:
        raise RuntimeError(f"expected 216 raw files, found {len(files)}")

    clean = _clean_lookup()

    cells: list[dict[str, Any]] = []

    lyapunov_reason_counts: Counter[str] = Counter()

    lyapunov_source_counts: Counter[str] = Counter()

    strict_steps = 0
    lyapunov_nonincrease_steps = 0
    selected_lower_steps = 0
    fallback_steps = 0

    robotics_grasped_steps = 0
    unsafe_while_grasped = 0
    interventions_while_grasped = 0

    gripper_proposed_to_perturbed = 0
    gripper_perturbed_to_executed = 0

    interface_adjustment_steps = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        summary = payload["seed_summary"]

        domain = str(payload["domain"])

        method = str(payload["method"])

        seed = int(payload["principal_seed"])

        clean_cell = clean[
            (
                domain,
                method,
                seed,
            )
        ]

        perturbed_rate = float(summary["perturbed_violation_step_rate"])

        executed_rate = float(summary["executed_violation_step_rate"])

        recovery_rate = summary["recovery_rate"]

        within_reduction = summary["within_filter_violation_reduction"]

        row = {
            "domain": domain,
            "method": method,
            "principal_seed": seed,
            "perturbation_name": str(payload["perturbation_name"]),
            "perturbation_family": str(payload["perturbation_family"]),
            "perturbation_updates": payload["perturbation_updates"],
            "episode_count": int(summary["episode_count"]),
            "total_environment_steps": int(summary["total_environment_steps"]),
            "perturbed_violation_step_rate": (perturbed_rate),
            "executed_violation_step_rate": (executed_rate),
            "critical_violation_step_rate": float(
                summary["critical_violation_step_rate"]
            ),
            "perturbed_constraint_violation_rate": float(
                summary["perturbed_constraint_violation_rate"]
            ),
            "executed_constraint_violation_rate": float(
                summary["executed_constraint_violation_rate"]
            ),
            "unsafe_perturbed_steps": int(summary["unsafe_perturbed_steps"]),
            "recovered_unsafe_steps": int(summary["recovered_unsafe_steps"]),
            "unresolved_unsafe_steps": int(summary["unresolved_unsafe_steps"]),
            "recovery_rate": (None if recovery_rate is None else float(recovery_rate)),
            "within_filter_violation_reduction": (
                None if within_reduction is None else float(within_reduction)
            ),
            "mean_reward": float(summary["mean_reward"]),
            "reward_sample_sd": float(summary["reward_sample_sd"]),
            "success_rate": float(summary["success_rate"]),
            "mean_episode_length": float(summary["mean_episode_length"]),
            "intervention_rate": float(summary["intervention_rate"]),
            "mean_action_perturbation_l2": float(
                summary["mean_action_perturbation_l2"]
            ),
            "max_action_perturbation_linf": float(
                summary["max_action_perturbation_linf"]
            ),
            "mean_safety_correction_l2": float(summary["mean_safety_correction_l2"]),
            "p95_safety_correction_l2": float(summary["p95_safety_correction_l2"]),
            "max_safety_correction_l2": float(summary["max_safety_correction_l2"]),
            "environment_interface_adjustment_rate": float(
                summary["environment_interface_adjustment_rate"]
            ),
            "mean_environment_interface_adjustment_l2": float(
                summary["mean_environment_interface_adjustment_l2"]
            ),
            "strict_lyapunov_decrease_rate": float(
                summary["strict_lyapunov_decrease_rate"]
            ),
            "lyapunov_nonincrease_rate": float(summary["lyapunov_nonincrease_rate"]),
            "selected_lower_than_perturbed_rate": float(
                summary["selected_lower_than_perturbed_rate"]
            ),
            "emergency_fallback_rate": float(summary["emergency_fallback_rate"]),
            "steps_object_grasped": int(summary["steps_object_grasped"]),
            "episodes_with_object_grasped": int(
                summary["episodes_with_object_grasped"]
            ),
            "perturbed_unsafe_steps_while_grasped": int(
                summary["perturbed_unsafe_steps_while_grasped"]
            ),
            "interventions_while_grasped": int(summary["interventions_while_grasped"]),
            "proposed_to_perturbed_gripper_semantic_change_rate": float(
                summary["proposed_to_perturbed_gripper_semantic_change_rate"]
            ),
            "perturbed_to_executed_gripper_semantic_change_rate": float(
                summary["perturbed_to_executed_gripper_semantic_change_rate"]
            ),
            "clean_violation_step_rate": clean_cell["violation"],
            "executed_violation_delta_from_clean": (
                executed_rate - clean_cell["violation"]
            ),
            "clean_critical_violation_step_rate": clean_cell["critical"],
            "critical_violation_delta_from_clean": (
                float(summary["critical_violation_step_rate"]) - clean_cell["critical"]
            ),
            "clean_constraint_violation_rate": clean_cell["constraint"],
            "executed_constraint_delta_from_clean": (
                float(summary["executed_constraint_violation_rate"])
                - clean_cell["constraint"]
            ),
            "clean_reward": clean_cell["reward"],
            "reward_delta_from_clean": (
                float(summary["mean_reward"]) - clean_cell["reward"]
            ),
            "clean_success_rate": clean_cell["success"],
            "success_delta_from_clean": (
                float(summary["success_rate"]) - clean_cell["success"]
            ),
            "clean_mean_episode_length": clean_cell["episode_length"],
            "episode_length_delta_from_clean": (
                float(summary["mean_episode_length"]) - clean_cell["episode_length"]
            ),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "source": str(path.relative_to(ROOT)),
        }

        cells.append(row)

        interface_adjustment_steps += round(
            row["environment_interface_adjustment_rate"]
            * row["total_environment_steps"]
        )

        if method == "lyapunov":
            for episode in payload["episodes"]:
                lyapunov_reason_counts.update(episode["intervention_reason_counts"])

                lyapunov_source_counts.update(
                    episode["selected_candidate_source_counts"]
                )

                strict_steps += int(episode["strict_lyapunov_decrease_count"])

                lyapunov_nonincrease_steps += int(episode["lyapunov_nonincrease_count"])

                selected_lower_steps += int(
                    episode["selected_lower_than_perturbed_count"]
                )

                fallback_steps += int(episode["emergency_fallback_count"])

            if domain == "robotics":
                robotics_grasped_steps += int(summary["steps_object_grasped"])

                unsafe_while_grasped += int(
                    summary["perturbed_unsafe_steps_while_grasped"]
                )

                interventions_while_grasped += int(
                    summary["interventions_while_grasped"]
                )

        if domain == "robotics":
            gripper_proposed_to_perturbed += round(
                float(summary["proposed_to_perturbed_gripper_semantic_change_rate"])
                * int(summary["total_environment_steps"])
            )

            gripper_perturbed_to_executed += round(
                float(summary["perturbed_to_executed_gripper_semantic_change_rate"])
                * int(summary["total_environment_steps"])
            )

    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in cells:
        groups[
            (
                row["domain"],
                row["perturbation_name"],
                row["method"],
            )
        ].append(row)

    aggregates: list[dict[str, Any]] = []

    for (
        domain,
        perturbation_name,
        method,
    ), rows in sorted(groups.items()):
        if len(rows) != 3:
            raise RuntimeError("each aggregate must contain three seeds")

        aggregates.append(
            {
                "domain": domain,
                "perturbation_name": (perturbation_name),
                "perturbation_family": rows[0]["perturbation_family"],
                "method": method,
                "seed_count": 3,
                "perturbed_violation_step_rate": (
                    _aggregate_metric(
                        rows,
                        "perturbed_violation_step_rate",
                    )
                ),
                "executed_violation_step_rate": (
                    _aggregate_metric(
                        rows,
                        "executed_violation_step_rate",
                    )
                ),
                "executed_violation_delta_from_clean": (
                    _aggregate_metric(
                        rows,
                        "executed_violation_delta_from_clean",
                    )
                ),
                "mean_reward": (
                    _aggregate_metric(
                        rows,
                        "mean_reward",
                    )
                ),
                "reward_delta_from_clean": (
                    _aggregate_metric(
                        rows,
                        "reward_delta_from_clean",
                    )
                ),
                "success_rate": (
                    _aggregate_metric(
                        rows,
                        "success_rate",
                    )
                ),
                "success_delta_from_clean": (
                    _aggregate_metric(
                        rows,
                        "success_delta_from_clean",
                    )
                ),
                "intervention_rate": (
                    _aggregate_metric(
                        rows,
                        "intervention_rate",
                    )
                ),
                "recovery_rate": (
                    _optional_aggregate(
                        rows,
                        "recovery_rate",
                    )
                ),
                "within_filter_violation_reduction": (
                    _optional_aggregate(
                        rows,
                        "within_filter_violation_reduction",
                    )
                ),
                "mean_safety_correction_l2": (
                    _aggregate_metric(
                        rows,
                        "mean_safety_correction_l2",
                    )
                ),
                "environment_interface_adjustment_rate": (
                    _aggregate_metric(
                        rows,
                        "environment_interface_adjustment_rate",
                    )
                ),
                "strict_lyapunov_decrease_rate": (
                    _aggregate_metric(
                        rows,
                        "strict_lyapunov_decrease_rate",
                    )
                ),
                "selected_lower_than_perturbed_rate": (
                    _aggregate_metric(
                        rows,
                        "selected_lower_than_perturbed_rate",
                    )
                ),
            }
        )

    if len(aggregates) != 72:
        raise RuntimeError(f"expected 72 aggregate rows, got {len(aggregates)}")

    family_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in cells:
        family_groups[
            (
                row["domain"],
                row["perturbation_family"],
                row["method"],
            )
        ].append(row)

    family_aggregates: list[dict[str, Any]] = []

    for (
        domain,
        family,
        method,
    ), rows in sorted(family_groups.items()):
        unsafe = sum(int(row["unsafe_perturbed_steps"]) for row in rows)

        recovered = sum(int(row["recovered_unsafe_steps"]) for row in rows)

        family_aggregates.append(
            {
                "domain": domain,
                "family": family,
                "method": method,
                "cell_count": len(rows),
                "executed_violation_delta_from_clean_mean": float(
                    mean(
                        [
                            float(row["executed_violation_delta_from_clean"])
                            for row in rows
                        ]
                    )
                ),
                "reward_delta_from_clean_mean": float(
                    mean([float(row["reward_delta_from_clean"]) for row in rows])
                ),
                "success_delta_from_clean_mean": float(
                    mean([float(row["success_delta_from_clean"]) for row in rows])
                ),
                "intervention_rate_mean": float(
                    mean([float(row["intervention_rate"]) for row in rows])
                ),
                "unsafe_perturbed_steps": (unsafe),
                "recovered_unsafe_steps": (recovered),
                "weighted_recovery_rate": (None if unsafe == 0 else recovered / unsafe),
            }
        )

    if len(family_aggregates) != 18:
        raise RuntimeError(
            "expected 18 family aggregate rows, " f"got {len(family_aggregates)}"
        )

    sensitivity: list[dict[str, Any]] = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            rows = [
                row
                for row in aggregates
                if row["domain"] == domain and row["method"] == method
            ]

            worst_safety = max(
                rows,
                key=lambda row: float(
                    row["executed_violation_delta_from_clean"]["mean"]
                ),
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

            best_recovery_candidates = [
                row for row in rows if row["recovery_rate"]["mean"] is not None
            ]

            best_recovery = (
                None
                if not best_recovery_candidates
                else max(
                    best_recovery_candidates,
                    key=lambda row: float(row["recovery_rate"]["mean"]),
                )
            )

            sensitivity.append(
                {
                    "domain": domain,
                    "method": method,
                    "worst_executed_safety": {
                        "perturbation": (worst_safety["perturbation_name"]),
                        "delta": (
                            worst_safety["executed_violation_delta_from_clean"]["mean"]
                        ),
                    },
                    "worst_reward": {
                        "perturbation": (worst_reward["perturbation_name"]),
                        "delta": (worst_reward["reward_delta_from_clean"]["mean"]),
                    },
                    "worst_success": {
                        "perturbation": (worst_success["perturbation_name"]),
                        "delta": (worst_success["success_delta_from_clean"]["mean"]),
                    },
                    "highest_intervention": {
                        "perturbation": (highest_intervention["perturbation_name"]),
                        "rate": (highest_intervention["intervention_rate"]["mean"]),
                    },
                    "highest_recovery": (
                        None
                        if best_recovery is None
                        else {
                            "perturbation": (best_recovery["perturbation_name"]),
                            "rate": (best_recovery["recovery_rate"]["mean"]),
                        }
                    ),
                }
            )

    total_unsafe = sum(int(row["unsafe_perturbed_steps"]) for row in cells)

    total_recovered = sum(int(row["recovered_unsafe_steps"]) for row in cells)

    total_unresolved = sum(int(row["unresolved_unsafe_steps"]) for row in cells)

    findings = {
        "unsafe_perturbed_steps": (total_unsafe),
        "recovered_unsafe_steps": (total_recovered),
        "unresolved_unsafe_steps": (total_unresolved),
        "overall_explicit_filter_recovery_fraction": (
            None if total_unsafe == 0 else total_recovered / total_unsafe
        ),
        "lyapunov_intervention_reason_counts": dict(
            sorted(lyapunov_reason_counts.items())
        ),
        "lyapunov_candidate_source_counts": dict(
            sorted(lyapunov_source_counts.items())
        ),
        "strict_lyapunov_decrease_steps": (strict_steps),
        "lyapunov_nonincrease_steps": (lyapunov_nonincrease_steps),
        "selected_lower_than_perturbed_steps": (selected_lower_steps),
        "emergency_fallback_steps": (fallback_steps),
        "lyapunov_decrease_interventions_observed": (
            lyapunov_reason_counts.get(
                "lyapunov_decrease",
                0,
            )
            > 0
        ),
        "strict_lyapunov_decrease_observed": (strict_steps > 0),
        "environment_interface_adjustment_steps": (interface_adjustment_steps),
        "lyapunov_robotics_grasped_steps": (robotics_grasped_steps),
        "lyapunov_robotics_unsafe_while_grasped": (unsafe_while_grasped),
        "lyapunov_robotics_interventions_while_grasped": (interventions_while_grasped),
        "robotics_proposed_to_perturbed_gripper_semantic_changes": (
            gripper_proposed_to_perturbed
        ),
        "robotics_perturbed_to_executed_gripper_semantic_changes": (
            gripper_perturbed_to_executed
        ),
    }

    claims = {
        "actuator_fault_certification": False,
        "production_robustness": False,
        "formal_lyapunov_stability": False,
        "formal_worst_case_robustness": False,
        "lyapunov_superiority": False,
        "iso_safety_compliance": False,
        "quantum_safety_advantage": False,
    }

    output = {
        "sprint": "5.12",
        "condition": "action_perturbation",
        "principal_cells": 216,
        "principal_episodes": 4320,
        "total_perturbation_count": 24,
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "methods": [
            "none",
            "clipping",
            "lyapunov",
        ],
        "principal_seeds": [
            42,
            123,
            456,
        ],
        "evaluation_seeds": list(
            range(
                20_000,
                20_020,
            )
        ),
        "cells": cells,
        "aggregates": aggregates,
        "family_aggregates": (family_aggregates),
        "feature_sensitivity": (sensitivity),
        "findings": findings,
        "claims": claims,
        "limitations": [
            (
                "Synthetic proxy environments only; "
                "this is not production actuator-fault evidence."
            ),
            (
                "The environments internally clip submitted actions "
                "to nominal bounds. This behavior is accounted "
                "separately and is not attributed to the explicit "
                "safety filters."
            ),
            (
                "Safety filters receive true simulator state; "
                "state-estimation uncertainty is not combined with "
                "action perturbation in Sprint 5.12."
            ),
            (
                "Strict Lyapunov decrease is empirical mechanism "
                "evidence and is not a formal stability guarantee."
            ),
        ],
    }

    JSON_OUT.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "domain",
        "perturbation_name",
        "perturbation_family",
        "method",
        "principal_seed",
        "perturbed_violation_step_rate",
        "executed_violation_step_rate",
        "executed_violation_delta_from_clean",
        "unsafe_perturbed_steps",
        "recovered_unsafe_steps",
        "unresolved_unsafe_steps",
        "recovery_rate",
        "within_filter_violation_reduction",
        "mean_reward",
        "reward_delta_from_clean",
        "success_rate",
        "success_delta_from_clean",
        "intervention_rate",
        "mean_action_perturbation_l2",
        "mean_safety_correction_l2",
        "p95_safety_correction_l2",
        "environment_interface_adjustment_rate",
        "strict_lyapunov_decrease_rate",
        "selected_lower_than_perturbed_rate",
        "steps_object_grasped",
        "proposed_to_perturbed_gripper_semantic_change_rate",
        "perturbed_to_executed_gripper_semantic_change_rate",
    ]

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in cells:
            writer.writerow({field: row[field] for field in csv_fields})

    markdown: list[str] = [
        "# Sprint 5.12 — Action Perturbation Robustness",
        "",
        "## Protocol",
        "",
        "- 24 deterministic structured action perturbations",
        "- 216 principal cells",
        "- 4,320 principal episodes",
        "- Frozen PPO checkpoints",
        "- Methods: NONE, CLIPPING, LYAPUNOV",
        "- Perturbation applied after PPO and before explicit safety filtering",
        "- No pre-filter clipping",
        "- Environment internal clipping accounted separately",
        "",
        "## Mechanism Findings",
        "",
        (f"- Unsafe perturbed steps: " f"{total_unsafe}"),
        (f"- Explicitly recovered unsafe steps: " f"{total_recovered}"),
        (f"- Unresolved unsafe steps: " f"{total_unresolved}"),
        (
            "- Overall explicit-filter recovery fraction: "
            + ("n/a" if total_unsafe == 0 else f"{total_recovered / total_unsafe:.6f}")
        ),
        (
            "- Lyapunov-decrease intervention reasons: "
            f"{lyapunov_reason_counts.get('lyapunov_decrease', 0)}"
        ),
        ("- Strict Lyapunov-decrease steps: " f"{strict_steps}"),
        ("- Selected lower than perturbed steps: " f"{selected_lower_steps}"),
        ("- Emergency fallback steps: " f"{fallback_steps}"),
        ("- Environment-interface adjustment steps: " f"{interface_adjustment_steps}"),
        "",
        "## Feature Sensitivity",
        "",
    ]

    for item in sensitivity:
        markdown.extend(
            [
                (f"### {item['domain']} / " f"{item['method']}"),
                "",
                (
                    "- Worst executed-safety delta: "
                    f"{item['worst_executed_safety']['perturbation']} "
                    f"({item['worst_executed_safety']['delta']:+.6f})"
                ),
                (
                    "- Worst reward delta: "
                    f"{item['worst_reward']['perturbation']} "
                    f"({item['worst_reward']['delta']:+.6f})"
                ),
                (
                    "- Worst success delta: "
                    f"{item['worst_success']['perturbation']} "
                    f"({item['worst_success']['delta']:+.6f})"
                ),
                (
                    "- Highest intervention: "
                    f"{item['highest_intervention']['perturbation']} "
                    f"({item['highest_intervention']['rate']:.6f})"
                ),
                "",
            ]
        )

    markdown.extend(
        [
            "## Interpretation Controls",
            "",
            (
                "- A `lyapunov_decrease` intervention reason is not "
                "identical to a strict decrease below current V."
            ),
            ("- Strict Lyapunov-decrease steps are reported " "separately."),
            (
                "- Environment-internal clipping is not counted "
                "as explicit safety-filter recovery."
            ),
            ("- Results are synthetic internal pilot evidence only."),
            "",
        ]
    )

    MD_OUT.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print(
        "Action seed cells:",
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
        "Unsafe perturbed steps:",
        total_unsafe,
    )

    print(
        "Recovered unsafe steps:",
        total_recovered,
    )

    print(
        "Unresolved unsafe steps:",
        total_unresolved,
    )

    print(
        "Overall recovery:",
        (None if total_unsafe == 0 else total_recovered / total_unsafe),
    )

    print()
    print(
        "Lyapunov reason counts:",
        dict(lyapunov_reason_counts),
    )

    print(
        "Strict Lyapunov-decrease steps:",
        strict_steps,
    )

    print(
        "Selected lower than perturbed:",
        selected_lower_steps,
    )

    print(
        "Emergency fallback steps:",
        fallback_steps,
    )

    print()
    print("SPRINT 5.12 ACTION ROBUSTNESS SUMMARY: PASS")


if __name__ == "__main__":
    main()
