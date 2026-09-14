"""Build Sprint 5.13E action robustness consolidation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.action_safety_consolidation import (
    PRINCIPAL_SEEDS,
    ActionSeedResult,
    summarize_rows,
    worst_condition,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "action-robustness"
    / "sprint5-action-robustness-summary.json"
)

OUTPUT_ROOT = ROOT / "results" / "safety" / "consolidated"

OUTPUT_JSON = OUTPUT_ROOT / "sprint5-action-three-seed-summary.json"

OUTPUT_CSV = OUTPUT_ROOT / "sprint5-action-three-seed-summary.csv"

OUTPUT_MD = OUTPUT_ROOT / "sprint5-action-three-seed-summary.md"

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
        raise TypeError("action summary must be a JSON object")

    return payload


def _optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    return float(value)


def _cell_to_result(
    cell: dict[str, Any],
) -> ActionSeedResult:
    return ActionSeedResult(
        domain=str(cell["domain"]),
        method=str(cell["method"]),
        perturbation_family=str(cell["perturbation_family"]),
        perturbation_name=str(cell["perturbation_name"]),
        principal_seed=int(cell["principal_seed"]),
        perturbed_violation_step_rate=float(cell["perturbed_violation_step_rate"]),
        executed_violation_step_rate=float(cell["executed_violation_step_rate"]),
        executed_constraint_violation_rate=float(
            cell["executed_constraint_violation_rate"]
        ),
        critical_violation_step_rate=float(cell["critical_violation_step_rate"]),
        mean_reward=float(cell["mean_reward"]),
        success_rate=float(cell["success_rate"]),
        intervention_rate=float(cell["intervention_rate"]),
        mean_safety_correction_l2=float(cell["mean_safety_correction_l2"]),
        p95_safety_correction_l2=float(cell["p95_safety_correction_l2"]),
        # These are legitimately undefined when no unsafe
        # perturbed steps exist in the corresponding cell.
        recovery_rate=_optional_float(cell["recovery_rate"]),
        within_filter_violation_reduction=_optional_float(
            cell["within_filter_violation_reduction"]
        ),
        executed_violation_delta_from_clean=float(
            cell["executed_violation_delta_from_clean"]
        ),
        reward_delta_from_clean=float(cell["reward_delta_from_clean"]),
        success_delta_from_clean=float(cell["success_delta_from_clean"]),
        selected_lower_than_perturbed_rate=float(
            cell["selected_lower_than_perturbed_rate"]
        ),
        strict_lyapunov_decrease_rate=float(cell["strict_lyapunov_decrease_rate"]),
        lyapunov_nonincrease_rate=float(cell["lyapunov_nonincrease_rate"]),
        environment_interface_adjustment_rate=float(
            cell["environment_interface_adjustment_rate"]
        ),
        proposed_to_perturbed_gripper_semantic_change_rate=float(
            cell["proposed_to_perturbed_gripper_semantic_change_rate"]
        ),
        perturbed_to_executed_gripper_semantic_change_rate=float(
            cell["perturbed_to_executed_gripper_semantic_change_rate"]
        ),
        unsafe_perturbed_steps=int(cell["unsafe_perturbed_steps"]),
        recovered_unsafe_steps=int(cell["recovered_unsafe_steps"]),
        unresolved_unsafe_steps=int(cell["unresolved_unsafe_steps"]),
        steps_object_grasped=int(cell["steps_object_grasped"]),
        interventions_while_grasped=int(cell["interventions_while_grasped"]),
        perturbed_unsafe_steps_while_grasped=int(
            cell["perturbed_unsafe_steps_while_grasped"]
        ),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = _load()

    cells_raw = source["cells"]

    if not isinstance(
        cells_raw,
        list,
    ):
        raise TypeError("cells must be a list")

    if len(cells_raw) != 216:
        raise RuntimeError(f"expected 216 cells, found {len(cells_raw)}")

    rows: list[ActionSeedResult] = []

    perturbation_updates: dict[
        tuple[str, str],
        Any,
    ] = {}

    for raw in cells_raw:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("action cell must be a dict")

        result = _cell_to_result(raw)

        rows.append(result)

        perturbation_updates[
            (
                result.domain,
                result.perturbation_name,
            )
        ] = raw["perturbation_updates"]

    identities = sorted(
        {
            (
                row.domain,
                row.perturbation_family,
                row.perturbation_name,
            )
            for row in rows
        }
    )

    if len(identities) != 24:
        raise RuntimeError(f"expected 24 perturbations, found {len(identities)}")

    aggregates: list[dict[str, Any]] = []

    for (
        domain,
        family,
        perturbation_name,
    ) in identities:
        for method in METHODS:
            group = [
                row
                for row in rows
                if (
                    row.domain == domain
                    and row.method == method
                    and row.perturbation_name == perturbation_name
                )
            ]

            if len(group) != 3:
                raise RuntimeError(
                    "incomplete action group: " f"{domain}/{method}/{perturbation_name}"
                )

            if {row.principal_seed for row in group} != set(PRINCIPAL_SEEDS):
                raise RuntimeError("principal seed mismatch")

            metrics = summarize_rows(group)

            aggregates.append(
                {
                    "domain": domain,
                    "method": method,
                    "perturbation_family": family,
                    "perturbation_name": perturbation_name,
                    "perturbation_updates": perturbation_updates[
                        (
                            domain,
                            perturbation_name,
                        )
                    ],
                    **{
                        key: asdict(value)
                        for (
                            key,
                            value,
                        ) in metrics.items()
                    },
                    "unsafe_perturbed_steps": sum(
                        row.unsafe_perturbed_steps for row in group
                    ),
                    "recovered_unsafe_steps": sum(
                        row.recovered_unsafe_steps for row in group
                    ),
                    "unresolved_unsafe_steps": sum(
                        row.unresolved_unsafe_steps for row in group
                    ),
                    "steps_object_grasped": sum(
                        row.steps_object_grasped for row in group
                    ),
                    "interventions_while_grasped": sum(
                        row.interventions_while_grasped for row in group
                    ),
                    "perturbed_unsafe_steps_while_grasped": sum(
                        row.perturbed_unsafe_steps_while_grasped for row in group
                    ),
                }
            )

    if len(aggregates) != 72:
        raise RuntimeError(f"expected 72 aggregates, found {len(aggregates)}")

    worst_cases: dict[
        str,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    for domain in DOMAINS:
        worst_cases[domain] = {}

        for method in METHODS:
            candidates = [
                row
                for row in aggregates
                if (row["domain"] == domain and row["method"] == method)
            ]

            worst_cases[domain][method] = {
                "largest_perturbed_violation": worst_condition(
                    candidates,
                    metric=("perturbed_violation_step_rate"),
                    mode="max",
                ),
                "largest_executed_violation_delta": worst_condition(
                    candidates,
                    metric=("executed_violation_delta_from_clean"),
                    mode="max",
                ),
                "largest_reward_decline": worst_condition(
                    candidates,
                    metric=("reward_delta_from_clean"),
                    mode="min",
                ),
                "highest_intervention_rate": worst_condition(
                    candidates,
                    metric=("intervention_rate"),
                    mode="max",
                ),
                "highest_strict_lyapunov_decrease_rate": worst_condition(
                    candidates,
                    metric=("strict_lyapunov_decrease_rate"),
                    mode="max",
                ),
            }

    findings = source["findings"]

    if not isinstance(
        findings,
        dict,
    ):
        raise TypeError("findings must be a dict")

    claims = source["claims"]

    if not isinstance(
        claims,
        dict,
    ):
        raise TypeError("claims must be a dict")

    mechanism: dict[
        str,
        Any,
    ] = {
        "unsafe_perturbed_steps": int(findings["unsafe_perturbed_steps"]),
        "recovered_unsafe_steps": int(findings["recovered_unsafe_steps"]),
        "unresolved_unsafe_steps": int(findings["unresolved_unsafe_steps"]),
        "overall_explicit_filter_recovery_fraction": float(
            findings["overall_explicit_filter_recovery_fraction"]
        ),
        "lyapunov_intervention_reason_counts": findings[
            "lyapunov_intervention_reason_counts"
        ],
        "lyapunov_candidate_source_counts": findings[
            "lyapunov_candidate_source_counts"
        ],
        "strict_lyapunov_decrease_steps": int(
            findings["strict_lyapunov_decrease_steps"]
        ),
        "strict_lyapunov_decrease_observed": bool(
            findings["strict_lyapunov_decrease_observed"]
        ),
        "lyapunov_nonincrease_steps": int(findings["lyapunov_nonincrease_steps"]),
        "selected_lower_than_perturbed_steps": int(
            findings["selected_lower_than_perturbed_steps"]
        ),
        "emergency_fallback_steps": int(findings["emergency_fallback_steps"]),
        "environment_interface_adjustment_steps": int(
            findings["environment_interface_adjustment_steps"]
        ),
        "lyapunov_decrease_interventions_observed": bool(
            findings["lyapunov_decrease_interventions_observed"]
        ),
        "lyapunov_robotics_grasped_steps": int(
            findings["lyapunov_robotics_grasped_steps"]
        ),
        "lyapunov_robotics_unsafe_while_grasped": int(
            findings["lyapunov_robotics_unsafe_while_grasped"]
        ),
        "lyapunov_robotics_interventions_while_grasped": int(
            findings["lyapunov_robotics_interventions_while_grasped"]
        ),
        "robotics_proposed_to_perturbed_gripper_semantic_changes": int(
            findings["robotics_proposed_to_perturbed_gripper_semantic_changes"]
        ),
        "robotics_perturbed_to_executed_gripper_semantic_changes": int(
            findings["robotics_perturbed_to_executed_gripper_semantic_changes"]
        ),
    }

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.13E",
        "artifact": "action-three-seed-consolidation",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "source": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "total_perturbation_count": int(source["total_perturbation_count"]),
        "principal_cells": int(source["principal_cells"]),
        "principal_episodes": int(source["principal_episodes"]),
        "seed_rows": [asdict(row) for row in rows],
        "three_seed_summary": aggregates,
        "worst_cases": worst_cases,
        "mechanism": mechanism,
        "claim_controls": claims,
        "reporting_limitations": [
            (
                "Action perturbations are synthetic actuator-"
                "command perturbations and are not calibrated "
                "physical actuator fault models."
            ),
            (
                "Undefined recovery metrics are retained as null "
                "when a cell contains no unsafe perturbed steps; "
                "they are not converted to zero."
            ),
            (
                "Environment-interface action adjustments are "
                "reported separately from explicit safety-filter "
                "interventions."
            ),
            (
                "Three principal policy seeds are the statistical "
                "experimental units; episode samples are nested."
            ),
            (
                "Observed Lyapunov-decrease interventions provide "
                "mechanism evidence only and do not establish "
                "formal Lyapunov stability."
            ),
            (
                "No production robustness, actuator certification, "
                "ISO compliance, or global Lyapunov superiority "
                "claim is supported."
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

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "method",
                "perturbation_family",
                "perturbation_name",
                "principal_seed",
                "perturbed_violation_step_rate",
                "executed_violation_step_rate",
                "executed_constraint_violation_rate",
                "critical_violation_step_rate",
                "mean_reward",
                "success_rate",
                "intervention_rate",
                "mean_safety_correction_l2",
                "p95_safety_correction_l2",
                "recovery_rate",
                "within_filter_violation_reduction",
                "executed_violation_delta_from_clean",
                "reward_delta_from_clean",
                "success_delta_from_clean",
                "selected_lower_than_perturbed_rate",
                "strict_lyapunov_decrease_rate",
                "lyapunov_nonincrease_rate",
                "environment_interface_adjustment_rate",
                ("proposed_to_perturbed_gripper_" "semantic_change_rate"),
                ("perturbed_to_executed_gripper_" "semantic_change_rate"),
                "unsafe_perturbed_steps",
                "recovered_unsafe_steps",
                "unresolved_unsafe_steps",
                "steps_object_grasped",
                "interventions_while_grasped",
                "perturbed_unsafe_steps_while_grasped",
            ],
        )

        writer.writeheader()

        writer.writerows([asdict(row) for row in rows])

    reasons = mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        reasons,
        dict,
    ):
        raise TypeError("Lyapunov reason counts must be a dict")

    lines = [
        ("# Sprint 5.13E — Action Robustness " "Consolidation"),
        "",
        ("Analysis only. No new policy training or " "environment execution."),
        "",
        "## Corpus",
        "",
        (f"- Action perturbations: " f"{payload['total_perturbation_count']}"),
        (f"- Principal cells: " f"{payload['principal_cells']}"),
        (f"- Principal episodes: " f"{payload['principal_episodes']}"),
        "",
        "## Explicit filter recovery",
        "",
        (f"- Unsafe perturbed steps: " f"{mechanism['unsafe_perturbed_steps']}"),
        (f"- Recovered unsafe steps: " f"{mechanism['recovered_unsafe_steps']}"),
        (f"- Unresolved unsafe steps: " f"{mechanism['unresolved_unsafe_steps']}"),
        (
            f"- Overall recovery fraction: "
            f"{mechanism['overall_explicit_filter_recovery_fraction']:.9f}"
        ),
        "",
        (
            "Undefined cell-level recovery rates remain null "
            "when no unsafe perturbed steps are present."
        ),
        "",
        "## Lyapunov mechanism",
        "",
        (f"- ACTION_BOUND: " f"{reasons.get('action_bound', 0)}"),
        (f"- DOMAIN_CONSTRAINT: " f"{reasons.get('domain_constraint', 0)}"),
        (f"- LYAPUNOV_DECREASE: " f"{reasons.get('lyapunov_decrease', 0)}"),
        (f"- NONE: " f"{reasons.get('none', 0)}"),
        (f"- Strict decrease steps: " f"{mechanism['strict_lyapunov_decrease_steps']}"),
        (
            f"- Selected-lower steps: "
            f"{mechanism['selected_lower_than_perturbed_steps']}"
        ),
        (f"- Emergency fallback steps: " f"{mechanism['emergency_fallback_steps']}"),
        "",
    ]

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    undefined_recovery_cells = sum(row.recovery_rate is None for row in rows)

    undefined_reduction_cells = sum(
        row.within_filter_violation_reduction is None for row in rows
    )

    print("=" * 80)
    print(" SPRINT 5.13E ACTION ROBUSTNESS CONSOLIDATION")
    print("=" * 80)
    print()

    print(
        "Seed cells:",
        len(rows),
        "/ 216",
    )

    print(
        "Three-seed aggregates:",
        len(aggregates),
        "/ 72",
    )

    print(
        "Perturbations:",
        len(identities),
        "/ 24",
    )

    print(
        "Episodes:",
        payload["principal_episodes"],
    )

    print()
    print(
        "Undefined recovery-rate cells:",
        undefined_recovery_cells,
    )

    print(
        "Undefined violation-reduction cells:",
        undefined_reduction_cells,
    )

    print()
    print(
        "Unsafe perturbed steps:",
        mechanism["unsafe_perturbed_steps"],
    )

    print(
        "Recovered unsafe steps:",
        mechanism["recovered_unsafe_steps"],
    )

    print(
        "Unresolved unsafe steps:",
        mechanism["unresolved_unsafe_steps"],
    )

    print(
        "Overall recovery fraction:",
        mechanism["overall_explicit_filter_recovery_fraction"],
    )

    print()
    print(
        "Lyapunov reasons:",
        reasons,
    )

    print(
        "Strict Lyapunov decreases:",
        mechanism["strict_lyapunov_decrease_steps"],
    )

    print(
        "Selected-lower steps:",
        mechanism["selected_lower_than_perturbed_steps"],
    )

    print(
        "Emergency fallback steps:",
        mechanism["emergency_fallback_steps"],
    )

    print(
        "Environment-interface adjustments:",
        mechanism["environment_interface_adjustment_steps"],
    )

    print()
    print("No new training: PASS")

    print("No new principal runs: PASS")

    print()
    print("SPRINT 5.13E ACTION CONSOLIDATION: PASS")


if __name__ == "__main__":
    main()
