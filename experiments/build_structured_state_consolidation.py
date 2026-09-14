"""Build Sprint 5.13D structured-state robustness consolidation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.structured_state_consolidation import (
    PRINCIPAL_SEEDS,
    StructuredStateSeedResult,
    summarize_rows,
    worst_condition,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "structured-state-robustness"
    / "sprint5-structured-state-robustness-summary.json"
)

OUTPUT_ROOT = ROOT / "results" / "safety" / "consolidated"

OUTPUT_JSON = OUTPUT_ROOT / "sprint5-structured-state-three-seed-summary.json"

OUTPUT_CSV = OUTPUT_ROOT / "sprint5-structured-state-three-seed-summary.csv"

OUTPUT_MD = OUTPUT_ROOT / "sprint5-structured-state-three-seed-summary.md"

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
        raise TypeError("structured-state summary must be a dict")

    return payload


def _cell_to_result(
    cell: dict[str, Any],
) -> StructuredStateSeedResult:
    return StructuredStateSeedResult(
        domain=str(cell["domain"]),
        method=str(cell["method"]),
        perturbation_family=str(cell["perturbation_family"]),
        perturbation_name=str(cell["perturbation_name"]),
        principal_seed=int(cell["principal_seed"]),
        violation_step_rate=float(cell["violation_step_rate"]),
        critical_violation_step_rate=float(cell["critical_violation_step_rate"]),
        constraint_violation_rate=float(cell["constraint_violation_rate"]),
        reward=float(cell["reward"]),
        success_rate=float(cell["success_rate"]),
        intervention_rate=float(cell["intervention_rate"]),
        mean_correction_l2=float(cell["mean_action_correction_l2"]),
        violation_delta_from_clean=float(cell["violation_delta_from_clean"]),
        critical_delta_from_clean=float(cell["critical_delta_from_clean"]),
        constraint_delta_from_clean=float(cell["constraint_delta_from_clean"]),
        reward_delta_from_clean=float(cell["reward_delta_from_clean"]),
        success_delta_from_clean=float(cell["success_delta_from_clean"]),
        strict_lyapunov_decrease_rate=float(cell["strict_lyapunov_decrease_rate"]),
        lyapunov_nonincrease_rate=float(cell["lyapunov_nonincrease_rate"]),
        steps_object_grasped=int(cell["steps_object_grasped"]),
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

    if len(cells_raw) != 198:
        raise RuntimeError(f"expected 198 cells, found {len(cells_raw)}")

    rows: list[StructuredStateSeedResult] = []

    perturbation_updates: dict[
        tuple[
            str,
            str,
        ],
        Any,
    ] = {}

    for raw in cells_raw:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("cell must be a dict")

        result = _cell_to_result(raw)

        rows.append(result)

        update_key = (
            result.domain,
            result.perturbation_name,
        )

        perturbation_updates[update_key] = raw["perturbation_updates"]

    aggregates: list[dict[str, Any]] = []

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

    if len(identities) != 22:
        raise RuntimeError(f"expected 22 perturbations, found {len(identities)}")

    for (
        domain,
        family,
        name,
    ) in identities:
        for method in METHODS:
            group = [
                row
                for row in rows
                if (
                    row.domain == domain
                    and row.method == method
                    and row.perturbation_name == name
                )
            ]

            if len(group) != 3:
                raise RuntimeError(f"incomplete group: " f"{domain}/{method}/{name}")

            if {row.principal_seed for row in group} != set(PRINCIPAL_SEEDS):
                raise RuntimeError("principal seed mismatch")

            metric_summary = summarize_rows(group)

            aggregates.append(
                {
                    "domain": domain,
                    "method": method,
                    "perturbation_family": family,
                    "perturbation_name": name,
                    "perturbation_updates": perturbation_updates[
                        (
                            domain,
                            name,
                        )
                    ],
                    **{
                        key: asdict(value)
                        for (
                            key,
                            value,
                        ) in metric_summary.items()
                    },
                }
            )

    if len(aggregates) != 66:
        raise RuntimeError(f"expected 66 aggregates, found {len(aggregates)}")

    worst_cases: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for domain in DOMAINS:
        worst_cases[domain] = {}

        for method in METHODS:
            domain_rows = [
                row
                for row in aggregates
                if (row["domain"] == domain and row["method"] == method)
            ]

            worst_cases[domain][method] = {
                "largest_violation_increase": worst_condition(
                    domain_rows,
                    metric=("violation_delta_from_clean"),
                    mode="max",
                ),
                "largest_critical_increase": worst_condition(
                    domain_rows,
                    metric=("critical_delta_from_clean"),
                    mode="max",
                ),
                "largest_constraint_increase": worst_condition(
                    domain_rows,
                    metric=("constraint_delta_from_clean"),
                    mode="max",
                ),
                "largest_reward_decline": worst_condition(
                    domain_rows,
                    metric=("reward_delta_from_clean"),
                    mode="min",
                ),
                "largest_success_decline": worst_condition(
                    domain_rows,
                    metric=("success_delta_from_clean"),
                    mode="min",
                ),
                "highest_intervention_rate": worst_condition(
                    domain_rows,
                    metric=("intervention_rate"),
                    mode="max",
                ),
            }

    findings = source["findings"]

    if not isinstance(
        findings,
        dict,
    ):
        raise TypeError("findings must be a dict")

    payload: dict[str, Any] = {
        "sprint": "5.13D",
        "artifact": "structured-state-three-seed-consolidation",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "source": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "driving_perturbation_count": int(source["driving_perturbation_count"]),
        "robotics_perturbation_count": int(source["robotics_perturbation_count"]),
        "total_perturbation_count": int(source["total_perturbation_count"]),
        "principal_cells": int(source["principal_cells"]),
        "principal_episodes": int(source["principal_episodes"]),
        "policy_uses_perturbed_observation": bool(
            source["policy_uses_perturbed_observation"]
        ),
        "safety_layer_uses_true_state": bool(source["safety_layer_uses_true_state"]),
        "environment_uses_true_state": bool(source["environment_uses_true_state"]),
        "seed_rows": [asdict(row) for row in rows],
        "three_seed_summary": aggregates,
        "worst_cases": worst_cases,
        "lyapunov_mechanism": {
            "intervention_reason_counts": findings[
                "lyapunov_intervention_reason_counts"
            ],
            "strict_decrease_steps": int(findings["strict_lyapunov_decrease_steps"]),
            "lyapunov_decrease_interventions_observed": bool(
                findings["lyapunov_decrease_interventions_observed"]
            ),
            "robotics_grasped_steps": int(
                findings["robotics_grasped_steps_in_lyapunov_runs"]
            ),
        },
        "reporting_limitations": [
            (
                "Structured state perturbations are synthetic "
                "semantic observation biases and are not "
                "calibrated sensor-fault models."
            ),
            (
                "The policy receives the perturbed observation, "
                "while the safety layer and environment retain "
                "true simulator state."
            ),
            (
                "Three principal policy seeds are the statistical "
                "experimental units; episode samples are nested."
            ),
            ("No formal robustness or safety certification " "claim is supported."),
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
                "violation_step_rate",
                "critical_violation_step_rate",
                "constraint_violation_rate",
                "reward",
                "success_rate",
                "intervention_rate",
                "mean_correction_l2",
                "violation_delta_from_clean",
                "critical_delta_from_clean",
                "constraint_delta_from_clean",
                "reward_delta_from_clean",
                "success_delta_from_clean",
                "strict_lyapunov_decrease_rate",
                "lyapunov_nonincrease_rate",
                "steps_object_grasped",
            ],
        )

        writer.writeheader()

        writer.writerows([asdict(row) for row in rows])

    mechanism = payload["lyapunov_mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("lyapunov_mechanism must be a dict")

    lines = [
        ("# Sprint 5.13D — Structured-State " "Robustness Consolidation"),
        "",
        ("Analysis only. No new policy or " "environment execution."),
        "",
        "## Corpus",
        "",
        (f"- Driving perturbations: " f"{payload['driving_perturbation_count']}"),
        (f"- Robotics perturbations: " f"{payload['robotics_perturbation_count']}"),
        (f"- Total perturbations: " f"{payload['total_perturbation_count']}"),
        (f"- Principal cells: " f"{payload['principal_cells']}"),
        (f"- Principal episodes: " f"{payload['principal_episodes']}"),
        "",
        "## State separation",
        "",
        (
            "- Policy uses perturbed observation: "
            f"{payload['policy_uses_perturbed_observation']}"
        ),
        (
            "- Safety layer uses true state: "
            f"{payload['safety_layer_uses_true_state']}"
        ),
        ("- Environment uses true state: " f"{payload['environment_uses_true_state']}"),
        "",
        "## Lyapunov mechanism",
        "",
        ("- Strict Lyapunov decreases: " f"{mechanism['strict_decrease_steps']}"),
        (
            "- Lyapunov-specific intervention observed: "
            f"{mechanism['lyapunov_decrease_interventions_observed']}"
        ),
        ("- Robotics grasped steps: " f"{mechanism['robotics_grasped_steps']}"),
        "",
    ]

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print(" SPRINT 5.13D STRUCTURED-STATE CONSOLIDATION")
    print("=" * 80)
    print()

    print(
        "Seed cells:",
        len(rows),
        "/ 198",
    )

    print(
        "Three-seed aggregates:",
        len(aggregates),
        "/ 66",
    )

    print(
        "Perturbations:",
        len(identities),
        "/ 22",
    )

    print(
        "Episodes:",
        payload["principal_episodes"],
    )

    print()
    print(
        "Lyapunov intervention reasons:",
        mechanism["intervention_reason_counts"],
    )

    print(
        "Strict Lyapunov decreases:",
        mechanism["strict_decrease_steps"],
    )

    print()
    print("No new training: PASS")

    print("No new principal runs: PASS")

    print()
    print("SPRINT 5.13D STRUCTURED-STATE CONSOLIDATION: PASS")


if __name__ == "__main__":
    main()
