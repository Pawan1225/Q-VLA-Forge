"""Build Sprint 5.13C Gaussian robustness consolidation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.gaussian_safety_consolidation import (
    PRINCIPAL_SEEDS,
    SIGMAS,
    GaussianSeedResult,
    summarize_gaussian_rows,
    worst_condition,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "gaussian-robustness"
    / "sprint5-gaussian-robustness-summary.json"
)

OUTPUT_ROOT = ROOT / "results" / "safety" / "consolidated"

OUTPUT_JSON = OUTPUT_ROOT / "sprint5-gaussian-three-seed-summary.json"

OUTPUT_CSV = OUTPUT_ROOT / "sprint5-gaussian-three-seed-summary.csv"

OUTPUT_MD = OUTPUT_ROOT / "sprint5-gaussian-three-seed-summary.md"

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
        raise TypeError("Gaussian summary must be a JSON object")

    return payload


def _cell_to_result(
    cell: dict[str, Any],
) -> GaussianSeedResult:
    return GaussianSeedResult(
        domain=str(cell["domain"]),
        method=str(cell["method"]),
        sigma=float(cell["sigma"]),
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

    if len(cells_raw) != 72:
        raise RuntimeError(f"expected 72 cells, found {len(cells_raw)}")

    rows: list[GaussianSeedResult] = []

    for raw in cells_raw:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("cell must be a dict")

        rows.append(_cell_to_result(raw))

    grouped_summary: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for method in METHODS:
            for sigma in SIGMAS:
                group = [
                    row
                    for row in rows
                    if (
                        row.domain == domain
                        and row.method == method
                        and row.sigma == sigma
                    )
                ]

                if len(group) != 3:
                    raise RuntimeError(
                        f"incomplete Gaussian group: " f"{domain}/{method}/{sigma}"
                    )

                if {row.principal_seed for row in group} != set(PRINCIPAL_SEEDS):
                    raise RuntimeError("principal seed mismatch")

                metrics = summarize_gaussian_rows(group)

                grouped_summary.append(
                    {
                        "domain": domain,
                        "method": method,
                        "sigma": sigma,
                        **{
                            key: asdict(value)
                            for (
                                key,
                                value,
                            ) in metrics.items()
                        },
                    }
                )

    if len(grouped_summary) != 24:
        raise RuntimeError("expected 24 Gaussian aggregate rows")

    worst_cases: dict[str, dict[str, dict[str, Any]]] = {}

    for domain in DOMAINS:
        worst_cases[domain] = {}

        for method in METHODS:
            noisy = [
                row
                for row in grouped_summary
                if (
                    row["domain"] == domain
                    and row["method"] == method
                    and float(row["sigma"]) > 0.0
                )
            ]

            worst_cases[domain][method] = {
                "largest_violation_increase": worst_condition(
                    noisy,
                    metric=("violation_delta_from_clean"),
                    mode="max",
                ),
                "largest_critical_increase": worst_condition(
                    noisy,
                    metric=("critical_delta_from_clean"),
                    mode="max",
                ),
                "largest_constraint_increase": worst_condition(
                    noisy,
                    metric=("constraint_delta_from_clean"),
                    mode="max",
                ),
                "largest_reward_decline": worst_condition(
                    noisy,
                    metric=("reward_delta_from_clean"),
                    mode="min",
                ),
                "largest_success_decline": worst_condition(
                    noisy,
                    metric=("success_delta_from_clean"),
                    mode="min",
                ),
                "highest_intervention_rate": worst_condition(
                    noisy,
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
        "sprint": "5.13C",
        "artifact": "gaussian-three-seed-consolidation",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "source": str(SOURCE.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "sigmas": list(SIGMAS),
        "conceptual_cells": 72,
        "new_noisy_cells": 54,
        "conceptual_episodes": int(source["conceptual_episodes"]),
        "new_noisy_episodes": int(source["new_noisy_episodes"]),
        "policy_uses_noisy_observation": bool(source["policy_uses_noisy_observation"]),
        "safety_layer_uses_true_state": bool(source["safety_layer_uses_true_state"]),
        "seed_rows": [asdict(row) for row in rows],
        "three_seed_summary": grouped_summary,
        "worst_cases": worst_cases,
        "lyapunov_mechanism": {
            "noisy_environment_steps": int(
                findings["noisy_lyapunov_environment_steps"]
            ),
            "noisy_grasped_steps": int(findings["noisy_lyapunov_grasped_steps"]),
            "intervention_reason_counts": findings[
                "noisy_lyapunov_intervention_reason_counts"
            ],
            "strict_decrease_steps": int(
                findings["noisy_lyapunov_strict_decrease_steps"]
            ),
            "lyapunov_decrease_interventions_observed": bool(
                findings["principal_lyapunov_decrease_interventions_observed"]
            ),
        },
        "reporting_limitations": [
            (
                "Gaussian observation noise is synthetic "
                "and is not calibrated to physical sensors."
            ),
            (
                "The safety layer retains privileged true "
                "simulator state while the policy receives "
                "the noisy observation."
            ),
            (
                "Episode-level samples are nested beneath "
                "three principal policy seeds and are not "
                "treated as independent experimental units."
            ),
            (
                "P95 correction magnitude is not exposed "
                "in the frozen Gaussian summary and is not "
                "fabricated here."
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
                "sigma",
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

    lyapunov_mechanism = payload["lyapunov_mechanism"]

    if not isinstance(
        lyapunov_mechanism,
        dict,
    ):
        raise TypeError("lyapunov_mechanism must be a dict")

    reporting_limitations = payload["reporting_limitations"]

    if not isinstance(
        reporting_limitations,
        list,
    ):
        raise TypeError("reporting_limitations must be a list")

    lines = [
        ("# Sprint 5.13C — Gaussian " "Robustness Consolidation"),
        "",
        ("Analysis only. No new Gaussian " "trajectories were generated."),
        "",
        "## Protocol",
        "",
        "- Policy input: noisy observation.",
        ("- Safety-layer input: true " "simulator state."),
        "- Principal seeds: 42, 123, 456.",
        ("- Sigma levels: " "0.00, 0.01, 0.05, 0.10."),
        "",
        "## Corpus",
        "",
        (f"- Conceptual cells: " f"{payload['conceptual_cells']}"),
        (f"- New noisy cells: " f"{payload['new_noisy_cells']}"),
        (f"- Conceptual episodes: " f"{payload['conceptual_episodes']}"),
        (f"- New noisy episodes: " f"{payload['new_noisy_episodes']}"),
        "",
        "## Lyapunov mechanism",
        "",
        (
            "- Noisy Lyapunov environment steps: "
            f"{lyapunov_mechanism['noisy_environment_steps']}"
        ),
        ("- Grasped steps: " f"{lyapunov_mechanism['noisy_grasped_steps']}"),
        (
            "- Strict Lyapunov decreases: "
            f"{lyapunov_mechanism['strict_decrease_steps']}"
        ),
        (
            "- Lyapunov-specific intervention observed: "
            f"{lyapunov_mechanism['lyapunov_decrease_interventions_observed']}"
        ),
        "",
        "## Limitations",
        "",
    ]

    for limitation in reporting_limitations:
        lines.append(f"- {limitation}")

    lines.append("")

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print(" SPRINT 5.13C GAUSSIAN ROBUSTNESS CONSOLIDATION")
    print("=" * 80)
    print()

    print(
        "Seed cells:",
        len(rows),
        "/ 72",
    )

    print(
        "Three-seed aggregates:",
        len(grouped_summary),
        "/ 24",
    )

    print(
        "New noisy cells:",
        payload["new_noisy_cells"],
        "/ 54",
    )

    print(
        "New noisy episodes:",
        payload["new_noisy_episodes"],
    )

    print()
    print(
        "Policy uses noisy observation:",
        payload["policy_uses_noisy_observation"],
    )

    print(
        "Safety uses true state:",
        payload["safety_layer_uses_true_state"],
    )

    print()
    print(
        "Lyapunov intervention reasons:",
        lyapunov_mechanism["intervention_reason_counts"],
    )

    print(
        "Strict Lyapunov decreases:",
        lyapunov_mechanism["strict_decrease_steps"],
    )

    print()
    print("No new training: PASS")

    print("No new principal runs: PASS")

    print()
    print("SPRINT 5.13C GAUSSIAN CONSOLIDATION: PASS")


if __name__ == "__main__":
    main()
