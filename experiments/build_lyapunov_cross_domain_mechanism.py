"""Build Sprint 5.14G cross-domain Lyapunov mechanism attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.cross_domain_safety import (
    exact_count_from_rate,
)

ROOT = Path(__file__).resolve().parents[1]

RAW_SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "action-robustness"
    / "sprint5-action-robustness-summary.json"
)

CONSOLIDATED_SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-action-three-seed-summary.json"
)

ATTRIBUTION_SOURCE = (
    ROOT
    / "results"
    / "safety"
    / "consolidated"
    / "sprint5-lyapunov-mechanism-attribution.json"
)

OUTPUT_DIR = ROOT / "results" / "safety" / "cross-domain"

OUTPUT_JSON = OUTPUT_DIR / "sprint5-cross-domain-lyapunov-mechanism.json"

OUTPUT_CSV = OUTPUT_DIR / "sprint5-cross-domain-lyapunov-mechanism.csv"

OUTPUT_MD = OUTPUT_DIR / "sprint5-cross-domain-lyapunov-mechanism.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

REASON_KEYS = (
    "action_bound",
    "domain_constraint",
    "lyapunov_decrease",
    "none",
)


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"expected JSON object: {path}")

    return payload


def _resolve_run_path(
    value: str,
) -> Path:
    normalized = value.replace(
        "\\",
        "/",
    )

    path = Path(normalized)

    if not path.is_absolute():
        path = ROOT / path

    return path


def _empty_domain_record(
    domain: str,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "run_cells": 0,
        "total_environment_steps": 0,
        "intervention_reason_counts": {key: 0 for key in REASON_KEYS},
        "intervention_steps": 0,
        "hard_guard_interventions": 0,
        "lyapunov_decrease_interventions": 0,
        "strict_lyapunov_decrease_steps": 0,
        "selected_lower_than_perturbed_steps": 0,
        "emergency_fallback_steps": 0,
    }


def main() -> None:
    raw = _load(RAW_SOURCE)

    consolidated = _load(CONSOLIDATED_SOURCE)

    attribution = _load(ATTRIBUTION_SOURCE)

    cells_raw = raw["cells"]

    if not isinstance(
        cells_raw,
        list,
    ):
        raise TypeError("raw action cells must be list")

    if len(cells_raw) != 216:
        raise RuntimeError("expected 216 raw action cells")

    lyapunov_cells = [
        cell
        for cell in cells_raw
        if isinstance(
            cell,
            dict,
        )
        and cell.get("method") == "lyapunov"
    ]

    if len(lyapunov_cells) != 72:
        raise RuntimeError("expected 72 Lyapunov action cells")

    domain_records = {domain: _empty_domain_record(domain) for domain in DOMAINS}

    run_sources: list[str] = []

    for cell in lyapunov_cells:
        domain = str(cell["domain"])

        if domain not in domain_records:
            raise RuntimeError(f"unexpected domain: {domain}")

        run_path = _resolve_run_path(str(cell["source"]))

        if not run_path.exists():
            raise FileNotFoundError(run_path)

        run = _load(run_path)

        if run["domain"] != domain:
            raise RuntimeError("run domain does not match summary cell")

        if run["method"] != "lyapunov":
            raise RuntimeError("expected Lyapunov run")

        seed_summary = run["seed_summary"]

        if not isinstance(
            seed_summary,
            dict,
        ):
            raise TypeError("seed_summary must be dict")

        total_steps = int(seed_summary["total_environment_steps"])

        if total_steps != int(cell["total_environment_steps"]):
            raise RuntimeError("run/summary environment-step mismatch")

        reason_rates = seed_summary["intervention_reason_rates"]

        if not isinstance(
            reason_rates,
            dict,
        ):
            raise TypeError("intervention_reason_rates must be dict")

        record = domain_records[domain]

        record["run_cells"] += 1

        record["total_environment_steps"] += total_steps

        for reason in REASON_KEYS:
            rate = float(
                reason_rates.get(
                    reason,
                    0.0,
                )
            )

            count = exact_count_from_rate(
                rate,
                total_steps,
            )

            record["intervention_reason_counts"][reason] += count

        strict_count = exact_count_from_rate(
            float(seed_summary["strict_lyapunov_decrease_rate"]),
            total_steps,
        )

        selected_lower_count = exact_count_from_rate(
            float(seed_summary["selected_lower_than_perturbed_rate"]),
            total_steps,
        )

        fallback_count = exact_count_from_rate(
            float(seed_summary["emergency_fallback_rate"]),
            total_steps,
        )

        record["strict_lyapunov_decrease_steps"] += strict_count

        record["selected_lower_than_perturbed_steps"] += selected_lower_count

        record["emergency_fallback_steps"] += fallback_count

        run_sources.append(
            str(run_path.relative_to(ROOT)).replace(
                "\\",
                "/",
            )
        )

    summaries: list[dict[str, Any]] = []

    for domain in DOMAINS:
        record = domain_records[domain]

        reasons = record["intervention_reason_counts"]

        if not isinstance(
            reasons,
            dict,
        ):
            raise TypeError("reason counts must be dict")

        intervention_steps = (
            int(reasons["action_bound"])
            + int(reasons["domain_constraint"])
            + int(reasons["lyapunov_decrease"])
        )

        hard_guard_interventions = int(reasons["action_bound"]) + int(
            reasons["domain_constraint"]
        )

        record["intervention_steps"] = intervention_steps

        record["hard_guard_interventions"] = hard_guard_interventions

        record["lyapunov_decrease_interventions"] = int(reasons["lyapunov_decrease"])

        record["hard_guard_fraction_of_interventions"] = (
            hard_guard_interventions / intervention_steps
            if intervention_steps > 0
            else None
        )

        record["lyapunov_fraction_of_interventions"] = (
            int(reasons["lyapunov_decrease"]) / intervention_steps
            if intervention_steps > 0
            else None
        )

        record["strict_decrease_rate_among_environment_steps"] = int(
            record["strict_lyapunov_decrease_steps"]
        ) / int(record["total_environment_steps"])

        record["strict_decrease_rate_among_interventions"] = (
            int(record["strict_lyapunov_decrease_steps"]) / intervention_steps
            if intervention_steps > 0
            else None
        )

        record["activation_observed"] = int(reasons["lyapunov_decrease"]) > 0

        summaries.append(record)

    global_reason_counts = {
        reason: sum(
            int(domain_records[domain]["intervention_reason_counts"][reason])
            for domain in DOMAINS
        )
        for reason in REASON_KEYS
    }

    global_strict = sum(
        int(domain_records[domain]["strict_lyapunov_decrease_steps"])
        for domain in DOMAINS
    )

    global_selected_lower = sum(
        int(domain_records[domain]["selected_lower_than_perturbed_steps"])
        for domain in DOMAINS
    )

    global_fallback = sum(
        int(domain_records[domain]["emergency_fallback_steps"]) for domain in DOMAINS
    )

    mechanism = consolidated["mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("consolidated mechanism must be dict")

    expected_reasons = mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        expected_reasons,
        dict,
    ):
        raise TypeError("expected reason counts must be dict")

    for reason in REASON_KEYS:
        if global_reason_counts[reason] != int(expected_reasons[reason]):
            raise RuntimeError(
                "reason-count reconstruction failed for "
                f"{reason}: "
                f"{global_reason_counts[reason]} != "
                f"{expected_reasons[reason]}"
            )

    if global_strict != int(mechanism["strict_lyapunov_decrease_steps"]):
        raise RuntimeError("strict-decrease reconstruction failed")

    if global_selected_lower != int(mechanism["selected_lower_than_perturbed_steps"]):
        raise RuntimeError("selected-lower reconstruction failed")

    if global_fallback != int(mechanism["emergency_fallback_steps"]):
        raise RuntimeError("fallback reconstruction failed")

    if global_reason_counts["lyapunov_decrease"] != 1584:
        raise RuntimeError("expected 1584 Lyapunov-decrease reasons")

    if global_strict != 70:
        raise RuntimeError("expected 70 strict Lyapunov decreases")

    if global_selected_lower != 4887:
        raise RuntimeError("expected 4887 selected-lower steps")

    if global_fallback != 0:
        raise RuntimeError("expected zero emergency fallback steps")

    driving_active = bool(domain_records["autonomous_driving"]["activation_observed"])

    robotics_active = bool(domain_records["robotics"]["activation_observed"])

    activation_consistent = driving_active and robotics_active

    if attribution["active_regimes"] != ["action_perturbation"]:
        raise RuntimeError("unexpected upstream active-regime attribution")

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14G",
        "artifact": "cross-domain-lyapunov-mechanism",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "new_safety_episodes": False,
        "source_artifacts": [
            str(RAW_SOURCE.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            str(CONSOLIDATED_SOURCE.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            str(ATTRIBUTION_SOURCE.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
        ],
        "frozen_run_sources": sorted(run_sources),
        "domain_mechanism_summary": summaries,
        "global_reconstruction": {
            "intervention_reason_counts": global_reason_counts,
            "strict_lyapunov_decrease_steps": global_strict,
            "selected_lower_than_perturbed_steps": global_selected_lower,
            "emergency_fallback_steps": global_fallback,
        },
        "activation_consistency": {
            "driving_activation_observed": driving_active,
            "robotics_activation_observed": robotics_active,
            "lyapunov_activation_cross_domain_consistent": activation_consistent,
        },
        "architecture_decomposition": [
            {
                "level": 1,
                "mechanism": "action_space_guard",
                "reason": "action_bound",
            },
            {
                "level": 2,
                "mechanism": "domain_constraint_guard",
                "reason": "domain_constraint",
            },
            {
                "level": 3,
                "mechanism": "lyapunov_candidate_selection",
                "reason": "lyapunov_decrease",
            },
        ],
        "interpretation": {
            "hard_guards_are_part_of_complete_lyapunov_filter": True,
            "lyapunov_decrease_reason_is_mechanism_activation": True,
            "strict_decrease_is_separate_from_reason_count": True,
            "formal_stability_claim_supported": False,
            "global_superiority_claim_supported": False,
        },
        "limitations": [
            (
                "Intervention-reason counts are reconstructed from "
                "frozen per-run rates using exact environment-step "
                "denominators and integer-integrity checks."
            ),
            (
                "Lyapunov-specific activation demonstrates use of "
                "the candidate-selection mechanism, not a formal "
                "closed-loop stability proof."
            ),
            (
                "Hard-guard and Lyapunov-specific intervention "
                "frequencies are empirical properties of the tested "
                "synthetic action perturbations."
            ),
            (
                "Cross-domain consistency refers to observed "
                "activation in both domains and does not imply equal "
                "activation magnitude."
            ),
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "domain,run_cells,total_environment_steps,"
            "action_bound,domain_constraint,lyapunov_decrease,none,"
            "intervention_steps,hard_guard_interventions,"
            "hard_guard_fraction_of_interventions,"
            "lyapunov_fraction_of_interventions,"
            "strict_lyapunov_decrease_steps,"
            "strict_decrease_rate_among_environment_steps,"
            "strict_decrease_rate_among_interventions,"
            "selected_lower_than_perturbed_steps,"
            "emergency_fallback_steps,"
            "activation_observed"
        )
    ]

    for row in summaries:
        reasons = row["intervention_reason_counts"]

        csv_lines.append(
            ",".join(
                [
                    str(row["domain"]),
                    str(row["run_cells"]),
                    str(row["total_environment_steps"]),
                    str(reasons["action_bound"]),
                    str(reasons["domain_constraint"]),
                    str(reasons["lyapunov_decrease"]),
                    str(reasons["none"]),
                    str(row["intervention_steps"]),
                    str(row["hard_guard_interventions"]),
                    str(row["hard_guard_fraction_of_interventions"]),
                    str(row["lyapunov_fraction_of_interventions"]),
                    str(row["strict_lyapunov_decrease_steps"]),
                    str(row["strict_decrease_rate_among_environment_steps"]),
                    str(row["strict_decrease_rate_among_interventions"]),
                    str(row["selected_lower_than_perturbed_steps"]),
                    str(row["emergency_fallback_steps"]),
                    str(row["activation_observed"]),
                ]
            )
        )

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Sprint 5.14G — Cross-Domain Lyapunov Mechanism",
        "",
        "## Safety architecture decomposition",
        "",
        "1. Action-space guard — ACTION_BOUND",
        "2. Domain constraint guard — DOMAIN_CONSTRAINT",
        "3. Lyapunov candidate selection — LYAPUNOV_DECREASE",
        "",
        "## Domain mechanism split",
        "",
        (
            "| Domain | Action bound | Domain constraint | "
            "Lyapunov decrease | Strict decrease | "
            "Selected lower | Fallback |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        reasons = row["intervention_reason_counts"]

        md_lines.append(
            f"| {row['domain']} | "
            f"{reasons['action_bound']} | "
            f"{reasons['domain_constraint']} | "
            f"{reasons['lyapunov_decrease']} | "
            f"{row['strict_lyapunov_decrease_steps']} | "
            f"{row['selected_lower_than_perturbed_steps']} | "
            f"{row['emergency_fallback_steps']} |"
        )

    md_lines.extend(
        [
            "",
            "## Activation consistency",
            "",
            (f"- Driving activation observed: " f"{driving_active}"),
            (f"- Robotics activation observed: " f"{robotics_active}"),
            (f"- Cross-domain activation consistent: " f"{activation_consistent}"),
            "",
            "## Global reconstruction",
            "",
            (f"- ACTION_BOUND: " f"{global_reason_counts['action_bound']}"),
            (f"- DOMAIN_CONSTRAINT: " f"{global_reason_counts['domain_constraint']}"),
            (f"- LYAPUNOV_DECREASE: " f"{global_reason_counts['lyapunov_decrease']}"),
            (f"- NONE: " f"{global_reason_counts['none']}"),
            (f"- Strict decreases: " f"{global_strict}"),
            (f"- Selected-lower steps: " f"{global_selected_lower}"),
            (f"- Emergency fallback steps: " f"{global_fallback}"),
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

    print("=" * 82)

    print(" SPRINT 5.14G LYAPUNOV MECHANISM DOMAIN SPLIT")

    print("=" * 82)

    print()

    print(
        "Lyapunov action cells:",
        len(lyapunov_cells),
    )

    print()

    for row in summaries:
        reasons = row["intervention_reason_counts"]

        print(row["domain"])

        print(
            "  ACTION_BOUND:",
            reasons["action_bound"],
        )

        print(
            "  DOMAIN_CONSTRAINT:",
            reasons["domain_constraint"],
        )

        print(
            "  LYAPUNOV_DECREASE:",
            reasons["lyapunov_decrease"],
        )

        print(
            "  STRICT DECREASE:",
            row["strict_lyapunov_decrease_steps"],
        )

        print(
            "  SELECTED LOWER:",
            row["selected_lower_than_perturbed_steps"],
        )

        print(
            "  FALLBACK:",
            row["emergency_fallback_steps"],
        )

        print(
            "  ACTIVATION:",
            row["activation_observed"],
        )

        print()

    print(
        "Global LYAPUNOV_DECREASE:",
        global_reason_counts["lyapunov_decrease"],
    )

    print(
        "Global strict decreases:",
        global_strict,
    )

    print(
        "Global selected-lower steps:",
        global_selected_lower,
    )

    print(
        "Global emergency fallback:",
        global_fallback,
    )

    print()

    print(
        "Driving activation:",
        driving_active,
    )

    print(
        "Robotics activation:",
        robotics_active,
    )

    print(
        "Cross-domain activation consistent:",
        activation_consistent,
    )

    print()

    print("Exact integer reconstruction: PASS")

    print("Global mechanism invariants: PASS")

    print("Hard-guard decomposition: PASS")

    print("Strict decrease separated from reason count: PASS")

    print("No formal stability claim: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14G LYAPUNOV MECHANISM ANALYSIS: PASS")


if __name__ == "__main__":
    main()
