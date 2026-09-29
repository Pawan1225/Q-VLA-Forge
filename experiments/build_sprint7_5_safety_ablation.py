"""Build Sprint 7.5 final clean-safety ablation artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.safety_ablation import (
    CLAIM_CONTROLS,
    DOMAINS,
    PRIVILEGED_STATE_LIMITATION,
    REQUIRED_SEEDS,
    SAFETY_METHODS,
    absolute_difference,
    load_canonical_safety_ablation,
    relative_reduction_percent,
    zero_observed_violations,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "safety-ablation"


def _format_mean_sd(
    mean: float | None,
    sample_std: float | None,
    *,
    decimals: int = 6,
) -> str:
    """Format reviewer-facing mean +/- sample SD text."""

    if mean is None:
        return "Not defined"

    if sample_std is None:
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} +/- " f"{sample_std:.{decimals}f}"


def _method_label(
    method: str,
) -> str:
    labels = {
        "none": "None",
        "clipping": "Clipping",
        "lyapunov": "Lyapunov",
    }

    return labels[method]


def _domain_label(
    domain: str,
) -> str:
    if domain == "autonomous_driving":
        return "Autonomous Driving"

    if domain == "robotics":
        return "Robotics"

    raise ValueError(f"Unsupported domain: {domain}")


def _enrich_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add descriptive changes relative to NONE."""

    enriched: list[dict[str, Any]] = []

    for domain in DOMAINS:
        domain_records = [record for record in records if record["domain"] == domain]

        reference = next(
            record for record in domain_records if record["method"] == "none"
        )

        reference_violation = float(reference["violation_rate_mean"])

        for record in domain_records:
            candidate_violation = float(record["violation_rate_mean"])

            if record["method"] == "none":
                relative_reduction = None
                absolute_delta = 0.0
            else:
                relative_reduction = relative_reduction_percent(
                    reference_violation,
                    candidate_violation,
                )

                absolute_delta = absolute_difference(
                    reference_violation,
                    candidate_violation,
                )

            output = dict(record)

            output["violation_absolute_difference_vs_none"] = absolute_delta

            output["violation_relative_reduction_percent_vs_none"] = relative_reduction

            output["zero_observed_violations"] = zero_observed_violations(
                candidate_violation
            )

            enriched.append(output)

    return enriched


def _build_payload() -> dict[str, Any]:
    evidence = load_canonical_safety_ablation(ROOT)

    canonical_records = [record.to_dict() for record in evidence.records]

    records = _enrich_records(canonical_records)

    mechanism_observations = [
        asdict(observation)
        for observation in (evidence.lyapunov_mechanism_observations)
    ]

    sources = sorted(
        {source for record in records for source in record["source_artifacts"]}
        | {
            source
            for observation in mechanism_observations
            for source in observation["source_artifacts"]
        }
    )

    return {
        "sprint": "7.5",
        "title": "Final Safety Ablation",
        "condition": "clean",
        "new_training_performed": False,
        "new_safety_evaluations_performed": False,
        "new_robustness_experiments_performed": False,
        "required_seeds": list(REQUIRED_SEEDS),
        "domains": list(DOMAINS),
        "safety_methods": list(SAFETY_METHODS),
        "primary_metric": ("violation_rate"),
        "comparison_semantics": {
            "none": "REFERENCE",
            "clipping": "EVALUATED",
            "lyapunov": "EVALUATED",
            "binary_pass_fail_threshold_used": False,
        },
        "records": records,
        "lyapunov_mechanism_observations": (mechanism_observations),
        "privileged_state_limitation": (PRIVILEGED_STATE_LIMITATION),
        "scientific_conclusion": (
            "Under the frozen Phase 1 synthetic clean-safety "
            "contracts, both clipping and the classical "
            "Lyapunov-guided filter produced zero observed "
            "violation-step rate in the evaluated runs, compared "
            "with nonzero violation rates for the no-safety "
            "reference. These empirical observations do not "
            "constitute a formal stability, forward-invariance, "
            "certification, production-safety, real-world-safety, "
            "or quantum-safety guarantee."
        ),
        "mechanism_conclusion": (
            "No principal LYAPUNOV_DECREASE intervention was "
            "observed in either clean domain evaluation. Observed "
            "interventions were attributable to the frozen "
            "domain hard-guard mechanisms under the evaluated "
            "principal trajectories."
        ),
        "claim_controls": dict(CLAIM_CONTROLS),
        "source_provenance": (sources),
    }


def _validate_payload(
    payload: dict[str, Any],
) -> None:
    """Builder-side integrity checks."""

    records = payload["records"]

    if len(records) != 6:
        raise ValueError("Expected six safety-ablation records.")

    observed_matrix = {
        (
            record["domain"],
            record["method"],
        )
        for record in records
    }

    expected_matrix = {
        (
            domain,
            method,
        )
        for domain in DOMAINS
        for method in SAFETY_METHODS
    }

    if observed_matrix != expected_matrix:
        raise ValueError("Safety-ablation matrix mismatch.")

    for record in records:
        if tuple(record["seeds"]) != REQUIRED_SEEDS:
            raise ValueError("Seed protocol changed.")

        if record["method"] == "none":
            if record["role"] != "REFERENCE":
                raise ValueError("NONE must remain REFERENCE.")

            if record["intervention_rate_mean"] is not None:
                raise ValueError("NONE intervention metric " "must remain undefined.")

        else:
            if record["role"] != "EVALUATED":
                raise ValueError("Safety method role mismatch.")

        if (
            record["method"] == "lyapunov"
            and record["lyapunov_method_classification"] != "classical"
        ):
            raise ValueError("Lyapunov method must remain classical.")

    evaluated = [
        record
        for record in records
        if record["method"]
        in (
            "clipping",
            "lyapunov",
        )
    ]

    if not all(record["violation_rate_mean"] == 0.0 for record in evaluated):
        raise ValueError("Frozen zero-observed-violation result changed.")

    mechanism = payload["lyapunov_mechanism_observations"]

    if len(mechanism) != 2:
        raise ValueError("Expected two Lyapunov mechanism observations.")

    if any(
        observation["lyapunov_decrease_interventions_observed"]
        for observation in mechanism
    ):
        raise ValueError(
            "Frozen clean runs did not observe " "LYAPUNOV_DECREASE interventions."
        )

    if not payload["privileged_state_limitation"].strip():
        raise ValueError("Privileged-state limitation missing.")

    for value in payload["claim_controls"].values():
        if value is not False:
            raise ValueError("All Sprint 7.5 claim controls " "must remain blocked.")

    for source in payload["source_provenance"]:
        if not source.startswith("results/safety/"):
            raise ValueError(f"Unexpected source: {source}")

        if any(
            fragment in source
            for fragment in (
                "gaussian-robustness",
                "structured-state-robustness",
                "action-robustness",
            )
        ):
            raise ValueError(
                "Robustness evidence contaminated " "the clean safety ablation."
            )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "domain",
        "method",
        "role",
        "violation_rate_mean",
        "violation_rate_sample_std",
        "violation_absolute_difference_vs_none",
        "violation_relative_reduction_percent_vs_none",
        "zero_observed_violations",
        "reward_mean",
        "reward_sample_std",
        "success_rate_mean",
        "success_rate_sample_std",
        "intervention_rate_mean",
        "intervention_rate_sample_std",
        "lyapunov_method_classification",
        "seeds",
        "source_artifacts",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in payload["records"]:
            writer.writerow(
                {
                    "domain": record["domain"],
                    "method": record["method"],
                    "role": record["role"],
                    "violation_rate_mean": record["violation_rate_mean"],
                    "violation_rate_sample_std": record["violation_rate_sample_std"],
                    "violation_absolute_difference_vs_none": (
                        record["violation_absolute_difference_vs_none"]
                    ),
                    "violation_relative_reduction_percent_vs_none": (
                        record["violation_relative_reduction_percent_vs_none"]
                    ),
                    "zero_observed_violations": record["zero_observed_violations"],
                    "reward_mean": record["reward_mean"],
                    "reward_sample_std": record["reward_sample_std"],
                    "success_rate_mean": record["success_rate_mean"],
                    "success_rate_sample_std": record["success_rate_sample_std"],
                    "intervention_rate_mean": record["intervention_rate_mean"],
                    "intervention_rate_sample_std": record[
                        "intervention_rate_sample_std"
                    ],
                    "lyapunov_method_classification": (
                        record["lyapunov_method_classification"]
                    ),
                    "seeds": ";".join(str(seed) for seed in record["seeds"]),
                    "source_artifacts": ";".join(record["source_artifacts"]),
                }
            )


def _write_plot_data(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "domain",
        "method",
        "role",
        "violation_rate_mean",
        "violation_rate_sample_std",
        "reward_mean",
        "reward_sample_std",
        "success_rate_mean",
        "success_rate_sample_std",
        "intervention_rate_mean",
        "intervention_rate_sample_std",
        "violation_relative_reduction_percent_vs_none",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in payload["records"]:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _build_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge - Sprint 7.5 Final Safety Ablation",
        "",
        (
            "Frozen Sprint 5 clean-safety evidence only. "
            "No new safety episodes, controller tuning, "
            "Lyapunov redesign, reward changes, or robustness "
            "experiments were performed."
        ),
        "",
        (
            "NONE is the descriptive reference. CLIPPING and "
            "LYAPUNOV are evaluated configurations. Sprint 7.5 "
            "does not define a new binary safety PASS/FAIL threshold."
        ),
        "",
    ]

    for domain in DOMAINS:
        lines.extend(
            [
                f"## {_domain_label(domain)}",
                "",
                (
                    "| Method | Violation Rate | Reward | "
                    "Success Rate | Intervention Rate | Role |"
                ),
                ("|---|---:|---:|---:|---:|---|"),
            ]
        )

        domain_records = [
            record for record in payload["records"] if record["domain"] == domain
        ]

        for method in SAFETY_METHODS:
            record = next(item for item in domain_records if item["method"] == method)

            violation = _format_mean_sd(
                record["violation_rate_mean"],
                record["violation_rate_sample_std"],
            )

            reward = _format_mean_sd(
                record["reward_mean"],
                record["reward_sample_std"],
            )

            success = _format_mean_sd(
                record["success_rate_mean"],
                record["success_rate_sample_std"],
            )

            intervention = _format_mean_sd(
                record["intervention_rate_mean"],
                record["intervention_rate_sample_std"],
            )

            lines.append(
                f"| {_method_label(method)} | "
                f"{violation} | "
                f"{reward} | "
                f"{success} | "
                f"{intervention} | "
                f"{record['role']} |"
            )

        lines.extend(
            [
                "",
                "### Violation Change Relative to NONE",
                "",
            ]
        )

        for method in (
            "clipping",
            "lyapunov",
        ):
            record = next(item for item in domain_records if item["method"] == method)

            relative = record["violation_relative_reduction_percent_vs_none"]

            absolute = record["violation_absolute_difference_vs_none"]

            relative_text = "Undefined" if relative is None else f"{relative:.2f}%"

            lines.extend(
                [
                    (
                        f"- {_method_label(method)} absolute "
                        f"difference vs NONE: "
                        f"{absolute:.6f}"
                    ),
                    (
                        f"- {_method_label(method)} relative "
                        f"reduction vs NONE: "
                        f"{relative_text}"
                    ),
                ]
            )

        lines.append("")

    lines.extend(
        [
            "## Lyapunov Mechanism Observation",
            "",
            (
                "The Sprint 5 Lyapunov-guided safety filter is "
                "classical. It is not a quantum safety algorithm."
            ),
            "",
        ]
    )

    for observation in payload["lyapunov_mechanism_observations"]:
        observed = (
            "yes" if observation["lyapunov_decrease_interventions_observed"] else "no"
        )

        lines.extend(
            [
                (
                    f"- {_domain_label(observation['domain'])}: "
                    f"LYAPUNOV_DECREASE intervention observed: "
                    f"{observed}."
                ),
                (f"  {observation['interpretation']}"),
            ]
        )

    lines.extend(
        [
            "",
            "## Scientific Conclusion",
            "",
            payload["scientific_conclusion"],
            "",
            payload["mechanism_conclusion"],
            "",
            "## Privileged-State Limitation",
            "",
            payload["privileged_state_limitation"],
            "",
            "## Claim Controls",
            "",
            "- Formal stability guarantee: blocked",
            "- Formal Lyapunov stability proof: blocked",
            "- Forward-invariance guarantee: blocked",
            "- Worst-case safety guarantee: blocked",
            "- ISO 26262 certification claim: blocked",
            "- Production-safety claim: blocked",
            "- Real-world-safety claim: blocked",
            "- Universal safety-controller claim: blocked",
            "- Quantum-safety advantage claim: blocked",
            "- Guaranteed-zero-violations claim: blocked",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = _build_payload()

    _validate_payload(payload)

    json_path = OUTPUT_DIR / "safety-ablation.json"

    csv_path = OUTPUT_DIR / "safety-ablation.csv"

    markdown_path = OUTPUT_DIR / "safety-ablation.md"

    plot_path = OUTPUT_DIR / "safety-ablation-plot-data.csv"

    _write_json(
        json_path,
        payload,
    )

    _write_csv(
        csv_path,
        payload,
    )

    _write_plot_data(
        plot_path,
        payload,
    )

    markdown_path.write_text(
        _build_markdown(payload),
        encoding="utf-8",
    )

    print("=" * 64)
    print(" SPRINT 7.5 SAFETY ABLATION BUILD")
    print("=" * 64)
    print()
    print("Artifacts:")
    print("  safety-ablation.json             PASS")
    print("  safety-ablation.csv              PASS")
    print("  safety-ablation.md               PASS")
    print("  safety-ablation-plot-data.csv    PASS")
    print()
    print("Canonical records:                 6")
    print("Safety methods:                    3")
    print("Domains:                           2")
    print("Robustness evidence excluded:      PASS")
    print("Lyapunov classification:           classical")
    print("Claim controls:                    PASS")
    print()
    print("SPRINT 7.5 SAFETY ABLATION BUILD: PASS")


if __name__ == "__main__":
    main()
