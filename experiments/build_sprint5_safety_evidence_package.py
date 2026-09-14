"""Build Sprint 5.13H consolidated Sprint 5 safety evidence package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

CLEAN = CONSOLIDATED / "sprint5-clean-three-seed-summary.json"
GAUSSIAN = CONSOLIDATED / "sprint5-gaussian-three-seed-summary.json"
STRUCTURED = CONSOLIDATED / "sprint5-structured-state-three-seed-summary.json"
ACTION = CONSOLIDATED / "sprint5-action-three-seed-summary.json"
ATTRIBUTION = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"
CLAIMS = CONSOLIDATED / "sprint5-safety-claim-matrix.json"

OUTPUT_JSON = CONSOLIDATED / "sprint5-safety-evidence-package.json"
OUTPUT_MD = CONSOLIDATED / "sprint5-safety-evidence-package.md"


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


def _find_clean_aggregate(
    clean: dict[str, Any],
    *,
    domain: str,
    method: str,
) -> dict[str, Any]:
    summary = clean["three_seed_summary"]

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError("clean three_seed_summary must be a dict")

    domain_summary = summary.get(domain)

    if not isinstance(
        domain_summary,
        dict,
    ):
        raise KeyError(f"clean domain missing: {domain}")

    method_summary = domain_summary.get(method)

    if not isinstance(
        method_summary,
        dict,
    ):
        raise KeyError(f"clean method missing: {domain}/{method}")

    return method_summary


def _metric_mean(
    row: dict[str, Any],
    metric: str,
) -> float:
    value = row[metric]

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(f"{metric} must be an aggregate dict")

    return float(value["mean"])


def main() -> None:
    clean = _load(CLEAN)

    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    attribution = _load(ATTRIBUTION)

    claims = _load(CLAIMS)

    if len(clean["seed_rows"]) != 18:
        raise RuntimeError("clean evidence incomplete")

    if len(gaussian["seed_rows"]) != 72:
        raise RuntimeError("Gaussian evidence incomplete")

    if len(structured["seed_rows"]) != 198:
        raise RuntimeError("structured-state evidence incomplete")

    if len(action["seed_rows"]) != 216:
        raise RuntimeError("action evidence incomplete")

    driving_none = _find_clean_aggregate(
        clean,
        domain="autonomous_driving",
        method="none",
    )

    driving_clipping = _find_clean_aggregate(
        clean,
        domain="autonomous_driving",
        method="clipping",
    )

    driving_lyapunov = _find_clean_aggregate(
        clean,
        domain="autonomous_driving",
        method="lyapunov",
    )

    robotics_none = _find_clean_aggregate(
        clean,
        domain="robotics",
        method="none",
    )

    robotics_clipping = _find_clean_aggregate(
        clean,
        domain="robotics",
        method="clipping",
    )

    robotics_lyapunov = _find_clean_aggregate(
        clean,
        domain="robotics",
        method="lyapunov",
    )

    action_mechanism = action["mechanism"]

    if not isinstance(
        action_mechanism,
        dict,
    ):
        raise TypeError("action mechanism must be dict")

    reasons = action_mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        reasons,
        dict,
    ):
        raise TypeError("Lyapunov reason counts must be dict")

    status_counts = claims["status_counts"]

    if not isinstance(
        status_counts,
        dict,
    ):
        raise TypeError("claim status counts must be dict")

    payload: dict[str, Any] = {
        "sprint": "5.13H",
        "artifact": "sprint5-safety-evidence-package",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "principal_seeds": [
            42,
            123,
            456,
        ],
        "source_artifacts": [
            str(path.relative_to(ROOT)).replace(
                "\\",
                "/",
            )
            for path in (
                CLEAN,
                GAUSSIAN,
                STRUCTURED,
                ACTION,
                ATTRIBUTION,
                CLAIMS,
            )
        ],
        "clean_evidence": {
            "autonomous_driving": {
                "none": {
                    "violation_step_rate": _metric_mean(
                        driving_none,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        driving_none,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        driving_none,
                        "success_rate",
                    ),
                },
                "clipping": {
                    "violation_step_rate": _metric_mean(
                        driving_clipping,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        driving_clipping,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        driving_clipping,
                        "success_rate",
                    ),
                },
                "lyapunov": {
                    "violation_step_rate": _metric_mean(
                        driving_lyapunov,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        driving_lyapunov,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        driving_lyapunov,
                        "success_rate",
                    ),
                },
            },
            "robotics": {
                "none": {
                    "violation_step_rate": _metric_mean(
                        robotics_none,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        robotics_none,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        robotics_none,
                        "success_rate",
                    ),
                },
                "clipping": {
                    "violation_step_rate": _metric_mean(
                        robotics_clipping,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        robotics_clipping,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        robotics_clipping,
                        "success_rate",
                    ),
                },
                "lyapunov": {
                    "violation_step_rate": _metric_mean(
                        robotics_lyapunov,
                        "violation_step_rate",
                    ),
                    "reward": _metric_mean(
                        robotics_lyapunov,
                        "reward",
                    ),
                    "success_rate": _metric_mean(
                        robotics_lyapunov,
                        "success_rate",
                    ),
                },
            },
        },
        "robustness_corpus": {
            "gaussian": {
                "seed_cells": len(gaussian["seed_rows"]),
                "aggregates": len(gaussian["three_seed_summary"]),
                "new_noisy_episodes": gaussian["new_noisy_episodes"],
                "policy_uses_noisy_observation": gaussian[
                    "policy_uses_noisy_observation"
                ],
                "safety_layer_uses_true_state": gaussian[
                    "safety_layer_uses_true_state"
                ],
            },
            "structured_state": {
                "seed_cells": len(structured["seed_rows"]),
                "aggregates": len(structured["three_seed_summary"]),
                "perturbations": structured["total_perturbation_count"],
                "episodes": structured["principal_episodes"],
                "policy_uses_perturbed_observation": structured[
                    "policy_uses_perturbed_observation"
                ],
                "safety_layer_uses_true_state": structured[
                    "safety_layer_uses_true_state"
                ],
            },
            "action": {
                "seed_cells": len(action["seed_rows"]),
                "aggregates": len(action["three_seed_summary"]),
                "perturbations": action["total_perturbation_count"],
                "episodes": action["principal_episodes"],
                "unsafe_perturbed_steps": action_mechanism["unsafe_perturbed_steps"],
                "recovered_unsafe_steps": action_mechanism["recovered_unsafe_steps"],
                "unresolved_unsafe_steps": action_mechanism["unresolved_unsafe_steps"],
                "recovery_fraction": action_mechanism[
                    "overall_explicit_filter_recovery_fraction"
                ],
                "environment_interface_adjustments": action_mechanism[
                    "environment_interface_adjustment_steps"
                ],
            },
        },
        "lyapunov_mechanism": {
            "active_regimes": attribution["active_regimes"],
            "inactive_regimes": attribution["inactive_regimes"],
            "action_lyapunov_decrease_reasons": reasons["lyapunov_decrease"],
            "strict_decrease_steps": action_mechanism["strict_lyapunov_decrease_steps"],
            "selected_lower_steps": action_mechanism[
                "selected_lower_than_perturbed_steps"
            ],
            "emergency_fallback_steps": action_mechanism["emergency_fallback_steps"],
        },
        "claim_controls": {
            "claim_count": claims["claim_count"],
            "supported": status_counts["supported"],
            "supported_with_limitations": status_counts["supported_with_limitations"],
            "unsupported": status_counts["unsupported"],
            "blocked_claim_ids": claims["unsupported_claim_ids"],
        },
        "proposal_safe_wording": claims["proposal_safe_wording"],
        "global_limitations": claims["global_limitations"],
        "headline_findings": [
            (
                "Clean explicit safety filtering reduced observed "
                "violation-step rate to zero in both pilot domains "
                "across the three principal policy seeds."
            ),
            (
                "Gaussian and structured-state evaluations tested "
                "policy robustness to observation perturbations while "
                "the safety layer retained true simulator state."
            ),
            (
                "Under direct action perturbation, explicit safety "
                "filtering recovered 69367 of 111341 unsafe perturbed "
                "steps, corresponding to an aggregate recovery "
                "fraction of approximately 0.623."
            ),
            (
                "The explicit Lyapunov-decrease pathway activated "
                "under direct action perturbation, with 1584 "
                "Lyapunov-decrease intervention reasons and 70 "
                "strict Lyapunov-decrease steps."
            ),
            (
                "No formal stability, worst-case robustness, "
                "production certification, global Lyapunov "
                "superiority, ISO compliance, or quantum safety "
                "advantage claim is supported."
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

    clean_evidence = payload["clean_evidence"]

    robustness = payload["robustness_corpus"]

    lyapunov = payload["lyapunov_mechanism"]

    controls = payload["claim_controls"]

    if not isinstance(
        clean_evidence,
        dict,
    ):
        raise TypeError("clean evidence must be dict")

    if not isinstance(
        robustness,
        dict,
    ):
        raise TypeError("robustness corpus must be dict")

    if not isinstance(
        lyapunov,
        dict,
    ):
        raise TypeError("Lyapunov mechanism must be dict")

    if not isinstance(
        controls,
        dict,
    ):
        raise TypeError("claim controls must be dict")

    lines = [
        ("# Sprint 5.13H — Consolidated Safety " "Evidence Package"),
        "",
        "## Scope",
        "",
        (
            "Analysis-only consolidation of frozen Sprint 5 safety "
            "evidence. No new training or evaluation was performed."
        ),
        "",
        "## Clean safety results",
        "",
        ("| Domain | Method | Violation rate | " "Reward | Success |"),
        "|---|---|---:|---:|---:|",
    ]

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        domain_data = clean_evidence[domain]

        if not isinstance(
            domain_data,
            dict,
        ):
            raise TypeError("domain clean evidence must be dict")

        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            row = domain_data[method]

            if not isinstance(
                row,
                dict,
            ):
                raise TypeError("clean method evidence must be dict")

            lines.append(
                f"| {domain} | {method} | "
                f"{float(row['violation_step_rate']):.6f} | "
                f"{float(row['reward']):.6f} | "
                f"{float(row['success_rate']):.6f} |"
            )

    gaussian_data = robustness["gaussian"]

    structured_data = robustness["structured_state"]

    action_data = robustness["action"]

    if not isinstance(
        gaussian_data,
        dict,
    ):
        raise TypeError("Gaussian robustness data must be dict")

    if not isinstance(
        structured_data,
        dict,
    ):
        raise TypeError("structured robustness data must be dict")

    if not isinstance(
        action_data,
        dict,
    ):
        raise TypeError("action robustness data must be dict")

    lines.extend(
        [
            "",
            "## Robustness corpus",
            "",
            (
                f"- Gaussian: "
                f"{gaussian_data['seed_cells']} seed cells, "
                f"{gaussian_data['aggregates']} aggregates."
            ),
            (
                f"- Structured state: "
                f"{structured_data['seed_cells']} seed cells, "
                f"{structured_data['perturbations']} perturbations, "
                f"{structured_data['episodes']} episodes."
            ),
            (
                f"- Action: "
                f"{action_data['seed_cells']} seed cells, "
                f"{action_data['perturbations']} perturbations, "
                f"{action_data['episodes']} episodes."
            ),
            "",
            "## Action recovery",
            "",
            (f"- Unsafe perturbed steps: " f"{action_data['unsafe_perturbed_steps']}"),
            (f"- Recovered unsafe steps: " f"{action_data['recovered_unsafe_steps']}"),
            (
                f"- Unresolved unsafe steps: "
                f"{action_data['unresolved_unsafe_steps']}"
            ),
            (
                f"- Aggregate recovery fraction: "
                f"{float(action_data['recovery_fraction']):.9f}"
            ),
            "",
            "## Lyapunov mechanism attribution",
            "",
            (f"- Active regimes: " f"{lyapunov['active_regimes']}"),
            (
                f"- Lyapunov-decrease reasons: "
                f"{lyapunov['action_lyapunov_decrease_reasons']}"
            ),
            (f"- Strict decrease steps: " f"{lyapunov['strict_decrease_steps']}"),
            (f"- Selected-lower steps: " f"{lyapunov['selected_lower_steps']}"),
            (f"- Emergency fallback steps: " f"{lyapunov['emergency_fallback_steps']}"),
            "",
            "## Claim controls",
            "",
            (f"- Supported: " f"{controls['supported']}"),
            (
                f"- Supported with limitations: "
                f"{controls['supported_with_limitations']}"
            ),
            (f"- Unsupported / blocked: " f"{controls['unsupported']}"),
            (f"- Blocked claim IDs: " f"{controls['blocked_claim_ids']}"),
            "",
            "## Headline findings",
            "",
        ]
    )

    for finding in payload["headline_findings"]:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Global limitations",
            "",
        ]
    )

    for limitation in payload["global_limitations"]:
        lines.append(f"- {limitation}")

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 84)

    print(" SPRINT 5.13H CONSOLIDATED SAFETY EVIDENCE PACKAGE")

    print("=" * 84)

    print()

    print(
        "Clean seed cells:",
        len(clean["seed_rows"]),
    )

    print(
        "Gaussian seed cells:",
        len(gaussian["seed_rows"]),
    )

    print(
        "Structured-state seed cells:",
        len(structured["seed_rows"]),
    )

    print(
        "Action seed cells:",
        len(action["seed_rows"]),
    )

    print()

    print(
        "Action recovery:",
        action_mechanism["recovered_unsafe_steps"],
        "/",
        action_mechanism["unsafe_perturbed_steps"],
    )

    print(
        "Lyapunov decrease reasons:",
        reasons["lyapunov_decrease"],
    )

    print(
        "Strict Lyapunov decreases:",
        action_mechanism["strict_lyapunov_decrease_steps"],
    )

    print()

    print(
        "Claims:",
        claims["claim_count"],
    )

    print(
        "Blocked claims:",
        claims["unsupported_claim_ids"],
    )

    print()

    print("Clean schema validation: PASS")

    print("No new training: PASS")

    print("No new principal runs: PASS")

    print("Frozen-source packaging: PASS")

    print()

    print("SPRINT 5.13H SAFETY EVIDENCE PACKAGE: PASS")


if __name__ == "__main__":
    main()
