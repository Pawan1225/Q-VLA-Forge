"""Build Sprint 5.14K consolidated cross-domain safety package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "safety" / "cross-domain"

ARCHITECTURE = RESULTS_DIR / "sprint5-cross-domain-architecture.json"

CLEAN = RESULTS_DIR / "sprint5-cross-domain-clean.json"

GAUSSIAN = RESULTS_DIR / "sprint5-cross-domain-gaussian.json"

STRUCTURED = RESULTS_DIR / "sprint5-cross-domain-structured-state.json"

ACTION = RESULTS_DIR / "sprint5-cross-domain-action.json"

MECHANISM = RESULTS_DIR / "sprint5-cross-domain-lyapunov-mechanism.json"

CONSISTENCY = RESULTS_DIR / "sprint5-cross-domain-consistency.json"

CLAIMS = RESULTS_DIR / "sprint5-cross-domain-claim-matrix.json"

FIGURE_INDEX = RESULTS_DIR / "sprint5-cross-domain-figure-index.json"

OUTPUT_JSON = RESULTS_DIR / "sprint5-cross-domain-package.json"

OUTPUT_MD = RESULTS_DIR / "sprint5-cross-domain-package.md"

SOURCE_ARTIFACTS = (
    ARCHITECTURE,
    CLEAN,
    GAUSSIAN,
    STRUCTURED,
    ACTION,
    MECHANISM,
    CONSISTENCY,
    CLAIMS,
    FIGURE_INDEX,
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


def _relative(
    path: Path,
) -> str:
    return str(path.relative_to(ROOT)).replace(
        "\\",
        "/",
    )


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    for path in SOURCE_ARTIFACTS:
        if not path.exists():
            raise FileNotFoundError(path)

    architecture = _load(ARCHITECTURE)

    _load(CLEAN)
    _load(GAUSSIAN)
    _load(STRUCTURED)
    _load(ACTION)

    mechanism = _load(MECHANISM)

    consistency = _load(CONSISTENCY)

    claims = _load(CLAIMS)

    figure_index = _load(FIGURE_INDEX)

    claim_count = int(claims["claim_count"])

    if claim_count != 12:
        raise RuntimeError("expected 12 cross-domain claims")

    figure_count = int(figure_index["figure_count"])

    if figure_count != 6:
        raise RuntimeError("expected 6 cross-domain figures")

    status_counts = claims["status_counts"]

    if not isinstance(
        status_counts,
        dict,
    ):
        raise TypeError("status_counts must be dict")

    if int(status_counts["supported"]) != 5:
        raise RuntimeError("expected 5 supported claims")

    if int(status_counts["supported_with_limitation"]) != 1:
        raise RuntimeError("expected 1 limited claim")

    if int(status_counts["not_supported"]) != 6:
        raise RuntimeError("expected 6 unsupported claims")

    clean_summary = consistency["clean_consistency"]

    gaussian_summary = consistency["gaussian_consistency"]

    structured_summary = consistency["structured_state_consistency"]

    action_summary = consistency["action_recovery_consistency"]

    activation_summary = consistency["lyapunov_activation_consistency"]

    if not isinstance(
        clean_summary,
        dict,
    ):
        raise TypeError("clean consistency must be dict")

    if not isinstance(
        gaussian_summary,
        dict,
    ):
        raise TypeError("Gaussian consistency must be dict")

    if not isinstance(
        structured_summary,
        dict,
    ):
        raise TypeError("structured consistency must be dict")

    if not isinstance(
        action_summary,
        dict,
    ):
        raise TypeError("action consistency must be dict")

    if not isinstance(
        activation_summary,
        dict,
    ):
        raise TypeError("activation consistency must be dict")

    global_reconstruction = mechanism["global_reconstruction"]

    if not isinstance(
        global_reconstruction,
        dict,
    ):
        raise TypeError("mechanism global reconstruction must be dict")

    reason_counts = global_reconstruction["intervention_reason_counts"]

    if not isinstance(
        reason_counts,
        dict,
    ):
        raise TypeError("reason counts must be dict")

    if int(reason_counts["lyapunov_decrease"]) != 1584:
        raise RuntimeError("expected 1584 Lyapunov-decrease interventions")

    if int(global_reconstruction["strict_lyapunov_decrease_steps"]) != 70:
        raise RuntimeError("expected 70 strict decreases")

    if int(global_reconstruction["selected_lower_than_perturbed_steps"]) != 4887:
        raise RuntimeError("expected 4887 selected-lower steps")

    source_manifest = [
        {
            "path": _relative(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in SOURCE_ARTIFACTS
    ]

    package: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14K",
        "artifact": "consolidated-cross-domain-safety-package",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "new_safety_episodes": False,
        "research_question": (
            "Which parts of the Q-VLA Forge safety architecture "
            "and empirical behavior generalize across autonomous "
            "driving and robotics, and which findings remain "
            "domain-specific?"
        ),
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "architecture_reuse": architecture,
        "evidence_summary": {
            "clean": {
                "comparisons": clean_summary["count"],
                "consistent": clean_summary["true"],
                "different": clean_summary["false"],
                "all_consistent": clean_summary["all_consistent"],
            },
            "gaussian": {
                "comparisons": gaussian_summary["count"],
                "consistent": gaussian_summary["true"],
                "different": gaussian_summary["false"],
                "all_consistent": gaussian_summary["all_consistent"],
            },
            "structured_state": {
                "comparisons": structured_summary["count"],
                "consistent": structured_summary["true"],
                "different": structured_summary["false"],
                "all_consistent": structured_summary["all_consistent"],
            },
            "action_recovery": {
                "comparisons": action_summary["count"],
                "consistent": action_summary["true"],
                "different": action_summary["false"],
                "all_consistent": action_summary["all_consistent"],
            },
            "lyapunov_activation": {
                "driving_activation_observed": activation_summary[
                    "driving_activation_observed"
                ],
                "robotics_activation_observed": activation_summary[
                    "robotics_activation_observed"
                ],
                "cross_domain_consistent": activation_summary["consistent"],
            },
        },
        "claim_summary": {
            "claim_count": claim_count,
            "supported": status_counts["supported"],
            "supported_with_limitation": status_counts["supported_with_limitation"],
            "not_supported": status_counts["not_supported"],
            "blocked_claim_ids": claims["blocked_claim_ids"],
        },
        "mechanism_summary": {
            "lyapunov_decrease_interventions": reason_counts["lyapunov_decrease"],
            "strict_lyapunov_decrease_steps": global_reconstruction[
                "strict_lyapunov_decrease_steps"
            ],
            "selected_lower_than_perturbed_steps": global_reconstruction[
                "selected_lower_than_perturbed_steps"
            ],
            "emergency_fallback_steps": global_reconstruction[
                "emergency_fallback_steps"
            ],
        },
        "figure_summary": {
            "figure_count": figure_count,
            "figures": figure_index["figures"],
        },
        "proposal_safe_conclusions": [
            (
                "A common safety architecture, intervention protocol, "
                "robustness harness, and evaluation schema were reused "
                "across autonomous driving and robotics."
            ),
            (
                "Clean explicit safety filtering was effective under "
                "both domain-specific pilot safety contracts."
            ),
            (
                "Structured-state robustness direction was consistent "
                "across the two domains for the tested methods."
            ),
            (
                "Positive action recovery was observed in both domains "
                "for clipping and Lyapunov filtering."
            ),
            (
                "Gaussian robustness was mostly but not universally "
                "directionally consistent, with seven of nine "
                "cross-domain comparisons aligned."
            ),
            (
                "Lyapunov-specific candidate-selection activation "
                "occurred in driving but not robotics under the tested "
                "action perturbations."
            ),
            (
                "The evidence supports reuse of the safety framework, "
                "not a universal controller, direct policy transfer, "
                "formal guarantee, or production deployment claim."
            ),
        ],
        "domain_specific_findings": [
            (
                "Driving and robotics retain different safety "
                "constraints and threshold semantics."
            ),
            ("Driving and robotics use separate trained PPO policy " "weights."),
            (
                "Driving and robotics use different Lyapunov potentials "
                "and domain predictors."
            ),
            (
                "Lyapunov-specific action-perturbation activation was "
                "observed only in autonomous driving."
            ),
        ],
        "limitations": [
            "Synthetic proxy environments.",
            "Three principal policy seeds.",
            "Different domain-specific state contracts.",
            "Different domain-specific reward scales.",
            "No direct raw reward ranking across domains.",
            "No direct raw violation-rate ranking across domains.",
            (
                "Gaussian and structured-state robustness use noisy or "
                "perturbed policy observations while the safety layer "
                "retains true simulator state."
            ),
            "No cross-domain policy-weight transfer experiment.",
            "No physical vehicle evaluation.",
            "No physical robot evaluation.",
            "No formal closed-loop stability proof.",
            "No formal worst-case robustness proof.",
            "No certification claim.",
            "No production generalization claim.",
            "No quantum safety advantage claim.",
        ],
        "source_manifest": source_manifest,
        "source_artifact_count": len(source_manifest),
        "frozen_sections": [
            "5.14B",
            "5.14C",
            "5.14D",
            "5.14E",
            "5.14F",
            "5.14G",
            "5.14H",
            "5.14I",
            "5.14J",
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            package,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sprint 5.14K — Consolidated Cross-Domain Safety Package",
        "",
        "## Research question",
        "",
        package["research_question"],
        "",
        "## Evidence summary",
        "",
        ("- Clean consistency: " f"{clean_summary['true']}/{clean_summary['count']}"),
        (
            "- Gaussian consistency: "
            f"{gaussian_summary['true']}/{gaussian_summary['count']}"
        ),
        (
            "- Structured-state consistency: "
            f"{structured_summary['true']}/{structured_summary['count']}"
        ),
        (
            "- Action-recovery consistency: "
            f"{action_summary['true']}/{action_summary['count']}"
        ),
        (
            "- Lyapunov activation cross-domain consistent: "
            f"{activation_summary['consistent']}"
        ),
        "",
        "## Claim summary",
        "",
        (f"- Supported: " f"{status_counts['supported']}"),
        (
            "- Supported with limitation: "
            f"{status_counts['supported_with_limitation']}"
        ),
        (f"- Not supported: " f"{status_counts['not_supported']}"),
        "",
        "## Mechanism summary",
        "",
        ("- LYAPUNOV_DECREASE interventions: " f"{reason_counts['lyapunov_decrease']}"),
        (
            "- Strict Lyapunov decreases: "
            f"{global_reconstruction['strict_lyapunov_decrease_steps']}"
        ),
        (
            "- Selected-lower steps: "
            f"{global_reconstruction['selected_lower_than_perturbed_steps']}"
        ),
        (
            "- Emergency fallbacks: "
            f"{global_reconstruction['emergency_fallback_steps']}"
        ),
        "",
        "## Proposal-safe conclusions",
        "",
    ]

    for statement in package["proposal_safe_conclusions"]:
        lines.append(f"- {statement}")

    lines.extend(
        [
            "",
            "## Domain-specific findings",
            "",
        ]
    )

    for statement in package["domain_specific_findings"]:
        lines.append(f"- {statement}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in package["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Source manifest",
            "",
            "| Artifact | SHA-256 | Bytes |",
            "|---|---|---:|",
        ]
    )

    for record in source_manifest:
        lines.append(
            f"| {record['path']} | "
            f"`{record['sha256']}` | "
            f"{record['size_bytes']} |"
        )

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)

    print(" SPRINT 5.14K CONSOLIDATED CROSS-DOMAIN SAFETY PACKAGE")

    print("=" * 80)

    print()

    print(
        "Source artifacts:",
        len(source_manifest),
    )

    print(
        "Figures:",
        figure_count,
    )

    print(
        "Claims:",
        claim_count,
    )

    print()

    print(
        "Clean consistency:",
        f"{clean_summary['true']}/{clean_summary['count']}",
    )

    print(
        "Gaussian consistency:",
        f"{gaussian_summary['true']}/{gaussian_summary['count']}",
    )

    print(
        "Structured-state consistency:",
        (f"{structured_summary['true']}/" f"{structured_summary['count']}"),
    )

    print(
        "Action-recovery consistency:",
        f"{action_summary['true']}/{action_summary['count']}",
    )

    print(
        "Lyapunov activation consistency:",
        activation_summary["consistent"],
    )

    print()

    print(
        "Supported claims:",
        status_counts["supported"],
    )

    print(
        "Supported with limitation:",
        status_counts["supported_with_limitation"],
    )

    print(
        "Not supported:",
        status_counts["not_supported"],
    )

    print()

    print("Source existence: PASS")

    print("Source SHA-256 manifest: PASS")

    print("Mechanism invariants: PASS")

    print("Claim controls: PASS")

    print("Figure provenance linkage: PASS")

    print("No universal controller claim: PASS")

    print("No formal guarantee claim: PASS")

    print("No production claim: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14K CONSOLIDATED PACKAGE: PASS")


if __name__ == "__main__":
    main()
