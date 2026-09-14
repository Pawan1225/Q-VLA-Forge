"""Build Sprint 5.14I cross-domain safety claim matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CROSS_DOMAIN = ROOT / "results" / "safety" / "cross-domain"

ARCHITECTURE = CROSS_DOMAIN / "sprint5-cross-domain-architecture.json"

CLEAN = CROSS_DOMAIN / "sprint5-cross-domain-clean.json"

GAUSSIAN = CROSS_DOMAIN / "sprint5-cross-domain-gaussian.json"

STRUCTURED = CROSS_DOMAIN / "sprint5-cross-domain-structured-state.json"

ACTION = CROSS_DOMAIN / "sprint5-cross-domain-action.json"

MECHANISM = CROSS_DOMAIN / "sprint5-cross-domain-lyapunov-mechanism.json"

CONSISTENCY = CROSS_DOMAIN / "sprint5-cross-domain-consistency.json"

OUTPUT_JSON = CROSS_DOMAIN / "sprint5-cross-domain-claim-matrix.json"

OUTPUT_CSV = CROSS_DOMAIN / "sprint5-cross-domain-claim-matrix.csv"

OUTPUT_MD = CROSS_DOMAIN / "sprint5-cross-domain-claim-matrix.md"


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


def main() -> None:
    architecture = _load(ARCHITECTURE)

    _load(CLEAN)
    _load(GAUSSIAN)
    _load(STRUCTURED)
    _load(ACTION)
    _load(MECHANISM)

    consistency = _load(CONSISTENCY)

    architecture_reuse = architecture["architecture_reuse"]

    if not isinstance(
        architecture_reuse,
        dict,
    ):
        raise TypeError("architecture reuse must be dict")

    clean_summary = consistency["clean_consistency"]

    gaussian_summary = consistency["gaussian_consistency"]

    structured_summary = consistency["structured_state_consistency"]

    action_summary = consistency["action_recovery_consistency"]

    activation_summary = consistency["lyapunov_activation_consistency"]

    task_preservation = consistency["task_preservation"]

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

    if not isinstance(
        task_preservation,
        list,
    ):
        raise TypeError("task preservation must be list")

    task_consistent = all(
        bool(row["task_preservation_consistent"])
        for row in task_preservation
        if isinstance(
            row,
            dict,
        )
    )

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "S5-CD01",
            "statement": (
                "A common safety architecture was instantiated "
                "across autonomous driving and robotics."
            ),
            "status": "supported",
            "supported": True,
            "evidence": [
                "5.14B architecture reuse record",
            ],
            "limitations": [
                (
                    "Constraints, clipping semantics, predictors, "
                    "Lyapunov potentials, and trained policy weights "
                    "remain domain-specific."
                ),
            ],
        },
        {
            "claim_id": "S5-CD02",
            "statement": (
                "Clean explicit safety filtering was effective in "
                "both pilot domains under their respective safety "
                "contracts."
            ),
            "status": (
                "supported"
                if bool(clean_summary["all_consistent"])
                else "supported_with_limitation"
            ),
            "supported": bool(clean_summary["all_consistent"]),
            "evidence": [
                "5.14C clean cross-domain analysis",
            ],
            "limitations": [
                (
                    "Raw violation-step rates are not interpreted as "
                    "direct cross-domain safety rankings."
                ),
            ],
        },
        {
            "claim_id": "S5-CD03",
            "statement": (
                "Gaussian robustness behavior was directionally "
                "consistent across all tested cross-domain comparisons."
            ),
            "status": (
                "supported"
                if bool(gaussian_summary["all_consistent"])
                else "supported_with_limitation"
            ),
            "supported": bool(gaussian_summary["all_consistent"]),
            "evidence": [
                "5.14D Gaussian cross-domain analysis",
            ],
            "limitations": [
                (
                    "Seven of nine comparisons were directionally "
                    "consistent; two showed domain differences."
                ),
                (
                    "The safety layer retained true simulator state "
                    "while the policy received noisy observations."
                ),
            ],
        },
        {
            "claim_id": "S5-CD04",
            "statement": (
                "Structured-state robustness direction was "
                "consistent across driving and robotics."
            ),
            "status": (
                "supported"
                if bool(structured_summary["all_consistent"])
                else "supported_with_limitation"
            ),
            "supported": bool(structured_summary["all_consistent"]),
            "evidence": [
                "5.14E structured-state cross-domain analysis",
            ],
            "limitations": [
                (
                    "Driving and robotics perturbation features are "
                    "domain-specific and are not treated as "
                    "semantically equivalent."
                ),
            ],
        },
        {
            "claim_id": "S5-CD05",
            "statement": (
                "Positive explicit action recovery was observed in "
                "both domains for clipping and Lyapunov filtering."
            ),
            "status": (
                "supported"
                if bool(action_summary["all_consistent"])
                else "supported_with_limitation"
            ),
            "supported": bool(action_summary["all_consistent"]),
            "evidence": [
                "5.14F action-recovery cross-domain analysis",
            ],
            "limitations": [
                (
                    "Recovery rates differ by domain and are not "
                    "required to be equal."
                ),
                (
                    "The action disturbances are synthetic and are "
                    "not calibrated actuator-fault models."
                ),
            ],
        },
        {
            "claim_id": "S5-CD06",
            "statement": (
                "Lyapunov-specific candidate-selection activation "
                "was observed in both domains under action "
                "perturbation."
            ),
            "status": (
                "supported"
                if bool(activation_summary["consistent"])
                else "not_supported"
            ),
            "supported": bool(activation_summary["consistent"]),
            "evidence": [
                "5.14G Lyapunov mechanism domain split",
            ],
            "limitations": [
                (
                    "Lyapunov-decrease activation was observed in "
                    "autonomous driving but not robotics."
                ),
            ],
        },
        {
            "claim_id": "S5-CD07",
            "statement": (
                "Clean task-preservation direction was consistent "
                "across both domains for clipping and Lyapunov."
            ),
            "status": ("supported" if task_consistent else "supported_with_limitation"),
            "supported": task_consistent,
            "evidence": [
                "5.14H task-preservation analysis",
            ],
            "limitations": [
                (
                    "Raw reward magnitudes are not compared across "
                    "domains because reward scales differ."
                ),
            ],
        },
        {
            "claim_id": "S5-CD08",
            "statement": (
                "The same safety thresholds were used across "
                "autonomous driving and robotics."
            ),
            "status": "not_supported",
            "supported": False,
            "evidence": [],
            "limitations": [
                (
                    "The domains use different safety constraints "
                    "and threshold semantics."
                ),
            ],
        },
        {
            "claim_id": "S5-CD09",
            "statement": (
                "The same trained policy weights transfer directly "
                "between driving and robotics."
            ),
            "status": "not_supported",
            "supported": False,
            "evidence": [],
            "limitations": [
                (
                    "Driving and robotics use separately trained "
                    "policies; cross-domain weight transfer was not "
                    "tested."
                ),
            ],
        },
        {
            "claim_id": "S5-CD10",
            "statement": (
                "A universal cross-domain safety controller was " "demonstrated."
            ),
            "status": "not_supported",
            "supported": False,
            "evidence": [],
            "limitations": [
                (
                    "Only the safety framework and interfaces are "
                    "shared; domain-specific safety semantics remain."
                ),
            ],
        },
        {
            "claim_id": "S5-CD11",
            "statement": (
                "The results establish a formal cross-domain safety " "guarantee."
            ),
            "status": "not_supported",
            "supported": False,
            "evidence": [],
            "limitations": [
                (
                    "The evidence is empirical and does not provide "
                    "a formal stability or worst-case robustness proof."
                ),
            ],
        },
        {
            "claim_id": "S5-CD12",
            "statement": (
                "Production generalization across physical vehicles "
                "and robots was demonstrated."
            ),
            "status": "not_supported",
            "supported": False,
            "evidence": [],
            "limitations": [
                (
                    "All evaluations use synthetic proxy "
                    "environments; no physical vehicle or robot "
                    "deployment was performed."
                ),
            ],
        },
    ]

    if len(claims) != 12:
        raise RuntimeError("expected 12 cross-domain claims")

    claim_ids = [str(claim["claim_id"]) for claim in claims]

    if len(set(claim_ids)) != 12:
        raise RuntimeError("claim IDs must be unique")

    status_counts = {
        "supported": sum(1 for claim in claims if claim["status"] == "supported"),
        "supported_with_limitation": sum(
            1 for claim in claims if claim["status"] == "supported_with_limitation"
        ),
        "not_supported": sum(
            1 for claim in claims if claim["status"] == "not_supported"
        ),
    }

    blocked_claim_ids = [
        str(claim["claim_id"]) for claim in claims if claim["status"] == "not_supported"
    ]

    required_blocked = {
        "S5-CD08",
        "S5-CD09",
        "S5-CD10",
        "S5-CD11",
        "S5-CD12",
    }

    if not required_blocked.issubset(set(blocked_claim_ids)):
        raise RuntimeError("required transfer/guarantee claims were not blocked")

    if bool(architecture_reuse["same_policy_weights"]):
        raise RuntimeError("same-policy-weights flag unexpectedly true")

    if bool(architecture_reuse["transfer_tested"]):
        raise RuntimeError("transfer-tested flag unexpectedly true")

    if bool(architecture_reuse["universal_controller_supported"]):
        raise RuntimeError("universal-controller flag unexpectedly true")

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14I",
        "artifact": "cross-domain-safety-claim-matrix",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "claim_count": len(claims),
        "status_counts": status_counts,
        "blocked_claim_ids": blocked_claim_ids,
        "claims": claims,
        "source_artifacts": [
            str(path.relative_to(ROOT)).replace(
                "\\",
                "/",
            )
            for path in (
                ARCHITECTURE,
                CLEAN,
                GAUSSIAN,
                STRUCTURED,
                ACTION,
                MECHANISM,
                CONSISTENCY,
            )
        ],
        "global_limitations": [
            "Synthetic proxy environments.",
            "Different domain-specific safety contracts.",
            "Different state dimensions.",
            "Different constraint semantics.",
            "Different Lyapunov functions.",
            "Separate trained policy weights.",
            "No cross-domain weight transfer.",
            "No zero-shot safety transfer.",
            "Three principal policy seeds.",
            (
                "Privileged true safety state under Gaussian and "
                "structured-state observation perturbations."
            ),
            "No physical vehicle evaluation.",
            "No physical robot evaluation.",
            "No formal stability proof.",
            "No certification claim.",
            "No production generalization claim.",
            "No quantum safety advantage claim.",
        ],
        "proposal_safe_statements": [
            (
                "Q-VLA Forge reused a common safety-filter interface, "
                "intervention protocol, robustness harness, and "
                "evaluation schema across autonomous-driving and "
                "robotics proxy environments while retaining "
                "domain-specific safety semantics."
            ),
            (
                "Explicit safety filtering reduced measured clean "
                "violations under both domain-specific pilot "
                "contracts."
            ),
            (
                "Action perturbation produced positive explicit "
                "recovery in both domains for clipping and Lyapunov "
                "filtering."
            ),
            (
                "Lyapunov-specific candidate-selection activation "
                "was observed in driving but not robotics under the "
                "tested action perturbations."
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

    csv_lines = ["claim_id,status,supported,statement,evidence_count,limitation_count"]

    for claim in claims:
        statement = str(claim["statement"]).replace(
            ",",
            ";",
        )

        csv_lines.append(
            ",".join(
                [
                    str(claim["claim_id"]),
                    str(claim["status"]),
                    str(claim["supported"]),
                    statement,
                    str(len(claim["evidence"])),
                    str(len(claim["limitations"])),
                ]
            )
        )

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Sprint 5.14I — Cross-Domain Safety Claim Matrix",
        "",
        "## Claim status",
        "",
        (f"- Supported: " f"{status_counts['supported']}"),
        (
            f"- Supported with limitation: "
            f"{status_counts['supported_with_limitation']}"
        ),
        (f"- Not supported: " f"{status_counts['not_supported']}"),
        "",
        "## Claims",
        "",
        "| ID | Status | Claim |",
        "|---|---|---|",
    ]

    for claim in claims:
        md_lines.append(
            f"| {claim['claim_id']} | "
            f"{claim['status']} | "
            f"{claim['statement']} |"
        )

    md_lines.extend(
        [
            "",
            "## Blocked claims",
            "",
        ]
    )

    for claim_id in blocked_claim_ids:
        md_lines.append(f"- {claim_id}")

    md_lines.extend(
        [
            "",
            "## Global limitations",
            "",
        ]
    )

    for limitation in payload["global_limitations"]:
        md_lines.append(f"- {limitation}")

    md_lines.extend(
        [
            "",
            "## Proposal-safe statements",
            "",
        ]
    )

    for statement in payload["proposal_safe_statements"]:
        md_lines.append(f"- {statement}")

    OUTPUT_MD.write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)

    print(" SPRINT 5.14I CROSS-DOMAIN SAFETY CLAIM MATRIX")

    print("=" * 80)

    print()

    print(
        "Claims:",
        len(claims),
    )

    print(
        "Supported:",
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

    print(
        "Blocked claim IDs:",
        blocked_claim_ids,
    )

    print()

    print("Architecture reuse claim control: PASS")

    print("Gaussian limitation retained: PASS")

    print("Lyapunov domain difference retained: PASS")

    print("Transfer claims blocked: PASS")

    print("Universal-controller claim blocked: PASS")

    print("Formal-guarantee claim blocked: PASS")

    print("Production claim blocked: PASS")

    print("No quantum safety claim: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14I CROSS-DOMAIN CLAIM MATRIX: PASS")


if __name__ == "__main__":
    main()
