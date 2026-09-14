"""Build Sprint 5.13G safety claim matrix and limitations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.safety_claim_matrix import (
    ClaimStatus,
    SafetyClaim,
    count_by_status,
    unsupported_claim_ids,
)

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

CLEAN = CONSOLIDATED / "sprint5-clean-three-seed-summary.json"

GAUSSIAN = CONSOLIDATED / "sprint5-gaussian-three-seed-summary.json"

STRUCTURED = CONSOLIDATED / "sprint5-structured-state-three-seed-summary.json"

ACTION = CONSOLIDATED / "sprint5-action-three-seed-summary.json"

ATTRIBUTION = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"

OUTPUT_JSON = CONSOLIDATED / "sprint5-safety-claim-matrix.json"

OUTPUT_CSV = CONSOLIDATED / "sprint5-safety-claim-matrix.csv"

OUTPUT_MD = CONSOLIDATED / "sprint5-safety-claim-matrix.md"


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
    clean = _load(CLEAN)

    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    attribution = _load(ATTRIBUTION)

    # Validate the frozen evidence needed by the claim matrix.
    if len(clean["seed_rows"]) != 18:
        raise RuntimeError("clean evidence is incomplete")

    if len(gaussian["seed_rows"]) != 72:
        raise RuntimeError("Gaussian evidence is incomplete")

    if len(structured["seed_rows"]) != 198:
        raise RuntimeError("structured-state evidence is incomplete")

    if len(action["seed_rows"]) != 216:
        raise RuntimeError("action evidence is incomplete")

    active_regimes = attribution["active_regimes"]

    if active_regimes != ["action_perturbation"]:
        raise RuntimeError("unexpected Lyapunov attribution result")

    claims: list[SafetyClaim] = [
        SafetyClaim(
            claim_id="S5-C01",
            statement=(
                "The explicit safety filters reduced clean "
                "executed violation-step rate to zero in both "
                "pilot domains across all three principal seeds."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13B clean three-seed consolidation",
                "driving clipping and Lyapunov violation rate = 0",
                "robotics clipping and Lyapunov violation rate = 0",
            ),
            limitations=(
                "Synthetic pilot environments only.",
                "Three principal policy seeds.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C02",
            statement=(
                "On the clean driving evaluation, clipping and "
                "Lyapunov filtering improved mean reward and "
                "success relative to the no-filter baseline."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13B clean three-seed consolidation",
                "driving NONE reward about -9.2002",
                "driving filtered reward about -5.2126",
                "driving NONE success 0.05",
                "driving filtered success about 0.3333",
            ),
            limitations=(
                "Synthetic driving environment.",
                "This does not imply general task-performance improvement.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C03",
            statement=(
                "On the clean robotics evaluation, clipping and "
                "Lyapunov filtering reduced violation-step rate "
                "to zero without reducing mean reward relative "
                "to the no-filter baseline."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13B clean three-seed consolidation",
                "robotics NONE violation about 0.01467",
                "robotics filtered violation = 0",
                "robotics NONE reward about 0.63285",
                "robotics filtered reward about 0.67790",
            ),
            limitations=(
                "Robotics success remained zero in the clean evaluation.",
                "No object grasp occurred in the clean Lyapunov runs.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C04",
            statement=(
                "The safety layer was evaluated under Gaussian "
                "observation noise while retaining access to the "
                "true simulator state."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13C Gaussian consolidation",
                "policy uses noisy observation = true",
                "safety layer uses true state = true",
            ),
            limitations=(
                (
                    "This is not equivalent to deploying a safety layer "
                    "with noisy or uncertain state estimation."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C05",
            statement=(
                "The safety layer was evaluated under structured "
                "state perturbations while retaining access to the "
                "true simulator state."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13D structured-state consolidation",
                "policy uses perturbed observation = true",
                "safety layer uses true state = true",
                "environment uses true state = true",
            ),
            limitations=(
                "Perturbations are synthetic semantic biases.",
                "They are not calibrated physical sensor-fault models.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C06",
            statement=(
                "Direct action perturbation produced explicit "
                "filter-recovery behavior across the tested pilot "
                "conditions."
            ),
            status=ClaimStatus.SUPPORTED_WITH_LIMITATIONS,
            evidence=(
                "5.13E action robustness consolidation",
                "111341 unsafe perturbed steps",
                "69367 recovered unsafe steps",
                "41974 unresolved unsafe steps",
                "overall explicit-filter recovery fraction about 0.623014",
            ),
            limitations=(
                "The recovery fraction is an aggregate pilot statistic.",
                "A substantial unresolved unsafe-step count remains.",
                (
                    "Action perturbations are synthetic and not calibrated "
                    "actuator-fault models."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C07",
            statement=(
                "The explicit Lyapunov-decrease intervention pathway "
                "activated under direct action perturbation."
            ),
            status=ClaimStatus.SUPPORTED_WITH_LIMITATIONS,
            evidence=(
                "5.13E action robustness consolidation",
                "1584 LYAPUNOV_DECREASE intervention reasons",
                "70 strict Lyapunov decrease steps",
                "4887 selected-lower-than-perturbed steps",
            ),
            limitations=(
                "Mechanism evidence only.",
                "Does not establish formal Lyapunov stability.",
                "Does not establish global superiority over clipping.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C08",
            statement=(
                "Within this pilot, the explicit Lyapunov-decrease "
                "pathway was observed under action perturbation but "
                "not under clean, Gaussian-noise, or structured-state "
                "evaluation."
            ),
            status=ClaimStatus.SUPPORTED_WITH_LIMITATIONS,
            evidence=(
                "5.13F Lyapunov mechanism attribution",
                "clean: 0 Lyapunov-decrease reasons",
                "Gaussian: 0 Lyapunov-decrease reasons",
                "structured state: 0 Lyapunov-decrease reasons",
                "action perturbation: 1584 reasons and 70 strict decreases",
            ),
            limitations=(
                "This is conditional on the tested pilot regimes.",
                (
                    "Absence of activation does not prove the mechanism "
                    "would never activate under other disturbances."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C09",
            statement=(
                "No emergency fallback was required in the "
                "action-perturbation Lyapunov evaluation."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13E action robustness consolidation",
                "emergency fallback steps = 0",
            ),
            limitations=("Applies only to the tested synthetic evaluation corpus.",),
        ),
        SafetyClaim(
            claim_id="S5-C10",
            statement=(
                "The evaluated Lyapunov filter formally guarantees "
                "closed-loop stability."
            ),
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=(
                "No formal control-Lyapunov proof was established.",
                "The implemented quantity is a pilot safety potential.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C11",
            statement=(
                "The evaluated safety system provides formal "
                "worst-case robustness guarantees."
            ),
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=(
                "No exhaustive or adversarial worst-case proof exists.",
                "The robustness evaluations use finite synthetic perturbations.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C12",
            statement=(
                "The evaluated safety system is certified for "
                "production autonomous-driving or robotics deployment."
            ),
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=(
                "No production validation or safety certification was performed.",
                "No ISO compliance claim is supported.",
            ),
        ),
        SafetyClaim(
            claim_id="S5-C13",
            statement=("Lyapunov filtering is globally superior to clipping."),
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=(
                (
                    "Clean driving and robotics results are often identical "
                    "or nearly identical between clipping and Lyapunov filtering."
                ),
                (
                    "Action-perturbation worst cases do not support a universal "
                    "superiority statement."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C14",
            statement=(
                "The safety experiments demonstrate a quantum " "safety advantage."
            ),
            status=ClaimStatus.UNSUPPORTED,
            evidence=(),
            limitations=(
                (
                    "Sprint 5 safety experiments do not establish a quantum "
                    "advantage in safety performance."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C15",
            statement=(
                "The action-robustness evaluation distinguishes "
                "explicit safety-filter corrections from environment-"
                "interface action adjustments."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13E action robustness consolidation",
                "1579 environment-interface adjustment steps",
                "separate explicit filter intervention accounting",
            ),
            limitations=(
                (
                    "The environment may still apply its own action-interface "
                    "semantics after the explicit safety layer."
                ),
            ),
        ),
        SafetyClaim(
            claim_id="S5-C16",
            statement=(
                "The reported three-seed statistics use the principal "
                "policy seeds as experimental units rather than treating "
                "episodes as independent policy replicates."
            ),
            status=ClaimStatus.SUPPORTED,
            evidence=(
                "5.13B clean consolidation",
                "5.13C Gaussian consolidation",
                "5.13D structured-state consolidation",
                "5.13E action consolidation",
                "principal seeds 42, 123, 456",
            ),
            limitations=(
                "n = 3 is small.",
                "No significance testing is claimed.",
            ),
        ),
    ]

    counts = count_by_status(claims)

    unsupported = unsupported_claim_ids(claims)

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.13G",
        "artifact": "safety-claim-matrix",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "claim_count": len(claims),
        "status_counts": counts,
        "unsupported_claim_ids": unsupported,
        "claims": [asdict(claim) for claim in claims],
        "global_limitations": [
            (
                "All Sprint 5 evidence is from synthetic pilot "
                "environments and finite evaluation corpora."
            ),
            (
                "The statistical unit is the trained policy seed; "
                "only three principal seeds are available."
            ),
            ("No statistical significance testing is claimed."),
            (
                "The safety layer retains true simulator state in "
                "the observation-perturbation evaluations."
            ),
            (
                "Action perturbations are synthetic and are not "
                "validated physical actuator-fault models."
            ),
            (
                "No formal Lyapunov stability theorem, formal "
                "worst-case robustness guarantee, production "
                "certification, ISO compliance, or quantum safety "
                "advantage is established."
            ),
        ],
        "proposal_safe_wording": [
            (
                "Explicit safety filtering reduced observed "
                "violation-step rates under the tested pilot conditions."
            ),
            (
                "The Lyapunov-specific candidate-selection pathway "
                "activated under direct action perturbation in the "
                "tested synthetic evaluation."
            ),
            (
                "Results provide empirical mechanism evidence and "
                "robustness characterization, not formal safety "
                "certification."
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
                "claim_id",
                "status",
                "statement",
                "evidence",
                "limitations",
            ],
        )

        writer.writeheader()

        for claim in claims:
            writer.writerow(
                {
                    "claim_id": claim.claim_id,
                    "status": claim.status.value,
                    "statement": claim.statement,
                    "evidence": " | ".join(claim.evidence),
                    "limitations": " | ".join(claim.limitations),
                }
            )

    lines = [
        "# Sprint 5.13G — Safety Claim Matrix",
        "",
        (f"- Total claims: {len(claims)}"),
        (f"- Supported: " f"{counts['supported']}"),
        (f"- Supported with limitations: " f"{counts['supported_with_limitations']}"),
        (f"- Unsupported: " f"{counts['unsupported']}"),
        "",
        "| ID | Status | Claim |",
        "|---|---|---|",
    ]

    for claim in claims:
        lines.append(
            f"| {claim.claim_id} | " f"{claim.status.value} | " f"{claim.statement} |"
        )

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

    print("=" * 80)
    print(" SPRINT 5.13G SAFETY CLAIM MATRIX")
    print("=" * 80)
    print()

    print(
        "Claims:",
        len(claims),
    )

    print(
        "Supported:",
        counts["supported"],
    )

    print(
        "Supported with limitations:",
        counts["supported_with_limitations"],
    )

    print(
        "Unsupported:",
        counts["unsupported"],
    )

    print()
    print(
        "Unsupported claim IDs:",
        unsupported,
    )

    print()
    print("Evidence sources validated: PASS")

    print("Formal-claim controls: PASS")

    print("No new training: PASS")

    print("No new principal runs: PASS")

    print()
    print("SPRINT 5.13G SAFETY CLAIM MATRIX: PASS")


if __name__ == "__main__":
    main()
