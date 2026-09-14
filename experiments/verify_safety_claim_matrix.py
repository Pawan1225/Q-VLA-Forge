"""Independently verify Sprint 5.13G safety claim matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

CLAIMS = CONSOLIDATED / "sprint5-safety-claim-matrix.json"

CLEAN = CONSOLIDATED / "sprint5-clean-three-seed-summary.json"

GAUSSIAN = CONSOLIDATED / "sprint5-gaussian-three-seed-summary.json"

STRUCTURED = CONSOLIDATED / "sprint5-structured-state-three-seed-summary.json"

ACTION = CONSOLIDATED / "sprint5-action-three-seed-summary.json"

ATTRIBUTION = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"


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


def _check(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def main() -> None:
    print("=" * 86)
    print(" Q-VLA FORGE - SPRINT 5.13G SAFETY CLAIM MATRIX VERIFICATION")
    print("=" * 86)
    print()

    claims_payload = _load(CLAIMS)

    clean = _load(CLEAN)

    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    attribution = _load(ATTRIBUTION)

    _check(
        claims_payload["analysis_only"] is True,
        "analysis-only scope",
    )

    _check(
        claims_payload["new_training"] is False,
        "no new training",
    )

    _check(
        claims_payload["new_principal_runs"] is False,
        "no new principal runs",
    )

    claims = claims_payload["claims"]

    if not isinstance(
        claims,
        list,
    ):
        raise TypeError("claims must be a list")

    _check(
        len(claims) == 16,
        "16 safety claims",
    )

    claim_map: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw in claims:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("claim row must be dict")

        claim_id = str(raw["claim_id"])

        claim_map[claim_id] = raw

    _check(
        len(claim_map) == 16,
        "16 unique claim IDs",
    )

    expected_ids = {
        f"S5-C{index:02d}"
        for index in range(
            1,
            17,
        )
    }

    _check(
        set(claim_map) == expected_ids,
        "claim IDs S5-C01 through S5-C16 complete",
    )

    status_counts = {
        "supported": 0,
        "supported_with_limitations": 0,
        "unsupported": 0,
    }

    for claim in claims:
        status = str(claim["status"])

        if status not in status_counts:
            raise AssertionError(f"unknown claim status: {status}")

        status_counts[status] += 1

    _check(
        status_counts["supported"] == 8,
        "8 supported claims",
    )

    _check(
        status_counts["supported_with_limitations"] == 3,
        "3 supported-with-limitations claims",
    )

    _check(
        status_counts["unsupported"] == 5,
        "5 unsupported claims",
    )

    _check(
        claims_payload["status_counts"] == status_counts,
        "reported status counts exact",
    )

    expected_unsupported = [
        "S5-C10",
        "S5-C11",
        "S5-C12",
        "S5-C13",
        "S5-C14",
    ]

    actual_unsupported = [
        claim_id
        for claim_id in sorted(claim_map)
        if claim_map[claim_id]["status"] == "unsupported"
    ]

    _check(
        actual_unsupported == expected_unsupported,
        "unsupported claim set exact",
    )

    _check(
        claims_payload["unsupported_claim_ids"] == expected_unsupported,
        "reported unsupported IDs exact",
    )

    # Clean evidence gate.
    _check(
        len(clean["seed_rows"]) == 18,
        "clean 18-cell evidence available",
    )

    # Gaussian evidence gate.
    _check(
        len(gaussian["seed_rows"]) == 72,
        "Gaussian 72-cell evidence available",
    )

    _check(
        gaussian["policy_uses_noisy_observation"] is True,
        "Gaussian policy uses noisy observation",
    )

    _check(
        gaussian["safety_layer_uses_true_state"] is True,
        "Gaussian safety layer uses true state",
    )

    # Structured-state evidence gate.
    _check(
        len(structured["seed_rows"]) == 198,
        "structured 198-cell evidence available",
    )

    _check(
        structured["policy_uses_perturbed_observation"] is True,
        "structured policy uses perturbed observation",
    )

    _check(
        structured["safety_layer_uses_true_state"] is True,
        "structured safety layer uses true state",
    )

    _check(
        structured["environment_uses_true_state"] is True,
        "structured environment uses true state",
    )

    # Action evidence gate.
    _check(
        len(action["seed_rows"]) == 216,
        "action 216-cell evidence available",
    )

    mechanism = action["mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("action mechanism must be dict")

    _check(
        int(mechanism["unsafe_perturbed_steps"]) == 111341,
        "action unsafe-step count exact",
    )

    _check(
        int(mechanism["recovered_unsafe_steps"]) == 69367,
        "action recovered-step count exact",
    )

    _check(
        int(mechanism["unresolved_unsafe_steps"]) == 41974,
        "action unresolved-step count exact",
    )

    reasons = mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        reasons,
        dict,
    ):
        raise TypeError("Lyapunov reasons must be dict")

    _check(
        int(reasons["lyapunov_decrease"]) == 1584,
        "1584 Lyapunov-decrease reasons",
    )

    _check(
        int(mechanism["strict_lyapunov_decrease_steps"]) == 70,
        "70 strict Lyapunov decreases",
    )

    _check(
        int(mechanism["selected_lower_than_perturbed_steps"]) == 4887,
        "4887 selected-lower steps",
    )

    _check(
        int(mechanism["emergency_fallback_steps"]) == 0,
        "zero emergency fallback steps",
    )

    _check(
        int(mechanism["environment_interface_adjustment_steps"]) == 1579,
        "1579 environment-interface adjustments",
    )

    # Cross-regime attribution.
    _check(
        attribution["active_regimes"] == ["action_perturbation"],
        "action perturbation is sole active Lyapunov regime",
    )

    _check(
        attribution["inactive_regimes"]
        == [
            "clean",
            "gaussian_state_perturbation",
            "structured_state_perturbation",
        ],
        "inactive Lyapunov regimes exact",
    )

    interpretation = attribution["interpretation"]

    if not isinstance(
        interpretation,
        dict,
    ):
        raise TypeError("attribution interpretation must be dict")

    _check(
        interpretation["formal_stability_claim"] is False,
        "formal stability claim disabled",
    )

    _check(
        interpretation["global_superiority_claim"] is False,
        "global superiority claim disabled",
    )

    # Explicitly inspect all prohibited claim statements.
    prohibited_expected = {
        "S5-C10": "formal Lyapunov stability",
        "S5-C11": "formal worst-case robustness",
        "S5-C12": "production certification",
        "S5-C13": "global Lyapunov superiority",
        "S5-C14": "quantum safety advantage",
    }

    for (
        claim_id,
        label,
    ) in prohibited_expected.items():
        claim = claim_map[claim_id]

        _check(
            claim["status"] == "unsupported",
            f"blocked claim: {label}",
        )

        limitations = claim["limitations"]

        _check(
            isinstance(
                limitations,
                list,
            )
            and len(limitations) > 0,
            f"blocked claim has limitation: {label}",
        )

    global_limitations = claims_payload["global_limitations"]

    _check(
        isinstance(
            global_limitations,
            list,
        )
        and len(global_limitations) >= 6,
        "global limitations present",
    )

    safe_wording = claims_payload["proposal_safe_wording"]

    _check(
        isinstance(
            safe_wording,
            list,
        )
        and len(safe_wording) >= 3,
        "proposal-safe wording present",
    )

    print()
    print("Claim inventory: PASS")
    print("Status classification: PASS")
    print("Evidence linkage: PASS")
    print("Lyapunov claim controls: PASS")
    print("Production/certification controls: PASS")
    print("Quantum-advantage control: PASS")
    print("Proposal-safe wording: PASS")
    print("No scientific reruns: PASS")

    print()
    print("SPRINT 5.13G SAFETY CLAIM MATRIX VERIFICATION: PASS")


if __name__ == "__main__":
    main()
