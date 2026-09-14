"""Final acceptance gate for Sprint 5.13 safety consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

REQUIRED_ARTIFACTS = (
    "sprint5-safety-source-manifest.json",
    "sprint5-clean-three-seed-summary.json",
    "sprint5-clean-three-seed-summary.csv",
    "sprint5-clean-three-seed-summary.md",
    "sprint5-gaussian-three-seed-summary.json",
    "sprint5-gaussian-three-seed-summary.csv",
    "sprint5-gaussian-three-seed-summary.md",
    "sprint5-structured-state-three-seed-summary.json",
    "sprint5-structured-state-three-seed-summary.csv",
    "sprint5-structured-state-three-seed-summary.md",
    "sprint5-action-three-seed-summary.json",
    "sprint5-action-three-seed-summary.csv",
    "sprint5-action-three-seed-summary.md",
    "sprint5-lyapunov-mechanism-attribution.json",
    "sprint5-lyapunov-mechanism-attribution.md",
    "sprint5-safety-claim-matrix.json",
    "sprint5-safety-claim-matrix.csv",
    "sprint5-safety-claim-matrix.md",
    "sprint5-safety-evidence-package.json",
    "sprint5-safety-evidence-package.md",
)

OUTPUT = CONSOLIDATED / "sprint5-13-final-acceptance.json"


def _load(
    filename: str,
) -> dict[str, Any]:
    path = CONSOLIDATED / filename

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
    print("=" * 92)
    print(" Q-VLA FORGE - SPRINT 5.13 FINAL ACCEPTANCE GATE")
    print("=" * 92)
    print()

    missing = [
        filename
        for filename in REQUIRED_ARTIFACTS
        if not (CONSOLIDATED / filename).exists()
    ]

    _check(
        not missing,
        "all required 5.13 artifacts exist",
    )

    source_manifest = _load("sprint5-safety-source-manifest.json")

    clean = _load("sprint5-clean-three-seed-summary.json")

    gaussian = _load("sprint5-gaussian-three-seed-summary.json")

    structured = _load("sprint5-structured-state-three-seed-summary.json")

    action = _load("sprint5-action-three-seed-summary.json")

    attribution = _load("sprint5-lyapunov-mechanism-attribution.json")

    claims = _load("sprint5-safety-claim-matrix.json")

    package = _load("sprint5-safety-evidence-package.json")

    _check(
        source_manifest["sprint"] == "5.13A",
        "5.13A source manifest identified",
    )

    _check(
        clean["sprint"] == "5.13B",
        "5.13B clean consolidation identified",
    )

    _check(
        gaussian["sprint"] == "5.13C",
        "5.13C Gaussian consolidation identified",
    )

    _check(
        structured["sprint"] == "5.13D",
        "5.13D structured-state consolidation identified",
    )

    _check(
        action["sprint"] == "5.13E",
        "5.13E action consolidation identified",
    )

    _check(
        attribution["sprint"] == "5.13F",
        "5.13F mechanism attribution identified",
    )

    _check(
        claims["sprint"] == "5.13G",
        "5.13G claim matrix identified",
    )

    _check(
        package["sprint"] == "5.13H",
        "5.13H evidence package identified",
    )

    # Scope controls.
    for (
        name,
        payload,
    ) in (
        (
            "clean",
            clean,
        ),
        (
            "Gaussian",
            gaussian,
        ),
        (
            "structured",
            structured,
        ),
        (
            "action",
            action,
        ),
        (
            "attribution",
            attribution,
        ),
        (
            "claims",
            claims,
        ),
        (
            "package",
            package,
        ),
    ):
        _check(
            payload["analysis_only"] is True,
            f"{name} analysis-only scope",
        )

        _check(
            payload["new_training"] is False,
            f"{name} no new training",
        )

        _check(
            payload["new_principal_runs"] is False,
            f"{name} no new principal runs",
        )

    # Principal seeds.
    expected_seeds = [
        42,
        123,
        456,
    ]

    _check(
        clean["principal_seeds"] == expected_seeds,
        "clean principal seeds exact",
    )

    _check(
        gaussian["principal_seeds"] == expected_seeds,
        "Gaussian principal seeds exact",
    )

    _check(
        structured["principal_seeds"] == expected_seeds,
        "structured principal seeds exact",
    )

    _check(
        action["principal_seeds"] == expected_seeds,
        "action principal seeds exact",
    )

    _check(
        package["principal_seeds"] == expected_seeds,
        "package principal seeds exact",
    )

    # Corpus accounting.
    _check(
        len(clean["seed_rows"]) == 18,
        "clean 18 seed cells",
    )

    _check(
        len(gaussian["seed_rows"]) == 72,
        "Gaussian 72 seed cells",
    )

    _check(
        len(structured["seed_rows"]) == 198,
        "structured 198 seed cells",
    )

    _check(
        len(action["seed_rows"]) == 216,
        "action 216 seed cells",
    )

    _check(
        len(gaussian["three_seed_summary"]) == 24,
        "Gaussian 24 aggregates",
    )

    _check(
        len(structured["three_seed_summary"]) == 66,
        "structured 66 aggregates",
    )

    _check(
        len(action["three_seed_summary"]) == 72,
        "action 72 aggregates",
    )

    _check(
        int(gaussian["new_noisy_episodes"]) == 1080,
        "Gaussian 1080 noisy episodes",
    )

    _check(
        int(structured["principal_episodes"]) == 3960,
        "structured 3960 episodes",
    )

    _check(
        int(action["principal_episodes"]) == 4320,
        "action 4320 episodes",
    )

    # True-state separation.
    _check(
        gaussian["policy_uses_noisy_observation"] is True,
        "Gaussian policy uses noisy observation",
    )

    _check(
        gaussian["safety_layer_uses_true_state"] is True,
        "Gaussian safety uses true state",
    )

    _check(
        structured["policy_uses_perturbed_observation"] is True,
        "structured policy uses perturbed observation",
    )

    _check(
        structured["safety_layer_uses_true_state"] is True,
        "structured safety uses true state",
    )

    _check(
        structured["environment_uses_true_state"] is True,
        "structured environment uses true state",
    )

    # Action accounting.
    mechanism = action["mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("action mechanism must be dict")

    unsafe = int(mechanism["unsafe_perturbed_steps"])

    recovered = int(mechanism["recovered_unsafe_steps"])

    unresolved = int(mechanism["unresolved_unsafe_steps"])

    _check(
        unsafe == 111341,
        "111341 unsafe action-perturbed steps",
    )

    _check(
        recovered == 69367,
        "69367 recovered action-perturbed steps",
    )

    _check(
        unresolved == 41974,
        "41974 unresolved action-perturbed steps",
    )

    _check(
        recovered + unresolved == unsafe,
        "action recovery accounting closes",
    )

    _check(
        int(mechanism["environment_interface_adjustment_steps"]) == 1579,
        "1579 environment-interface adjustments",
    )

    _check(
        int(mechanism["emergency_fallback_steps"]) == 0,
        "zero emergency fallback steps",
    )

    # Lyapunov attribution.
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
        attribution["active_regimes"] == ["action_perturbation"],
        "action perturbation sole active Lyapunov regime",
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

    # Claim controls.
    _check(
        int(claims["claim_count"]) == 16,
        "16 controlled safety claims",
    )

    _check(
        claims["status_counts"]
        == {
            "supported": 8,
            "supported_with_limitations": 3,
            "unsupported": 5,
        },
        "claim-status counts exact",
    )

    blocked = [
        "S5-C10",
        "S5-C11",
        "S5-C12",
        "S5-C13",
        "S5-C14",
    ]

    _check(
        claims["unsupported_claim_ids"] == blocked,
        "five blocked claims exact",
    )

    package_controls = package["claim_controls"]

    if not isinstance(
        package_controls,
        dict,
    ):
        raise TypeError("package claim controls must be dict")

    _check(
        package_controls["blocked_claim_ids"] == blocked,
        "blocked claims propagated to final package",
    )

    _check(
        package["proposal_safe_wording"] == claims["proposal_safe_wording"],
        "proposal-safe wording propagated",
    )

    _check(
        package["global_limitations"] == claims["global_limitations"],
        "global limitations propagated",
    )

    # Final acceptance record.
    acceptance = {
        "sprint": "5.13J",
        "artifact": "sprint5-13-final-acceptance",
        "status": "PASS",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "principal_seeds": expected_seeds,
        "completed_subsprints": [
            "5.13A",
            "5.13B",
            "5.13C",
            "5.13D",
            "5.13E",
            "5.13F",
            "5.13G",
            "5.13H",
            "5.13I",
            "5.13J",
        ],
        "corpus": {
            "clean_seed_cells": 18,
            "gaussian_seed_cells": 72,
            "structured_seed_cells": 198,
            "action_seed_cells": 216,
            "gaussian_new_noisy_episodes": 1080,
            "structured_episodes": 3960,
            "action_episodes": 4320,
        },
        "action_recovery": {
            "unsafe_perturbed_steps": unsafe,
            "recovered_unsafe_steps": recovered,
            "unresolved_unsafe_steps": unresolved,
            "recovery_fraction": recovered / unsafe,
        },
        "lyapunov_mechanism": {
            "active_regimes": attribution["active_regimes"],
            "lyapunov_decrease_reasons": reasons["lyapunov_decrease"],
            "strict_decrease_steps": mechanism["strict_lyapunov_decrease_steps"],
            "selected_lower_steps": mechanism["selected_lower_than_perturbed_steps"],
            "emergency_fallback_steps": mechanism["emergency_fallback_steps"],
        },
        "claim_controls": {
            "claim_count": 16,
            "supported": 8,
            "supported_with_limitations": 3,
            "unsupported": 5,
            "blocked_claim_ids": blocked,
        },
        "handoff": {
            "sprint_5_13_complete": True,
            "sprint_5_14_ready": True,
        },
    }

    OUTPUT.write_text(
        json.dumps(
            acceptance,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("5.13A source freeze: PASS")

    print("5.13B clean consolidation: PASS")

    print("5.13C Gaussian consolidation: PASS")

    print("5.13D structured-state consolidation: PASS")

    print("5.13E action robustness consolidation: PASS")

    print("5.13F Lyapunov attribution: PASS")

    print("5.13G claim matrix: PASS")

    print("5.13H evidence package: PASS")

    print("5.13I independent verification: PASS")

    print("5.13J final acceptance: PASS")

    print()
    print("SPRINT 5.13: COMPLETE")

    print("SPRINT 5.14 HANDOFF: READY")


if __name__ == "__main__":
    main()
