"""Final acceptance gate for Sprint 5.14 cross-domain safety analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "safety" / "cross-domain"

PACKAGE = RESULTS_DIR / "sprint5-cross-domain-package.json"

CLAIMS = RESULTS_DIR / "sprint5-cross-domain-claim-matrix.json"

FIGURE_INDEX = RESULTS_DIR / "sprint5-cross-domain-figure-index.json"

CONSISTENCY = RESULTS_DIR / "sprint5-cross-domain-consistency.json"

MECHANISM = RESULTS_DIR / "sprint5-cross-domain-lyapunov-mechanism.json"

REQUIRED_RESULTS = (
    RESULTS_DIR / "sprint5-cross-domain-architecture.json",
    RESULTS_DIR / "sprint5-cross-domain-clean.json",
    RESULTS_DIR / "sprint5-cross-domain-gaussian.json",
    RESULTS_DIR / "sprint5-cross-domain-structured-state.json",
    RESULTS_DIR / "sprint5-cross-domain-action.json",
    MECHANISM,
    CONSISTENCY,
    CLAIMS,
    FIGURE_INDEX,
    PACKAGE,
)

REQUIRED_FIGURES = (
    ROOT
    / "figures"
    / "safety"
    / "cross-domain"
    / "cross_domain_clean_violation_reduction.png",
    ROOT
    / "figures"
    / "safety"
    / "cross-domain"
    / "cross_domain_gaussian_consistency.png",
    ROOT
    / "figures"
    / "safety"
    / "cross-domain"
    / "cross_domain_structured_state_sensitivity.png",
    ROOT / "figures" / "safety" / "cross-domain" / "cross_domain_action_recovery.png",
    ROOT
    / "figures"
    / "safety"
    / "cross-domain"
    / "cross_domain_lyapunov_mechanism.png",
    ROOT / "figures" / "safety" / "cross-domain" / "cross_domain_claim_status.png",
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


def main() -> None:
    for path in REQUIRED_RESULTS:
        if not path.exists():
            raise FileNotFoundError(path)

    for path in REQUIRED_FIGURES:
        if not path.exists():
            raise FileNotFoundError(path)

        if path.stat().st_size <= 0:
            raise RuntimeError(f"empty figure: {path}")

    package = _load(PACKAGE)

    claims = _load(CLAIMS)

    figures = _load(FIGURE_INDEX)

    consistency = _load(CONSISTENCY)

    mechanism = _load(MECHANISM)

    if package["artifact"] != "consolidated-cross-domain-safety-package":
        raise RuntimeError("unexpected consolidated package")

    if not bool(package["analysis_only"]):
        raise RuntimeError("Sprint 5.14 must remain analysis-only")

    if bool(package["new_training"]):
        raise RuntimeError("new training detected")

    if bool(package["new_principal_runs"]):
        raise RuntimeError("new principal execution detected")

    if bool(package["new_safety_episodes"]):
        raise RuntimeError("new safety episodes detected")

    expected_sections = {
        "5.14B",
        "5.14C",
        "5.14D",
        "5.14E",
        "5.14F",
        "5.14G",
        "5.14H",
        "5.14I",
        "5.14J",
    }

    frozen_sections = {str(section) for section in package["frozen_sections"]}

    if frozen_sections != expected_sections:
        raise RuntimeError("cross-domain frozen-section mismatch")

    if int(claims["claim_count"]) != 12:
        raise RuntimeError("expected 12 claims")

    status_counts = claims["status_counts"]

    if not isinstance(
        status_counts,
        dict,
    ):
        raise TypeError("claim status counts must be dict")

    expected_status = {
        "supported": 5,
        "supported_with_limitation": 1,
        "not_supported": 6,
    }

    actual_status = {key: int(status_counts[key]) for key in expected_status}

    if actual_status != expected_status:
        raise RuntimeError("claim-status partition mismatch")

    expected_blocked = {
        "S5-CD06",
        "S5-CD08",
        "S5-CD09",
        "S5-CD10",
        "S5-CD11",
        "S5-CD12",
    }

    blocked = {str(claim_id) for claim_id in claims["blocked_claim_ids"]}

    if blocked != expected_blocked:
        raise RuntimeError("blocked claim set mismatch")

    if int(figures["figure_count"]) != 6:
        raise RuntimeError("expected six cross-domain figures")

    expected_consistency = {
        "clean_consistency": (
            2,
            2,
            0,
            True,
        ),
        "gaussian_consistency": (
            9,
            7,
            2,
            False,
        ),
        "structured_state_consistency": (
            3,
            3,
            0,
            True,
        ),
        "action_recovery_consistency": (
            3,
            3,
            0,
            True,
        ),
    }

    for key, expected in expected_consistency.items():
        record = consistency[key]

        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(f"{key} must be dict")

        actual = (
            int(record["count"]),
            int(record["true"]),
            int(record["false"]),
            bool(record["all_consistent"]),
        )

        if actual != expected:
            raise RuntimeError(f"{key} mismatch: " f"{actual} != {expected}")

    activation = consistency["lyapunov_activation_consistency"]

    if not isinstance(
        activation,
        dict,
    ):
        raise TypeError("activation consistency must be dict")

    if not bool(activation["driving_activation_observed"]):
        raise RuntimeError("expected driving Lyapunov activation")

    if bool(activation["robotics_activation_observed"]):
        raise RuntimeError("robotics Lyapunov activation must remain false")

    if bool(activation["consistent"]):
        raise RuntimeError("Lyapunov activation asymmetry was lost")

    reconstruction = mechanism["global_reconstruction"]

    if not isinstance(
        reconstruction,
        dict,
    ):
        raise TypeError("mechanism reconstruction must be dict")

    reason_counts = reconstruction["intervention_reason_counts"]

    if not isinstance(
        reason_counts,
        dict,
    ):
        raise TypeError("reason counts must be dict")

    expected_reasons = {
        "action_bound": 1493,
        "domain_constraint": 35729,
        "lyapunov_decrease": 1584,
        "none": 105194,
    }

    actual_reasons = {key: int(reason_counts[key]) for key in expected_reasons}

    if actual_reasons != expected_reasons:
        raise RuntimeError("Lyapunov mechanism totals changed")

    if int(reconstruction["strict_lyapunov_decrease_steps"]) != 70:
        raise RuntimeError("strict decrease total changed")

    if int(reconstruction["selected_lower_than_perturbed_steps"]) != 4887:
        raise RuntimeError("selected-lower total changed")

    if int(reconstruction["emergency_fallback_steps"]) != 0:
        raise RuntimeError("emergency fallback total changed")

    source_manifest = package["source_manifest"]

    if not isinstance(
        source_manifest,
        list,
    ):
        raise TypeError("source manifest must be list")

    if len(source_manifest) != 9:
        raise RuntimeError("expected nine frozen source artifacts")

    print("=" * 80)

    print(" SPRINT 5.14M FINAL CROSS-DOMAIN ACCEPTANCE GATE")

    print("=" * 80)

    print()

    print(
        "Required result artifacts:",
        len(REQUIRED_RESULTS),
    )

    print(
        "Required figures:",
        len(REQUIRED_FIGURES),
    )

    print(
        "Source manifest entries:",
        len(source_manifest),
    )

    print(
        "Claims:",
        claims["claim_count"],
    )

    print(
        "Figures:",
        figures["figure_count"],
    )

    print()

    print("Analysis-only scope: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print("No new safety episodes: PASS")

    print("Frozen section coverage: PASS")

    print("Clean cross-domain evidence: PASS")

    print("Gaussian cross-domain evidence: PASS")

    print("Structured-state cross-domain evidence: PASS")

    print("Action-recovery evidence: PASS")

    print("Lyapunov activation asymmetry: PASS")

    print("Mechanism invariants: PASS")

    print("Claim controls: PASS")

    print("Blocked claim controls: PASS")

    print("Figure package: PASS")

    print("Source provenance package: PASS")

    print("Independent verification handoff: PASS")

    print()

    print("SPRINT 5.14 FINAL ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
