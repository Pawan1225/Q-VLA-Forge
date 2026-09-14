"""Independently verify Sprint 5.13H consolidated safety evidence package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

PACKAGE = CONSOLIDATED / "sprint5-safety-evidence-package.json"

CLEAN = CONSOLIDATED / "sprint5-clean-three-seed-summary.json"

GAUSSIAN = CONSOLIDATED / "sprint5-gaussian-three-seed-summary.json"

STRUCTURED = CONSOLIDATED / "sprint5-structured-state-three-seed-summary.json"

ACTION = CONSOLIDATED / "sprint5-action-three-seed-summary.json"

ATTRIBUTION = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"

CLAIMS = CONSOLIDATED / "sprint5-safety-claim-matrix.json"


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


def _close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return abs(actual - expected) <= tolerance


def _clean_mean(
    clean: dict[str, Any],
    *,
    domain: str,
    method: str,
    metric: str,
) -> float:
    summary = clean["three_seed_summary"]

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError("clean three_seed_summary must be dict")

    domain_summary = summary[domain]

    if not isinstance(
        domain_summary,
        dict,
    ):
        raise TypeError("clean domain summary must be dict")

    method_summary = domain_summary[method]

    if not isinstance(
        method_summary,
        dict,
    ):
        raise TypeError("clean method summary must be dict")

    metric_summary = method_summary[metric]

    if not isinstance(
        metric_summary,
        dict,
    ):
        raise TypeError("clean metric summary must be dict")

    return float(metric_summary["mean"])


def main() -> None:
    print("=" * 92)
    print(" Q-VLA FORGE - SPRINT 5.13I SAFETY EVIDENCE PACKAGE VERIFICATION")
    print("=" * 92)
    print()

    package = _load(PACKAGE)

    clean = _load(CLEAN)

    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    attribution = _load(ATTRIBUTION)

    claims = _load(CLAIMS)

    _check(
        package["analysis_only"] is True,
        "analysis-only package",
    )

    _check(
        package["new_training"] is False,
        "no new training",
    )

    _check(
        package["new_principal_runs"] is False,
        "no new principal runs",
    )

    _check(
        package["principal_seeds"]
        == [
            42,
            123,
            456,
        ],
        "principal seeds exact",
    )

    expected_sources = {
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
    }

    actual_sources = set(package["source_artifacts"])

    _check(
        actual_sources == expected_sources,
        "source artifact set exact",
    )

    _check(
        len(clean["seed_rows"]) == 18,
        "clean evidence has 18 seed rows",
    )

    _check(
        len(gaussian["seed_rows"]) == 72,
        "Gaussian evidence has 72 seed rows",
    )

    _check(
        len(structured["seed_rows"]) == 198,
        "structured evidence has 198 seed rows",
    )

    _check(
        len(action["seed_rows"]) == 216,
        "action evidence has 216 seed rows",
    )

    clean_package = package["clean_evidence"]

    if not isinstance(
        clean_package,
        dict,
    ):
        raise TypeError("clean package must be dict")

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        domain_package = clean_package[domain]

        if not isinstance(
            domain_package,
            dict,
        ):
            raise TypeError("domain package must be dict")

        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            method_package = domain_package[method]

            if not isinstance(
                method_package,
                dict,
            ):
                raise TypeError("method package must be dict")

            for metric in (
                "violation_step_rate",
                "reward",
                "success_rate",
            ):
                expected = _clean_mean(
                    clean,
                    domain=domain,
                    method=method,
                    metric=metric,
                )

                actual = float(method_package[metric])

                _check(
                    _close(
                        actual,
                        expected,
                    ),
                    (f"clean {domain}/{method}/" f"{metric} exact"),
                )

    robustness = package["robustness_corpus"]

    if not isinstance(
        robustness,
        dict,
    ):
        raise TypeError("robustness corpus must be dict")

    gaussian_package = robustness["gaussian"]

    structured_package = robustness["structured_state"]

    action_package = robustness["action"]

    if not isinstance(
        gaussian_package,
        dict,
    ):
        raise TypeError("Gaussian package must be dict")

    if not isinstance(
        structured_package,
        dict,
    ):
        raise TypeError("structured package must be dict")

    if not isinstance(
        action_package,
        dict,
    ):
        raise TypeError("action package must be dict")

    _check(
        int(gaussian_package["seed_cells"]) == 72,
        "Gaussian seed-cell count exact",
    )

    _check(
        int(gaussian_package["aggregates"]) == 24,
        "Gaussian aggregate count exact",
    )

    _check(
        int(gaussian_package["new_noisy_episodes"]) == 1080,
        "Gaussian noisy episode count exact",
    )

    _check(
        gaussian_package["policy_uses_noisy_observation"] is True,
        "Gaussian policy-noise separation exact",
    )

    _check(
        gaussian_package["safety_layer_uses_true_state"] is True,
        "Gaussian true-state safety exact",
    )

    _check(
        int(structured_package["seed_cells"]) == 198,
        "structured seed-cell count exact",
    )

    _check(
        int(structured_package["aggregates"]) == 66,
        "structured aggregate count exact",
    )

    _check(
        int(structured_package["perturbations"]) == 22,
        "structured perturbation count exact",
    )

    _check(
        int(structured_package["episodes"]) == 3960,
        "structured episode count exact",
    )

    _check(
        structured_package["policy_uses_perturbed_observation"] is True,
        "structured perturbed-observation policy exact",
    )

    _check(
        structured_package["safety_layer_uses_true_state"] is True,
        "structured true-state safety exact",
    )

    _check(
        int(action_package["seed_cells"]) == 216,
        "action seed-cell count exact",
    )

    _check(
        int(action_package["aggregates"]) == 72,
        "action aggregate count exact",
    )

    _check(
        int(action_package["perturbations"]) == 24,
        "action perturbation count exact",
    )

    _check(
        int(action_package["episodes"]) == 4320,
        "action episode count exact",
    )

    action_mechanism = action["mechanism"]

    if not isinstance(
        action_mechanism,
        dict,
    ):
        raise TypeError("action mechanism must be dict")

    _check(
        int(action_package["unsafe_perturbed_steps"])
        == int(action_mechanism["unsafe_perturbed_steps"])
        == 111341,
        "unsafe perturbed-step count exact",
    )

    _check(
        int(action_package["recovered_unsafe_steps"])
        == int(action_mechanism["recovered_unsafe_steps"])
        == 69367,
        "recovered unsafe-step count exact",
    )

    _check(
        int(action_package["unresolved_unsafe_steps"])
        == int(action_mechanism["unresolved_unsafe_steps"])
        == 41974,
        "unresolved unsafe-step count exact",
    )

    _check(
        int(action_package["recovered_unsafe_steps"])
        + int(action_package["unresolved_unsafe_steps"])
        == int(action_package["unsafe_perturbed_steps"]),
        "action recovery accounting closes",
    )

    expected_recovery = 69367 / 111341

    _check(
        _close(
            float(action_package["recovery_fraction"]),
            expected_recovery,
        ),
        "action recovery fraction exact",
    )

    _check(
        int(action_package["environment_interface_adjustments"]) == 1579,
        "environment-interface adjustment count exact",
    )

    mechanism = package["lyapunov_mechanism"]

    if not isinstance(
        mechanism,
        dict,
    ):
        raise TypeError("package Lyapunov mechanism must be dict")

    _check(
        mechanism["active_regimes"] == ["action_perturbation"],
        "action perturbation is sole active regime",
    )

    _check(
        mechanism["inactive_regimes"]
        == [
            "clean",
            "gaussian_state_perturbation",
            "structured_state_perturbation",
        ],
        "inactive regimes exact",
    )

    _check(
        mechanism["active_regimes"] == attribution["active_regimes"],
        "active-regime attribution source exact",
    )

    _check(
        mechanism["inactive_regimes"] == attribution["inactive_regimes"],
        "inactive-regime attribution source exact",
    )

    _check(
        int(mechanism["action_lyapunov_decrease_reasons"]) == 1584,
        "1584 Lyapunov-decrease reasons",
    )

    _check(
        int(mechanism["strict_decrease_steps"]) == 70,
        "70 strict Lyapunov decreases",
    )

    _check(
        int(mechanism["selected_lower_steps"]) == 4887,
        "4887 selected-lower steps",
    )

    _check(
        int(mechanism["emergency_fallback_steps"]) == 0,
        "zero emergency fallback steps",
    )

    controls = package["claim_controls"]

    if not isinstance(
        controls,
        dict,
    ):
        raise TypeError("claim controls must be dict")

    _check(
        int(controls["claim_count"]) == 16,
        "16 claims exact",
    )

    _check(
        int(controls["supported"]) == 8,
        "8 supported claims exact",
    )

    _check(
        int(controls["supported_with_limitations"]) == 3,
        "3 limited claims exact",
    )

    _check(
        int(controls["unsupported"]) == 5,
        "5 unsupported claims exact",
    )

    expected_blocked = [
        "S5-C10",
        "S5-C11",
        "S5-C12",
        "S5-C13",
        "S5-C14",
    ]

    _check(
        controls["blocked_claim_ids"] == expected_blocked,
        "blocked claim IDs exact",
    )

    _check(
        controls["blocked_claim_ids"] == claims["unsupported_claim_ids"],
        "blocked claims source-linked",
    )

    _check(
        package["proposal_safe_wording"] == claims["proposal_safe_wording"],
        "proposal-safe wording source-exact",
    )

    _check(
        package["global_limitations"] == claims["global_limitations"],
        "global limitations source-exact",
    )

    headline_findings = package["headline_findings"]

    _check(
        isinstance(
            headline_findings,
            list,
        )
        and len(headline_findings) == 5,
        "five headline findings",
    )

    joined_findings = " ".join(str(item) for item in headline_findings).lower()

    for prohibited_phrase in (
        "formal stability",
        "worst-case robustness",
        "production certification",
        "global lyapunov superiority",
        "quantum safety advantage",
    ):
        _check(
            prohibited_phrase in joined_findings,
            ("headline limitations explicitly mention: " f"{prohibited_phrase}"),
        )

    print()
    print("Source linkage: PASS")

    print("Clean evidence reconstruction: PASS")

    print("Gaussian corpus accounting: PASS")

    print("Structured-state corpus accounting: PASS")

    print("Action corpus accounting: PASS")

    print("Recovery accounting: PASS")

    print("Lyapunov attribution: PASS")

    print("Claim controls: PASS")

    print("Limitations propagation: PASS")

    print("No scientific reruns: PASS")

    print()
    print("SPRINT 5.13I SAFETY EVIDENCE VERIFICATION: PASS")


if __name__ == "__main__":
    main()
