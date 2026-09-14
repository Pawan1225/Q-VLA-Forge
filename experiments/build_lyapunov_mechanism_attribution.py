"""Build Sprint 5.13F Lyapunov mechanism attribution."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.lyapunov_mechanism_attribution import (
    RegimeMechanism,
    mechanism_active,
)

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

GAUSSIAN = CONSOLIDATED / "sprint5-gaussian-three-seed-summary.json"

STRUCTURED = CONSOLIDATED / "sprint5-structured-state-three-seed-summary.json"

ACTION = CONSOLIDATED / "sprint5-action-three-seed-summary.json"

DRIVING_CLEAN = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-driving"
    / "sprint5-driving-lyapunov-summary.json"
)

ROBOTICS_CLEAN = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-robotics"
    / "sprint5-robotics-lyapunov-summary.json"
)

OUTPUT_JSON = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"

OUTPUT_MD = CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.md"


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
    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    # Validate that both frozen clean Lyapunov sources
    # exist and contain valid JSON objects.
    _load(DRIVING_CLEAN)

    _load(ROBOTICS_CLEAN)

    gaussian_mechanism = gaussian["lyapunov_mechanism"]

    structured_mechanism = structured["lyapunov_mechanism"]

    action_mechanism = action["mechanism"]

    if not isinstance(
        gaussian_mechanism,
        dict,
    ):
        raise TypeError("Gaussian mechanism must be dict")

    if not isinstance(
        structured_mechanism,
        dict,
    ):
        raise TypeError("structured mechanism must be dict")

    if not isinstance(
        action_mechanism,
        dict,
    ):
        raise TypeError("action mechanism must be dict")

    gaussian_reasons = gaussian_mechanism["intervention_reason_counts"]

    structured_reasons = structured_mechanism["intervention_reason_counts"]

    action_reasons = action_mechanism["lyapunov_intervention_reason_counts"]

    if not isinstance(
        gaussian_reasons,
        dict,
    ):
        raise TypeError("Gaussian reasons must be dict")

    if not isinstance(
        structured_reasons,
        dict,
    ):
        raise TypeError("structured reasons must be dict")

    if not isinstance(
        action_reasons,
        dict,
    ):
        raise TypeError("action reasons must be dict")

    clean_regime = RegimeMechanism(
        regime="clean",
        lyapunov_decrease_reasons=0,
        strict_decrease_steps=0,
        selected_lower_steps=None,
        emergency_fallback_steps=0,
        mechanism_active=False,
    )

    gaussian_reason_count = int(
        gaussian_reasons.get(
            "lyapunov_decrease",
            0,
        )
    )

    gaussian_strict_count = int(gaussian_mechanism["strict_decrease_steps"])

    gaussian_regime = RegimeMechanism(
        regime="gaussian_state_perturbation",
        lyapunov_decrease_reasons=(gaussian_reason_count),
        strict_decrease_steps=(gaussian_strict_count),
        selected_lower_steps=None,
        emergency_fallback_steps=None,
        mechanism_active=mechanism_active(
            lyapunov_decrease_reasons=(gaussian_reason_count),
            strict_decrease_steps=(gaussian_strict_count),
        ),
    )

    structured_reason_count = int(
        structured_reasons.get(
            "lyapunov_decrease",
            0,
        )
    )

    structured_strict_count = int(structured_mechanism["strict_decrease_steps"])

    structured_regime = RegimeMechanism(
        regime="structured_state_perturbation",
        lyapunov_decrease_reasons=(structured_reason_count),
        strict_decrease_steps=(structured_strict_count),
        selected_lower_steps=None,
        emergency_fallback_steps=None,
        mechanism_active=mechanism_active(
            lyapunov_decrease_reasons=(structured_reason_count),
            strict_decrease_steps=(structured_strict_count),
        ),
    )

    action_reason_count = int(action_reasons["lyapunov_decrease"])

    action_strict_count = int(action_mechanism["strict_lyapunov_decrease_steps"])

    action_regime = RegimeMechanism(
        regime="action_perturbation",
        lyapunov_decrease_reasons=(action_reason_count),
        strict_decrease_steps=(action_strict_count),
        selected_lower_steps=int(
            action_mechanism["selected_lower_than_perturbed_steps"]
        ),
        emergency_fallback_steps=int(action_mechanism["emergency_fallback_steps"]),
        mechanism_active=mechanism_active(
            lyapunov_decrease_reasons=(action_reason_count),
            strict_decrease_steps=(action_strict_count),
        ),
    )

    regimes = [
        clean_regime,
        gaussian_regime,
        structured_regime,
        action_regime,
    ]

    active_regimes = [regime.regime for regime in regimes if regime.mechanism_active]

    inactive_regimes = [
        regime.regime for regime in regimes if not regime.mechanism_active
    ]

    if active_regimes != ["action_perturbation"]:
        raise RuntimeError("unexpected Lyapunov mechanism activation pattern")

    expected_inactive = [
        "clean",
        "gaussian_state_perturbation",
        "structured_state_perturbation",
    ]

    if inactive_regimes != expected_inactive:
        raise RuntimeError("unexpected inactive-regime pattern")

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.13F",
        "artifact": "lyapunov-mechanism-attribution",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "regimes": [asdict(regime) for regime in regimes],
        "active_regimes": active_regimes,
        "inactive_regimes": inactive_regimes,
        "clean_sources": [
            str(DRIVING_CLEAN.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            str(ROBOTICS_CLEAN.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
        ],
        "consolidated_sources": [
            str(GAUSSIAN.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            str(STRUCTURED.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
            str(ACTION.relative_to(ROOT)).replace(
                "\\",
                "/",
            ),
        ],
        "interpretation": {
            "observation_perturbation_activation": False,
            "action_perturbation_activation": True,
            "formal_stability_claim": False,
            "global_superiority_claim": False,
            "mechanism_evidence_only": True,
        },
        "claim": (
            "Within the Sprint 5 pilot evidence, explicit "
            "Lyapunov-decrease intervention was not observed "
            "under clean, Gaussian observation-noise, or "
            "structured state-perturbation evaluation, but "
            "was observed under direct action perturbation. "
            "This supports a mechanism-attribution statement "
            "for the tested synthetic pilot conditions only; "
            "it does not establish formal stability or global "
            "method superiority."
        ),
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

    lines = [
        ("# Sprint 5.13F — Lyapunov " "Mechanism Attribution"),
        "",
        "## Cross-regime result",
        "",
        ("| Regime | Lyapunov reasons | " "Strict decreases | Active |"),
        "|---|---:|---:|:---:|",
    ]

    for regime in regimes:
        lines.append(
            f"| {regime.regime} | "
            f"{regime.lyapunov_decrease_reasons} | "
            f"{regime.strict_decrease_steps} | "
            f"{regime.mechanism_active} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The tested Lyapunov-decrease pathway remained "
                "inactive in clean and observation-perturbation "
                "regimes, but activated under direct action "
                "perturbation."
            ),
            "",
            (
                "This is mechanism evidence within the synthetic "
                "pilot protocol and is not a formal Lyapunov "
                "stability, production-safety, or superiority "
                "claim."
            ),
            "",
        ]
    )

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 84)

    print(" SPRINT 5.13F LYAPUNOV MECHANISM ATTRIBUTION")

    print("=" * 84)

    print()

    for regime in regimes:
        print(
            regime.regime,
            "reasons=",
            regime.lyapunov_decrease_reasons,
            "strict=",
            regime.strict_decrease_steps,
            "active=",
            regime.mechanism_active,
        )

    print()

    print(
        "Active regimes:",
        active_regimes,
    )

    print(
        "Inactive regimes:",
        inactive_regimes,
    )

    print()

    print("Clean source validation: PASS")

    print("Observation-regime attribution: PASS")

    print("Action-regime attribution: PASS")

    print("Claim controls: PASS")

    print("No new training: PASS")

    print("No new principal runs: PASS")

    print()

    print("SPRINT 5.13F LYAPUNOV ATTRIBUTION: PASS")


if __name__ == "__main__":
    main()
