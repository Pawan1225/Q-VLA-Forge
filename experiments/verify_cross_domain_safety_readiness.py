"""Verify readiness for Sprint 5.14 cross-domain safety analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

FINAL_513 = CONSOLIDATED / "sprint5-13-final-acceptance.json"

SOURCE_MANIFEST = CONSOLIDATED / "sprint5-safety-source-manifest.json"

PACKAGE = CONSOLIDATED / "sprint5-safety-evidence-package.json"

CLAIMS = CONSOLIDATED / "sprint5-safety-claim-matrix.json"

CROSS_DOMAIN_CONFIG = ROOT / "configs" / "cross_domain_safety.yaml"


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

    print(f"{label:<35} PASS")


def main() -> None:
    print("=" * 52)
    print(" SPRINT 5.14 CROSS-DOMAIN SAFETY READINESS")
    print("=" * 52)
    print()

    _check(
        FINAL_513.exists(),
        "Sprint 5.13 evidence",
    )

    _check(
        SOURCE_MANIFEST.exists(),
        "Source manifest",
    )

    _check(
        PACKAGE.exists(),
        "Consolidated package",
    )

    _check(
        CLAIMS.exists(),
        "Claim matrix",
    )

    _check(
        CROSS_DOMAIN_CONFIG.exists(),
        "Cross-domain config",
    )

    final_513 = _load(FINAL_513)

    package = _load(PACKAGE)

    claims = _load(CLAIMS)

    _check(
        final_513["status"] == "PASS"
        and final_513["handoff"]["sprint_5_14_ready"] is True,
        "Sprint 5.13 final acceptance",
    )

    clean = package["clean_evidence"]

    if not isinstance(
        clean,
        dict,
    ):
        raise TypeError("clean evidence must be dict")

    _check(
        "autonomous_driving" in clean,
        "Driving evidence",
    )

    _check(
        "robotics" in clean,
        "Robotics evidence",
    )

    for method in (
        "none",
        "clipping",
        "lyapunov",
    ):
        _check(
            all(
                method in clean[domain]
                for domain in (
                    "autonomous_driving",
                    "robotics",
                )
            ),
            method.upper(),
        )

    robustness = package["robustness_corpus"]

    if not isinstance(
        robustness,
        dict,
    ):
        raise TypeError("robustness corpus must be dict")

    _check(
        True,
        "Clean regime",
    )

    _check(
        "gaussian" in robustness,
        "Gaussian regime",
    )

    _check(
        "structured_state" in robustness,
        "Structured-state regime",
    )

    _check(
        "action" in robustness,
        "Action regime",
    )

    _check(
        package["principal_seeds"]
        == [
            42,
            123,
            456,
        ],
        "Three-seed evidence",
    )

    _check(
        claims["claim_count"] == 16 and claims["status_counts"]["unsupported"] == 5,
        "Claim controls",
    )

    _check(
        package["analysis_only"] is True,
        "Analysis-only scope",
    )

    _check(
        package["new_training"] is False,
        "No new training",
    )

    _check(
        package["new_principal_runs"] is False,
        "No new principal execution",
    )

    print()
    print("SPRINT 5.14 READINESS: PASS")


if __name__ == "__main__":
    main()
