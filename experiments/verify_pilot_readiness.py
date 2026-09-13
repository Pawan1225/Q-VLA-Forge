"""Independently verify Sprint 3.13 pilot-readiness artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

READINESS_PATH = Path("results/pilot-readiness/sprint1-3-readiness.json")

DATASET_AUDIT_PATH = Path("results/pilot-readiness/dataset-audit.json")

ACCOUNTING_PATH = Path("results/pilot-readiness/representation-accounting.json")

CLAIM_REGISTRY_PATH = Path("results/pilot-readiness/claim-registry.json")


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def require_true(
    condition: bool,
    message: str,
) -> None:
    """Raise if one readiness condition is false."""
    if not condition:
        raise AssertionError(message)


def find_claim(
    registry: dict[str, Any],
    claim_id: str,
) -> dict[str, Any]:
    """Return one uniquely registered claim."""
    matches = [item for item in registry["claims"] if item["claim_id"] == claim_id]

    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one claim: {claim_id}")

    return matches[0]


def main() -> None:
    readiness = load_json(READINESS_PATH)

    dataset = load_json(DATASET_AUDIT_PATH)

    accounting = load_json(ACCOUNTING_PATH)

    claims = load_json(CLAIM_REGISTRY_PATH)

    expected_checks = {
        "seed_consistency",
        "domain_consistency",
        "baseline_identity",
        "dataset_integrity",
        "representation_accounting",
        "quantum_taxonomy",
        "architecture_terminology",
        "claim_controls",
    }

    require_true(
        set(readiness["checks"]) == expected_checks,
        "Unexpected readiness-check set",
    )

    for (
        name,
        passed,
    ) in readiness["checks"].items():
        require_true(
            passed is True,
            f"Readiness check failed: {name}",
        )

    scope = readiness["scope"]

    require_true(
        scope["new_training_performed"] is False,
        "Audit unexpectedly performed new training",
    )

    require_true(
        scope["new_model_tuning_performed"] is False,
        "Audit unexpectedly performed tuning",
    )

    require_true(
        scope["new_algorithm_introduced"] is False,
        "Audit unexpectedly introduced a new algorithm",
    )

    require_true(
        scope["frozen_sprint_results_recomputed"] is False,
        "Frozen Sprint results were unexpectedly recomputed",
    )

    require_true(
        dataset["overall_passed"] is True,
        "Dataset audit failed",
    )

    require_true(
        accounting["overall_passed"] is True,
        "Representation accounting audit failed",
    )

    require_true(
        accounting["sprint2_runtime_controls"]["native_compressed_runtime_used"]
        is False,
        "Unexpected native compressed runtime claim",
    )

    require_true(
        accounting["sprint3_runtime_controls"]["quantum_advantage_claimed"] is False,
        "Unexpected quantum-advantage claim",
    )

    require_true(
        accounting["sprint3_runtime_controls"]["quantum_speedup_claimed"] is False,
        "Unexpected quantum-speedup claim",
    )

    require_true(
        accounting["sprint3_runtime_controls"]["native_tt_runtime_speedup_claimed"]
        is False,
        "Unexpected native TT speedup claim",
    )

    require_true(
        claims["overall_passed"] is True,
        "Claim registry failed",
    )

    required_claim_statuses = {
        "quantum-advantage": "NOT_SUPPORTED",
        "quantum-speedup": "NOT_SUPPORTED",
        "native-tt-runtime-speedup": "NOT_SUPPORTED",
        "native-int8-runtime-speedup": "NOT_SUPPORTED",
        "production-vla-validation": "NOT_SUPPORTED",
        "functional-safety-certification": "NOT_SUPPORTED",
        "proxy-latency-under-100ms": ("SUPPORTED_WITH_LIMITATION"),
        "ppo-sample-efficiency": "NOT_YET_TESTED",
        "pqc-vqc-sample-efficiency": "NOT_YET_TESTED",
        "lyapunov-safety-improvement": "NOT_YET_TESTED",
        "perturbation-robustness": "NOT_YET_TESTED",
    }

    for (
        claim_id,
        expected_status,
    ) in required_claim_statuses.items():
        actual = find_claim(
            claims,
            claim_id,
        )["status"]

        require_true(
            actual == expected_status,
            (
                f"Claim status mismatch for {claim_id}: "
                f"{actual} != {expected_status}"
            ),
        )

    handoff = readiness["handoff"]

    require_true(
        handoff["classical_rl"] == "PPO",
        "Unexpected classical RL handoff",
    )

    require_true(
        handoff["seeds"]
        == [
            42,
            123,
            456,
        ],
        "Unexpected Sprint 4 seed handoff",
    )

    require_true(
        set(handoff["domains"])
        == {
            "autonomous_driving",
            "robotics",
        },
        "Unexpected Sprint 4 domain handoff",
    )

    require_true(
        readiness["sprint4_ready"] is True,
        "Sprint 4 readiness is false",
    )

    print()
    print("============================================")
    print(" SPRINT 3.13 INDEPENDENT VERIFIER")
    print("============================================")
    print("Readiness checks: PASS")
    print("Audit scope controls: PASS")
    print("Dataset integrity: PASS")
    print("Representation accounting: PASS")
    print("Claim controls: PASS")
    print("Sprint 4 handoff: PASS")
    print()
    print("INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
