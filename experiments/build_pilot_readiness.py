"""Build the final pre-Sprint-4 pilot-readiness artifact."""

from __future__ import annotations

import json
from pathlib import Path

SPRINT1_PATH = Path("results/sprint1-baseline-manifest.json")

SPRINT2_PATH = Path("results/compression/sprint2-compression-manifest.json")

SPRINT3_PATH = Path("results/training/sprint3-training-manifest.json")

DATASET_AUDIT_PATH = Path("results/pilot-readiness/dataset-audit.json")

ACCOUNTING_PATH = Path("results/pilot-readiness/representation-accounting.json")

CLAIM_REGISTRY_PATH = Path("results/pilot-readiness/claim-registry.json")

OUTPUT_PATH = Path("results/pilot-readiness/sprint1-3-readiness.json")


def load_json(
    path: Path,
) -> dict:
    """Load one required JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def claim_status(
    claims: dict,
    claim_id: str,
) -> str:
    """Return the status for one registered proposal claim."""
    matches = [item for item in claims["claims"] if item["claim_id"] == claim_id]

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one claim: {claim_id}")

    return matches[0]["status"]


def main() -> None:
    sprint1 = load_json(SPRINT1_PATH)

    sprint2 = load_json(SPRINT2_PATH)

    sprint3 = load_json(SPRINT3_PATH)

    dataset = load_json(DATASET_AUDIT_PATH)

    accounting = load_json(ACCOUNTING_PATH)

    claims = load_json(CLAIM_REGISTRY_PATH)

    expected_seeds = [
        42,
        123,
        456,
    ]

    expected_domains = {
        "autonomous_driving",
        "robotics",
    }

    seed_consistency = (
        sprint1["seeds"] == expected_seeds
        and sprint2["validation_seeds"] == expected_seeds
        and sprint3["protocol"]["seeds"] == expected_seeds
    )

    domain_consistency = (
        set(sprint2["domains"]) == expected_domains
        and set(sprint3["protocol"]["domains"]) == expected_domains
    )

    baseline_consistency = (
        sprint1["baseline_name"] == "shared_vla_fp32"
        and sprint3["methods"]["reference"] == "shared_vla_fp32"
        and sprint1["parameters"] == 76179
        and sprint1["fp32_model_size_bytes"] == 304716
    )

    quantum = sprint3["quantum_claim_control"]

    quantum_taxonomy_consistent = (
        sprint2["quantum_hardware_used"] is False
        and quantum["tt_mps_is_quantum_inspired"] is True
        and quantum["quantum_hardware_used"] is False
        and quantum["quantum_advantage_claimed"] is False
        and quantum["quantum_speedup_claimed"] is False
        and quantum["native_tt_runtime_speedup_claimed"] is False
    )

    architecture = sprint3["architecture_claim_control"]

    architecture_terminology_consistent = (
        architecture["shared_architecture"] is True
        and architecture["universal_trained_model"] is False
        and architecture["language_component"] == "lightweight trainable text encoder"
    )

    dataset_integrity_passed = (
        dataset["overall_passed"] is True
        and dataset["all_split_audits_passed"] is True
        and dataset["finite_values_passed"] is True
        and dataset["action_constraints_passed"] is True
        and dataset["state_variance_passed"] is True
        and dataset["action_variance_passed"] is True
    )

    accounting_passed = (
        accounting["overall_passed"] is True
        and accounting["audit_checks"]["baseline_parameter_identity"] is True
        and accounting["audit_checks"]["baseline_byte_identity"] is True
        and accounting["audit_checks"]["sprint2_storage_vs_runtime_distinguished"]
        is True
        and accounting["audit_checks"]["sprint3_trainable_vs_storage_distinguished"]
        is True
        and accounting["audit_checks"]["quantum_taxonomy_consistent"] is True
        and accounting["audit_checks"]["native_runtime_speedup_not_claimed"] is True
    )

    claim_registry_passed = (
        claims["overall_passed"] is True
        and claim_status(
            claims,
            "quantum-advantage",
        )
        == "NOT_SUPPORTED"
        and claim_status(
            claims,
            "quantum-speedup",
        )
        == "NOT_SUPPORTED"
        and claim_status(
            claims,
            "native-tt-runtime-speedup",
        )
        == "NOT_SUPPORTED"
        and claim_status(
            claims,
            "production-vla-validation",
        )
        == "NOT_SUPPORTED"
        and claim_status(
            claims,
            "proxy-latency-under-100ms",
        )
        == "SUPPORTED_WITH_LIMITATION"
        and claim_status(
            claims,
            "ppo-sample-efficiency",
        )
        == "NOT_YET_TESTED"
        and claim_status(
            claims,
            "pqc-vqc-sample-efficiency",
        )
        == "NOT_YET_TESTED"
    )

    sprint4_ready = all(
        (
            seed_consistency,
            domain_consistency,
            baseline_consistency,
            quantum_taxonomy_consistent,
            architecture_terminology_consistent,
            dataset_integrity_passed,
            accounting_passed,
            claim_registry_passed,
        )
    )

    payload = {
        "project": "Q-VLA Forge",
        "audit": ("Sprint 3.13 Pilot Readiness " "and Scientific Audit"),
        "scope": {
            "new_training_performed": False,
            "new_model_tuning_performed": False,
            "new_algorithm_introduced": False,
            "frozen_sprint_results_recomputed": False,
            "purpose": (
                "Audit the frozen Sprint 1-3 evidence chain "
                "before Sprint 4 reinforcement-learning and "
                "hybrid-QML experiments."
            ),
        },
        "sprint_manifests": {
            "sprint1": str(SPRINT1_PATH),
            "sprint2": str(SPRINT2_PATH),
            "sprint3": str(SPRINT3_PATH),
        },
        "audit_artifacts": {
            "dataset_integrity": str(DATASET_AUDIT_PATH),
            "representation_accounting": str(ACCOUNTING_PATH),
            "claim_registry": str(CLAIM_REGISTRY_PATH),
        },
        "checks": {
            "seed_consistency": seed_consistency,
            "domain_consistency": domain_consistency,
            "baseline_identity": baseline_consistency,
            "dataset_integrity": dataset_integrity_passed,
            "representation_accounting": accounting_passed,
            "quantum_taxonomy": quantum_taxonomy_consistent,
            "architecture_terminology": (architecture_terminology_consistent),
            "claim_controls": claim_registry_passed,
        },
        "pilot_scope_statement": claims["pilot_scope_statement"],
        "handoff": {
            "next_sprint": ("Sprint 4 - PPO + Hybrid QML"),
            "classical_rl": "PPO",
            "qml_path": ("angle encoding + PQC/VQC + " "hybrid classical-QML policy"),
            "domains": [
                "autonomous_driving",
                "robotics",
            ],
            "seeds": expected_seeds,
        },
        "sprint4_ready": sprint4_ready,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" SPRINT 3.13 PILOT READINESS")
    print("============================================")

    for (
        name,
        passed,
    ) in payload["checks"].items():
        print(f"{name}: " f"{'PASS' if passed else 'FAIL'}")

    print()
    print(
        "SPRINT 4 HANDOFF:",
        "READY" if sprint4_ready else "NOT READY",
    )

    print(f"Artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
