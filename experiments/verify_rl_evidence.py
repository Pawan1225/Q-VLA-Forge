"""Independently verify the Sprint 4.14 RL evidence package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = ROOT / "results" / "rl" / "evidence" / "sprint4-rl-evidence.json"

VALIDATION_PATH = (
    ROOT / "results" / "rl" / "validation" / "sprint4-three-seed-validation.json"
)

SAMPLE_EFFICIENCY_PATH = (
    ROOT / "results" / "rl" / "analysis" / "sprint4-sample-efficiency.json"
)

ABLATION_PATH = (
    ROOT / "results" / "rl" / "ablation" / "sprint4-classical-vs-qml-ablation.json"
)

CROSS_DOMAIN_PATH = (
    ROOT / "results" / "rl" / "cross-domain" / "sprint4-cross-domain.json"
)

CLAIM_REGISTRY_PATH = ROOT / "results" / "pilot-readiness" / "claim-registry.json"


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: " f"{actual!r} != {expected!r}")


def require_close(
    actual: float,
    expected: float,
    *,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    if abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label} mismatch: " f"{actual} != {expected}")


def cross_domain_record(
    data: dict[str, Any],
    *,
    domain: str,
    method: str,
) -> dict[str, Any]:
    for record in data["method_records"]:
        if record["domain"] == domain and record["method"] == method:
            return record

    raise RuntimeError(f"missing source record: " f"{domain}/{method}")


def main() -> None:
    evidence = load_json(EVIDENCE_PATH)

    validation = load_json(VALIDATION_PATH)

    sample_efficiency = load_json(SAMPLE_EFFICIENCY_PATH)

    ablation = load_json(ABLATION_PATH)

    cross_domain = load_json(CROSS_DOMAIN_PATH)

    claim_registry = load_json(CLAIM_REGISTRY_PATH)

    print()
    print("==========================================")
    print(" SPRINT 4.14 RL EVIDENCE VERIFICATION")
    print("==========================================")
    print()

    # -------------------------------------------------
    # Protocol boundary
    # -------------------------------------------------

    require_equal(
        evidence["new_training_performed"],
        False,
        label="new training",
    )

    require_equal(
        evidence["new_metrics_introduced"],
        False,
        label="new metrics",
    )

    require_equal(
        evidence["principal_seeds"],
        [
            42,
            123,
            456,
        ],
        label="principal seeds",
    )

    require_equal(
        validation["principal_seeds"],
        [
            42,
            123,
            456,
        ],
        label="validation seeds",
    )

    # -------------------------------------------------
    # Reconstruct all six method records from 4.13
    # -------------------------------------------------

    source_records: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        ):
            source_records[
                (
                    domain,
                    method,
                )
            ] = cross_domain_record(
                cross_domain,
                domain=domain,
                method=method,
            )

    # -------------------------------------------------
    # Global target reach
    # -------------------------------------------------

    ppo_reach = sum(
        int(
            source_records[
                (
                    domain,
                    "full_ppo",
                )
            ]["target_reach_count"]
        )
        for domain in (
            "autonomous_driving",
            "robotics",
        )
    )

    matched_reach = sum(
        int(
            source_records[
                (
                    domain,
                    "matched_classical",
                )
            ]["target_reach_count"]
        )
        for domain in (
            "autonomous_driving",
            "robotics",
        )
    )

    qml_reach = sum(
        int(
            source_records[
                (
                    domain,
                    "hybrid_qml",
                )
            ]["target_reach_count"]
        )
        for domain in (
            "autonomous_driving",
            "robotics",
        )
    )

    require_equal(
        ppo_reach,
        6,
        label="source PPO reach",
    )

    require_equal(
        matched_reach,
        1,
        label="source matched reach",
    )

    require_equal(
        qml_reach,
        0,
        label="source QML reach",
    )

    require_equal(
        evidence["target_reach"]["full_ppo"],
        {
            "reached": ppo_reach,
            "total": 6,
        },
        label="evidence PPO reach",
    )

    require_equal(
        evidence["target_reach"]["matched_classical"],
        {
            "reached": matched_reach,
            "total": 6,
        },
        label="evidence matched reach",
    )

    require_equal(
        evidence["target_reach"]["hybrid_qml"],
        {
            "reached": qml_reach,
            "total": 6,
        },
        label="evidence QML reach",
    )

    # -------------------------------------------------
    # Parameter counts and method metrics
    # -------------------------------------------------

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        ):
            source = source_records[
                (
                    domain,
                    method,
                )
            ]

            packaged = evidence["domains"][domain]["methods"][method]

            require_equal(
                packaged["actor_parameters"],
                source["actor_parameters"],
                label=(f"{domain}/{method} " "actor parameters"),
            )

            require_equal(
                packaged["target_reach_count"],
                source["target_reach_count"],
                label=(f"{domain}/{method} " "target reach"),
            )

            for key in (
                "normalized_auc_mean",
                "normalized_auc_sd",
                "best_progress_mean",
                "best_progress_sd",
                "final_progress_mean",
                "final_progress_sd",
            ):
                require_close(
                    float(packaged[key]),
                    float(source[key]),
                    label=(f"{domain}/{method} " f"{key}"),
                )

    driving_ppo = source_records[
        (
            "autonomous_driving",
            "full_ppo",
        )
    ]

    driving_qml = source_records[
        (
            "autonomous_driving",
            "hybrid_qml",
        )
    ]

    robotics_ppo = source_records[
        (
            "robotics",
            "full_ppo",
        )
    ]

    robotics_qml = source_records[
        (
            "robotics",
            "hybrid_qml",
        )
    ]

    require_equal(
        driving_ppo["actor_parameters"],
        1318,
        label="driving PPO actor",
    )

    require_equal(
        driving_qml["actor_parameters"],
        54,
        label="driving QML actor",
    )

    require_equal(
        robotics_ppo["actor_parameters"],
        1382,
        label="robotics PPO actor",
    )

    require_equal(
        robotics_qml["actor_parameters"],
        62,
        label="robotics QML actor",
    )

    # -------------------------------------------------
    # Compactness reconstructed from parameter counts
    # -------------------------------------------------

    driving_reduction = (
        1.0 - (driving_qml["actor_parameters"] / driving_ppo["actor_parameters"])
    ) * 100.0

    robotics_reduction = (
        1.0 - (robotics_qml["actor_parameters"] / robotics_ppo["actor_parameters"])
    ) * 100.0

    require_close(
        evidence["domains"]["autonomous_driving"]["compactness"][
            "parameter_reduction_percent"
        ],
        driving_reduction,
        label="driving compactness",
    )

    require_close(
        evidence["domains"]["robotics"]["compactness"]["parameter_reduction_percent"],
        robotics_reduction,
        label="robotics compactness",
    )

    # -------------------------------------------------
    # Matched-budget directions reconstructed
    # -------------------------------------------------

    driving_matched = source_records[
        (
            "autonomous_driving",
            "matched_classical",
        )
    ]

    robotics_matched = source_records[
        (
            "robotics",
            "matched_classical",
        )
    ]

    driving_delta = (
        driving_matched["normalized_auc_mean"] - driving_qml["normalized_auc_mean"]
    )

    robotics_delta = (
        robotics_matched["normalized_auc_mean"] - robotics_qml["normalized_auc_mean"]
    )

    driving_direction = (
        "matched_classical"
        if driving_delta > 0
        else "hybrid_qml" if driving_delta < 0 else "tie"
    )

    robotics_direction = (
        "matched_classical"
        if robotics_delta > 0
        else "hybrid_qml" if robotics_delta < 0 else "tie"
    )

    require_equal(
        driving_direction,
        "hybrid_qml",
        label="driving direction",
    )

    require_equal(
        robotics_direction,
        "matched_classical",
        label="robotics direction",
    )

    require_equal(
        evidence["matched_budget_ablation"]["driving_direction"],
        driving_direction,
        label="packaged driving direction",
    )

    require_equal(
        evidence["matched_budget_ablation"]["robotics_direction"],
        robotics_direction,
        label="packaged robotics direction",
    )

    require_equal(
        evidence["matched_budget_ablation"]["direction_consistent_across_domains"],
        False,
        label="direction consistency",
    )

    # -------------------------------------------------
    # Cross-domain architecture boundary
    # -------------------------------------------------

    source_conclusion = cross_domain["cross_domain_conclusion"]

    require_equal(
        source_conclusion["architecture_reused"],
        True,
        label="source architecture reuse",
    )

    require_equal(
        evidence["global_results"]["cross_domain_architecture_reuse"],
        True,
        label="packaged architecture reuse",
    )

    require_equal(
        source_conclusion["matched_auc_direction_consistent_across_domains"],
        False,
        label="source representation consistency",
    )

    require_equal(
        evidence["global_results"]["cross_domain_representation_direction_consistent"],
        False,
        label="packaged representation consistency",
    )

    # -------------------------------------------------
    # Sample-efficiency source linkage
    # -------------------------------------------------

    require_equal(
        evidence["sample_efficiency"]["source_sprint"],
        sample_efficiency["sprint"],
        label="sample-efficiency source sprint",
    )

    require_equal(
        evidence["matched_budget_ablation"]["source_sprint"],
        ablation["sprint"],
        label="ablation source sprint",
    )

    # -------------------------------------------------
    # Claim registry boundary
    # -------------------------------------------------

    require_equal(
        claim_registry["overall_passed"],
        True,
        label="claim registry",
    )

    packaged_claim_ids = {claim["claim_id"] for claim in evidence["claims"]}

    required_claim_ids = {
        "sprint4-classical-ppo-baseline",
        "sprint4-hybrid-pqc-ppo-implementation",
        "sprint4-hybrid-actor-compactness",
        "sprint4-robust-qml-sample-efficiency-advantage",
        "sprint4-quantum-computational-speedup",
        "sprint4-quantum-hardware-advantage",
        "sprint4-cross-domain-architecture-reuse",
        "sprint4-cross-domain-representation-consistency",
    }

    if not required_claim_ids.issubset(packaged_claim_ids):
        raise RuntimeError("required Sprint 4 claims missing " "from evidence package")

    claims_by_id = {claim["claim_id"]: claim for claim in evidence["claims"]}

    require_equal(
        claims_by_id["sprint4-classical-ppo-baseline"]["status"],
        "SUPPORTED",
        label="PPO claim status",
    )

    require_equal(
        claims_by_id["sprint4-hybrid-actor-compactness"]["status"],
        "SUPPORTED",
        label="compactness claim status",
    )

    require_equal(
        claims_by_id["sprint4-robust-qml-sample-efficiency-advantage"]["status"],
        "NOT_SUPPORTED",
        label="QML advantage status",
    )

    require_equal(
        claims_by_id["sprint4-quantum-computational-speedup"]["status"],
        "NOT_SUPPORTED",
        label="speedup status",
    )

    require_equal(
        claims_by_id["sprint4-quantum-hardware-advantage"]["status"],
        "NOT_SUPPORTED",
        label="hardware advantage status",
    )

    # -------------------------------------------------
    # Reproducibility
    # -------------------------------------------------

    require_equal(
        validation["global_validation"]["driving_seed42_reproduction_exists"],
        True,
        label="driving QML seed42 reproduction",
    )

    require_equal(
        validation["global_validation"]["robotics_seed42_reproduction_exists"],
        True,
        label="robotics QML seed42 reproduction",
    )

    require_equal(
        evidence["reproducibility"]["driving_matched_classical_seed42_reproduction"],
        True,
        label="driving matched seed42 reproduction",
    )

    require_equal(
        evidence["reproducibility"]["robotics_matched_classical_seed42_reproduction"],
        True,
        label="robotics matched seed42 reproduction",
    )

    print(
        "Full PPO target reach:      ",
        ppo_reach,
        "/ 6",
    )

    print(
        "Matched target reach:       ",
        matched_reach,
        "/ 6",
    )

    print(
        "Hybrid QML target reach:    ",
        qml_reach,
        "/ 6",
    )

    print()

    print("Driving compactness:        PASS")

    print("Robotics compactness:       PASS")

    print()

    print(
        "Driving representation:    ",
        driving_direction,
    )

    print(
        "Robotics representation:   ",
        robotics_direction,
    )

    print()

    print("Architecture reuse:         PASS")

    print("Cross-domain consistency:   FALSE")

    print("Claim boundaries:           PASS")

    print()

    print("SPRINT 4.14 RL EVIDENCE: PASS")


if __name__ == "__main__":
    main()
