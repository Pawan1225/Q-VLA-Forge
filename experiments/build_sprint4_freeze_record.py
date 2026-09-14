"""Build the Sprint 4 scientific freeze record."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.rl_final_gate import file_sha256

MANIFEST_PATH = Path("results/rl/final/sprint4-final-manifest.json")

EVIDENCE_PATH = Path("results/rl/evidence/sprint4-rl-evidence.json")

CROSS_DOMAIN_PATH = Path("results/rl/cross-domain/sprint4-cross-domain.json")

OUTPUT_PATH = Path("results/rl/final/sprint4-freeze-record.json")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def build_freeze_record() -> dict[str, object]:
    manifest = load_json(MANIFEST_PATH)

    evidence = load_json(EVIDENCE_PATH)

    cross_domain = load_json(CROSS_DOMAIN_PATH)

    records = {
        (
            record["domain"],
            record["method"],
        ): record
        for record in cross_domain["method_records"]
    }

    driving_full = int(
        records[
            (
                "autonomous_driving",
                "full_ppo",
            )
        ]["actor_parameters"]
    )

    driving_qml = int(
        records[
            (
                "autonomous_driving",
                "hybrid_qml",
            )
        ]["actor_parameters"]
    )

    robotics_full = int(
        records[
            (
                "robotics",
                "full_ppo",
            )
        ]["actor_parameters"]
    )

    robotics_qml = int(
        records[
            (
                "robotics",
                "hybrid_qml",
            )
        ]["actor_parameters"]
    )

    driving_reduction = (1.0 - driving_qml / driving_full) * 100.0

    robotics_reduction = (1.0 - robotics_qml / robotics_full) * 100.0

    conclusion = cross_domain["cross_domain_conclusion"]

    boundary = cross_domain["scientific_boundary"]

    claim_status = {
        claim["claim_id"]: claim["status"]
        for claim in evidence["claim_matrix"]["claims"]
    }

    return {
        "sprint": "4",
        "freeze_sprint": "4.15",
        "scientific_state": "frozen",
        "final_acceptance": "pending",
        "primary_metric": ("frozen paired-target reach " "under the Sprint 4 protocol"),
        "principal_seeds": [
            42,
            123,
            456,
        ],
        "principal_runs": {
            "full_ppo": 6,
            "matched_classical": 6,
            "hybrid_qml": 6,
        },
        "reproduction_runs": {
            "hybrid_qml": 2,
            "matched_classical": 2,
            "total": 4,
        },
        "target_reach": {
            "full_ppo": {
                "driving": 3,
                "robotics": 3,
                "total": 6,
                "out_of": 6,
            },
            "matched_classical": {
                "driving": 0,
                "robotics": 1,
                "total": 1,
                "out_of": 6,
            },
            "hybrid_qml": {
                "driving": 0,
                "robotics": 0,
                "total": 0,
                "out_of": 6,
            },
        },
        "actor_compactness": {
            "autonomous_driving": {
                "full_ppo_actor_parameters": (driving_full),
                "compact_actor_parameters": (driving_qml),
                "reduction_percent": (driving_reduction),
            },
            "robotics": {
                "full_ppo_actor_parameters": (robotics_full),
                "compact_actor_parameters": (robotics_qml),
                "reduction_percent": (robotics_reduction),
            },
        },
        "matched_budget_representation": {
            "autonomous_driving": {
                "direction": (conclusion["matched_auc_direction_driving"]),
            },
            "robotics": {
                "direction": (conclusion["matched_auc_direction_robotics"]),
            },
            "direction_consistent_across_domains": (
                conclusion["matched_auc_direction_consistent_across_domains"]
            ),
        },
        "cross_domain": {
            "architecture_reused": (conclusion["architecture_reused"]),
            "shared_trained_weights": (conclusion["shared_trained_weights"]),
            "robust_cross_domain_qml_advantage": (
                conclusion["robust_cross_domain_qml_advantage"]
            ),
            "robust_cross_domain_matched_classical_advantage": (
                conclusion["robust_cross_domain_matched_classical_advantage"]
            ),
        },
        "claim_boundaries": {
            "qml_sample_efficiency_advantage": False,
            "robust_cross_domain_qml_advantage": False,
            "quantum_speedup": False,
            "quantum_hardware_advantage": False,
            "universal_policy": False,
            "transfer_learning": False,
            "zero_shot_transfer": False,
            "shared_trained_weights": False,
        },
        "claim_matrix_status": {
            "qml_sample_efficiency": (claim_status["qml-sample-efficiency"]),
            "quantum_speedup": (claim_status["quantum-speedup"]),
            "quantum_hardware_advantage": (claim_status["quantum-hardware-advantage"]),
        },
        "scientific_boundary": {
            "transfer_learning_performed": (boundary["transfer_learning_performed"]),
            "zero_shot_transfer_performed": (boundary["zero_shot_transfer_performed"]),
            "universal_policy_claimed": (boundary["universal_policy_claimed"]),
            "quantum_speedup_claimed": (boundary["quantum_speedup_claimed"]),
        },
        "safety": {
            "sprint4_safety_status": ("NOT_YET_TESTED_IN_SPRINT_4"),
            "safety_guarantee_established": False,
            "interpretation": (
                "Sprint 4 does not test formal "
                "safety filtering or safety guarantees. "
                "Safety is handed off to Sprint 5."
            ),
        },
        "manifest": {
            "path": MANIFEST_PATH.as_posix(),
            "sha256": file_sha256(MANIFEST_PATH),
            "artifact_count": manifest["artifact_count"],
            "group_count": manifest["group_count"],
        },
        "quality_gate": {
            "full_test_count": None,
            "final_verifier_state": "pending",
        },
    }


def main() -> None:
    record = build_freeze_record()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Freeze record:",
        OUTPUT_PATH.as_posix(),
    )


if __name__ == "__main__":
    main()
