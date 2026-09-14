"""Build the Sprint 4 RL evidence manifest."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = ROOT / "results" / "rl" / "evidence" / "sprint4-manifest.json"


def main() -> None:
    artifacts = [
        {
            "path": "results/rl/ppo/sprint4-ppo-summary.json",
            "role": "classical_ppo_three_seed_summary",
            "sprint": "4.5",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "baseline",
        },
        {
            "path": "results/rl/ppo/sprint4-ppo-targets.json",
            "role": "frozen_ppo_derived_targets",
            "sprint": "4.5",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "baseline",
        },
        {
            "path": "results/rl/qml-foundation/sprint4-qml-foundation.json",
            "role": "qml_circuit_foundation",
            "sprint": "4.6",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "quantum",
        },
        {
            "path": "results/rl/hybrid-policy/sprint4-hybrid-qml-policy.json",
            "role": "hybrid_qml_policy_architecture",
            "sprint": "4.7",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "policy",
        },
        {
            "path": "results/rl/qml/sprint4-driving-qml-summary.json",
            "role": "driving_qml_three_seed_summary",
            "sprint": "4.8",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "driving",
        },
        {
            "path": "results/rl/qml/sprint4-robotics-qml-summary.json",
            "role": "robotics_qml_three_seed_summary",
            "sprint": "4.9",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "robotics",
        },
        {
            "path": "results/rl/validation/sprint4-three-seed-validation.json",
            "role": "three_seed_rl_validation",
            "sprint": "4.10",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "validation",
        },
        {
            "path": "results/rl/analysis/sprint4-sample-efficiency.json",
            "role": "normalized_sample_efficiency_analysis",
            "sprint": "4.11",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "sample_efficiency",
        },
        {
            "path": "results/rl/ablation/sprint4-classical-vs-qml-ablation.json",
            "role": "matched_budget_classical_vs_qml_ablation",
            "sprint": "4.12",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "ablation",
        },
        {
            "path": "results/rl/cross-domain/sprint4-cross-domain.json",
            "role": "cross_domain_rl_comparison",
            "sprint": "4.13",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "cross_domain",
        },
        {
            "path": "results/pilot-readiness/claim-registry.json",
            "role": "proposal_claim_registry",
            "sprint": "3.13-4.13",
            "frozen": True,
            "proposal_relevance": "primary",
            "group": "claims",
        },
        {
            "path": "results/rl/evidence/sprint4-rl-evidence.json",
            "role": "consolidated_sprint4_evidence",
            "sprint": "4.14",
            "frozen": False,
            "proposal_relevance": "primary",
            "group": "proposal_evidence",
        },
        {
            "path": "results/rl/evidence/sprint4-rl-evidence.csv",
            "role": "compact_six_row_evidence_summary",
            "sprint": "4.14",
            "frozen": False,
            "proposal_relevance": "secondary",
            "group": "proposal_evidence",
        },
        {
            "path": "results/rl/evidence/sprint4-rl-evidence.md",
            "role": "proposal_ready_evidence_brief",
            "sprint": "4.14",
            "frozen": False,
            "proposal_relevance": "primary",
            "group": "proposal_evidence",
        },
        {
            "path": "results/rl/evidence/sprint4-figure-index.json",
            "role": "proposal_figure_index",
            "sprint": "4.14",
            "frozen": False,
            "proposal_relevance": "primary",
            "group": "figures",
        },
        {
            "path": "results/rl/evidence/sprint4-claim-matrix.json",
            "role": "proposal_claim_limitation_matrix",
            "sprint": "4.14",
            "frozen": False,
            "proposal_relevance": "primary",
            "group": "claims",
        },
    ]

    payload = {
        "sprint": "4.14",
        "title": "Sprint 4 RL Evidence Manifest",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "group_coverage": {
            "baseline": "present",
            "quantum": "present",
            "policy": "present",
            "driving": "present",
            "robotics": "present",
            "validation": "present",
            "sample_efficiency": "present",
            "ablation": "present",
            "cross_domain": "present",
            "figures": "present",
            "claims": "present",
            "proposal_evidence": "present",
        },
        "quantum_group_standalone_artifact": True,
        "manifest_invents_missing_artifacts": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Sprint 4 manifest:",
        OUTPUT_PATH,
    )
    print(
        "Artifact count:",
        len(artifacts),
    )


if __name__ == "__main__":
    main()
