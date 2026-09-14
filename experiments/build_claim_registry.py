"""Build the Q-VLA Forge proposal claim and limitation registry."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path("results/pilot-readiness/claim-registry.json")


def claim(
    *,
    claim_id: str,
    status: str,
    statement: str,
    evidence_basis: str,
) -> dict[str, str]:
    """Create one proposal claim registry item."""
    return {
        "claim_id": claim_id,
        "status": status,
        "statement": statement,
        "evidence_basis": evidence_basis,
    }


def main() -> None:
    claims = [
        claim(
            claim_id="shared-vla-architecture-both-domains",
            status="SUPPORTED",
            statement=(
                "The shared VLA proxy architecture operates in both "
                "autonomous-driving and robotics proxy domains."
            ),
            evidence_basis=(
                "Sprint 1 established the same shared architecture "
                "with domain-specific state adapters and one common "
                "multimodal latent/action stack."
            ),
        ),
        claim(
            claim_id="three-seed-fp32-baseline",
            status="SUPPORTED",
            statement=(
                "The FP32 baseline was validated using deterministic "
                "seeds 42, 123, and 456 in both proxy domains."
            ),
            evidence_basis=("Sprint 1 baseline manifest and validation summary."),
        ),
        claim(
            claim_id="int8-effective-compression",
            status="SUPPORTED",
            statement=(
                "INT8 achieved approximately 3.846x effective "
                "whole-model storage compression."
            ),
            evidence_basis=("Sprint 2 three-seed compression validation."),
        ),
        claim(
            claim_id="int8-pilot-criterion",
            status="SUPPORTED",
            statement=(
                "INT8 satisfied the frozen >=2x compression and "
                "<=5% relative MSE-change criterion in all three "
                "seeds in both proxy domains."
            ),
            evidence_basis=("Sprint 2 compression validation summary."),
        ),
        claim(
            claim_id="tt-mps-parameter-reduction",
            status="SUPPORTED",
            statement=(
                "Trainable TT/MPS reduced trainable parameter count "
                "by approximately 44.46% in both proxy domains."
            ),
            evidence_basis=("Sprint 3 three-seed structured-training evidence."),
        ),
        claim(
            claim_id="matched-tt-vs-svd-quality",
            status="SUPPORTED_WITH_LIMITATION",
            statement=(
                "In the seed-42 matched-parameter ablation, TT/MPS "
                "produced lower validation and test MSE than matched "
                "SVD in both proxy domains."
            ),
            evidence_basis=(
                "Sprint 3 matched-budget ablation; seed 42 only, "
                "and neither representation reached the paired "
                "FP32 convergence target."
            ),
        ),
        claim(
            claim_id="shared-architecture-reuse",
            status="SUPPORTED_WITH_LIMITATION",
            statement=(
                "The same architecture can reuse common computational "
                "modules across both proxy domains."
            ),
            evidence_basis=(
                "Cross-domain Sprint 1-3 architecture evidence. "
                "This does not imply one universally trained model."
            ),
        ),
        claim(
            claim_id="proxy-latency-under-100ms",
            status="SUPPORTED_WITH_LIMITATION",
            statement=(
                "Measured latency for the compact synthetic proxy model "
                "was below 100 ms in both domains."
            ),
            evidence_basis=(
                "Sprint 1 mean and p95 latency measurements on the "
                "76,179-parameter proxy model. This is not production "
                "deployment validation."
            ),
        ),
        claim(
            claim_id="quantum-advantage",
            status="NOT_SUPPORTED",
            statement=("The pilot does not establish quantum advantage."),
            evidence_basis=(
                "TT/MPS was executed classically and no quantum "
                "hardware comparison was performed."
            ),
        ),
        claim(
            claim_id="quantum-speedup",
            status="NOT_SUPPORTED",
            statement=("The pilot does not establish quantum speedup."),
            evidence_basis=("No quantum-runtime benchmark has been performed."),
        ),
        claim(
            claim_id="native-tt-runtime-speedup",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish native TT/MPS runtime " "acceleration."
            ),
            evidence_basis=(
                "Sprint 2 TT/MPS inference reconstructs dense FP32 "
                "weights, and Sprint 3 TT/MPS does not claim native "
                "runtime speedup."
            ),
        ),
        claim(
            claim_id="native-int8-runtime-speedup",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish native INT8 runtime " "acceleration."
            ),
            evidence_basis=("Sprint 2 uses FP32-dequantized weights for inference."),
        ),
        claim(
            claim_id="svd-training-efficiency-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish an SVD training-efficiency " "advantage."
            ),
            evidence_basis=(
                "Driving SVD required more optimizer steps on average, "
                "and robotics SVD reached the paired target in 0/3 seeds."
            ),
        ),
        claim(
            claim_id="tt-mps-training-efficiency-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a TT/MPS "
                "training-efficiency advantage."
            ),
            evidence_basis=(
                "TT/MPS reached the paired FP32 target in 0/3 seeds "
                "in both proxy domains."
            ),
        ),
        claim(
            claim_id="production-vla-validation",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish performance on a " "production-scale VLA."
            ),
            evidence_basis=("Experiments use a compact synthetic proxy architecture."),
        ),
        claim(
            claim_id="production-autonomous-driving-validation",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish production autonomous-driving "
                "performance."
            ),
            evidence_basis=(
                "No production driving stack or real-world benchmark " "was evaluated."
            ),
        ),
        claim(
            claim_id="production-robotics-validation",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish production robotics " "performance."
            ),
            evidence_basis=(
                "No production robotics stack or real-world benchmark " "was evaluated."
            ),
        ),
        claim(
            claim_id="functional-safety-certification",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish ISO 26262, IEC 62061, "
                "or ISO 10218 compliance or certification."
            ),
            evidence_basis=("Functional-safety certification has not been performed."),
        ),
        claim(
            claim_id="ppo-sample-efficiency",
            status="SUPPORTED",
            statement=(
                "Classical PPO sample efficiency was evaluated under "
                "the frozen Sprint 4 protocol."
            ),
            evidence_basis=(
                "Sprint 4.5 established the PPO baseline and Sprint 4.11 "
                "measured target reach and normalized learning-curve progress. "
                "Classical PPO reached all six paired frozen targets."
            ),
        ),
        claim(
            claim_id="pqc-vqc-sample-efficiency",
            status="SUPPORTED_WITH_LIMITATION",
            statement=(
                "Hybrid PQC/VQC policy sample efficiency was evaluated "
                "under the frozen Sprint 4 protocol."
            ),
            evidence_basis=(
                "Sprints 4.8-4.11 evaluated six principal hybrid-QML runs. "
                "The hybrid policy reached 0/6 paired frozen PPO targets, "
                "so evaluation is supported but an efficiency advantage is not."
            ),
        ),
        claim(
            claim_id="hybrid-qml-policy-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a robust hybrid-QML "
                "policy advantage over classical controls."
            ),
            evidence_basis=(
                "Sprint 4.10 found 0/6 QML target reaches versus 6/6 "
                "for full PPO, and Sprint 4.12-4.13 found "
                "domain-dependent matched-budget representation effects."
            ),
        ),
        claim(
            claim_id="lyapunov-safety-improvement",
            status="NOT_YET_TESTED",
            statement=("Lyapunov safety-filter improvement has not yet been tested."),
            evidence_basis=("Planned for Sprint 5."),
        ),
        claim(
            claim_id="perturbation-robustness",
            status="NOT_YET_TESTED",
            statement=(
                "Robustness under Gaussian, state, and action perturbations "
                "has not yet been tested."
            ),
            evidence_basis=("Planned for Sprint 5."),
        ),
        claim(
            claim_id="cross-domain-rl-consistency",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish directionally consistent "
                "matched-budget RL representation behavior across the "
                "autonomous-driving and robotics proxy domains."
            ),
            evidence_basis=(
                "Sprint 4.13 cross-domain comparison: at exactly matched "
                "compact actor budgets, hybrid QML achieved higher mean "
                "normalized AUC in autonomous driving, while matched "
                "classical achieved higher mean normalized AUC in robotics. "
                "The representation-effect direction therefore reversed "
                "across domains."
            ),
        ),
        claim(
            claim_id="sprint4-classical-ppo-baseline",
            status="SUPPORTED",
            statement=(
                "Classical PPO baselines were established for "
                "autonomous-driving and robotics proxy domains using "
                "principal seeds 42, 123, and 456."
            ),
            evidence_basis=(
                "Sprint 4.5 classical PPO evidence; all six domain-seed "
                "runs completed the frozen 20,000-step protocol and all "
                "six paired PPO-derived targets were reached."
            ),
        ),
        claim(
            claim_id="sprint4-hybrid-pqc-ppo-implementation",
            status="SUPPORTED",
            statement=(
                "A hybrid quantum-classical PPO policy with a variational "
                "quantum actor and classical value critic was implemented "
                "and evaluated in both proxy domains."
            ),
            evidence_basis=(
                "Sprints 4.6-4.10: four-qubit PennyLane PQC, hybrid policy "
                "integration, six principal QML runs, and independent "
                "three-seed validation."
            ),
        ),
        claim(
            claim_id="sprint4-hybrid-actor-compactness",
            status="SUPPORTED",
            statement=(
                "The hybrid PQC actor substantially reduced actor parameter "
                "count relative to the classical PPO actor in both proxy domains."
            ),
            evidence_basis=(
                "Driving actor parameters reduced from 1318 to 54 "
                "(approximately 95.90%); robotics reduced from 1382 to 62 "
                "(approximately 95.51%). Counts were identical across "
                "seeds 42, 123, and 456."
            ),
        ),
        claim(
            claim_id="sprint4-driving-qml-sample-efficiency-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a hybrid-QML "
                "sample-efficiency advantage for autonomous driving."
            ),
            evidence_basis=(
                "Sprint 4.8 and Sprint 4.10 validation: the hybrid QML "
                "policy reached 0/3 paired frozen driving PPO reward "
                "targets within 20,000 environment steps."
            ),
        ),
        claim(
            claim_id="sprint4-robotics-qml-sample-efficiency-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a hybrid-QML "
                "sample-efficiency advantage for robotics."
            ),
            evidence_basis=(
                "Sprint 4.9 and Sprint 4.10 validation: the hybrid QML "
                "policy reached 0/3 paired frozen robotics PPO reward "
                "targets within 20,000 environment steps."
            ),
        ),
        claim(
            claim_id="sprint4-robust-qml-sample-efficiency-advantage",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a robust hybrid-QML "
                "sample-efficiency advantage across the six principal "
                "domain-seed comparisons."
            ),
            evidence_basis=(
                "Sprint 4.10 three-seed validation reconstructed 0/6 QML "
                "target reaches versus 6/6 classical PPO target reaches. "
                "The pre-specified requirement of all three seeds reaching "
                "target with mean >=10% sample-efficiency improvement was "
                "not satisfied in either domain."
            ),
        ),
        claim(
            claim_id="sprint4-quantum-computational-speedup",
            status="NOT_SUPPORTED",
            statement=("Sprint 4 does not establish quantum computational speedup."),
            evidence_basis=(
                "Hybrid QML experiments used PennyLane default.qubit with "
                "shots=None. Wall-clock runtime was treated as diagnostic "
                "only and was not used for a speedup claim."
            ),
        ),
        claim(
            claim_id="sprint4-quantum-hardware-advantage",
            status="NOT_SUPPORTED",
            statement=("Sprint 4 does not establish quantum hardware advantage."),
            evidence_basis=(
                "No quantum hardware execution or hardware-vs-classical "
                "comparison was performed; all QML experiments used simulation."
            ),
        ),
        claim(
            claim_id="sprint4-cross-domain-architecture-reuse",
            status="SUPPORTED",
            statement=(
                "A common hybrid QML policy architecture and PPO evaluation "
                "framework was reused across both autonomous-driving and "
                "robotics proxy domains."
            ),
            evidence_basis=(
                "Sprint 4.13 cross-domain analysis verified reuse of the "
                "same four-qubit, two-layer, 16-parameter PQC core, "
                "three-dimensional action interface, PPO protocol, "
                "20,000-step budget, principal seed set, and evaluation "
                "methodology. Observation dimensions and trained weights "
                "remained domain-specific."
            ),
        ),
        claim(
            claim_id="sprint4-cross-domain-representation-consistency",
            status="NOT_SUPPORTED",
            statement=(
                "The pilot does not establish a directionally consistent "
                "matched-budget representation advantage across both proxy domains."
            ),
            evidence_basis=(
                "Sprint 4.13 reconstructed a matched-classical minus "
                "hybrid-QML mean normalized-AUC difference of approximately "
                "-0.334467 in autonomous driving and +0.079151 in robotics. "
                "The sign reversal means neither hybrid QML nor matched "
                "classical was the consistent AUC winner across both domains."
            ),
        ),
    ]

    valid_statuses = {
        "SUPPORTED",
        "SUPPORTED_WITH_LIMITATION",
        "NOT_SUPPORTED",
        "NOT_YET_TESTED",
    }

    for item in claims:
        if item["status"] not in valid_statuses:
            raise ValueError(f"Invalid claim registry status: {item['status']}")

    claim_ids = [item["claim_id"] for item in claims]

    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Duplicate claim registry claim_id detected.")

    if len(claims) != 34:
        raise ValueError(f"Expected 34 claims, found {len(claims)}.")

    payload = {
        "audit": ("Q-VLA Forge Proposal Claim and Limitation Registry"),
        "statuses": sorted(valid_statuses),
        "pilot_scope_statement": (
            "Q-VLA Forge Phase-1 is a pilot-scale controlled "
            "experimental framework using synthetic autonomous-driving "
            "and robotics proxy tasks. It isolates and compares "
            "classical, quantum-inspired, and hybrid QML components "
            "under reproducible conditions. Results must not be interpreted "
            "as validation of a production-scale VLA, autonomous-driving "
            "stack, robotic system, or functional-safety certification."
        ),
        "claims": claims,
        "future_validation_pathways": {
            "driving": [
                "CARLA",
                "nuScenes",
                "larger VLA backbone",
            ],
            "robotics": [
                "RLBench",
                "LIBERO",
                "Open X-Embodiment",
                "OpenVLA or comparable backbone",
            ],
            "quantum": [
                "shallow hardware-compatible PQC",
                "simulator-first validation",
                "hardware pathway after controlled simulation",
            ],
        },
        "overall_passed": True,
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

    counts = {
        status: sum(item["status"] == status for item in claims)
        for status in valid_statuses
    }

    print()
    print("==================================================")
    print(" Q-VLA FORGE CLAIM REGISTRY")
    print("==================================================")

    for status in (
        "SUPPORTED",
        "SUPPORTED_WITH_LIMITATION",
        "NOT_SUPPORTED",
        "NOT_YET_TESTED",
    ):
        print(f"{status}: {counts[status]}")

    print(f"Total claims: {len(claims)}")

    print("Overall: PASS")

    print(f"Artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
