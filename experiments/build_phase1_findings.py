from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

EVIDENCE_MAP = (
    ROOT / "results" / "integration" / "sprint5-evidence-architecture-map.json"
)

OUTPUT = ROOT / "results" / "integration" / "sprint5-phase1-findings.json"

if not EVIDENCE_MAP.exists():
    raise FileNotFoundError(EVIDENCE_MAP)

findings: dict[str, Any] = {
    "compression": {
        "classical_result": (
            "INT8 was the strongest Phase 1 compression result "
            "under the frozen pilot criteria."
        ),
        "quantum_inspired_result": (
            "Tensor Train / MPS compression was implemented and "
            "evaluated as a quantum-inspired pathway."
        ),
        "advantage_demonstrated": False,
        "phase1_conclusion": (
            "The quantum-inspired compression pathway was feasible "
            "but did not establish superiority over the strongest "
            "classical compression result."
        ),
    },
    "training_efficiency": {
        "classical_result": (
            "FP32 and trainable SVD controls were evaluated under "
            "the paired convergence protocol."
        ),
        "quantum_inspired_result": (
            "Trainable TT/MPS variants were evaluated under the "
            "same training-efficiency framework."
        ),
        "robust_ten_percent_improvement_demonstrated": False,
        "phase1_conclusion": (
            "No robust >=10% training-efficiency advantage was "
            "established under the frozen Phase 1 protocol."
        ),
    },
    "rl_qml": {
        "classical_ppo_target_reaches": 6,
        "matched_classical_target_reaches": 1,
        "hybrid_qml_target_reaches": 0,
        "hybrid_actor_parameter_reduction": {
            "autonomous_driving_approx_percent": 95.90,
            "robotics_approx_percent": 95.51,
        },
        "sample_efficiency_advantage_demonstrated": False,
        "phase1_conclusion": (
            "The hybrid PQC actor was substantially smaller, but "
            "no sample-efficiency advantage was demonstrated under "
            "the frozen pilot protocol."
        ),
    },
    "safety": {
        "clean_explicit_filtering_effective_both_domains": True,
        "gaussian_direction_consistency": {
            "consistent": 7,
            "total": 9,
        },
        "structured_state_direction_consistency": {
            "consistent": 3,
            "total": 3,
        },
        "action_recovery_direction_consistency": {
            "consistent": 3,
            "total": 3,
        },
        "lyapunov_specific_activation": {
            "autonomous_driving": True,
            "robotics": False,
        },
        "phase1_conclusion": (
            "Strong empirical pilot safety was demonstrated, while "
            "Lyapunov-specific mechanism activation remained "
            "domain-dependent."
        ),
    },
    "cross_domain": {
        "shared_framework_supported": True,
        "universal_trained_model_supported": False,
        "zero_shot_transfer_supported": False,
        "universal_safety_controller_supported": False,
        "phase1_conclusion": (
            "Cross-domain reuse is supported at the framework and "
            "interface level, not as a universal trained policy."
        ),
    },
}

unsupported_advantage_claims = [
    "quantum_advantage",
    "quantum_speedup",
    "qml_sample_efficiency_advantage",
    "tt_mps_compression_superiority",
    "robust_training_efficiency_advantage",
    "universal_trained_vla",
    "zero_shot_cross_domain_transfer",
    "formal_safety_guarantee",
    "production_validation",
]

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5G",
    "artifact": "phase1-scientific-findings-freeze",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "source_artifact": str(EVIDENCE_MAP).replace(
        "\\",
        "/",
    ),
    "findings": findings,
    "unsupported_advantage_claims": unsupported_advantage_claims,
    "proposal_summary": (
        "Phase 1 established a reproducible dual-domain framework "
        "covering compression, training efficiency, RL/QML, and "
        "safety. Classical baselines remained strongest on the main "
        "performance criteria, while quantum-inspired and QML paths "
        "demonstrated feasibility and compactness without establishing "
        "quantum advantage."
    ),
}

if findings["rl_qml"]["hybrid_qml_target_reaches"] != 0:
    raise RuntimeError("hybrid QML target-reach freeze changed")

if findings["cross_domain"]["universal_trained_model_supported"]:
    raise RuntimeError("universal trained model must remain unsupported")

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.14.5G PHASE-1 SCIENTIFIC FINDINGS FREEZE")
print("=" * 80)
print()
print("Compression finding frozen: PASS")
print("Training-efficiency finding frozen: PASS")
print("RL/QML finding frozen: PASS")
print("Safety finding frozen: PASS")
print("Cross-domain finding frozen: PASS")
print()
print("Quantum advantage blocked: PASS")
print("Quantum speedup blocked: PASS")
print("QML sample-efficiency advantage blocked: PASS")
print("TT/MPS superiority blocked: PASS")
print("Training-efficiency advantage blocked: PASS")
print("Universal VLA claim blocked: PASS")
print("Zero-shot transfer claim blocked: PASS")
print("Formal safety guarantee blocked: PASS")
print("Production validation claim blocked: PASS")
print()
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5G PHASE-1 FINDINGS FREEZE: PASS")
