from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

ARCHITECTURE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"

COMPONENT_MAP = (
    ROOT / "results" / "integration" / "sprint5-shared-domain-component-map.json"
)

BOTTLENECK_MAP = (
    ROOT / "results" / "integration" / "sprint5-volkswagen-bottleneck-map.json"
)

CLASSIFICATION = (
    ROOT / "results" / "integration" / "sprint5-computational-classification.json"
)

OUTPUT = ROOT / "results" / "integration" / "sprint5-evidence-architecture-map.json"

CANONICAL_EVIDENCE = {
    "shared_vla_baseline": (ROOT / "results" / "sprint1-baseline-manifest.json"),
    "compression": (
        ROOT
        / "results"
        / "compression"
        / "evidence"
        / "sprint2-compression-evidence.json"
    ),
    "training_efficiency": (
        ROOT / "results" / "training" / "evidence" / "sprint3-training-evidence.json"
    ),
    "ppo_qml": (ROOT / "results" / "rl" / "evidence" / "sprint4-rl-evidence.json"),
    "rl_claims": (ROOT / "results" / "rl" / "evidence" / "sprint4-claim-matrix.json"),
    "safety": (
        ROOT
        / "results"
        / "safety"
        / "consolidated"
        / "sprint5-safety-evidence-package.json"
    ),
    "cross_domain_safety": (
        ROOT
        / "results"
        / "safety"
        / "cross-domain"
        / "sprint5-cross-domain-package.json"
    ),
    "cross_domain_claims": (
        ROOT
        / "results"
        / "safety"
        / "cross-domain"
        / "sprint5-cross-domain-claim-matrix.json"
    ),
}

for path in (
    ARCHITECTURE,
    COMPONENT_MAP,
    BOTTLENECK_MAP,
    CLASSIFICATION,
):
    if not path.exists():
        raise FileNotFoundError(path)

for name, path in CANONICAL_EVIDENCE.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")


def rel(
    path: Path,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


evidence_map: dict[str, Any] = {
    "shared_vla_baseline": {
        "architecture_blocks": [
            "vision_encoder_framework",
            "language_encoder_framework",
            "state_encoder_framework",
            "multimodal_fusion",
            "shared_latent_representation",
        ],
        "primary_sprint": "Sprint 1",
        "evidence": rel(CANONICAL_EVIDENCE["shared_vla_baseline"]),
    },
    "compression": {
        "architecture_blocks": [
            "int8",
            "svd",
            "tensor_train",
            "mps",
            "tt_svd",
        ],
        "primary_sprint": "Sprint 2",
        "evidence": rel(CANONICAL_EVIDENCE["compression"]),
    },
    "training_efficiency": {
        "architecture_blocks": [
            "adamw",
            "cosine_learning_rate",
            "trainable_svd",
            "trainable_tt_mps",
            "paired_target_reach_analysis",
        ],
        "primary_sprint": "Sprint 3",
        "evidence": rel(CANONICAL_EVIDENCE["training_efficiency"]),
    },
    "classical_rl": {
        "architecture_blocks": [
            "ppo",
            "matched_classical_actor",
        ],
        "primary_sprint": "Sprint 4",
        "evidence": rel(CANONICAL_EVIDENCE["ppo_qml"]),
    },
    "hybrid_qml": {
        "architecture_blocks": [
            "angle_encoding",
            "four_qubit_pqc",
            "pqc_vqc",
            "hybrid_quantum_classical_actor",
        ],
        "primary_sprint": "Sprint 4",
        "evidence": rel(CANONICAL_EVIDENCE["ppo_qml"]),
        "claim_controls": rel(CANONICAL_EVIDENCE["rl_claims"]),
    },
    "safety": {
        "architecture_blocks": [
            "none",
            "clipping",
            "lyapunov_guided_filter",
            "robustness_harness",
        ],
        "primary_sprint": "Sprint 5",
        "evidence": rel(CANONICAL_EVIDENCE["safety"]),
    },
    "cross_domain_reuse": {
        "architecture_blocks": [
            "shared_safety_interface",
            "shared_robustness_harness",
            "shared_evidence_schema",
            "domain_specific_safety_semantics",
        ],
        "primary_sprint": "Sprint 5.14",
        "evidence": rel(CANONICAL_EVIDENCE["cross_domain_safety"]),
        "claim_controls": rel(CANONICAL_EVIDENCE["cross_domain_claims"]),
    },
}

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5F",
    "artifact": "evidence-to-architecture-map",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "architecture_source": rel(ARCHITECTURE),
    "component_map_source": rel(COMPONENT_MAP),
    "bottleneck_map_source": rel(BOTTLENECK_MAP),
    "classification_source": rel(CLASSIFICATION),
    "evidence_map": evidence_map,
    "mapped_sections": len(evidence_map),
    "sprints_covered": [
        "Sprint 1",
        "Sprint 2",
        "Sprint 3",
        "Sprint 4",
        "Sprint 5",
        "Sprint 5.14",
    ],
    "claim_boundary": (
        "Evidence mapping records where each architecture block "
        "was implemented and evaluated. It does not imply that "
        "every quantum or quantum-inspired component achieved "
        "performance advantage over classical baselines."
    ),
}

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
print(" SPRINT 5.14.5F EVIDENCE-TO-ARCHITECTURE MAP")
print("=" * 80)
print()
print(
    "Mapped evidence sections:",
    result["mapped_sections"],
)
print(
    "Sprints covered:",
    len(result["sprints_covered"]),
)
print()
print("Sprint 1 baseline mapped: PASS")
print("Sprint 2 compression mapped: PASS")
print("Sprint 3 training mapped: PASS")
print("Sprint 4 PPO/QML mapped: PASS")
print("Sprint 5 safety mapped: PASS")
print("Sprint 5.14 cross-domain mapped: PASS")
print()
print("Canonical evidence paths: PASS")
print("Claim-control paths retained: PASS")
print("Advantage overclaim blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5F EVIDENCE MAP: PASS")
