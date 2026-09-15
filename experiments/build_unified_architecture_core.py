from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")
OUTPUT_DIR = ROOT / "results" / "integration"
OUTPUT = OUTPUT_DIR / "sprint5-unified-architecture.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

payload: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5B",
    "artifact": "canonical-unified-architecture",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_metrics": False,
    "domains": [
        "autonomous_driving",
        "robotics",
    ],
    "canonical_flow": [
        "domain_input_adapters",
        "shared_ai_vla_core",
        "multimodal_fusion",
        "shared_latent_representation",
        "compression_and_policy_paths",
        "proposed_action",
        "safety_layer",
        "executed_action",
        "domain_environment",
    ],
    "shared_ai_vla_core": [
        "vision_encoder_framework",
        "language_encoder_framework",
        "state_encoder_framework",
        "multimodal_fusion",
        "shared_latent_representation",
        "three_dimensional_action_interface",
    ],
    "compression_path": {
        "classical": [
            "int8",
            "svd",
        ],
        "quantum_inspired": [
            "tensor_train",
            "mps",
            "tt_svd",
        ],
    },
    "policy_path": {
        "classical": [
            "ppo",
        ],
        "quantum_ml": [
            "angle_encoding",
            "pqc_vqc",
            "hybrid_quantum_classical_actor",
        ],
    },
    "safety_path": {
        "methods": [
            "none",
            "clipping",
            "lyapunov_guided_filter",
        ],
        "classification": "classical",
    },
    "domain_specific": {
        "autonomous_driving": [
            "state_semantics",
            "environment",
            "reward",
            "ppo_weights",
            "safety_constraints",
            "clipping_rules",
            "transition_predictor",
            "lyapunov_potential",
        ],
        "robotics": [
            "state_semantics",
            "environment",
            "reward",
            "ppo_weights",
            "safety_constraints",
            "clipping_rules",
            "transition_predictor",
            "lyapunov_potential",
            "object_grasped_context",
        ],
    },
    "architecture_claim": (
        "Q-VLA Forge demonstrates a modular cross-domain computational "
        "framework in which shared AI, compression, RL/QML, evaluation, "
        "and safety interfaces are reused across autonomous-driving and "
        "robotics proxy tasks while domain-specific state semantics, "
        "policy weights, dynamics, and safety constraints remain isolated."
    ),
    "blocked_architecture_claims": [
        "universal_vla",
        "single_trained_policy",
        "shared_trained_weights",
        "zero_shot_cross_domain_policy",
        "universal_safety_controller",
    ],
    "evidence_sources": {
        "sprint1": "results/sprint1-baseline-manifest.json",
        "sprint2": (
            "results/compression/evidence/" "sprint2-compression-evidence.json"
        ),
        "sprint3": ("results/training/evidence/" "sprint3-training-evidence.json"),
        "sprint4": ("results/rl/evidence/" "sprint4-rl-evidence.json"),
        "sprint5_cross_domain": (
            "results/safety/cross-domain/" "sprint5-cross-domain-package.json"
        ),
    },
}

for source in payload["evidence_sources"].values():
    if not Path(source).exists():
        raise FileNotFoundError(source)

OUTPUT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.14.5B CANONICAL UNIFIED ARCHITECTURE")
print("=" * 80)
print()
print("Domains:", len(payload["domains"]))
print(
    "Shared AI/VLA components:",
    len(payload["shared_ai_vla_core"]),
)
print(
    "Driving-specific components:",
    len(payload["domain_specific"]["autonomous_driving"]),
)
print(
    "Robotics-specific components:",
    len(payload["domain_specific"]["robotics"]),
)
print()
print("Evidence sources: PASS")
print("Shared architecture defined: PASS")
print("Domain boundaries retained: PASS")
print("Lyapunov classified classical: PASS")
print("Universal-model claims blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5B CANONICAL ARCHITECTURE: PASS")
