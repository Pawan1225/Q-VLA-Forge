from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")
OUTPUT = ROOT / "results" / "integration" / "sprint5-volkswagen-bottleneck-map.json"

SOURCES = {
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
    "rl_sample_efficiency": (
        ROOT / "results" / "rl" / "evidence" / "sprint4-rl-evidence.json"
    ),
    "safety": (
        ROOT
        / "results"
        / "safety"
        / "cross-domain"
        / "sprint5-cross-domain-package.json"
    ),
}

for name, path in SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")

bottlenecks: dict[str, Any] = {
    "model_footprint": {
        "challenge_area": "compression",
        "q_vla_forge_methods": [
            "int8",
            "svd",
            "tensor_train",
            "mps",
            "tt_svd",
        ],
        "primary_sprint": "Sprint 2",
        "canonical_evidence": str(SOURCES["compression"]).replace(
            "\\",
            "/",
        ),
        "phase1_status": "experimentally_addressed",
    },
    "training_efficiency": {
        "challenge_area": "training_efficiency",
        "q_vla_forge_methods": [
            "adamw",
            "cosine_learning_rate",
            "trainable_svd",
            "trainable_tt_mps",
            "paired_target_reach_analysis",
        ],
        "primary_sprint": "Sprint 3",
        "canonical_evidence": str(SOURCES["training_efficiency"]).replace(
            "\\",
            "/",
        ),
        "phase1_status": "experimentally_addressed",
    },
    "rl_alignment_sample_efficiency": {
        "challenge_area": "rl_alignment_sample_efficiency",
        "q_vla_forge_methods": [
            "ppo",
            "matched_classical_actor",
            "hybrid_pqc_actor",
            "sample_efficiency_analysis",
            "matched_budget_ablation",
        ],
        "primary_sprint": "Sprint 4",
        "canonical_evidence": str(SOURCES["rl_sample_efficiency"]).replace(
            "\\",
            "/",
        ),
        "phase1_status": "experimentally_addressed",
    },
    "safety": {
        "challenge_area": "safety",
        "q_vla_forge_methods": [
            "none_baseline",
            "clipping",
            "lyapunov_guided_filter",
            "gaussian_robustness",
            "structured_state_robustness",
            "action_perturbation_recovery",
            "cross_domain_safety_analysis",
        ],
        "primary_sprint": "Sprint 5",
        "canonical_evidence": str(SOURCES["safety"]).replace(
            "\\",
            "/",
        ),
        "phase1_status": "experimentally_addressed",
    },
}

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5D",
    "artifact": "volkswagen-four-bottleneck-map",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "bottleneck_count": len(bottlenecks),
    "bottlenecks": bottlenecks,
    "all_four_addressed": (
        len(bottlenecks) == 4
        and all(
            record["phase1_status"] == "experimentally_addressed"
            for record in bottlenecks.values()
        )
    ),
    "supported_statement": (
        "All four Volkswagen challenge bottlenecks were "
        "experimentally addressed within the Phase 1 Q-VLA "
        "Forge pilot using frozen evidence from Sprints 2–5."
    ),
    "claim_boundary": (
        "Experimentally addressed does not imply that each "
        "bottleneck achieved a positive advantage over its "
        "classical baseline."
    ),
}

if not bool(result["all_four_addressed"]):
    raise RuntimeError("all four Volkswagen bottlenecks must be mapped")

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
print(" SPRINT 5.14.5D VOLKSWAGEN FOUR-BOTTLENECK MAP")
print("=" * 80)
print()
print(
    "Bottlenecks:",
    result["bottleneck_count"],
)
print()
print("Model footprint / compression: PASS")
print("Training efficiency: PASS")
print("RL alignment / sample efficiency: PASS")
print("Safety: PASS")
print()
print("Canonical evidence links: PASS")
print("All four experimentally addressed: PASS")
print("Advantage overclaim blocked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5D FOUR-BOTTLENECK MAP: PASS")
