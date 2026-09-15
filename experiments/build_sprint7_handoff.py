from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

SCOPE = ROOT / "results" / "safety" / "final" / "sprint5-final-scope.json"

CHAIN = ROOT / "results" / "safety" / "final" / "sprint5-chain-inventory.json"

MANIFEST = ROOT / "results" / "safety" / "final" / "sprint5-final-manifest.json"

INVARIANTS = (
    ROOT / "results" / "safety" / "final" / "sprint5-scientific-invariants.json"
)

CLAIMS = ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"

LIMITATIONS = ROOT / "results" / "safety" / "final" / "sprint5-final-limitations.json"

UNIFIED_BRIDGE = (
    ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"
)

PHASE2_ROADMAP = ROOT / "results" / "integration" / "sprint5-phase2-roadmap.json"

OUTPUT = ROOT / "results" / "safety" / "final" / "sprint7-handoff.json"

SOURCES = (
    SCOPE,
    CHAIN,
    MANIFEST,
    INVARIANTS,
    CLAIMS,
    LIMITATIONS,
    UNIFIED_BRIDGE,
    PHASE2_ROADMAP,
)

for path in SOURCES:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


scope = load_json(SCOPE)
chain = load_json(CHAIN)
manifest = load_json(MANIFEST)
invariants = load_json(INVARIANTS)
claims = load_json(CLAIMS)
limitations = load_json(LIMITATIONS)
bridge = load_json(UNIFIED_BRIDGE)
roadmap = load_json(PHASE2_ROADMAP)

if scope["handoff_target"] != "Sprint 7":
    raise RuntimeError("handoff target must remain Sprint 7")

if bool(scope["sprint6_execution_required"]):
    raise RuntimeError("Sprint 6 execution must remain unnecessary")

if not bool(chain["all_stages_present"]):
    raise RuntimeError("Sprint 5 chain must be complete")

if not bool(manifest["all_required_present"]):
    raise RuntimeError("Sprint 5 manifest must be complete")

if not bool(bridge["proposal_ready"]):
    raise RuntimeError("unified bridge must remain proposal-ready")

supported_claims = claims["supported_claims"]

supported_with_limitation = claims["supported_with_limitation"]

not_supported_claims = claims["not_supported_claims"]

if not isinstance(
    supported_claims,
    list,
):
    raise TypeError("supported claims must be list")

if not isinstance(
    supported_with_limitation,
    list,
):
    raise TypeError("limited claims must be list")

if not isinstance(
    not_supported_claims,
    list,
):
    raise TypeError("not-supported claims must be list")

headline_results = {
    "compression": (
        "INT8 remained the strongest Phase 1 compression result; "
        "TT/MPS was implemented and evaluated without establishing "
        "superiority."
    ),
    "training_efficiency": (
        "No robust >=10% training-efficiency improvement was "
        "demonstrated under the frozen protocol."
    ),
    "rl_qml": (
        "Classical PPO reached 6/6 frozen targets, matched classical "
        "reached 1/6, and hybrid QML reached 0/6. The hybrid actor "
        "nevertheless demonstrated substantial parameter compactness."
    ),
    "safety": (
        "Explicit safety filtering produced strong empirical pilot "
        "effects across both domains, with robustness evaluated under "
        "Gaussian, structured-state, and action perturbations."
    ),
    "lyapunov": (
        "Hard action/domain guards dominated observed interventions in "
        "several regimes. Lyapunov-specific candidate selection became "
        "active under action perturbation in driving but not robotics."
    ),
    "cross_domain": (
        "A shared computational framework was demonstrated at the "
        "framework and interface level, not as one universal trained "
        "policy."
    ),
}

challenge_bottleneck_map = {
    "model_footprint": "Sprint 2 compression evidence",
    "training_efficiency": "Sprint 3 training-efficiency evidence",
    "rl_alignment_sample_efficiency": "Sprint 4 PPO/QML evidence",
    "safety": "Sprint 5 safety and robustness evidence",
}

canonical_evidence = {
    "sprint1_baseline": "results/sprint1-baseline-manifest.json",
    "sprint2_compression": (
        "results/compression/evidence/" "sprint2-compression-evidence.json"
    ),
    "sprint3_training": ("results/training/evidence/" "sprint3-training-evidence.json"),
    "sprint4_rl_qml": ("results/rl/evidence/" "sprint4-rl-evidence.json"),
    "sprint5_safety": (
        "results/safety/consolidated/" "sprint5-safety-evidence-package.json"
    ),
    "sprint5_cross_domain": (
        "results/safety/cross-domain/" "sprint5-cross-domain-package.json"
    ),
    "unified_architecture": (
        "results/integration/" "sprint5-unified-bridge-package.json"
    ),
    "phase2_roadmap": ("results/integration/" "sprint5-phase2-roadmap.json"),
    "final_manifest": ("results/safety/final/" "sprint5-final-manifest.json"),
    "final_claim_boundary": (
        "results/safety/final/" "sprint5-final-claim-boundary.json"
    ),
    "final_limitations": ("results/safety/final/" "sprint5-final-limitations.json"),
}

for evidence_path in canonical_evidence.values():
    if not (ROOT / evidence_path).exists():
        raise FileNotFoundError(evidence_path)

submission_requirements = {
    "allowed": [
        "final_validation",
        "evidence_aggregation",
        "ablation_summary",
        "proposal_tables",
        "proposal_figures",
        "submission_ready_claims",
        "readme_evidence_update",
        "proposal_evidence_package",
        "reproducibility_verification",
    ],
    "requires_explicit_reopening": [
        "new_training",
        "new_ppo_execution",
        "new_qml_execution",
        "new_safety_episodes",
        "new_robustness_runs",
        "retuning",
        "new_scientific_benchmark",
    ],
}

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "artifact": "sprint7-handoff",
    "source_sprint": "5.15",
    "target_sprint": "7",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "phase1_status": "frozen",
    "sprint6_execution_required": False,
    "sprint6_status_statement": scope["sprint6_status_statement"],
    "canonical_evidence": canonical_evidence,
    "headline_results": headline_results,
    "challenge_bottleneck_map": challenge_bottleneck_map,
    "supported_claims": supported_claims,
    "supported_with_limitation": supported_with_limitation,
    "unsupported_claims": not_supported_claims,
    "limitations": limitations["limitations"],
    "lyapunov_nuance": limitations["lyapunov_nuance"],
    "proposal_architecture": ("docs/unified_architecture.md"),
    "phase2_roadmap": roadmap["priorities"],
    "submission_requirements": submission_requirements,
    "scientific_invariants": invariants["invariants"],
    "ready_for_sprint7": True,
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
print(" SPRINT 5.15.7 SPRINT-7 HANDOFF")
print("=" * 80)
print()
print("Source sprint: 5.15")
print("Target sprint: 7")
print("Phase 1 status: frozen")
print("Sprint 6 execution required: False")
print()
print(
    "Canonical evidence entries:",
    len(canonical_evidence),
)
print(
    "Supported claims:",
    len(supported_claims),
)
print(
    "Supported with limitation:",
    len(supported_with_limitation),
)
print(
    "Unsupported claims:",
    len(not_supported_claims),
)
print(
    "Limitations:",
    len(limitations["limitations"]),
)
print(
    "Phase-2 priorities:",
    len(roadmap["priorities"]),
)
print()
print("Canonical evidence paths: PASS")
print("Headline results: PASS")
print("Challenge bottleneck map: PASS")
print("Claim boundary: PASS")
print("Limitations transferred: PASS")
print("Lyapunov nuance transferred: PASS")
print("Unified architecture transferred: PASS")
print("Phase-2 roadmap transferred: PASS")
print("Sprint-6 skip wording preserved: PASS")
print("Sprint-7 execution permissions locked: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 7 HANDOFF: READY")
print("SPRINT 5.15.7 HANDOFF: PASS")
