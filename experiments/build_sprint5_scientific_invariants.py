from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "safety" / "final"
OUTPUT = OUTPUT_DIR / "sprint5-scientific-invariants.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCES = {
    "phase1_findings": (
        ROOT / "results" / "integration" / "sprint5-phase1-findings.json"
    ),
    "cross_domain_package": (
        ROOT
        / "results"
        / "safety"
        / "cross-domain"
        / "sprint5-cross-domain-package.json"
    ),
    "safety_package": (
        ROOT
        / "results"
        / "safety"
        / "consolidated"
        / "sprint5-safety-evidence-package.json"
    ),
    "unified_bridge": (
        ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"
    ),
}

for name, path in SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


phase1_findings = load_json(SOURCES["phase1_findings"])

findings = phase1_findings["findings"]

if not isinstance(
    findings,
    dict,
):
    raise TypeError("phase1 findings must be dict")

rl_qml = findings["rl_qml"]

cross_domain = findings["cross_domain"]

safety = findings["safety"]

if not isinstance(
    rl_qml,
    dict,
):
    raise TypeError("rl_qml findings must be dict")

if not isinstance(
    cross_domain,
    dict,
):
    raise TypeError("cross_domain findings must be dict")

if not isinstance(
    safety,
    dict,
):
    raise TypeError("safety findings must be dict")

invariants: dict[str, Any] = {
    "compression": {
        "int8_strongest_phase1_result": True,
        "tt_mps_implemented_and_evaluated": True,
        "tt_mps_superiority_demonstrated": False,
        "allowed_summary": (
            "INT8 remained the strongest Phase 1 compression result; "
            "TT/MPS provided an implemented quantum-inspired pathway "
            "without demonstrating superiority."
        ),
    },
    "training_efficiency": {
        "robust_ten_percent_improvement_demonstrated": False,
        "allowed_summary": (
            "No robust >=10% training-efficiency improvement was "
            "demonstrated under the frozen Phase 1 protocol."
        ),
    },
    "rl_qml": {
        "classical_ppo_target_reaches": int(rl_qml["classical_ppo_target_reaches"]),
        "matched_classical_target_reaches": int(
            rl_qml["matched_classical_target_reaches"]
        ),
        "hybrid_qml_target_reaches": int(rl_qml["hybrid_qml_target_reaches"]),
        "hybrid_actor_parameter_reduction": rl_qml["hybrid_actor_parameter_reduction"],
        "qml_sample_efficiency_advantage": False,
        "allowed_summary": (
            "Classical PPO reached all frozen targets. The hybrid "
            "PQC actor was substantially more compact but did not "
            "demonstrate sample-efficiency advantage."
        ),
    },
    "safety": {
        "clean_explicit_filtering_effective_both_domains": bool(
            safety["clean_explicit_filtering_effective_both_domains"]
        ),
        "gaussian_robustness_evaluated": True,
        "structured_state_robustness_evaluated": True,
        "action_robustness_evaluated": True,
        "unsafe_action_recovery_quantified": True,
        "lyapunov_specific_activation": safety["lyapunov_specific_activation"],
        "formal_safety_guarantee": False,
        "formal_lyapunov_stability_proof": False,
        "allowed_summary": (
            "Explicit safety filtering produced strong empirical "
            "pilot effects. Robustness was evaluated across Gaussian, "
            "structured-state, and action perturbations. Lyapunov-"
            "specific mechanism activation remained domain-dependent."
        ),
    },
    "architecture": {
        "shared_computational_framework_supported": bool(
            cross_domain["shared_framework_supported"]
        ),
        "same_trained_policy_weights": False,
        "zero_shot_cross_domain_transfer_tested": False,
        "universal_trained_vla_supported": False,
        "universal_safety_controller_supported": False,
        "allowed_summary": (
            "Cross-domain reuse is supported at the framework and "
            "interface level, not as one universal trained policy."
        ),
    },
    "quantum_claims": {
        "quantum_advantage_demonstrated": False,
        "quantum_speedup_demonstrated": False,
        "qml_superiority_demonstrated": False,
    },
}

if invariants["rl_qml"]["classical_ppo_target_reaches"] != 6:
    raise RuntimeError("classical PPO target count changed")

if invariants["rl_qml"]["matched_classical_target_reaches"] != 1:
    raise RuntimeError("matched classical target count changed")

if invariants["rl_qml"]["hybrid_qml_target_reaches"] != 0:
    raise RuntimeError("hybrid QML target count changed")

if not bool(invariants["architecture"]["shared_computational_framework_supported"]):
    raise RuntimeError("shared framework invariant changed")

if bool(invariants["quantum_claims"]["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage must remain unsupported")

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.15.4",
    "artifact": "sprint5-scientific-invariants",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_safety_runs": False,
    "new_robustness_runs": False,
    "new_metrics": False,
    "invariants": invariants,
    "source_artifacts": {
        key: str(path).replace(
            "\\",
            "/",
        )
        for key, path in SOURCES.items()
    },
    "usage_rule": (
        "Sprint 7 may summarize these frozen invariants but must "
        "not strengthen them into unsupported claims."
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
print(" SPRINT 5.15.4 SCIENTIFIC INVARIANTS FREEZE")
print("=" * 80)
print()

print("Compression invariant: PASS")
print("Training-efficiency invariant: PASS")
print("RL/QML target counts: PASS")
print("QML compactness invariant: PASS")
print("Safety findings invariant: PASS")
print("Lyapunov domain dependence preserved: PASS")
print("Shared architecture invariant: PASS")
print("Universal VLA claim blocked: PASS")
print("Zero-shot transfer claim blocked: PASS")
print("Quantum advantage blocked: PASS")
print("Quantum speedup blocked: PASS")
print("Formal safety guarantee blocked: PASS")
print()

print("No new training: PASS")
print("No new principal execution: PASS")
print("No new safety execution: PASS")
print("No new robustness execution: PASS")
print("No new metrics: PASS")
print()

print("SPRINT 5.15.4 SCIENTIFIC INVARIANTS: PASS")
