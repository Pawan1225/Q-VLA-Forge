from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

INVARIANTS = (
    ROOT / "results" / "safety" / "final" / "sprint5-scientific-invariants.json"
)

CLAIMS = ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"

LYAPUNOV_MECHANISM = (
    ROOT
    / "results"
    / "safety"
    / "cross-domain"
    / "sprint5-cross-domain-lyapunov-mechanism.json"
)

OUTPUT = ROOT / "results" / "safety" / "final" / "sprint5-final-limitations.json"

for path in (
    INVARIANTS,
    CLAIMS,
    LYAPUNOV_MECHANISM,
):
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


invariants = load_json(INVARIANTS)
claims = load_json(CLAIMS)
mechanism = load_json(LYAPUNOV_MECHANISM)

scientific_invariants = invariants["invariants"]

if not isinstance(
    scientific_invariants,
    dict,
):
    raise TypeError("scientific invariants must be dict")

safety_invariants = scientific_invariants["safety"]

if not isinstance(
    safety_invariants,
    dict,
):
    raise TypeError("safety invariants must be dict")

lyapunov_activation = safety_invariants["lyapunov_specific_activation"]

if not isinstance(
    lyapunov_activation,
    dict,
):
    raise TypeError("Lyapunov activation must be dict")

limitations = [
    {
        "id": "L01",
        "category": "environment",
        "statement": "Phase 1 used synthetic proxy environments.",
    },
    {
        "id": "L02",
        "category": "model_scale",
        "statement": (
            "The shared Phase 1 VLA baseline is compact and does not "
            "represent a production-scale pretrained VLA backbone."
        ),
    },
    {
        "id": "L03",
        "category": "policy_integration",
        "statement": (
            "Sprint 4 RL policies consume compact environment state "
            "rather than the Sprint 1 shared VLA latent representation."
        ),
    },
    {
        "id": "L04",
        "category": "statistics",
        "statement": ("The principal trained-policy protocol uses three seeds."),
    },
    {
        "id": "L05",
        "category": "evaluation_size",
        "statement": (
            "Safety evaluations use 20 held-out episodes per "
            "seed/condition in the principal protocol."
        ),
    },
    {
        "id": "L06",
        "category": "qml_scale",
        "statement": (
            "The Phase 1 hybrid QML pathway uses a small simulated "
            "4-qubit, 2-layer PQC."
        ),
    },
    {
        "id": "L07",
        "category": "quantum_hardware",
        "statement": "No quantum hardware execution was performed.",
    },
    {
        "id": "L08",
        "category": "tensor_network_scale",
        "statement": ("TT/MPS compression was evaluated at pilot-model scale."),
    },
    {
        "id": "L09",
        "category": "driving_validation",
        "statement": (
            "No principal CARLA, nuScenes, or physical-vehicle "
            "evaluation was performed."
        ),
    },
    {
        "id": "L10",
        "category": "robotics_validation",
        "statement": (
            "No principal RLBench, LIBERO, Open X-Embodiment, or "
            "physical-robot evaluation was performed."
        ),
    },
    {
        "id": "L11",
        "category": "state_estimation",
        "statement": (
            "During perception-perturbation experiments, the policy "
            "used perturbed/noisy observations while the safety layer "
            "retained access to true simulator state."
        ),
    },
    {
        "id": "L12",
        "category": "action_interface",
        "statement": (
            "Environment action-bound handling is accounted for "
            "separately from explicit safety-filter intervention."
        ),
    },
    {
        "id": "L13",
        "category": "lyapunov",
        "statement": (
            "The implemented Lyapunov quantity is an empirical safety "
            "potential and does not constitute a formal closed-loop "
            "stability proof."
        ),
    },
    {
        "id": "L14",
        "category": "certification",
        "statement": (
            "No functional-safety certification or standards " "compliance is claimed."
        ),
    },
    {
        "id": "L15",
        "category": "deployment",
        "statement": (
            "No production deployment or full-scale embedded VLA "
            "validation was performed."
        ),
    },
]

lyapunov_nuance: dict[str, Any] = {
    "complete_filter_is_not_equivalent_to_lyapunov_decrease_only": True,
    "hard_guards_material_to_observed_interventions": True,
    "clean_regime_lyapunov_specific_activation": False,
    "gaussian_regime_lyapunov_specific_activation": False,
    "structured_state_regime_lyapunov_specific_activation": False,
    "action_regime_lyapunov_specific_activation": True,
    "driving_action_activation": bool(lyapunov_activation["autonomous_driving"]),
    "robotics_action_activation": bool(lyapunov_activation["robotics"]),
    "formal_stability_claim": False,
    "canonical_statement": (
        "Hard action/domain guards accounted for the observed "
        "interventions in clean, Gaussian, and structured-state "
        "regimes. Lyapunov-specific candidate selection became "
        "empirically active under action perturbation, with activation "
        "observed in autonomous driving but not robotics."
    ),
}

if not bool(lyapunov_nuance["driving_action_activation"]):
    raise RuntimeError("driving Lyapunov activation invariant changed")

if bool(lyapunov_nuance["robotics_action_activation"]):
    raise RuntimeError("robotics Lyapunov activation invariant changed")

if bool(lyapunov_nuance["formal_stability_claim"]):
    raise RuntimeError("formal Lyapunov stability must remain unsupported")

not_supported_count = int(claims["status_counts"]["not_supported"])

if not_supported_count != 13:
    raise RuntimeError("final claim-boundary partition changed")

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.15.6",
    "artifact": "sprint5-final-limitations",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_safety_runs": False,
    "new_robustness_runs": False,
    "limitation_count": len(limitations),
    "limitations": limitations,
    "lyapunov_nuance": lyapunov_nuance,
    "source_artifacts": [
        str(INVARIANTS).replace("\\", "/"),
        str(CLAIMS).replace("\\", "/"),
        str(LYAPUNOV_MECHANISM).replace("\\", "/"),
    ],
    "usage_rule": (
        "Sprint 7 must preserve these limitations and the Lyapunov "
        "mechanism nuance whenever related proposal claims are used."
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
print(" SPRINT 5.15.6 LYAPUNOV NUANCE + LIMITATIONS FREEZE")
print("=" * 80)
print()
print(
    "Limitations:",
    result["limitation_count"],
)
print()
print("Synthetic-environment limitation: PASS")
print("Compact-VLA limitation: PASS")
print("Shared-latent RL gap preserved: PASS")
print("Three-seed limitation: PASS")
print("Evaluation-size limitation: PASS")
print("Small-PQC limitation: PASS")
print("No quantum-hardware limitation: PASS")
print("Pilot-scale TT/MPS limitation: PASS")
print("No real driving validation: PASS")
print("No physical robotics validation: PASS")
print("Privileged safety-state limitation: PASS")
print("Environment action handling boundary: PASS")
print("No formal Lyapunov proof: PASS")
print("No certification claim: PASS")
print("No production deployment claim: PASS")
print()
print("Hard-guard contribution preserved: PASS")
print("Action-regime Lyapunov activation preserved: PASS")
print("Driving/robotics mechanism asymmetry preserved: PASS")
print("Formal stability claim blocked: PASS")
print()
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.15.6 LIMITATIONS FREEZE: PASS")
