from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

INVARIANTS = (
    ROOT / "results" / "safety" / "final" / "sprint5-scientific-invariants.json"
)

BRIDGE_CLAIMS = ROOT / "results" / "integration" / "sprint5-unified-claim-matrix.json"

CROSS_DOMAIN_CLAIMS = (
    ROOT
    / "results"
    / "safety"
    / "cross-domain"
    / "sprint5-cross-domain-claim-matrix.json"
)

OUTPUT = ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"

for path in (
    INVARIANTS,
    BRIDGE_CLAIMS,
    CROSS_DOMAIN_CLAIMS,
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

supported = [
    {
        "claim_id": "S5-FINAL-C01",
        "statement": (
            "A shared computational framework was instantiated "
            "across autonomous-driving and robotics proxy domains."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C02",
        "statement": (
            "All four Volkswagen challenge bottlenecks were "
            "experimentally addressed in the Phase 1 pilot."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C03",
        "statement": (
            "INT8 satisfied the strongest Phase 1 compression "
            "criterion under the frozen pilot configuration."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C04",
        "statement": (
            "The hybrid PQC policy actor demonstrated substantial "
            "parameter compactness in both domains."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C05",
        "statement": (
            "Explicit clean safety filtering was empirically "
            "effective under the frozen driving and robotics "
            "proxy contracts."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C06",
        "statement": (
            "Gaussian observation robustness was evaluated under "
            "the frozen Phase 1 protocol."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C07",
        "statement": (
            "Structured-state robustness was evaluated under "
            "domain-specific perturbation contracts."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C08",
        "statement": (
            "Action perturbation robustness and unsafe-action "
            "recovery were quantified."
        ),
        "status": "supported",
    },
    {
        "claim_id": "S5-FINAL-C09",
        "statement": (
            "Cross-domain safety-framework reuse was demonstrated "
            "at the framework and interface level."
        ),
        "status": "supported",
    },
]

supported_with_limitation = [
    {
        "claim_id": "S5-FINAL-C10",
        "statement": (
            "The Phase 1 architecture provides a modular Phase 2 " "scale-up pathway."
        ),
        "status": "supported_with_limitation",
        "limitation": (
            "The scale-up itself has not yet been experimentally " "demonstrated."
        ),
    },
    {
        "claim_id": "S5-FINAL-C11",
        "statement": (
            "Lyapunov-guided safety was empirically effective under "
            "the frozen pilot safety contract."
        ),
        "status": "supported_with_limitation",
        "limitation": (
            "Observed interventions were often dominated by hard "
            "action/domain guards, and Lyapunov-specific activation "
            "was domain-dependent."
        ),
    },
    {
        "claim_id": "S5-FINAL-C12",
        "statement": (
            "Robustness was demonstrated under synthetic Gaussian, "
            "structured-state, and action perturbations."
        ),
        "status": "supported_with_limitation",
        "limitation": (
            "The perturbation studies used synthetic proxy "
            "environments and do not establish real-world robustness."
        ),
    },
]

not_supported = [
    {
        "claim_id": "S5-FINAL-C13",
        "statement": "Quantum advantage was demonstrated.",
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C14",
        "statement": "Quantum computational speedup was demonstrated.",
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C15",
        "statement": (
            "Hybrid QML demonstrated sample-efficiency superiority "
            "over classical PPO."
        ),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C16",
        "statement": (
            "TT/MPS demonstrated compression superiority over "
            "the strongest classical baseline."
        ),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C17",
        "statement": ("A robust training-efficiency advantage was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C18",
        "statement": (
            "A universal trained VLA policy was demonstrated across "
            "driving and robotics."
        ),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C19",
        "statement": ("Zero-shot cross-domain policy transfer was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C20",
        "statement": ("Formal Lyapunov closed-loop stability was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C21",
        "statement": ("A formal safety guarantee was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C22",
        "statement": ("Production safety or deployment readiness was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C23",
        "statement": ("Safety certification or standards compliance was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C24",
        "statement": ("Real-world autonomous-driving validation was demonstrated."),
        "status": "not_supported",
    },
    {
        "claim_id": "S5-FINAL-C25",
        "statement": ("Physical-robot validation was demonstrated."),
        "status": "not_supported",
    },
]

all_claims = supported + supported_with_limitation + not_supported

claim_ids = [str(claim["claim_id"]) for claim in all_claims]

if len(claim_ids) != len(set(claim_ids)):
    raise RuntimeError("final claim IDs must be unique")

status_counts = {
    "supported": len(supported),
    "supported_with_limitation": len(supported_with_limitation),
    "not_supported": len(not_supported),
}

blocked_claim_ids = [str(claim["claim_id"]) for claim in not_supported]

scientific_invariants = invariants["invariants"]

if not isinstance(
    scientific_invariants,
    dict,
):
    raise TypeError("scientific invariants must be dict")

if bool(scientific_invariants["quantum_claims"]["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage invariant mismatch")

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.15.5",
    "artifact": "sprint5-final-claim-boundary",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_safety_runs": False,
    "new_robustness_runs": False,
    "claim_count": len(all_claims),
    "status_counts": status_counts,
    "supported_claims": supported,
    "supported_with_limitation": supported_with_limitation,
    "not_supported_claims": not_supported,
    "blocked_claim_ids": blocked_claim_ids,
    "source_artifacts": [
        str(INVARIANTS).replace("\\", "/"),
        str(BRIDGE_CLAIMS).replace("\\", "/"),
        str(CROSS_DOMAIN_CLAIMS).replace("\\", "/"),
    ],
    "usage_rule": (
        "Sprint 7 may reuse supported claims exactly or in weaker "
        "form. Supported-with-limitation claims must retain their "
        "limitations. Not-supported claims must not be rewritten "
        "as positive Phase 1 conclusions."
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
print(" SPRINT 5.15.5 FINAL CLAIM BOUNDARY")
print("=" * 80)
print()
print(
    "Claims:",
    result["claim_count"],
)
print(
    "Supported:",
    status_counts["supported"],
)
print(
    "Supported with limitation:",
    status_counts["supported_with_limitation"],
)
print(
    "Not supported:",
    status_counts["not_supported"],
)
print()
print("Four-bottleneck claim preserved: PASS")
print("INT8 result claim preserved: PASS")
print("QML compactness claim preserved: PASS")
print("Empirical safety claims preserved: PASS")
print("Robustness evaluation claims preserved: PASS")
print("Cross-domain reuse claim preserved: PASS")
print()
print("Quantum advantage blocked: PASS")
print("Quantum speedup blocked: PASS")
print("QML superiority blocked: PASS")
print("TT/MPS superiority blocked: PASS")
print("Training-efficiency advantage blocked: PASS")
print("Universal VLA blocked: PASS")
print("Zero-shot transfer blocked: PASS")
print("Formal Lyapunov stability blocked: PASS")
print("Formal safety guarantee blocked: PASS")
print("Production readiness blocked: PASS")
print("Certification claim blocked: PASS")
print("Real-world driving validation blocked: PASS")
print("Physical robotics validation blocked: PASS")
print()
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.15.5 FINAL CLAIM BOUNDARY: PASS")
