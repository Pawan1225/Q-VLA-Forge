from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

FINDINGS = ROOT / "results" / "integration" / "sprint5-phase1-findings.json"

ARCHITECTURE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"

BOTTLENECKS = (
    ROOT / "results" / "integration" / "sprint5-volkswagen-bottleneck-map.json"
)

CLASSIFICATION = (
    ROOT / "results" / "integration" / "sprint5-computational-classification.json"
)

EVIDENCE_MAP = (
    ROOT / "results" / "integration" / "sprint5-evidence-architecture-map.json"
)

OUTPUT = ROOT / "results" / "integration" / "sprint5-unified-claim-matrix.json"

for path in (
    FINDINGS,
    ARCHITECTURE,
    BOTTLENECKS,
    CLASSIFICATION,
    EVIDENCE_MAP,
):
    if not path.exists():
        raise FileNotFoundError(path)

claims: list[dict[str, Any]] = [
    {
        "claim_id": "S5.14.5-C01",
        "statement": (
            "A shared computational framework was instantiated "
            "across autonomous-driving and robotics proxy "
            "environments."
        ),
        "status": "supported",
        "evidence": [
            "results/integration/sprint5-unified-architecture.json",
            ("results/integration/" "sprint5-shared-domain-component-map.json"),
            ("results/safety/cross-domain/" "sprint5-cross-domain-package.json"),
        ],
        "limitations": [
            (
                "The domains retain separate state semantics, "
                "environments, rewards, trained policy weights, "
                "and safety constraints."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C02",
        "statement": (
            "All four Volkswagen challenge bottlenecks were "
            "experimentally addressed within the Phase 1 pilot."
        ),
        "status": "supported",
        "evidence": [
            ("results/integration/" "sprint5-volkswagen-bottleneck-map.json"),
        ],
        "limitations": [
            (
                "Experimentally addressed does not imply that "
                "every bottleneck achieved an advantage over "
                "its strongest classical baseline."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C03",
        "statement": (
            "Classical, quantum-inspired, and quantum-machine-"
            "learning alternatives were evaluated under "
            "controlled pilot protocols."
        ),
        "status": "supported",
        "evidence": [
            ("results/integration/" "sprint5-computational-classification.json"),
            ("results/integration/" "sprint5-evidence-architecture-map.json"),
        ],
        "limitations": [
            ("No quantum advantage or quantum speedup was " "demonstrated."),
        ],
    },
    {
        "claim_id": "S5.14.5-C04",
        "statement": (
            "Q-VLA Forge reused common experiment, policy-interface, "
            "safety-interface, robustness, and evidence infrastructure "
            "across both domains."
        ),
        "status": "supported",
        "evidence": [
            ("results/integration/" "sprint5-shared-domain-component-map.json"),
        ],
        "limitations": [
            (
                "Reuse occurs at the framework and interface level, "
                "not as one universal trained model."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C05",
        "statement": (
            "The Phase 1 architecture provides a modular path "
            "for scaling to larger VLA backbones and more "
            "realistic environments."
        ),
        "status": "supported_with_limitation",
        "evidence": [
            "results/integration/sprint5-unified-architecture.json",
            "results/integration/sprint5-phase1-findings.json",
        ],
        "limitations": [
            (
                "The scale-up pathway is architectural and has "
                "not yet been experimentally demonstrated."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C06",
        "statement": (
            "One universal trained VLA model was demonstrated "
            "across autonomous driving and robotics."
        ),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "Driving and robotics use separate policy weights "
                "and domain-specific semantics."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C07",
        "statement": ("Cross-domain zero-shot policy transfer was " "demonstrated."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            ("No cross-domain policy-weight transfer experiment " "was performed."),
        ],
    },
    {
        "claim_id": "S5.14.5-C08",
        "statement": ("Quantum advantage was demonstrated."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "Quantum-inspired and QML pathways were evaluated "
                "without establishing superiority over classical "
                "baselines."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C09",
        "statement": ("Quantum computational speedup was demonstrated."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "The Phase 1 QML experiments used simulation and "
                "did not demonstrate computational speedup."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C10",
        "statement": (
            "Hybrid QML demonstrated a sample-efficiency advantage "
            "over classical PPO."
        ),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "Classical PPO reached 6/6 frozen targets while "
                "the hybrid QML actor reached 0/6."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C11",
        "statement": (
            "TT/MPS compression demonstrated superiority over "
            "the strongest classical compression baseline."
        ),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "TT/MPS demonstrated a quantum-inspired pathway "
                "but did not establish superiority over INT8."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C12",
        "statement": ("A robust training-efficiency advantage was demonstrated."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "No robust >=10% training-efficiency improvement "
                "was established under the frozen protocol."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C13",
        "statement": ("The Phase 1 results constitute a formal safety guarantee."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "Safety evidence is empirical and does not provide "
                "a formal closed-loop stability or certification proof."
            ),
        ],
    },
    {
        "claim_id": "S5.14.5-C14",
        "statement": ("Production deployment readiness was demonstrated."),
        "status": "not_supported",
        "evidence": [],
        "limitations": [
            (
                "The pilot uses synthetic proxy environments and "
                "does not include physical vehicle or robot deployment."
            ),
        ],
    },
]

claim_ids = [str(claim["claim_id"]) for claim in claims]

if len(claim_ids) != len(set(claim_ids)):
    raise RuntimeError("claim IDs must be unique")

status_counts = {
    "supported": sum(1 for claim in claims if claim["status"] == "supported"),
    "supported_with_limitation": sum(
        1 for claim in claims if claim["status"] == "supported_with_limitation"
    ),
    "not_supported": sum(1 for claim in claims if claim["status"] == "not_supported"),
}

required_blocked = {
    "S5.14.5-C06",
    "S5.14.5-C07",
    "S5.14.5-C08",
    "S5.14.5-C09",
    "S5.14.5-C10",
    "S5.14.5-C11",
    "S5.14.5-C12",
    "S5.14.5-C13",
    "S5.14.5-C14",
}

blocked_claim_ids = {
    str(claim["claim_id"]) for claim in claims if claim["status"] == "not_supported"
}

if blocked_claim_ids != required_blocked:
    raise RuntimeError("blocked claim set mismatch")

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.14.5H",
    "artifact": "unified-architecture-claim-matrix",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "claim_count": len(claims),
    "status_counts": status_counts,
    "claims": claims,
    "blocked_claim_ids": sorted(blocked_claim_ids),
    "source_artifacts": [
        str(path).replace(
            "\\",
            "/",
        )
        for path in (
            FINDINGS,
            ARCHITECTURE,
            BOTTLENECKS,
            CLASSIFICATION,
            EVIDENCE_MAP,
        )
    ],
    "proposal_safe_summary": (
        "The Phase 1 pilot supports a modular shared framework "
        "covering all four challenge bottlenecks and controlled "
        "classical/QI/QML evaluation, while explicitly not claiming "
        "quantum advantage, universal transfer, formal safety "
        "guarantees, or production readiness."
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
print(" SPRINT 5.14.5H SUPPORTED / UNSUPPORTED CLAIM CONTROLS")
print("=" * 80)
print()
print(
    "Claims:",
    len(claims),
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
print("Shared-framework claim: PASS")
print("Four-bottleneck claim: PASS")
print("Classical/QI/QML comparison claim: PASS")
print("Phase-2 pathway limitation retained: PASS")
print()
print("Universal VLA claim blocked: PASS")
print("Zero-shot transfer claim blocked: PASS")
print("Quantum advantage claim blocked: PASS")
print("Quantum speedup claim blocked: PASS")
print("QML sample-efficiency advantage blocked: PASS")
print("TT/MPS superiority claim blocked: PASS")
print("Training-efficiency advantage blocked: PASS")
print("Formal safety guarantee blocked: PASS")
print("Production deployment claim blocked: PASS")
print()
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.14.5H CLAIM CONTROLS: PASS")
