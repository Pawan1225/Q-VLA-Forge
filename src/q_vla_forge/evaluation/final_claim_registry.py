"""Sprint 7.11 — final Phase 1 claim registry freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HANDOFF = "results/safety/final/" "sprint7-handoff.json"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_claim_registry(
    root: Path,
) -> dict[str, Any]:
    """Freeze final supported and blocked Phase 1 claims."""

    handoff_path = root / HANDOFF

    if not handoff_path.exists():
        raise FileNotFoundError(handoff_path)

    handoff = load_json(handoff_path)

    compression = load_json(
        root / "results/final-validation/compression/" "final-compression-summary.json"
    )

    training = load_json(
        root / "results/final-validation/training-efficiency/"
        "final-training-efficiency-summary.json"
    )

    rl_qml = load_json(
        root / "results/final-validation/rl-qml/" "final-rl-qml-summary.json"
    )

    safety = load_json(
        root / "results/final-validation/safety-robustness/"
        "final-safety-robustness-summary.json"
    )

    cross_domain = load_json(
        root / "results/final-validation/cross-domain/"
        "final-cross-domain-summary.json"
    )

    claims = [
        {
            "claim_id": "S7-C01",
            "area": "architecture",
            "status": "supported",
            "statement": (
                "A shared computational framework was "
                "instantiated across autonomous-driving "
                "and robotics proxy domains."
            ),
            "evidence": cross_domain["source"],
        },
        {
            "claim_id": "S7-C02",
            "area": "compression",
            "status": "supported",
            "statement": (
                "INT8 was the only evaluated compression "
                "method on the Pareto frontier in both "
                "Phase 1 proxy domains under the frozen "
                "compression criterion."
            ),
            "evidence": compression["source"],
        },
        {
            "claim_id": "S7-C03",
            "area": "training_efficiency",
            "status": "supported",
            "statement": (
                "Phase 1 did not demonstrate a robust "
                "10% or greater optimizer-step efficiency "
                "improvement across all three locked seeds."
            ),
            "evidence": training["source"],
        },
        {
            "claim_id": "S7-C04",
            "area": "rl",
            "status": "supported",
            "statement": (
                "Classical PPO reached all six frozen "
                "domain-seed targets in the final "
                "matched comparison."
            ),
            "evidence": rl_qml["source"],
        },
        {
            "claim_id": "S7-C05",
            "area": "qml",
            "status": "supported_with_limitation",
            "statement": (
                "The evaluated PQC/QML actor was "
                "substantially more parameter-compact "
                "than the full PPO actor."
            ),
            "limitation": (
                "No QML sample-efficiency, computational, "
                "quantum-speedup, or quantum-hardware "
                "advantage was demonstrated."
            ),
            "evidence": rl_qml["source"],
        },
        {
            "claim_id": "S7-C06",
            "area": "safety",
            "status": "supported_with_limitation",
            "statement": (
                "Explicit safety filtering reduced the "
                "observed clean violation-step rate to "
                "zero in both evaluated proxy domains."
            ),
            "limitation": (
                "This is empirical pilot evidence and "
                "does not establish a formal safety "
                "guarantee."
            ),
            "evidence": safety["source"],
        },
        {
            "claim_id": "S7-C07",
            "area": "robustness",
            "status": "supported",
            "statement": (
                "Gaussian observation, structured-state, "
                "and direct action perturbation robustness "
                "were evaluated under the frozen Phase 1 "
                "protocol."
            ),
            "evidence": safety["source"],
        },
        {
            "claim_id": "S7-C08",
            "area": "robustness",
            "status": "supported",
            "statement": (
                "Explicit filtering recovered a measurable "
                "fraction of unsafe directly perturbed "
                "actions under the evaluated synthetic "
                "action-perturbation protocol."
            ),
            "evidence": safety["source"],
        },
        {
            "claim_id": "S7-C09",
            "area": "cross_domain",
            "status": "supported_with_limitation",
            "statement": (
                "Cross-domain reuse was demonstrated at "
                "the framework, interface, evaluation, "
                "and protocol levels."
            ),
            "limitation": (
                "The Phase 1 evidence does not demonstrate "
                "one universal trained policy, zero-shot "
                "policy transfer, or one universal safety "
                "controller."
            ),
            "evidence": cross_domain["source"],
        },
    ]

    blocked_claims = [
        {
            "claim_id": "S7-B01",
            "statement": ("Quantum advantage was demonstrated."),
            "reason": (
                "No quantum advantage evidence exists " "in the frozen Phase 1 results."
            ),
        },
        {
            "claim_id": "S7-B02",
            "statement": ("QML demonstrated superior sample efficiency."),
            "reason": ("QML reached 0/6 frozen targets in the " "matched comparison."),
        },
        {
            "claim_id": "S7-B03",
            "statement": ("TT/MPS demonstrated superior compression."),
            "reason": (
                "Evaluated TT/MPS configurations did not "
                "satisfy the frozen joint compression "
                "criterion."
            ),
        },
        {
            "claim_id": "S7-B04",
            "statement": (
                "The Lyapunov implementation provides a "
                "formal closed-loop stability guarantee."
            ),
            "reason": (
                "The implemented quantity is an empirical "
                "safety potential; no formal proof was "
                "established."
            ),
        },
        {
            "claim_id": "S7-B05",
            "statement": ("Q-VLA Forge is production ready."),
            "reason": (
                "Phase 1 used compact synthetic proxy "
                "environments and no production deployment."
            ),
        },
        {
            "claim_id": "S7-B06",
            "statement": (
                "One trained policy transfers zero-shot "
                "between driving and robotics."
            ),
            "reason": (
                "Separate domain policy weights were used "
                "and zero-shot transfer was not tested."
            ),
        },
        {
            "claim_id": "S7-B07",
            "statement": ("Formal functional-safety certification " "was achieved."),
            "reason": (
                "No certification or standards-compliance " "claim is supported."
            ),
        },
    ]

    supported = sum(claim["status"] == "supported" for claim in claims)

    supported_with_limitation = sum(
        claim["status"] == "supported_with_limitation" for claim in claims
    )

    return {
        "sprint": "7.11",
        "protocol": "final_claim_registry_freeze",
        "status": "FROZEN",
        "phase1_status": handoff["phase1_status"],
        "claims": claims,
        "blocked_claims": blocked_claims,
        "summary": {
            "supported": supported,
            "supported_with_limitation": (supported_with_limitation),
            "blocked": len(blocked_claims),
        },
        "scientific_invariants": handoff.get(
            "scientific_invariants",
            {},
        ),
        "global_limitations": handoff.get(
            "limitations",
            [],
        ),
        "new_training": False,
        "new_experiments": False,
        "frozen": True,
    }
