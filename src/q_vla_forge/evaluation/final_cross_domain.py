"""Sprint 7.8 — final frozen cross-domain comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CROSS_DOMAIN_PACKAGE = (
    "results/safety/cross-domain/" "sprint5-cross-domain-package.json"
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_cross_domain(
    root: Path,
) -> dict[str, Any]:
    """Build Sprint 7.8 cross-domain evidence classification."""

    source_path = root / CROSS_DOMAIN_PACKAGE

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source = load_json(source_path)

    evidence = source["evidence_summary"]

    classifications = {
        "framework_reuse": {
            "classification": "Cross-domain supported",
            "evidence": source["architecture_reuse"]["supported_statement"],
        },
        "clean_safety_direction": {
            "classification": "Cross-domain supported",
            "evidence": evidence["clean"],
        },
        "structured_state_direction": {
            "classification": "Cross-domain supported",
            "evidence": evidence["structured_state"],
        },
        "action_recovery_direction": {
            "classification": "Cross-domain supported",
            "evidence": evidence["action_recovery"],
        },
        "gaussian_robustness_direction": {
            "classification": "Domain-dependent",
            "evidence": evidence["gaussian"],
        },
        "lyapunov_specific_activation": {
            "classification": "Domain-dependent",
            "evidence": evidence["lyapunov_activation"],
        },
        "same_trained_policy_weights": {
            "classification": "Not demonstrated",
            "evidence": False,
        },
        "zero_shot_cross_domain_transfer": {
            "classification": "Not demonstrated",
            "evidence": False,
        },
        "universal_safety_controller": {
            "classification": "Not demonstrated",
            "evidence": False,
        },
    }

    return {
        "sprint": "7.8",
        "protocol": "final_cross_domain_comparison",
        "status": "FROZEN",
        "source": CROSS_DOMAIN_PACKAGE,
        "domains": source["domains"],
        "classifications": classifications,
        "domain_specific_findings": source["domain_specific_findings"],
        "proposal_safe_conclusions": source["proposal_safe_conclusions"],
        "limitations": source["limitations"],
        "claim": (
            "Cross-domain reuse is supported at the framework, "
            "interface, robustness-harness, metric-schema, and "
            "protocol levels, but not as one universal trained "
            "policy or universal safety controller."
        ),
        "new_training": False,
        "new_principal_runs": False,
    }
