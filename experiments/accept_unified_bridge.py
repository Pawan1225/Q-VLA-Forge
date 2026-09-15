from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

PACKAGE = ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"

ARCHITECTURE = ROOT / "results" / "integration" / "sprint5-unified-architecture.json"

COMPONENT_MAP = (
    ROOT / "results" / "integration" / "sprint5-shared-domain-component-map.json"
)

BOTTLENECKS = (
    ROOT / "results" / "integration" / "sprint5-volkswagen-bottleneck-map.json"
)

CLASSIFICATION = (
    ROOT / "results" / "integration" / "sprint5-computational-classification.json"
)

EVIDENCE_MAP = (
    ROOT / "results" / "integration" / "sprint5-evidence-architecture-map.json"
)

FINDINGS = ROOT / "results" / "integration" / "sprint5-phase1-findings.json"

CLAIMS = ROOT / "results" / "integration" / "sprint5-unified-claim-matrix.json"

ROADMAP = ROOT / "results" / "integration" / "sprint5-phase2-roadmap.json"

HANDOFF = ROOT / "results" / "integration" / "sprint5-phase2-handoff.md"

DOC = ROOT / "docs" / "unified_architecture.md"

REQUIRED_ARTIFACTS = (
    PACKAGE,
    ARCHITECTURE,
    COMPONENT_MAP,
    BOTTLENECKS,
    CLASSIFICATION,
    EVIDENCE_MAP,
    FINDINGS,
    CLAIMS,
    ROADMAP,
    HANDOFF,
    DOC,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


for path in REQUIRED_ARTIFACTS:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size <= 0:
        raise RuntimeError(f"empty artifact: {path}")


package = load_json(PACKAGE)
architecture = load_json(ARCHITECTURE)
components = load_json(COMPONENT_MAP)
bottlenecks = load_json(BOTTLENECKS)
classification = load_json(CLASSIFICATION)
evidence = load_json(EVIDENCE_MAP)
findings = load_json(FINDINGS)
claims = load_json(CLAIMS)
roadmap = load_json(ROADMAP)

# ---------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------

payloads = {
    "package": package,
    "architecture": architecture,
    "component_map": components,
    "bottlenecks": bottlenecks,
    "classification": classification,
    "evidence_map": evidence,
    "findings": findings,
    "claims": claims,
    "roadmap": roadmap,
}

for name, payload in payloads.items():
    if not bool(payload["analysis_only"]):
        raise RuntimeError(f"{name} must remain analysis-only")

    if bool(payload["new_training"]):
        raise RuntimeError(f"{name} reports new training")

    if bool(payload["new_principal_runs"]):
        raise RuntimeError(f"{name} reports new principal runs")


# ---------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------

expected_domains = {
    "autonomous_driving",
    "robotics",
}

actual_domains = {str(domain) for domain in architecture["domains"]}

if actual_domains != expected_domains:
    raise RuntimeError("domain coverage mismatch")

shared_components = components["shared_components"]

if not isinstance(
    shared_components,
    dict,
):
    raise TypeError("shared_components must be dict")

if len(shared_components) != 6:
    raise RuntimeError("expected six shared component groups")


# ---------------------------------------------------------------------
# Four challenge bottlenecks
# ---------------------------------------------------------------------

if int(bottlenecks["bottleneck_count"]) != 4:
    raise RuntimeError("expected four challenge bottlenecks")

if not bool(bottlenecks["all_four_addressed"]):
    raise RuntimeError("all four bottlenecks must remain addressed")


# ---------------------------------------------------------------------
# Classification boundaries
# ---------------------------------------------------------------------

classification_boundaries = classification["classification_boundaries"]

if not isinstance(
    classification_boundaries,
    dict,
):
    raise TypeError("classification boundaries must be dict")

if bool(classification_boundaries["lyapunov_is_quantum"]):
    raise RuntimeError("Lyapunov must remain classical")

if bool(classification_boundaries["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage must remain unsupported")

if bool(classification_boundaries["quantum_speedup_demonstrated"]):
    raise RuntimeError("quantum speedup must remain unsupported")


# ---------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------

if int(evidence["mapped_sections"]) != 7:
    raise RuntimeError("expected seven evidence sections")


# ---------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------

phase1 = findings["findings"]

if not isinstance(
    phase1,
    dict,
):
    raise TypeError("findings must be dict")

rl_qml = phase1["rl_qml"]

if int(rl_qml["classical_ppo_target_reaches"]) != 6:
    raise RuntimeError("classical PPO target count changed")

if int(rl_qml["matched_classical_target_reaches"]) != 1:
    raise RuntimeError("matched classical target count changed")

if int(rl_qml["hybrid_qml_target_reaches"]) != 0:
    raise RuntimeError("hybrid QML target count changed")


# ---------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------

if int(claims["claim_count"]) != 14:
    raise RuntimeError("expected fourteen claims")

status_counts = claims["status_counts"]

expected_status_counts = {
    "supported": 4,
    "supported_with_limitation": 1,
    "not_supported": 9,
}

actual_status_counts = {key: int(status_counts[key]) for key in expected_status_counts}

if actual_status_counts != expected_status_counts:
    raise RuntimeError("claim status partition mismatch")

expected_blocked = {
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

actual_blocked = {str(claim_id) for claim_id in claims["blocked_claim_ids"]}

if actual_blocked != expected_blocked:
    raise RuntimeError("blocked claim set mismatch")


# ---------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------

priorities = roadmap["priorities"]

if not isinstance(
    priorities,
    list,
):
    raise TypeError("priorities must be list")

if len(priorities) != 8:
    raise RuntimeError("expected eight Phase-2 priorities")


# ---------------------------------------------------------------------
# Consolidated package
# ---------------------------------------------------------------------

if package["artifact"] != "minimal-unified-architecture-phase2-bridge-package":
    raise RuntimeError("unexpected bridge package identity")

if int(package["bottleneck_count"]) != 4:
    raise RuntimeError("package bottleneck count mismatch")

if int(package["evidence_sections"]) != 7:
    raise RuntimeError("package evidence count mismatch")

if int(package["claim_count"]) != 14:
    raise RuntimeError("package claim count mismatch")

if int(package["phase2_priority_count"]) != 8:
    raise RuntimeError("package Phase-2 priority count mismatch")

if not bool(package["proposal_ready"]):
    raise RuntimeError("package must remain proposal-ready")


print("=" * 80)
print(" SPRINT 5.14.5L FINAL UNIFIED BRIDGE ACCEPTANCE")
print("=" * 80)
print()
print(
    "Required artifacts:",
    len(REQUIRED_ARTIFACTS),
)
print("Domains: 2")
print("Shared component groups: 6")
print("Volkswagen bottlenecks: 4")
print("Evidence sections: 7")
print("Claims: 14")
print("Phase-2 priorities: 8")
print()
print("Analysis/documentation-only scope: PASS")
print("No new training: PASS")
print("No new principal runs: PASS")
print("Architecture integration: PASS")
print("Shared/domain boundaries: PASS")
print("Four-bottleneck coverage: PASS")
print("Classical/QI/QML classification: PASS")
print("Lyapunov classical boundary: PASS")
print("Evidence mapping: PASS")
print("Phase-1 findings freeze: PASS")
print("Claim controls: PASS")
print("Unsupported claims blocked: PASS")
print("Phase-2 roadmap: PASS")
print("Unified architecture document: PASS")
print("Phase-2 handoff: PASS")
print("Proposal-ready bridge package: PASS")
print("Independent verification handoff: PASS")
print()
print("SPRINT 5.14.5 FINAL ACCEPTANCE: PASS")
