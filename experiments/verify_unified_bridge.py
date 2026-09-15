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

doc_text = DOC.read_text(encoding="utf-8-sig")

handoff_text = HANDOFF.read_text(encoding="utf-8-sig")

# ---------------------------------------------------------------------
# Scope controls
# ---------------------------------------------------------------------

for payload_name, payload in {
    "package": package,
    "architecture": architecture,
    "component_map": components,
    "bottlenecks": bottlenecks,
    "classification": classification,
    "evidence_map": evidence,
    "findings": findings,
    "claims": claims,
    "roadmap": roadmap,
}.items():
    if not bool(payload["analysis_only"]):
        raise RuntimeError(f"{payload_name} must remain analysis-only")

    if bool(payload["new_training"]):
        raise RuntimeError(f"{payload_name} reports new training")

    if bool(payload["new_principal_runs"]):
        raise RuntimeError(f"{payload_name} reports new principal runs")


# ---------------------------------------------------------------------
# Domain coverage
# ---------------------------------------------------------------------

domains = {str(domain) for domain in architecture["domains"]}

expected_domains = {
    "autonomous_driving",
    "robotics",
}

if domains != expected_domains:
    raise RuntimeError("domain set mismatch")


# ---------------------------------------------------------------------
# Shared vs domain-specific boundaries
# ---------------------------------------------------------------------

boundary = components["shared_boundary"]

if not isinstance(boundary, dict):
    raise TypeError("shared_boundary must be dict")

required_true = {
    "shared_framework",
    "shared_interfaces",
    "shared_action_dimension",
    "shared_training_protocol",
    "shared_safety_api",
    "shared_robustness_harness",
}

for key in required_true:
    if not bool(boundary[key]):
        raise RuntimeError(f"{key} must remain true")

required_false = {
    "same_environment",
    "same_state_semantics",
    "same_reward",
    "same_policy_weights",
    "same_safety_constraints",
    "same_lyapunov_potential",
    "cross_domain_weight_transfer_tested",
    "universal_controller_supported",
}

for key in required_false:
    if bool(boundary[key]):
        raise RuntimeError(f"{key} must remain false")


# ---------------------------------------------------------------------
# Four Volkswagen bottlenecks
# ---------------------------------------------------------------------

if int(bottlenecks["bottleneck_count"]) != 4:
    raise RuntimeError("expected exactly four bottlenecks")

if not bool(bottlenecks["all_four_addressed"]):
    raise RuntimeError("all four challenge bottlenecks must be addressed")

expected_bottlenecks = {
    "model_footprint",
    "training_efficiency",
    "rl_alignment_sample_efficiency",
    "safety",
}

actual_bottlenecks = {str(key) for key in bottlenecks["bottlenecks"]}

if actual_bottlenecks != expected_bottlenecks:
    raise RuntimeError("bottleneck mapping mismatch")


# ---------------------------------------------------------------------
# Computational classification
# ---------------------------------------------------------------------

boundaries = classification["classification_boundaries"]

if not isinstance(boundaries, dict):
    raise TypeError("classification boundaries must be dict")

if bool(boundaries["lyapunov_is_quantum"]):
    raise RuntimeError("Lyapunov must remain classical")

if bool(boundaries["clipping_is_quantum"]):
    raise RuntimeError("clipping must remain classical")

if bool(boundaries["ppo_is_quantum"]):
    raise RuntimeError("PPO must remain classical")

if bool(boundaries["svd_is_quantum_inspired"]):
    raise RuntimeError("SVD must remain classical")

if not bool(boundaries["tensor_train_is_quantum_inspired"]):
    raise RuntimeError("Tensor Train must remain quantum-inspired")

if not bool(boundaries["mps_is_quantum_inspired"]):
    raise RuntimeError("MPS must remain quantum-inspired")

if not bool(boundaries["pqc_is_qml"]):
    raise RuntimeError("PQC must remain QML")

if bool(boundaries["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage must remain unsupported")

if bool(boundaries["quantum_speedup_demonstrated"]):
    raise RuntimeError("quantum speedup must remain unsupported")


# ---------------------------------------------------------------------
# Evidence mapping
# ---------------------------------------------------------------------

if int(evidence["mapped_sections"]) != 7:
    raise RuntimeError("expected seven evidence-map sections")

expected_sprints = {
    "Sprint 1",
    "Sprint 2",
    "Sprint 3",
    "Sprint 4",
    "Sprint 5",
    "Sprint 5.14",
}

actual_sprints = {str(item) for item in evidence["sprints_covered"]}

if actual_sprints != expected_sprints:
    raise RuntimeError("evidence sprint coverage mismatch")


# ---------------------------------------------------------------------
# Phase-1 findings
# ---------------------------------------------------------------------

phase1 = findings["findings"]

if not isinstance(phase1, dict):
    raise TypeError("findings must be dict")

rl_qml = phase1["rl_qml"]

if int(rl_qml["classical_ppo_target_reaches"]) != 6:
    raise RuntimeError("classical PPO target count changed")

if int(rl_qml["matched_classical_target_reaches"]) != 1:
    raise RuntimeError("matched classical target count changed")

if int(rl_qml["hybrid_qml_target_reaches"]) != 0:
    raise RuntimeError("hybrid QML target count changed")

if bool(rl_qml["sample_efficiency_advantage_demonstrated"]):
    raise RuntimeError("QML sample-efficiency advantage must remain unsupported")

cross_domain = phase1["cross_domain"]

if bool(cross_domain["universal_trained_model_supported"]):
    raise RuntimeError("universal trained model must remain unsupported")

if bool(cross_domain["zero_shot_transfer_supported"]):
    raise RuntimeError("zero-shot transfer must remain unsupported")


# ---------------------------------------------------------------------
# Claim matrix
# ---------------------------------------------------------------------

if int(claims["claim_count"]) != 14:
    raise RuntimeError("expected fourteen bridge claims")

status_counts = claims["status_counts"]

expected_status_counts = {
    "supported": 4,
    "supported_with_limitation": 1,
    "not_supported": 9,
}

for key, expected in expected_status_counts.items():
    if int(status_counts[key]) != expected:
        raise RuntimeError(f"claim count mismatch for {key}")

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

blocked = {str(claim_id) for claim_id in claims["blocked_claim_ids"]}

if blocked != expected_blocked:
    raise RuntimeError("blocked bridge claim set mismatch")


# ---------------------------------------------------------------------
# Phase-2 roadmap
# ---------------------------------------------------------------------

priorities = roadmap["priorities"]

if not isinstance(priorities, list):
    raise TypeError("Phase-2 priorities must be list")

if len(priorities) != 8:
    raise RuntimeError("expected eight Phase-2 priorities")

priority_ids = [int(record["priority"]) for record in priorities]

if priority_ids != list(
    range(
        1,
        9,
    )
):
    raise RuntimeError("Phase-2 priority order mismatch")


# ---------------------------------------------------------------------
# Proposal document controls
# ---------------------------------------------------------------------

required_doc_sections = (
    "## 1. Purpose",
    "## 2. Canonical Architecture",
    "## 3. Shared Components",
    "## 4. Domain-Specific Components",
    "## 5. Volkswagen Challenge Bottlenecks",
    "## 6. Classical / Quantum-Inspired / Quantum Components",
    "## 7. Evidence Mapping",
    "## 8. Phase 1 Findings",
    "## 9. Supported Claims",
    "## 10. Unsupported Claims",
    "## 11. Phase 1 Limitations",
    "## 12. Phase 2 Roadmap",
)

for section in required_doc_sections:
    if section not in doc_text:
        raise RuntimeError(f"missing documentation section: {section}")


# ---------------------------------------------------------------------
# Anti-overclaim checks
# ---------------------------------------------------------------------

required_negative_phrases = (
    "does not support",
    "quantum advantage",
    "quantum speedup",
    "formal safety guarantee",
    "production deployment readiness",
)

lower_doc = doc_text.lower()

for phrase in required_negative_phrases:
    if phrase.lower() not in lower_doc:
        raise RuntimeError(f"missing anti-overclaim wording: {phrase}")

if "quantum advantage was demonstrated" in lower_doc:
    raise RuntimeError("positive quantum-advantage claim detected")

if "quantum speedup was demonstrated" in lower_doc:
    raise RuntimeError("positive quantum-speedup claim detected")

if "production-ready" in lower_doc:
    raise RuntimeError("production-ready claim detected")

if "certified safe" in lower_doc:
    raise RuntimeError("certified-safe claim detected")

if "zero-shot cross-domain transfer was demonstrated" in lower_doc:
    raise RuntimeError("positive zero-shot-transfer claim detected")


# ---------------------------------------------------------------------
# Handoff controls
# ---------------------------------------------------------------------

handoff_required = (
    "## Current Phase 1 Status",
    "## What Is Frozen",
    "## What Worked",
    "## What Did Not Establish Advantage",
    "## Critical Phase 1 Limitations",
    "## Phase 2 Priority Sequence",
    "## Phase 2 Entry Principle",
    "## Claim Boundary",
)

for section in handoff_required:
    if section not in handoff_text:
        raise RuntimeError(f"missing handoff section: {section}")


# ---------------------------------------------------------------------
# Package integrity
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
    raise RuntimeError("bridge package must be proposal-ready")


print("=" * 80)
print(" SPRINT 5.14.5K INDEPENDENT UNIFIED BRIDGE VERIFICATION")
print("=" * 80)
print()
print(
    "Required artifacts:",
    len(REQUIRED_ARTIFACTS),
)
print("Domains: 2")
print("Volkswagen bottlenecks: 4")
print("Evidence sections: 7")
print("Claims: 14")
print("Phase-2 priorities: 8")
print()
print("Artifact existence: PASS")
print("Artifact non-empty checks: PASS")
print("Analysis-only scope: PASS")
print("No new training: PASS")
print("No new principal runs: PASS")
print("Domain coverage: PASS")
print("Shared/domain boundary controls: PASS")
print("Four-bottleneck coverage: PASS")
print("Classical/QI/QML classification: PASS")
print("Lyapunov classical classification: PASS")
print("Evidence mapping: PASS")
print("Phase-1 finding invariants: PASS")
print("Claim partition: PASS")
print("Blocked claim set: PASS")
print("Phase-2 roadmap: PASS")
print("Unified document structure: PASS")
print("Phase-2 handoff structure: PASS")
print("Anti-overclaim controls: PASS")
print("Proposal-ready package: PASS")
print()
print("SPRINT 5.14.5K INDEPENDENT VERIFICATION: PASS")
