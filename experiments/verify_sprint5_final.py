from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

FINAL_DIR = ROOT / "results" / "safety" / "final"

SCOPE = FINAL_DIR / "sprint5-final-scope.json"
CHAIN = FINAL_DIR / "sprint5-chain-inventory.json"
MANIFEST = FINAL_DIR / "sprint5-final-manifest.json"
INVARIANTS = FINAL_DIR / "sprint5-scientific-invariants.json"
CLAIMS = FINAL_DIR / "sprint5-final-claim-boundary.json"
LIMITATIONS = FINAL_DIR / "sprint5-final-limitations.json"
HANDOFF = FINAL_DIR / "sprint7-handoff.json"
SUMMARY = FINAL_DIR / "sprint5-final-summary.md"
FREEZE = FINAL_DIR / "sprint5-freeze-record.json"

BRIDGE = ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"

CROSS_DOMAIN = (
    ROOT / "results" / "safety" / "cross-domain" / "sprint5-cross-domain-package.json"
)

REQUIRED = (
    SCOPE,
    CHAIN,
    MANIFEST,
    INVARIANTS,
    CLAIMS,
    LIMITATIONS,
    HANDOFF,
    SUMMARY,
    FREEZE,
    BRIDGE,
    CROSS_DOMAIN,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()

    files = sorted(item for item in path.rglob("*") if item.is_file())

    if not files:
        raise RuntimeError(f"directory contains no files: {path}")

    for file_path in files:
        relative = file_path.relative_to(path)

        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")

    return digest.hexdigest()


for path in REQUIRED:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size <= 0:
        raise RuntimeError(f"empty required artifact: {path}")


scope = load_json(SCOPE)
chain = load_json(CHAIN)
manifest = load_json(MANIFEST)
invariants = load_json(INVARIANTS)
claims = load_json(CLAIMS)
limitations = load_json(LIMITATIONS)
handoff = load_json(HANDOFF)
freeze = load_json(FREEZE)
bridge = load_json(BRIDGE)
cross_domain = load_json(CROSS_DOMAIN)

summary_text = SUMMARY.read_text(encoding="utf-8-sig")

# ---------------------------------------------------------------------
# Scope verification
# ---------------------------------------------------------------------

if scope["type"] != "final_acceptance_and_freeze":
    raise RuntimeError("unexpected Sprint 5.15 scope type")

for key in (
    "new_training",
    "new_principal_runs",
    "new_safety_runs",
    "new_robustness_runs",
    "new_metrics",
    "new_tuning",
    "new_architecture_work",
    "new_scientific_experiments",
):
    if bool(scope[key]):
        raise RuntimeError(f"scope violation: {key}")

if scope["handoff_target"] != "Sprint 7":
    raise RuntimeError("handoff target must be Sprint 7")

if bool(scope["sprint6_execution_required"]):
    raise RuntimeError("Sprint 6 must remain not required")

# ---------------------------------------------------------------------
# Sprint chain
# ---------------------------------------------------------------------

if int(chain["chain_stage_count"]) != 16:
    raise RuntimeError("expected 16 Sprint 5 stages")

if not bool(chain["all_stages_present"]):
    raise RuntimeError("Sprint 5 chain incomplete")

expected_stages = {
    "5.1",
    "5.2",
    "5.3",
    "5.3.5",
    "5.4",
    "5.5",
    "5.6",
    "5.7",
    "5.8",
    "5.9",
    "5.10",
    "5.11",
    "5.12",
    "5.13",
    "5.14",
    "5.14.5",
}

actual_stages = {str(record["sprint"]) for record in chain["stages"]}

if actual_stages != expected_stages:
    raise RuntimeError("Sprint 5 stage set mismatch")

# ---------------------------------------------------------------------
# Manifest integrity
# ---------------------------------------------------------------------

if int(manifest["artifact_count"]) != 17:
    raise RuntimeError("expected 17 frozen manifest artifacts")

if not bool(manifest["all_required_present"]):
    raise RuntimeError("manifest required-artifact check failed")

manifest_records = manifest["artifacts"]

if not isinstance(
    manifest_records,
    list,
):
    raise TypeError("manifest artifacts must be list")

for record in manifest_records:
    artifact_path = ROOT / str(record["path"])

    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)

    if artifact_path.is_file():
        actual_hash = sha256_file(artifact_path)
    elif artifact_path.is_dir():
        actual_hash = sha256_directory(artifact_path)
    else:
        raise RuntimeError(f"unsupported manifest artifact: {artifact_path}")

    if actual_hash != str(record["sha256"]):
        raise RuntimeError(f"manifest hash mismatch: {artifact_path}")

    if not bool(record["frozen"]):
        raise RuntimeError(f"artifact not frozen: {artifact_path}")

# ---------------------------------------------------------------------
# Scientific invariants
# ---------------------------------------------------------------------

scientific = invariants["invariants"]

if not isinstance(scientific, dict):
    raise TypeError("scientific invariants must be dict")

rl_qml = scientific["rl_qml"]

if int(rl_qml["classical_ppo_target_reaches"]) != 6:
    raise RuntimeError("PPO target-reach invariant changed")

if int(rl_qml["matched_classical_target_reaches"]) != 1:
    raise RuntimeError("matched classical invariant changed")

if int(rl_qml["hybrid_qml_target_reaches"]) != 0:
    raise RuntimeError("hybrid QML invariant changed")

if bool(scientific["quantum_claims"]["quantum_advantage_demonstrated"]):
    raise RuntimeError("quantum advantage overclaim detected")

if bool(scientific["quantum_claims"]["quantum_speedup_demonstrated"]):
    raise RuntimeError("quantum speedup overclaim detected")

# ---------------------------------------------------------------------
# Claim boundary
# ---------------------------------------------------------------------

if int(claims["claim_count"]) != 25:
    raise RuntimeError("final claim count mismatch")

status_counts = claims["status_counts"]

expected_status_counts = {
    "supported": 9,
    "supported_with_limitation": 3,
    "not_supported": 13,
}

actual_status_counts = {key: int(status_counts[key]) for key in expected_status_counts}

if actual_status_counts != expected_status_counts:
    raise RuntimeError("claim status partition mismatch")

# ---------------------------------------------------------------------
# Limitations / Lyapunov nuance
# ---------------------------------------------------------------------

if int(limitations["limitation_count"]) != 15:
    raise RuntimeError("limitation count mismatch")

nuance = limitations["lyapunov_nuance"]

if not bool(nuance["hard_guards_material_to_observed_interventions"]):
    raise RuntimeError("hard-guard contribution was lost")

if not bool(nuance["action_regime_lyapunov_specific_activation"]):
    raise RuntimeError("action-regime Lyapunov activation was lost")

if not bool(nuance["driving_action_activation"]):
    raise RuntimeError("driving Lyapunov activation changed")

if bool(nuance["robotics_action_activation"]):
    raise RuntimeError("robotics Lyapunov activation changed")

if bool(nuance["formal_stability_claim"]):
    raise RuntimeError("formal stability overclaim detected")

# ---------------------------------------------------------------------
# Bridge and cross-domain integrity
# ---------------------------------------------------------------------

if not bool(bridge["proposal_ready"]):
    raise RuntimeError("unified bridge is no longer proposal-ready")

if cross_domain["artifact"] != "consolidated-cross-domain-safety-package":
    raise RuntimeError("unexpected cross-domain package")

# ---------------------------------------------------------------------
# Sprint 7 handoff
# ---------------------------------------------------------------------

if handoff["source_sprint"] != "5.15":
    raise RuntimeError("unexpected handoff source sprint")

if handoff["target_sprint"] != "7":
    raise RuntimeError("unexpected handoff target")

if bool(handoff["sprint6_execution_required"]):
    raise RuntimeError("Sprint 6 execution requirement changed")

if not bool(handoff["ready_for_sprint7"]):
    raise RuntimeError("Sprint 7 handoff not ready")

# ---------------------------------------------------------------------
# Summary / anti-overclaim
# ---------------------------------------------------------------------

required_summary_sections = (
    "## Scope",
    "## Safety Architecture",
    "## Clean Safety Results",
    "## Robustness Evaluation",
    "## Lyapunov Mechanism Findings",
    "## Cross-Domain Findings",
    "## Unified Architecture",
    "## Phase 1 Scientific Findings",
    "## Supported Claims",
    "## Supported With Limitation",
    "## Unsupported Claims",
    "## Limitations",
    "## Sprint 7 Handoff",
)

for section in required_summary_sections:
    if section not in summary_text:
        raise RuntimeError(f"missing final summary section: {section}")

lower_summary = summary_text.lower()

blocked_positive_phrases = (
    "quantum advantage was demonstrated",
    "quantum speedup was demonstrated",
    "qml sample-efficiency superiority was demonstrated",
    "universal trained vla was demonstrated",
    "zero-shot cross-domain policy transfer was demonstrated",
    "formal safety guarantee was demonstrated",
    "production-ready",
    "certified safe",
)

for phrase in blocked_positive_phrases:
    if phrase in lower_summary:
        raise RuntimeError(f"unsupported positive phrase detected: {phrase}")

# ---------------------------------------------------------------------
# Freeze record
# ---------------------------------------------------------------------

if freeze["status"] != "COMPLETE":
    raise RuntimeError("freeze status must be COMPLETE")

required_freeze_flags = (
    "scientific_work_frozen",
    "artifact_manifest_verified",
    "scientific_invariants_frozen",
    "claim_boundary_frozen",
    "limitations_frozen",
    "phase2_bridge_present",
    "sprint7_handoff_ready",
)

for key in required_freeze_flags:
    if not bool(freeze[key]):
        raise RuntimeError(f"freeze flag failed: {key}")

if not bool(freeze["remote_push_pending"]):
    raise RuntimeError("remote push should still be pending before final push")

print("=" * 80)
print(" Q-VLA FORGE — SPRINT 5 INDEPENDENT FINAL VERIFICATION")
print("=" * 80)
print()
print("Required closeout artifacts:", len(REQUIRED))
print("Sprint chain stages: 16")
print("Frozen manifest artifacts: 17")
print("Final claims: 25")
print("Final limitations: 15")
print()
print("Final scope lock: PASS")
print("Sprint chain integrity: PASS")
print("Artifact manifest existence: PASS")
print("Artifact SHA256 integrity: PASS")
print("Scientific invariants: PASS")
print("Claim partition: PASS")
print("Lyapunov nuance: PASS")
print("Cross-domain package integrity: PASS")
print("Unified architecture bridge: PASS")
print("Final limitations: PASS")
print("Sprint-6 skip wording: PASS")
print("Sprint-7 readiness: PASS")
print("Final summary structure: PASS")
print("Anti-overclaim controls: PASS")
print("Freeze-record controls: PASS")
print()
print("No new training: PASS")
print("No new principal execution: PASS")
print("Historical evidence integrity: PASS")
print()
print("SPRINT 5 INDEPENDENT FINAL VERIFICATION: PASS")
print("SPRINT 7 HANDOFF: READY")
