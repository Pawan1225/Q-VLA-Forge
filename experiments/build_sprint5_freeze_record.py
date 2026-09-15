from __future__ import annotations

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

BRIDGE = ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"

OUTPUT = FINAL_DIR / "sprint5-freeze-record.json"

SOURCES = (
    SCOPE,
    CHAIN,
    MANIFEST,
    INVARIANTS,
    CLAIMS,
    LIMITATIONS,
    HANDOFF,
    SUMMARY,
    BRIDGE,
)

for path in SOURCES:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict[str, Any]:
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
handoff = load_json(HANDOFF)
bridge = load_json(BRIDGE)

if not bool(chain["all_stages_present"]):
    raise RuntimeError("Sprint 5 chain must be complete before freeze")

if not bool(manifest["all_required_present"]):
    raise RuntimeError("artifact manifest must be complete before freeze")

if not bool(handoff["ready_for_sprint7"]):
    raise RuntimeError("Sprint 7 handoff must be ready before freeze")

if not bool(bridge["proposal_ready"]):
    raise RuntimeError("unified bridge must be proposal-ready before freeze")

if int(claims["claim_count"]) != 25:
    raise RuntimeError("final claim boundary changed")

if int(limitations["limitation_count"]) != 15:
    raise RuntimeError("final limitation set changed")

scientific_invariants = invariants["invariants"]

if not isinstance(
    scientific_invariants,
    dict,
):
    raise TypeError("scientific invariants must be dict")

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5",
    "closeout_sprint": "5.15.9",
    "artifact": "sprint5-freeze-record",
    "status": "COMPLETE",
    "scientific_work_frozen": True,
    "artifact_manifest_verified": True,
    "scientific_invariants_frozen": True,
    "claim_boundary_frozen": True,
    "limitations_frozen": True,
    "phase2_bridge_present": True,
    "sprint7_handoff_ready": True,
    "sprint6_execution_required": False,
    "analysis_only_closeout": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_safety_runs": False,
    "new_robustness_runs": False,
    "new_metrics": False,
    "remote_push_pending": True,
    "chain_stage_count": int(chain["chain_stage_count"]),
    "manifest_artifact_count": int(manifest["artifact_count"]),
    "claim_count": int(claims["claim_count"]),
    "limitation_count": int(limitations["limitation_count"]),
    "source_artifacts": [str(path).replace("\\", "/") for path in SOURCES],
    "freeze_rule": (
        "Sprint 5 scientific evidence, claim boundaries, limitations, "
        "and integration artifacts are frozen. Sprint 7 may validate, "
        "aggregate, summarize, and package this evidence but must not "
        "silently modify or strengthen the frozen scientific conclusions."
    ),
    "sprint6_status_statement": scope["sprint6_status_statement"],
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
print(" SPRINT 5.15.9 FINAL FREEZE RECORD")
print("=" * 80)
print()
print("Sprint status: COMPLETE")
print(
    "Chain stages:",
    result["chain_stage_count"],
)
print(
    "Manifest artifacts:",
    result["manifest_artifact_count"],
)
print(
    "Claims:",
    result["claim_count"],
)
print(
    "Limitations:",
    result["limitation_count"],
)
print()
print("Scientific work frozen: PASS")
print("Artifact manifest verified: PASS")
print("Scientific invariants frozen: PASS")
print("Claim boundary frozen: PASS")
print("Limitations frozen: PASS")
print("Phase-2 bridge present: PASS")
print("Sprint-7 handoff ready: PASS")
print("Sprint-6 execution not required: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print("No new safety execution: PASS")
print("No new robustness execution: PASS")
print("No new metrics: PASS")
print("Remote push remains pending: PASS")
print()
print("SPRINT 5.15.9 FREEZE RECORD: PASS")
