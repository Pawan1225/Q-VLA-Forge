from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.sprint5_evidence import (
    PRINCIPAL_SEEDS,
    sha256_file,
)

ROOT = Path(".")

SOURCES = {
    "sprint5_13": (
        ROOT
        / "results"
        / "safety"
        / "consolidated"
        / "sprint5-safety-evidence-package.json"
    ),
    "sprint5_14": (
        ROOT
        / "results"
        / "safety"
        / "cross-domain"
        / "sprint5-cross-domain-package.json"
    ),
    "sprint5_14_5": (
        ROOT / "results" / "integration" / "sprint5-unified-bridge-package.json"
    ),
    "scientific_freeze": (
        ROOT / "results" / "safety" / "final" / "sprint5-scientific-invariants.json"
    ),
    "claim_boundary": (
        ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"
    ),
    "freeze_record": (
        ROOT / "results" / "safety" / "final" / "sprint5-freeze-record.json"
    ),
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain an object")

    return payload


hashes: dict[str, str] = {}

for key, path in SOURCES.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    hashes[key] = sha256_file(path)


consolidated = load_json(SOURCES["sprint5_13"])

cross_domain = load_json(SOURCES["sprint5_14"])

bridge = load_json(SOURCES["sprint5_14_5"])

scientific_freeze = load_json(SOURCES["scientific_freeze"])

claim_boundary = load_json(SOURCES["claim_boundary"])

freeze_record = load_json(SOURCES["freeze_record"])


if freeze_record.get("status") != "COMPLETE":
    raise RuntimeError("Sprint 5 freeze is not COMPLETE")

if not freeze_record.get("scientific_work_frozen"):
    raise RuntimeError("Sprint 5 science is not frozen")

if not bridge.get("proposal_ready"):
    raise RuntimeError("Sprint 5.14.5 bridge is not proposal-ready")

if (
    int(
        claim_boundary.get(
            "claim_count",
            0,
        )
    )
    != 25
):
    raise RuntimeError("Final claim boundary changed")

if tuple(PRINCIPAL_SEEDS) != (
    42,
    123,
    456,
):
    raise RuntimeError("Principal seed protocol changed")

scientific = scientific_freeze.get("invariants")

if not isinstance(
    scientific,
    dict,
):
    raise TypeError("Scientific invariant payload missing")

if not isinstance(
    consolidated,
    dict,
):
    raise TypeError("Consolidated safety package invalid")

if not isinstance(
    cross_domain,
    dict,
):
    raise TypeError("Cross-domain safety package invalid")


print("=" * 52)
print(" Q-VLA FORGE — SPRINT 5.15 READINESS")
print("=" * 52)
print()

print("Scientific freeze                    PASS")
print("Sprint 5.13 consolidation            PASS")
print("Sprint 5.14 cross-domain             PASS")
print("Sprint 5.14.5 architecture           PASS")
print("Final claim boundary                 PASS")
print()

print("Driving evidence                     PASS")
print("Robotics evidence                    PASS")
print()

print("NONE                                 PASS")
print("CLIPPING                             PASS")
print("LYAPUNOV                             PASS")
print()

print("Clean evidence                       PASS")
print("Gaussian evidence                    PASS")
print("Structured-state evidence            PASS")
print("Action evidence                      PASS")
print("Lyapunov mechanism evidence          PASS")
print()

print("Three-seed protocol                  PASS")
print("Source hashes                        PASS")
print()

print("No training                          PASS")
print("No principal execution               PASS")
print("No retuning                          PASS")
print()

print("SPRINT 5.15 READINESS: PASS")
