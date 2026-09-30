from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "final-validation" / "phase-separation"

CSV_PATH = OUTPUT_DIR / "phase1-vs-phase2.csv"

JSON_PATH = OUTPUT_DIR / "phase1-vs-phase2.json"

MARKDOWN_PATH = OUTPUT_DIR / "phase1-vs-phase2.md"

MANIFEST_PATH = OUTPUT_DIR / "phase-separation-manifest.json"

FULL_SYSTEM_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "full-system-ablation"
    / "full-system-ablation.json"
)

PROPOSAL_EVIDENCE_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "proposal-evidence"
    / "proposal-evidence-matrix.json"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(
    path: Path,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    require(
        isinstance(
            payload,
            dict,
        ),
        f"{path} must contain a JSON object",
    )

    return payload


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


required_files = (
    CSV_PATH,
    JSON_PATH,
    MARKDOWN_PATH,
    MANIFEST_PATH,
    FULL_SYSTEM_PATH,
    PROPOSAL_EVIDENCE_PATH,
)

for path in required_files:
    require(
        path.is_file(),
        f"Missing required file: {path}",
    )


rows = load_csv(CSV_PATH)

payload = load_json(JSON_PATH)

manifest = load_json(MANIFEST_PATH)

full_system = load_json(FULL_SYSTEM_PATH)

proposal = load_json(PROPOSAL_EVIDENCE_PATH)


expected_areas = [
    "shared_architecture",
    "compression",
    "training_efficiency",
    "rl",
    "qml",
    "safety",
    "robustness",
    "full_system",
]

require(
    len(rows) == 8,
    "Phase separation CSV must contain 8 rows",
)

require(
    payload["row_count"] == 8,
    "Phase separation JSON row_count must be 8",
)

require(
    manifest["row_count"] == 8,
    "Manifest row_count must be 8",
)

require(
    [row["area"] for row in rows] == expected_areas,
    "Phase separation areas changed",
)

require(
    manifest["areas"] == expected_areas,
    "Manifest areas changed",
)


require(
    payload["phase1_status"] == "evaluated_component_evidence",
    "Phase 1 status changed",
)

require(
    payload["phase2_status"] == "candidate_integrated_validation",
    "Phase 2 status changed",
)

require(
    payload["principal_seeds"]
    == [
        42,
        123,
        456,
    ],
    "Principal seed protocol changed",
)

require(
    payload["direct_full_system_runs"] == 0,
    "DIRECT count changed",
)

require(
    payload["component_only_cells"] == 16,
    "COMPONENT_ONLY count changed",
)


require(
    full_system["phase1"]["direct_count"] == 0,
    "Frozen full-system DIRECT count changed",
)

require(
    full_system["phase1"]["component_only_count"] == 16,
    ("Frozen full-system COMPONENT_ONLY " "count changed"),
)

require(
    full_system["phase1"]["status"] == payload["phase1_status"],
    "Phase 1 status does not match frozen evidence",
)

require(
    full_system["phase2"]["status"] == payload["phase2_status"],
    "Phase 2 status does not match frozen evidence",
)


require(
    proposal["direct_full_system_runs"] == 0,
    "Proposal evidence DIRECT count changed",
)

require(
    proposal["component_only_cells"] == 16,
    "Proposal evidence COMPONENT_ONLY count changed",
)

require(
    proposal["new_experiments"] is False,
    "Proposal evidence unexpectedly contains new experiments",
)

require(
    proposal["new_training"] is False,
    "Proposal evidence unexpectedly contains new training",
)

require(
    proposal["new_scientific_results"] is False,
    ("Proposal evidence unexpectedly contains " "new scientific results"),
)


controls = payload["claim_controls"]

for name, value in controls.items():
    require(
        value is False,
        ("Phase-boundary claim control must " f"remain false: {name}"),
    )


require(
    payload["new_experiments"] is False,
    "Sprint 7.13 must not add experiments",
)

require(
    payload["new_training"] is False,
    "Sprint 7.13 must not add training",
)

require(
    payload["new_scientific_results"] is False,
    "Sprint 7.13 must not add scientific results",
)

require(
    payload["phase2_predictions_reported"] is False,
    "Phase 2 predictions must remain blocked",
)


lookup = {row["area"]: row for row in rows}


training = lookup["training_efficiency"]

require(
    "did not demonstrate" in training["phase1_boundary"],
    "Training negative result missing",
)

require(
    "10%" in training["phase1_boundary"],
    "Training efficiency threshold missing",
)


qml = lookup["qml"]

require(
    "0 of 6" in qml["phase1_boundary"],
    "QML target result missing",
)

require(
    "No QML sample-efficiency" in qml["phase1_boundary"],
    "QML sample-efficiency boundary missing",
)

require(
    "quantum-speedup" in qml["phase1_boundary"],
    "Quantum-speedup boundary missing",
)

require(
    "quantum-hardware advantage" in qml["phase1_boundary"],
    "Quantum-hardware boundary missing",
)


safety = lookup["safety"]

require(
    "empirical pilot evidence only" in safety["phase1_boundary"],
    "Safety empirical boundary missing",
)

require(
    "formal safety" in safety["phase1_boundary"],
    "Formal safety boundary missing",
)

require(
    "certification" in safety["phase1_boundary"],
    "Certification boundary missing",
)

require(
    "production readiness" in safety["phase1_boundary"],
    "Production boundary missing",
)


shared = lookup["shared_architecture"]

require(
    "No universal trained policy" in shared["phase1_boundary"],
    "Universal-policy boundary missing",
)

require(
    "zero-shot transfer" in shared["phase1_boundary"],
    "Zero-shot transfer boundary missing",
)

require(
    "universal safety controller" in shared["phase1_boundary"],
    "Universal safety-controller boundary missing",
)


full = lookup["full_system"]

require(
    "DIRECT = 0" in full["phase1_boundary"],
    "Full-system DIRECT boundary missing",
)

require(
    "COMPONENT_ONLY = 16" in full["phase1_boundary"],
    "Full-system component-only boundary missing",
)

require(
    "No synthetic full-system metric" in full["phase1_boundary"],
    "Synthetic metric boundary missing",
)

require(
    "interaction effect" in full["phase1_boundary"],
    "Interaction-effect boundary missing",
)

require(
    "full-system superiority" in full["phase1_boundary"],
    "Full-system superiority boundary missing",
)


markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

require(
    "Q-VLA Forge — Phase 1 vs Phase 2 Separation" in markdown,
    "Markdown UTF-8 heading invalid",
)

require(
    "Direct integrated full-system runs: 0" in markdown,
    "Markdown DIRECT boundary missing",
)

require(
    "Component-only factorial cells: 16" in markdown,
    "Markdown COMPONENT_ONLY boundary missing",
)

require(
    "No predicted Phase 2 performance claim." in markdown,
    "Phase 2 prediction boundary missing",
)

require(
    "16 Compression × QML × Safety" in markdown,
    "Integrated Phase 2 carry-over missing",
)

require(
    (
        "Phase 2 does not assume that quantum or "
        "quantum-inspired methods will outperform "
        "matched classical baselines."
    )
    in markdown,
    "Phase 2 neutrality statement missing",
)


mojibake = (
    "Ã—",
    "â‰¥",
    "â€”",
    "Â±",
    "ï¸",
)

for token in mojibake:
    require(
        token not in markdown,
        ("UTF-8 corruption detected in " f"Markdown: {token}"),
    )


require(
    manifest["generated_from_frozen_evidence"] is True,
    "Manifest frozen-evidence flag changed",
)

require(
    manifest["new_experiments"] is False,
    "Manifest new_experiments must be false",
)

require(
    manifest["new_training"] is False,
    "Manifest new_training must be false",
)

require(
    manifest["new_scientific_results"] is False,
    ("Manifest new_scientific_results " "must be false"),
)

require(
    manifest["phase2_predictions_reported"] is False,
    ("Manifest Phase 2 predictions " "must remain false"),
)


print("=" * 72)

print(" SPRINT 7.13 — PHASE 1 VS PHASE 2 VERIFICATION")

print("=" * 72)

print()

print("Structure")
print("  [PASS] 8 boundary rows")
print("  [PASS] CSV / JSON / Markdown / manifest")

print()

print("Frozen Phase 1")
print("  [PASS] seeds 42 / 123 / 456")
print("  [PASS] DIRECT = 0")
print("  [PASS] COMPONENT_ONLY = 16")
print("  [PASS] no new experiments")
print("  [PASS] no new training")
print("  [PASS] no new scientific results")

print()

print("Scientific separation")
print("  [PASS] training negative retained")
print("  [PASS] QML advantage claims blocked")
print("  [PASS] safety remains empirical")
print("  [PASS] cross-domain transfer bounded")
print("  [PASS] full-system effects remain Phase 2")
print("  [PASS] Phase 2 predictions blocked")

print()

print("Artifact integrity")
print("  [PASS] UTF-8")
print("  [PASS] frozen evidence aligned")

print()

print("Independent verifier: PASS")

print("SPRINT 7.13: PASS")
