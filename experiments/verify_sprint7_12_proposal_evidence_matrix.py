from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "final-validation" / "proposal-evidence"

CSV_PATH = OUTPUT_DIR / "proposal-evidence-matrix.csv"

JSON_PATH = OUTPUT_DIR / "proposal-evidence-matrix.json"

MARKDOWN_PATH = OUTPUT_DIR / "proposal-evidence-matrix.md"

MANIFEST_PATH = OUTPUT_DIR / "proposal-evidence-manifest.json"

CLAIM_REGISTRY = (
    ROOT / "results" / "final-validation" / "tables" / "final-claim-registry.csv"
)

SCORECARD = (
    ROOT
    / "results"
    / "final-validation"
    / "tables"
    / "challenge-bottleneck-scorecard.csv"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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


required_files = (
    CSV_PATH,
    JSON_PATH,
    MARKDOWN_PATH,
    MANIFEST_PATH,
    CLAIM_REGISTRY,
    SCORECARD,
)

for path in required_files:
    require(
        path.is_file(),
        f"Missing required file: {path}",
    )


matrix_csv = load_csv(CSV_PATH)

matrix_json = load_json(JSON_PATH)

manifest = load_json(MANIFEST_PATH)

claim_registry = load_csv(CLAIM_REGISTRY)

scorecard = load_csv(SCORECARD)


require(
    len(matrix_csv) == 9,
    "Proposal evidence CSV must contain 9 rows",
)

require(
    matrix_json["row_count"] == 9,
    "Proposal evidence JSON row_count must be 9",
)

require(
    manifest["row_count"] == 9,
    "Proposal evidence manifest row_count must be 9",
)


expected_ids = [
    f"S7-C{index:02d}"
    for index in range(
        1,
        10,
    )
]

csv_ids = [row["evidence_id"] for row in matrix_csv]

json_ids = [row["evidence_id"] for row in matrix_json["rows"]]

manifest_ids = manifest["claim_ids"]

require(
    csv_ids == expected_ids,
    "CSV claim IDs changed",
)

require(
    json_ids == expected_ids,
    "JSON claim IDs changed",
)

require(
    manifest_ids == expected_ids,
    "Manifest claim IDs changed",
)


registry_lookup = {row["claim_id"]: row for row in claim_registry}

require(
    set(registry_lookup) == set(expected_ids),
    "Frozen claim registry IDs changed",
)

for row in matrix_csv:
    evidence_id = row["evidence_id"]

    frozen = registry_lookup[evidence_id]

    require(
        row["proposal_safe_claim"] == frozen["statement"],
        ("Proposal-safe claim changed for " f"{evidence_id}"),
    )

    frozen_limitation = frozen["limitation"].strip()

    if frozen_limitation:
        require(
            frozen_limitation in row["limitation"],
            ("Frozen limitation not preserved for " f"{evidence_id}"),
        )


scorecard_bottlenecks = {row["bottleneck"] for row in scorecard}

matrix_bottlenecks = {row["challenge_bottleneck"] for row in matrix_csv}

require(
    scorecard_bottlenecks.issubset(matrix_bottlenecks),
    ("Challenge bottleneck scorecard is not " "fully represented in proposal evidence"),
)


require(
    matrix_json["phase"] == "Phase 1",
    "Proposal evidence phase changed",
)

require(
    matrix_json["evidence_frozen"] is True,
    "Evidence must remain frozen",
)

require(
    matrix_json["new_experiments"] is False,
    "Sprint 7.12 must not introduce experiments",
)

require(
    matrix_json["new_training"] is False,
    "Sprint 7.12 must not introduce training",
)

require(
    matrix_json["new_scientific_results"] is False,
    ("Sprint 7.12 must not introduce " "new scientific results"),
)


require(
    matrix_json["principal_seeds"]
    == [
        42,
        123,
        456,
    ],
    "Principal seed protocol changed",
)

require(
    matrix_json["direct_full_system_runs"] == 0,
    "DIRECT full-system count changed",
)

require(
    matrix_json["component_only_cells"] == 16,
    "COMPONENT_ONLY count changed",
)


controls = matrix_json["claim_controls"]

for name, value in controls.items():
    require(
        value is False,
        ("Claim control must remain false: " f"{name}"),
    )


training = next(row for row in matrix_csv if row["evidence_id"] == "S7-C03")

require(
    training["phase1_status"] == "not_demonstrated",
    ("Training-efficiency negative result " "was not preserved"),
)

require(
    "did not demonstrate" in training["proposal_safe_claim"],
    ("Training-efficiency negative wording " "was not preserved"),
)


qml = next(row for row in matrix_csv if row["evidence_id"] == "S7-C05")

require(
    "95.90%" in qml["observed_result"],
    "Driving QML compactness missing",
)

require(
    "95.51%" in qml["observed_result"],
    "Robotics QML compactness missing",
)

require(
    "No QML sample-efficiency" in qml["limitation"],
    "QML sample-efficiency boundary missing",
)


safety = next(row for row in matrix_csv if row["evidence_id"] == "S7-C06")

require(
    "zero" in safety["proposal_safe_claim"],
    "Safety observed-zero result missing",
)

require(
    "no formal safety" in safety["limitation"],
    "Formal-safety boundary missing",
)


cross_domain = next(row for row in matrix_csv if row["evidence_id"] == "S7-C09")

require(
    "zero-shot policy transfer" in cross_domain["limitation"],
    "Cross-domain transfer boundary missing",
)


markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

require(
    "Q-VLA Forge — Phase 1 Proposal Evidence Matrix" in markdown,
    "Markdown UTF-8 heading is invalid",
)

require(
    "S7-C01" in markdown and "S7-C09" in markdown,
    "Markdown claim coverage incomplete",
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
    "No quantum advantage claim." in markdown,
    "Markdown quantum-advantage boundary missing",
)

require(
    "No QML sample-efficiency superiority claim." in markdown,
    "Markdown QML boundary missing",
)

require(
    "No TT/MPS superiority claim." in markdown,
    "Markdown TT/MPS boundary missing",
)

require(
    "No formal safety or Lyapunov stability guarantee." in markdown,
    "Markdown safety boundary missing",
)

require(
    "No full-system superiority claim." in markdown,
    "Markdown full-system boundary missing",
)

require(
    "16 Compression × QML × Safety" in markdown,
    "Phase 2 integrated carry-over missing",
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
        ("UTF-8 corruption detected in Markdown: " f"{token}"),
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


print("=" * 72)

print(" SPRINT 7.12 — PROPOSAL EVIDENCE MATRIX VERIFICATION")

print("=" * 72)

print()

print("Structure")
print("  [PASS] 9 evidence rows")
print("  [PASS] S7-C01 through S7-C09")
print("  [PASS] CSV / JSON / Markdown / manifest")

print()

print("Frozen evidence alignment")
print("  [PASS] claim registry aligned")
print("  [PASS] bottleneck scorecard represented")
print("  [PASS] seeds 42 / 123 / 456")
print("  [PASS] DIRECT = 0")
print("  [PASS] COMPONENT_ONLY = 16")

print()

print("Scientific boundaries")
print("  [PASS] training negative retained")
print("  [PASS] QML compactness separated from advantage")
print("  [PASS] safety remains empirical")
print("  [PASS] cross-domain transfer bounded")
print("  [PASS] claim controls blocked")

print()

print("Artifact integrity")
print("  [PASS] UTF-8")
print("  [PASS] Phase 2 carry-over retained")
print("  [PASS] no new scientific results")

print()

print("Independent verifier: PASS")

print("SPRINT 7.12: PASS")
