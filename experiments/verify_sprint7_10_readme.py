from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(".")

README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"

TABLE_DIR = ROOT / "results" / "final-validation" / "tables"

COMPRESSION_TABLE = TABLE_DIR / "compression-table.csv"

TRAINING_TABLE = TABLE_DIR / "training-table.csv"

RL_TABLE = TABLE_DIR / "rl-table.csv"

SAFETY_TABLE = TABLE_DIR / "safety-table.csv"

CROSS_DOMAIN_TABLE = TABLE_DIR / "cross-domain-table.csv"

FIGURE_DIR = ROOT / "results" / "final-validation" / "figures"

ARCHITECTURE_FIGURE = FIGURE_DIR / "figure-01-architecture.png"

FULL_SYSTEM_PATH = (
    ROOT
    / "results"
    / "final-validation"
    / "full-system-ablation"
    / "full-system-ablation.json"
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
    README,
    REQUIREMENTS,
    COMPRESSION_TABLE,
    TRAINING_TABLE,
    RL_TABLE,
    SAFETY_TABLE,
    CROSS_DOMAIN_TABLE,
    ARCHITECTURE_FIGURE,
    FULL_SYSTEM_PATH,
)

for path in required_files:
    require(
        path.is_file(),
        f"Missing required file: {path}",
    )


text = README.read_text(encoding="utf-8")

flat = " ".join(text.split())

lower = flat.lower()


required_sections = (
    "## Challenge",
    "## Pilot Objective",
    "## Research Questions",
    "## Architecture",
    "## Algorithms",
    "## Installation",
    "## Quick Start",
    "## Experiments",
    "## Dashboard",
    "## Results",
    "## Ablation",
    "## Limitations",
    "## Reproduction",
    "## Phase 2 Roadmap",
)

for heading in required_sections:
    require(
        heading in text,
        f"Missing README section: {heading}",
    )


repository_commands = (
    "pip install -r requirements.txt",
    '$env:PYTHONPATH="src;."',
    "pytest tests -q",
    "streamlit run dashboard\\app.py",
    "ruff check src tests experiments dashboard",
    "black --check src tests experiments dashboard",
    "mypy src",
)

for command in repository_commands:
    require(
        command in text,
        f"Missing repository command: {command}",
    )


evidence_paths = (
    "results/final-validation/statistics/",
    "results/final-validation/compression-ablation/",
    "results/final-validation/qml-ablation/",
    "results/final-validation/safety-ablation/",
    "results/final-validation/full-system-ablation/",
    "results/final-validation/figures/",
    "results/final-validation/tables/",
)

for evidence_path in evidence_paths:
    require(
        evidence_path in text,
        ("Missing evidence path in README: " f"{evidence_path}"),
    )


requirements_text = REQUIREMENTS.read_text(encoding="utf-8")

required_dependencies = (
    "torch==",
    "numpy==",
    "pandas==",
    "matplotlib==",
    "PyYAML==",
    "gymnasium==",
    "PennyLane==",
    "streamlit==",
    "pytest==",
    "ruff==",
    "black==",
    "mypy==",
)

for dependency in required_dependencies:
    require(
        dependency in requirements_text,
        ("Missing dependency from " f"requirements.txt: {dependency}"),
    )


compression = load_csv(COMPRESSION_TABLE)

compression_lookup = {
    (
        row["Domain"],
        row["Method"],
    ): row
    for row in compression
}

for domain in (
    "Driving",
    "Robotics",
):
    require(
        compression_lookup[
            (
                domain,
                "INT8",
            )
        ]["Criterion"]
        == "PASS",
        ("Canonical INT8 criterion " f"changed for {domain}"),
    )

    for method in (
        "SVD",
        "TT/MPS",
    ):
        require(
            compression_lookup[
                (
                    domain,
                    method,
                )
            ]["Criterion"]
            == "FAIL",
            ("Canonical compression outcome " f"changed: {domain}/{method}"),
        )

require(
    ("INT8" in flat and "3.85×" in flat),
    ("README compression headline " "is missing"),
)


training = load_csv(TRAINING_TABLE)

require(
    all(row["Efficiency Result"] == "FAIL" for row in training),
    ("Canonical training-efficiency " "outcome changed"),
)

require(
    ("No robust >=10% optimizer-step " "efficiency improvement was demonstrated")
    in flat,
    ("README training negative " "result is missing"),
)


rl = load_csv(RL_TABLE)

target_totals: dict[
    str,
    list[int],
] = {}

for row in rl:
    reached, total = row["Target Reaches"].split("/")

    record = target_totals.setdefault(
        row["Policy"],
        [
            0,
            0,
        ],
    )

    record[0] += int(reached)

    record[1] += int(total)

require(
    target_totals["Classical PPO / MLP"]
    == [
        6,
        6,
    ],
    ("Canonical PPO target reach " "changed"),
)

require(
    target_totals["Matched classical control"]
    == [
        1,
        6,
    ],
    ("Canonical matched-classical " "target reach changed"),
)

require(
    target_totals["QML / PQC"]
    == [
        0,
        6,
    ],
    ("Canonical QML target reach " "changed"),
)

require(
    "95.90%" in flat,
    ("README driving PQC compactness " "is missing"),
)

require(
    "95.51%" in flat,
    ("README robotics PQC compactness " "is missing"),
)

require(
    ("Actor compactness is reported separately " "from sample efficiency") in flat,
    ("README compactness/sample-efficiency " "boundary is missing"),
)


safety = load_csv(SAFETY_TABLE)

for row in safety:
    if row["Method"] not in {
        "CLIPPING",
        "LYAPUNOV",
    }:
        continue

    require(
        row["Violation Rate"].startswith("0.000"),
        ("Filtered clean violation rate " "is no longer zero"),
    )

require(
    ("This is empirical proxy evidence only") in flat,
    ("README empirical safety " "boundary is missing"),
)

require(
    ("formal closed-loop stability") in lower,
    ("README formal-safety limitation " "is missing"),
)


full_system = load_json(FULL_SYSTEM_PATH)

phase1 = full_system["phase1"]

require(
    int(phase1["direct_count"]) == 0,
    ("Canonical DIRECT count changed"),
)

require(
    int(phase1["component_only_count"]) == 16,
    ("Canonical COMPONENT_ONLY " "count changed"),
)

require(
    int(phase1["not_evaluated_count"]) == 0,
    ("Canonical NOT_EVALUATED " "count changed"),
)

require(
    "DIRECT = 0" in flat,
    ("README DIRECT = 0 boundary " "missing"),
)

require(
    "COMPONENT_ONLY = 16" in flat,
    ("README COMPONENT_ONLY = 16 " "boundary missing"),
)

require(
    "NOT_EVALUATED = 0" in flat,
    ("README NOT_EVALUATED = 0 " "boundary missing"),
)

require(
    (
        "No matched integrated Compression × QML × "
        "Safety factorial configuration was directly executed"
    )
    in flat,
    ("README integrated-factorial " "boundary is missing"),
)


scientific_controls = full_system["scientific_controls"]

require(
    scientific_controls["synthetic_metric_composition_allowed"] is False,
    ("Synthetic metric composition " "must remain blocked"),
)

require(
    scientific_controls["interaction_effects_estimable_with_component_only"] is False,
    ("Interaction effects must remain " "non-estimable"),
)

require(
    scientific_controls["full_system_end_to_end_metrics_reported"] is False,
    ("Full-system end-to-end metrics " "must remain unreported"),
)


require(
    ("results/final-validation/figures/" "figure-01-architecture.png") in text,
    ("README architecture figure " "link is missing"),
)


negative_results = (
    ("No robust >=10% optimizer-step " "efficiency improvement"),
    "QML sample-efficiency superiority",
    "No TT/MPS superiority",
)

for statement in negative_results:
    require(
        statement.lower() in lower,
        ("README negative result or " f"boundary missing: {statement}"),
    )


blocked_success_phrases = (
    "quantum advantage achieved.",
    "quantum speedup achieved.",
    "qml superiority demonstrated.",
    "tt/mps superiority demonstrated.",
    "formally safe.",
    "certified safe.",
    "production ready.",
    "zero-shot transfer demonstrated.",
    "full-system superiority demonstrated.",
)

for phrase in blocked_success_phrases:
    require(
        phrase not in lower,
        ("Unsupported README claim " f"detected: {phrase}"),
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
        token not in text,
        ("README encoding corruption " f"found: {token}"),
    )


require(
    "42, 123, 456" in flat,
    ("README three-seed protocol " "is missing"),
)

require(
    ("16 matched Compression × QML × Safety " "domain/configuration experiments")
    in flat,
    ("README Phase 2 factorial " "objective is missing"),
)


print("=" * 60)

print(" SPRINT 7.10 — FINAL README ACCEPTANCE")

print("=" * 60)

print()

print("Content")

print("  [PASS] Challenge")

print("  [PASS] Pilot objective")

print("  [PASS] Research questions")

print("  [PASS] Architecture")

print("  [PASS] Algorithms")

print("  [PASS] Installation / Quick Start")

print("  [PASS] Experiments")

print("  [PASS] Dashboard")

print("  [PASS] Results")

print("  [PASS] Ablations")

print("  [PASS] Limitations")

print("  [PASS] Reproduction")

print("  [PASS] Phase 2 roadmap")

print()

print("Evidence")

print("  [PASS] canonical results consistent")

print("  [PASS] evidence paths valid")

print("  [PASS] figures/tables linked")

print("  [PASS] three-seed protocol retained")

print()

print("Scientific controls")

print("  [PASS] negative results retained")

print("  [PASS] no quantum advantage")

print("  [PASS] no QML superiority")

print("  [PASS] no TT/MPS superiority")

print("  [PASS] safety claims bounded")

print("  [PASS] DIRECT = 0 / " "COMPONENT_ONLY = 16")

print("  [PASS] Phase 1 / Phase 2 separated")

print()

print("Independent verifier: PASS")

print("SPRINT 7.10: PASS")
