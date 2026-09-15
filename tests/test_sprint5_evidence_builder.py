from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_DIR = ROOT / "results" / "safety" / "evidence"

JSON_PATH = EVIDENCE_DIR / "sprint5-safety-evidence.json"

CSV_PATH = EVIDENCE_DIR / "sprint5-safety-evidence.csv"

MARKDOWN_PATH = EVIDENCE_DIR / "sprint5-safety-evidence.md"

CLAIM_MATRIX = EVIDENCE_DIR / "sprint5-safety-claim-matrix.json"

FIGURE_INDEX = EVIDENCE_DIR / "sprint5-safety-figure-index.json"

MANIFEST = EVIDENCE_DIR / "sprint5-safety-manifest.json"


def load_json(
    path: Path,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(
        payload,
        dict,
    )

    return payload


def test_core_outputs_exist() -> None:
    for path in (
        JSON_PATH,
        CSV_PATH,
        MARKDOWN_PATH,
        CLAIM_MATRIX,
        FIGURE_INDEX,
        MANIFEST,
    ):
        assert path.is_file()

        assert path.stat().st_size > 0


def test_evidence_scope_frozen() -> None:
    payload = load_json(JSON_PATH)

    metadata = payload["metadata"]

    assert metadata["new_training"] is False

    assert metadata["new_principal_execution"] is False

    assert metadata["new_scientific_experiment"] is False

    assert metadata["scientific_scope_frozen"] is True


def test_csv_has_expected_schema() -> None:
    with CSV_PATH.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        rows = list(reader)

    assert rows

    assert set(rows[0]) == {
        "domain",
        "regime",
        "condition",
        "method",
        "seed",
        "metric",
        "value",
        "reference_value",
        "delta",
        "unit",
        "source_artifact",
        "source_sha256",
    }


def test_csv_principal_seeds_only() -> None:
    with CSV_PATH.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert {row["seed"] for row in rows} == {
        "42",
        "123",
        "456",
    }


def test_clean_filters_zero() -> None:
    payload = load_json(JSON_PATH)

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "clipping",
            "lyapunov",
        ):
            value = float(
                payload["clean_safety"][domain][method]["violation_step_rate"]["mean"]
            )

            assert abs(value) <= 1e-12


def test_action_arithmetic() -> None:
    payload = load_json(JSON_PATH)

    action = payload["action_robustness"]["global"]

    unsafe = int(action["unsafe_perturbed_steps"])

    recovered = int(action["recovered_unsafe_steps"])

    unresolved = int(action["unresolved_unsafe_steps"])

    assert unsafe == (recovered + unresolved)


def test_figure_count() -> None:
    payload = load_json(FIGURE_INDEX)

    assert payload["figure_count"] == 5

    assert len(payload["figures"]) == 5


def test_claim_matrix_ids() -> None:
    payload = load_json(CLAIM_MATRIX)

    ids = {row["claim_id"] for row in payload["proposal_claims"]}

    assert ids == {
        "S5-E01",
        "S5-E02",
        "S5-E03",
        "S5-E04",
        "S5-E05",
        "S5-E06",
    }
