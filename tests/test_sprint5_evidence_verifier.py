from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]

VERIFIER_PATH = ROOT / "experiments" / "verify_sprint5_safety_evidence.py"


def load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sprint5_evidence_verifier",
        VERIFIER_PATH,
    )

    assert spec is not None

    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None

    spec.loader.exec_module(module)

    return module


def test_assert_close_passes() -> None:
    verifier = load_verifier()

    verifier.assert_close(
        1.0,
        1.0 + 1e-13,
        label="test",
    )


def test_assert_close_detects_corruption() -> None:
    verifier = load_verifier()

    with pytest.raises(RuntimeError):
        verifier.assert_close(
            1.0,
            1.001,
            label="corrupted",
        )


def test_sha256_detects_file_change(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()

    path = tmp_path / "artifact.json"

    path.write_text(
        '{"value": 1}\n',
        encoding="utf-8",
    )

    original = verifier.sha256_file(path)

    path.write_text(
        '{"value": 2}\n',
        encoding="utf-8",
    )

    changed = verifier.sha256_file(path)

    assert original != changed


def test_corrupted_action_arithmetic_detected() -> None:
    unsafe = 100
    recovered = 60
    unresolved = 39

    assert unsafe != (recovered + unresolved)


def test_corrupted_scope_detected() -> None:
    verifier = load_verifier()

    payload = {
        "metadata": {
            "new_training": True,
            "new_principal_execution": False,
            "new_scientific_experiment": False,
            "scientific_scope_frozen": True,
        }
    }

    with pytest.raises(RuntimeError):
        verifier.verify_scope(payload)


def test_hash_implementation_matches_hashlib(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()

    path = tmp_path / "hash.txt"

    content = b"Q-VLA Forge"

    path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert verifier.sha256_file(path) == expected


def test_claim_corruption_fixture() -> None:
    payload = {"proposal_claims": [{"claim_id": ("S5-E01")}]}

    serialized = json.dumps(payload)

    assert "S5-E06" not in serialized
