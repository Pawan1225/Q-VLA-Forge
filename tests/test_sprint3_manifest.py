from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.sprint3_manifest import (
    sha256_file,
    validate_ablation_run_count,
    validate_primary_run_count,
    validate_seed_set,
    validate_unique_run_keys,
)

MANIFEST = Path("results") / "training" / "sprint3-training-manifest.json"


def test_sha256_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "q-vla-forge",
        encoding="utf-8",
    )

    result = sha256_file(path)

    assert len(result.sha256) == 64

    assert result.size_bytes > 0


def test_locked_seed_set() -> None:
    validate_seed_set(
        {
            42,
            123,
            456,
        }
    )


def test_wrong_seed_set_rejected() -> None:
    with pytest.raises(ValueError):
        validate_seed_set(
            {
                42,
                123,
            }
        )


def test_primary_run_count() -> None:
    validate_primary_run_count(
        fp32_count=6,
        structured_count=12,
    )


def test_invalid_primary_run_count_rejected() -> None:
    with pytest.raises(ValueError):
        validate_primary_run_count(
            fp32_count=6,
            structured_count=11,
        )


def test_ablation_run_count() -> None:
    validate_ablation_run_count(4)


def test_duplicate_run_key_rejected() -> None:
    with pytest.raises(ValueError):
        validate_unique_run_keys(
            [
                (
                    "robotics",
                    "trainable_svd",
                    42,
                ),
                (
                    "robotics",
                    "trainable_svd",
                    42,
                ),
            ]
        )


def test_generated_manifest_primary_matrix() -> None:
    if not MANIFEST.exists():
        pytest.skip("Sprint 3 manifest not generated")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    matrix = payload["primary_run_matrix"]

    assert matrix["fp32_runs"] == 6

    assert matrix["structured_runs"] == 12

    assert matrix["total_runs"] == 18


def test_generated_manifest_total_training_runs() -> None:
    if not MANIFEST.exists():
        pytest.skip("Sprint 3 manifest not generated")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["total_scientific_training_runs"] == 22


def test_generated_manifest_claim_controls() -> None:
    if not MANIFEST.exists():
        pytest.skip("Sprint 3 manifest not generated")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    control = payload["quantum_claim_control"]

    assert control["quantum_hardware_used"] is False

    assert control["quantum_advantage_claimed"] is False

    assert control["quantum_speedup_claimed"] is False


def test_generated_manifest_language_encoder_wording() -> None:
    if not MANIFEST.exists():
        pytest.skip("Sprint 3 manifest not generated")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert (
        payload["architecture_claim_control"]["language_component"]
        == "lightweight trainable text encoder"
    )


def test_generated_manifest_contains_hashes() -> None:
    if not MANIFEST.exists():
        pytest.skip("Sprint 3 manifest not generated")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["artifacts"]

    for artifact in payload["artifacts"]:
        assert len(artifact["sha256"]) == 64
