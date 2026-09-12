from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.sprint2_manifest import (
    artifact_record,
    build_sprint2_manifest,
    save_sprint2_manifest,
    sha256_file,
)


def test_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.txt"

    path.write_text(
        "q-vla-forge",
        encoding="utf-8",
    )

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_existing_artifact_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    record = artifact_record(path)

    assert record.exists
    assert record.size_bytes > 0
    assert record.sha256


def test_missing_artifact_record(
    tmp_path: Path,
) -> None:
    record = artifact_record(tmp_path / "missing.json")

    assert not record.exists
    assert record.size_bytes == 0
    assert record.sha256 == ""


def test_manifest_builds(
    tmp_path: Path,
) -> None:
    paths = []

    for index in range(3):
        path = tmp_path / f"artifact-{index}.json"

        path.write_text(
            json.dumps({"index": index}),
            encoding="utf-8",
        )

        paths.append(path)

    manifest = build_sprint2_manifest(
        artifact_paths=paths,
        validation_result_count=18,
        ablation_point_count=20,
    )

    assert manifest.status == "complete"

    assert manifest.validation_seeds == (
        42,
        123,
        456,
    )

    assert manifest.validation_result_count == 18

    assert manifest.ablation_point_count == 20


def test_manifest_has_two_classical_methods(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_sprint2_manifest(
        artifact_paths=[path],
        validation_result_count=18,
        ablation_point_count=20,
    )

    assert manifest.classical_methods == (
        "INT8",
        "SVD",
    )


def test_tt_mps_is_one_family(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_sprint2_manifest(
        artifact_paths=[path],
        validation_result_count=18,
        ablation_point_count=20,
    )

    assert len(manifest.quantum_inspired_methods) == 1


def test_quantum_hardware_false(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_sprint2_manifest(
        artifact_paths=[path],
        validation_result_count=18,
        ablation_point_count=20,
    )

    assert not manifest.quantum_hardware_used


def test_native_runtime_false(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_sprint2_manifest(
        artifact_paths=[path],
        validation_result_count=18,
        ablation_point_count=20,
    )

    assert not (manifest.native_compressed_runtime_used)


def test_wrong_validation_count_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        build_sprint2_manifest(
            artifact_paths=[path],
            validation_result_count=17,
            ablation_point_count=20,
        )


def test_wrong_ablation_count_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        build_sprint2_manifest(
            artifact_paths=[path],
            validation_result_count=18,
            ablation_point_count=19,
        )


def test_missing_artifact_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        build_sprint2_manifest(
            artifact_paths=[tmp_path / "missing.json"],
            validation_result_count=18,
            ablation_point_count=20,
        )


def test_manifest_can_be_saved(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"

    artifact.write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_sprint2_manifest(
        artifact_paths=[artifact],
        validation_result_count=18,
        ablation_point_count=20,
    )

    output = tmp_path / "manifest.json"

    save_sprint2_manifest(
        manifest,
        output,
    )

    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "complete"
