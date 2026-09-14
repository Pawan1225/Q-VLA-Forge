from __future__ import annotations

from pathlib import Path

import pytest

from q_vla_forge.evaluation.rl_final_gate import (
    artifact_to_dict,
    build_frozen_artifact,
    file_sha256,
)


def test_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "Q-VLA Forge",
        encoding="utf-8",
    )

    first = file_sha256(path)
    second = file_sha256(path)

    assert first == second
    assert len(first) == 64
    assert first == first.upper()


def test_artifact_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    record = build_frozen_artifact(
        path=path,
        sprint="4.15",
        role="test",
    )

    assert record.path == path.as_posix()
    assert record.sprint == "4.15"
    assert record.role == "test"
    assert record.required is True
    assert record.frozen is True
    assert record.size_bytes > 0
    assert len(record.sha256) == 64


def test_required_missing_artifact_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        build_frozen_artifact(
            path=path,
            sprint="4.15",
            role="required-test",
        )


def test_optional_missing_artifact_is_recorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "optional.json"

    record = build_frozen_artifact(
        path=path,
        sprint="4.15",
        role="optional-test",
        required=False,
    )

    assert record.required is False
    assert record.sha256 == ""
    assert record.size_bytes == 0


def test_artifact_to_dict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "freeze",
        encoding="utf-8",
    )

    record = build_frozen_artifact(
        path=path,
        sprint="4.15",
        role="serialization-test",
    )

    data = artifact_to_dict(record)

    assert data["path"] == path.as_posix()
    assert data["sprint"] == "4.15"
    assert data["role"] == "serialization-test"
    assert data["required"] is True
    assert data["frozen"] is True
    assert len(data["sha256"]) == 64
    assert data["size_bytes"] == path.stat().st_size
