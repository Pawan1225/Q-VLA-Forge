from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRecord:
    """One frozen Sprint 2 evidence artifact."""

    path: str
    exists: bool
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Sprint2Manifest:
    """Machine-readable Sprint 2 compression manifest."""

    sprint: str
    status: str

    domains: tuple[str, ...]
    validation_seeds: tuple[int, ...]

    classical_methods: tuple[str, ...]
    quantum_inspired_methods: tuple[str, ...]

    minimum_compression_ratio: float
    maximum_mse_increase_percent: float

    primary_size_metric: str
    primary_quality_metric: str

    quantum_hardware_used: bool
    native_compressed_runtime_used: bool

    validation_result_count: int
    ablation_point_count: int

    artifacts: tuple[ArtifactRecord, ...]


def sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 digest for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def artifact_record(
    path: Path,
) -> ArtifactRecord:
    """Build one immutable artifact record."""
    exists = path.exists()

    if not exists:
        return ArtifactRecord(
            path=str(path),
            exists=False,
            size_bytes=0,
            sha256="",
        )

    if not path.is_file():
        raise ValueError(f"artifact is not a file: {path}")

    return ArtifactRecord(
        path=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def build_sprint2_manifest(
    *,
    artifact_paths: Sequence[Path],
    validation_result_count: int,
    ablation_point_count: int,
) -> Sprint2Manifest:
    """Build the frozen Sprint 2 manifest."""
    if validation_result_count != 18:
        raise ValueError(
            "Sprint 2 must contain exactly " "18 multi-seed validation results"
        )

    if ablation_point_count != 20:
        raise ValueError(
            "Sprint 2 must contain exactly " "20 target-level ablation points"
        )

    records = tuple(artifact_record(path) for path in artifact_paths)

    missing = tuple(record.path for record in records if not record.exists)

    if missing:
        raise FileNotFoundError("missing Sprint 2 artifacts: " + ", ".join(missing))

    empty = tuple(record.path for record in records if record.size_bytes <= 0)

    if empty:
        raise ValueError("empty Sprint 2 artifacts: " + ", ".join(empty))

    return Sprint2Manifest(
        sprint="2",
        status="complete",
        domains=(
            "autonomous_driving",
            "robotics",
        ),
        validation_seeds=(
            42,
            123,
            456,
        ),
        classical_methods=(
            "INT8",
            "SVD",
        ),
        quantum_inspired_methods=("TT/open-boundary MPS via TT-SVD",),
        minimum_compression_ratio=2.0,
        maximum_mse_increase_percent=5.0,
        primary_size_metric=("effective representation payload bytes"),
        primary_quality_metric=("relative test-MSE change"),
        quantum_hardware_used=False,
        native_compressed_runtime_used=False,
        validation_result_count=(validation_result_count),
        ablation_point_count=(ablation_point_count),
        artifacts=records,
    )


def save_sprint2_manifest(
    manifest: Sprint2Manifest,
    path: Path,
) -> None:
    """Save Sprint 2 manifest."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
        ),
        encoding="utf-8",
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))
