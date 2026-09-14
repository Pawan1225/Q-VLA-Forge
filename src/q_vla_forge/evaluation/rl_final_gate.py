"""Final Sprint 4 freeze and verification utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenArtifact:
    path: str
    sprint: str
    role: str
    required: bool
    frozen: bool
    sha256: str
    size_bytes: int


def file_sha256(
    path: Path,
) -> str:
    """Return the uppercase SHA256 digest for a file."""

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def build_frozen_artifact(
    *,
    path: Path,
    sprint: str,
    role: str,
    required: bool = True,
    frozen: bool = True,
) -> FrozenArtifact:
    """Build a deterministic freeze record for an artifact."""

    if required and not path.exists():
        raise FileNotFoundError(path)

    normalized_path = path.as_posix()

    if not path.exists():
        return FrozenArtifact(
            path=normalized_path,
            sprint=sprint,
            role=role,
            required=required,
            frozen=frozen,
            sha256="",
            size_bytes=0,
        )

    if not path.is_file():
        raise ValueError(f"freeze artifact must be a file: {path}")

    return FrozenArtifact(
        path=normalized_path,
        sprint=sprint,
        role=role,
        required=required,
        frozen=frozen,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def artifact_to_dict(
    artifact: FrozenArtifact,
) -> dict[str, Any]:
    """Convert a frozen artifact record to JSON-compatible data."""

    return asdict(artifact)
