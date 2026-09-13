"""Sprint 3 final-manifest and artifact-integrity utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactHash:
    """SHA-256 identity for one Sprint 3 artifact."""

    path: str
    sha256: str
    size_bytes: int


def sha256_file(
    path: Path,
) -> ArtifactHash:
    """Return SHA-256 and size for one file."""
    if not path.exists():
        raise FileNotFoundError(f"artifact does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"artifact is not a file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return ArtifactHash(
        path=path.as_posix(),
        sha256=digest.hexdigest(),
        size_bytes=path.stat().st_size,
    )


def validate_unique_run_keys(
    keys: list[
        tuple[
            str,
            str,
            int,
        ]
    ],
) -> None:
    """Require unique domain/method/seed run keys."""
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate domain/method/seed run key detected")


def validate_seed_set(
    seeds: set[int],
) -> None:
    """Require the locked three-seed protocol."""
    expected = {
        42,
        123,
        456,
    }

    if seeds != expected:
        raise ValueError(
            f"expected seeds {sorted(expected)}, " f"found {sorted(seeds)}"
        )


def validate_primary_run_count(
    *,
    fp32_count: int,
    structured_count: int,
) -> None:
    """Require the complete 18-run validation matrix."""
    if fp32_count != 6:
        raise ValueError(f"expected 6 FP32 runs, found {fp32_count}")

    if structured_count != 12:
        raise ValueError("expected 12 structured runs, " f"found {structured_count}")

    if fp32_count + structured_count != 18:
        raise ValueError("expected 18 principal runs")


def validate_ablation_run_count(
    count: int,
) -> None:
    """Require four controlled SVD-vs-TT/MPS ablation runs."""
    if count != 4:
        raise ValueError(f"expected 4 ablation runs, found {count}")
