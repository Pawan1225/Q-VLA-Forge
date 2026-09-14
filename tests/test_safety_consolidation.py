"""Tests for Sprint 5.13 safety consolidation foundation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from q_vla_forge.evaluation.safety_consolidation import (
    make_manifest_entry,
    paired_seed_deltas,
    recovery_rate,
    sample_sd,
    three_seed_metric,
)


def test_sample_sd_three_values() -> None:
    assert sample_sd(
        [
            1.0,
            2.0,
            3.0,
        ]
    ) == pytest.approx(1.0)


def test_sample_sd_single_value() -> None:
    assert sample_sd([5.0]) == 0.0


def test_three_seed_metric() -> None:
    metric = three_seed_metric(
        seed42=1.0,
        seed123=2.0,
        seed456=3.0,
    )

    assert metric.mean == pytest.approx(2.0)

    assert metric.sample_sd == pytest.approx(1.0)


def test_paired_seed_deltas() -> None:
    result = paired_seed_deltas(
        baseline={
            42: 3.0,
            123: 4.0,
            456: 5.0,
        },
        comparison={
            42: 2.0,
            123: 4.5,
            456: 1.0,
        },
    )

    assert result == {
        42: -1.0,
        123: 0.5,
        456: -4.0,
    }


def test_recovery_rate() -> None:
    assert recovery_rate(
        unsafe_steps=10,
        recovered_steps=7,
    ) == pytest.approx(0.7)


def test_zero_denominator_recovery() -> None:
    assert (
        recovery_rate(
            unsafe_steps=0,
            recovered_steps=0,
        )
        is None
    )


def test_invalid_recovery() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        recovery_rate(
            unsafe_steps=5,
            recovered_steps=6,
        )


def test_manifest_hash(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"

    artifact.write_text(
        '{"ok": true}\n',
        encoding="utf-8",
    )

    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    entry = make_manifest_entry(
        root=tmp_path,
        path=artifact,
        sprint="5.12",
        role="test artifact",
        group="action_robustness",
    )

    assert entry.path == "artifact.json"
    assert entry.sha256 == expected
    assert entry.size_bytes > 0
    assert entry.required is True
    assert entry.frozen is True
