from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.evaluation.pilot_readiness import (
    SplitAudit,
    overlap_count,
    require_locked_domains,
    require_locked_seeds,
    sample_fingerprint,
    summarize_distribution,
)


def test_sample_fingerprint_is_deterministic() -> None:
    visual = np.zeros(
        (3, 4, 4),
        dtype=np.float32,
    )

    state = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    action = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float32,
    )

    first = sample_fingerprint(
        visual=visual,
        state=state,
        action=action,
        language_goal="test goal",
    )

    second = sample_fingerprint(
        visual=visual,
        state=state,
        action=action,
        language_goal="test goal",
    )

    assert first == second
    assert len(first) == 64


def test_sample_fingerprint_changes() -> None:
    visual = np.zeros(
        (3, 4, 4),
        dtype=np.float32,
    )

    state = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    action = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float32,
    )

    first = sample_fingerprint(
        visual=visual,
        state=state,
        action=action,
        language_goal="goal one",
    )

    second = sample_fingerprint(
        visual=visual,
        state=state,
        action=action,
        language_goal="goal two",
    )

    assert first != second


def test_overlap_count() -> None:
    assert (
        overlap_count(
            {"a", "b"},
            {"b", "c"},
        )
        == 1
    )


def test_split_audit_passes_clean_split() -> None:
    audit = SplitAudit(
        domain="robotics",
        seed=42,
        train_size=512,
        validation_size=128,
        test_size=128,
        train_unique=512,
        validation_unique=128,
        test_unique=128,
        train_validation_overlap=0,
        train_test_overlap=0,
        validation_test_overlap=0,
    )

    assert audit.passed


def test_split_audit_rejects_overlap() -> None:
    audit = SplitAudit(
        domain="robotics",
        seed=42,
        train_size=512,
        validation_size=128,
        test_size=128,
        train_unique=512,
        validation_unique=128,
        test_unique=128,
        train_validation_overlap=1,
        train_test_overlap=0,
        validation_test_overlap=0,
    )

    assert not audit.passed


def test_distribution_summary() -> None:
    values = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_distribution(values)

    assert summary.minimum == [
        0.0,
        1.0,
    ]

    assert summary.maximum == [
        2.0,
        3.0,
    ]

    assert summary.mean == [
        1.0,
        2.0,
    ]


def test_locked_seeds() -> None:
    require_locked_seeds(
        {
            42,
            123,
            456,
        }
    )


def test_wrong_seeds_fail() -> None:
    with pytest.raises(ValueError):
        require_locked_seeds(
            {
                42,
                123,
            }
        )


def test_locked_domains() -> None:
    require_locked_domains(
        {
            "autonomous_driving",
            "robotics",
        }
    )
