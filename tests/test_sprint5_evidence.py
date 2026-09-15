from __future__ import annotations

import math
from pathlib import Path

import pytest

from q_vla_forge.evaluation.sprint5_evidence import (
    ActionRecoveryResult,
    LyapunovMechanismResult,
    SeedMetric,
    aggregate_seed_values,
    artifact_record,
    recovery_rate,
    relative_reduction,
    validate_action_recovery,
    validate_mechanism_result,
    validate_scope_flags,
)


def test_sample_sd_uses_three_principal_seeds() -> None:
    result = aggregate_seed_values(
        (
            SeedMetric(
                seed=42,
                value=1.0,
            ),
            SeedMetric(
                seed=123,
                value=2.0,
            ),
            SeedMetric(
                seed=456,
                value=3.0,
            ),
        )
    )

    assert result.mean == 2.0
    assert result.sample_sd == pytest.approx(1.0)


def test_seed_order_is_normalized() -> None:
    result = aggregate_seed_values(
        (
            SeedMetric(
                seed=456,
                value=3.0,
            ),
            SeedMetric(
                seed=42,
                value=1.0,
            ),
            SeedMetric(
                seed=123,
                value=2.0,
            ),
        )
    )

    assert tuple(item.seed for item in result.seed_values) == (
        42,
        123,
        456,
    )


def test_wrong_seed_protocol_fails() -> None:
    with pytest.raises(
        ValueError,
        match="Expected principal seeds",
    ):
        aggregate_seed_values(
            (
                SeedMetric(
                    seed=1,
                    value=1.0,
                ),
                SeedMetric(
                    seed=2,
                    value=2.0,
                ),
                SeedMetric(
                    seed=3,
                    value=3.0,
                ),
            )
        )


def test_relative_reduction() -> None:
    assert relative_reduction(
        0.5,
        0.25,
    ) == pytest.approx(0.5)


def test_zero_denominator_relative_reduction() -> None:
    assert (
        relative_reduction(
            0.0,
            0.0,
        )
        is None
    )


def test_recovery_rate() -> None:
    assert recovery_rate(
        100,
        75,
    ) == pytest.approx(0.75)


def test_zero_denominator_recovery() -> None:
    assert (
        recovery_rate(
            0,
            0,
        )
        is None
    )


def test_recovery_cannot_exceed_unsafe() -> None:
    with pytest.raises(ValueError):
        recovery_rate(
            10,
            11,
        )


def test_action_recovery_arithmetic() -> None:
    result = ActionRecoveryResult(
        domain="autonomous_driving",
        method="clipping",
        unsafe_perturbed_steps=10,
        recovered_steps=7,
        unresolved_steps=3,
        recovery_rate=0.7,
    )

    validate_action_recovery(result)


def test_action_recovery_bad_arithmetic_fails() -> None:
    result = ActionRecoveryResult(
        domain="autonomous_driving",
        method="clipping",
        unsafe_perturbed_steps=10,
        recovered_steps=7,
        unresolved_steps=2,
        recovery_rate=0.7,
    )

    with pytest.raises(ValueError):
        validate_action_recovery(result)


def test_mechanism_decomposition() -> None:
    result = LyapunovMechanismResult(
        domain="autonomous_driving",
        regime="action",
        action_bound=10,
        domain_constraint=20,
        lyapunov_decrease=5,
        emergency_fallback=0,
        strict_decrease=2,
        total_interventions=35,
    )

    validate_mechanism_result(result)


def test_strict_decrease_is_separate() -> None:
    result = LyapunovMechanismResult(
        domain="autonomous_driving",
        regime="action",
        action_bound=0,
        domain_constraint=0,
        lyapunov_decrease=100,
        emergency_fallback=0,
        strict_decrease=3,
        total_interventions=100,
    )

    validate_mechanism_result(result)

    assert result.lyapunov_decrease != result.strict_decrease


def test_scope_flags_pass() -> None:
    validate_scope_flags(
        {
            "new_training": False,
            "new_principal_execution": False,
            "new_scientific_experiment": False,
            "scientific_scope_frozen": True,
        }
    )


def test_scope_training_violation_fails() -> None:
    with pytest.raises(ValueError):
        validate_scope_flags(
            {
                "new_training": True,
                "new_principal_execution": False,
                "new_scientific_experiment": False,
                "scientific_scope_frozen": True,
            }
        )


def test_artifact_hashing(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "evidence.json"

    artifact.write_text(
        '{"value": 1}\n',
        encoding="utf-8",
    )

    record = artifact_record(
        artifact,
        role="test",
    )

    assert len(record.sha256) == 64
    assert record.size_bytes > 0
    assert record.required is True
    assert not math.isnan(float(record.size_bytes))
