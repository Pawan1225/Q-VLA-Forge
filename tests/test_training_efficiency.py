from __future__ import annotations

import pytest

from q_vla_forge.training.efficiency import (
    EpochEfficiencyRecord,
    TargetReachResult,
    build_target_definition,
    compare_target_efficiency,
    find_target_reach,
    percentage_reduction,
)


def _record(
    epoch: int,
    validation_loss: float,
) -> EpochEfficiencyRecord:
    return EpochEfficiencyRecord(
        epoch=epoch,
        train_loss=validation_loss + 0.01,
        validation_loss=validation_loss,
        learning_rate=0.001,
        epoch_seconds=float(epoch),
        cumulative_seconds=float(sum(range(1, epoch + 1))),
        optimizer_steps=epoch * 16,
        samples_processed=epoch * 512,
    )


def test_target_uses_five_percent_tolerance() -> None:
    target = build_target_definition(0.0200)

    assert target.target_validation_loss == pytest.approx(0.0210)


def test_custom_target_tolerance() -> None:
    target = build_target_definition(
        1.0,
        tolerance_fraction=0.10,
    )

    assert target.target_validation_loss == pytest.approx(1.10)


def test_negative_reference_loss_rejected() -> None:
    with pytest.raises(ValueError):
        build_target_definition(-0.1)


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        build_target_definition(
            1.0,
            tolerance_fraction=-0.01,
        )


def test_find_first_target_epoch() -> None:
    history = [
        _record(
            1,
            0.10,
        ),
        _record(
            2,
            0.07,
        ),
        _record(
            3,
            0.04,
        ),
        _record(
            4,
            0.03,
        ),
    ]

    result = find_target_reach(
        history,
        0.05,
    )

    assert result.reached_target
    assert result.epoch_to_target == 3
    assert result.steps_to_target == 48
    assert result.samples_to_target == 1536


def test_target_equality_counts_as_reached() -> None:
    history = [
        _record(
            1,
            0.05,
        )
    ]

    result = find_target_reach(
        history,
        0.05,
    )

    assert result.reached_target
    assert result.epoch_to_target == 1


def test_unreached_target_returns_none_metrics() -> None:
    history = [
        _record(
            1,
            0.10,
        ),
        _record(
            2,
            0.08,
        ),
    ]

    result = find_target_reach(
        history,
        0.05,
    )

    assert not result.reached_target
    assert result.epoch_to_target is None
    assert result.steps_to_target is None
    assert result.samples_to_target is None
    assert result.seconds_to_target is None


def test_empty_history_rejected() -> None:
    with pytest.raises(ValueError):
        find_target_reach(
            [],
            0.05,
        )


def test_percentage_reduction_positive() -> None:
    assert percentage_reduction(
        20.0,
        10.0,
    ) == pytest.approx(50.0)


def test_percentage_reduction_negative_when_worse() -> None:
    assert percentage_reduction(
        10.0,
        12.0,
    ) == pytest.approx(-20.0)


def test_zero_reference_rejected() -> None:
    with pytest.raises(ValueError):
        percentage_reduction(
            0.0,
            1.0,
        )


def test_compare_target_efficiency() -> None:
    reference = TargetReachResult(
        reached_target=True,
        epoch_to_target=10,
        steps_to_target=160,
        samples_to_target=5120,
        seconds_to_target=100.0,
    )

    candidate = TargetReachResult(
        reached_target=True,
        epoch_to_target=8,
        steps_to_target=128,
        samples_to_target=4096,
        seconds_to_target=90.0,
    )

    result = compare_target_efficiency(
        domain="autonomous_driving",
        seed=42,
        reference_method="fp32",
        candidate_method="tt_mps",
        reference=reference,
        candidate=candidate,
    )

    assert result.epoch_reduction_percent == pytest.approx(20.0)

    assert result.step_reduction_percent == pytest.approx(20.0)

    assert result.sample_reduction_percent == pytest.approx(20.0)

    assert result.wall_time_reduction_percent == pytest.approx(10.0)


def test_unreached_candidate_has_no_reduction_claim() -> None:
    reference = TargetReachResult(
        reached_target=True,
        epoch_to_target=10,
        steps_to_target=160,
        samples_to_target=5120,
        seconds_to_target=100.0,
    )

    candidate = TargetReachResult(
        reached_target=False,
        epoch_to_target=None,
        steps_to_target=None,
        samples_to_target=None,
        seconds_to_target=None,
    )

    result = compare_target_efficiency(
        domain="robotics",
        seed=42,
        reference_method="fp32",
        candidate_method="svd",
        reference=reference,
        candidate=candidate,
    )

    assert not (result.candidate_reached_target)

    assert result.epoch_reduction_percent is None

    assert result.step_reduction_percent is None

    assert result.sample_reduction_percent is None

    assert result.wall_time_reduction_percent is None


def test_epoch_must_start_at_one() -> None:
    with pytest.raises(ValueError):
        EpochEfficiencyRecord(
            epoch=0,
            train_loss=1.0,
            validation_loss=1.0,
            learning_rate=0.001,
            epoch_seconds=1.0,
            cumulative_seconds=1.0,
            optimizer_steps=16,
            samples_processed=512,
        )
