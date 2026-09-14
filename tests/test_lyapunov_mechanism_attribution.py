"""Tests for Sprint 5.13F Lyapunov mechanism attribution."""

from q_vla_forge.evaluation.lyapunov_mechanism_attribution import (
    RegimeMechanism,
    attribution_label,
    mechanism_active,
)


def test_inactive_mechanism() -> None:
    assert (
        mechanism_active(
            lyapunov_decrease_reasons=0,
            strict_decrease_steps=0,
        )
        is False
    )


def test_reason_activity() -> None:
    assert (
        mechanism_active(
            lyapunov_decrease_reasons=1,
            strict_decrease_steps=0,
        )
        is True
    )


def test_strict_decrease_activity() -> None:
    assert (
        mechanism_active(
            lyapunov_decrease_reasons=0,
            strict_decrease_steps=1,
        )
        is True
    )


def test_attribution_label() -> None:
    regime = RegimeMechanism(
        regime="action_perturbation",
        lyapunov_decrease_reasons=1584,
        strict_decrease_steps=70,
        selected_lower_steps=4887,
        emergency_fallback_steps=0,
        mechanism_active=True,
    )

    assert attribution_label(regime) == "active"
