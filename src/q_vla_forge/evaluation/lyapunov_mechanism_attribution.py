"""Sprint 5.13F cross-regime Lyapunov mechanism attribution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeMechanism:
    regime: str
    lyapunov_decrease_reasons: int
    strict_decrease_steps: int
    selected_lower_steps: int | None
    emergency_fallback_steps: int | None
    mechanism_active: bool


def mechanism_active(
    *,
    lyapunov_decrease_reasons: int,
    strict_decrease_steps: int,
) -> bool:
    return lyapunov_decrease_reasons > 0 or strict_decrease_steps > 0


def attribution_label(
    regime: RegimeMechanism,
) -> str:
    if regime.mechanism_active:
        return "active"

    return "inactive"
