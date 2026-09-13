"""Matched classical-vs-quantum-inspired training ablation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterBudget:
    """One structured representation parameter budget."""

    method: str
    configuration: str
    trainable_parameters: int


@dataclass(frozen=True)
class MatchedParameterPair:
    """Classical SVD and QI TT/MPS parameter-budget match."""

    svd: ParameterBudget
    tt_mps: ParameterBudget
    absolute_parameter_difference: int
    relative_difference_percent: float


def relative_parameter_difference_percent(
    left: int,
    right: int,
) -> float:
    """Return symmetric-ish relative difference using larger budget."""
    if left <= 0 or right <= 0:
        raise ValueError("parameter counts must be positive")

    denominator = max(
        left,
        right,
    )

    return (abs(left - right) / denominator) * 100.0


def select_closest_parameter_pair(
    svd_budgets: Iterable[ParameterBudget],
    tt_budgets: Iterable[ParameterBudget],
) -> MatchedParameterPair:
    """
    Select the closest SVD/TT budget without using performance data.

    Ties are resolved deterministically by:
    1. smaller relative parameter difference
    2. smaller absolute difference
    3. fewer combined parameters
    4. lexicographic configuration labels
    """
    svd_values = list(svd_budgets)

    tt_values = list(tt_budgets)

    if not svd_values:
        raise ValueError("svd_budgets must not be empty")

    if not tt_values:
        raise ValueError("tt_budgets must not be empty")

    candidates: list[
        tuple[
            tuple[
                float,
                int,
                int,
                str,
                str,
            ],
            MatchedParameterPair,
        ]
    ] = []

    for svd in svd_values:
        if svd.method != "trainable_svd":
            raise ValueError("invalid SVD method label")

        for tt in tt_values:
            if tt.method != "trainable_tt_mps":
                raise ValueError("invalid TT/MPS method label")

            absolute = abs(svd.trainable_parameters - tt.trainable_parameters)

            relative = relative_parameter_difference_percent(
                svd.trainable_parameters,
                tt.trainable_parameters,
            )

            pair = MatchedParameterPair(
                svd=svd,
                tt_mps=tt,
                absolute_parameter_difference=(absolute),
                relative_difference_percent=(relative),
            )

            key = (
                relative,
                absolute,
                (svd.trainable_parameters + tt.trainable_parameters),
                svd.configuration,
                tt.configuration,
            )

            candidates.append(
                (
                    key,
                    pair,
                )
            )

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]
