"""Tests for Tensor Train / Matrix Product State equivalence."""

from __future__ import annotations

import math

import pytest
import torch

from q_vla_forge.compression import (
    mps_to_tt,
    reconstruct_mps,
    reconstruct_tt,
    tt_svd,
    tt_to_mps,
    verify_tt_mps_equivalence,
)


def _decomposition(
    rank: int = 4,
):
    """Return a deterministic TT decomposition for mapping tests."""
    torch.manual_seed(42)

    tensor = torch.randn(
        8,
        8,
        8,
    )

    return tt_svd(
        tensor,
        max_rank=rank,
    )


def test_tt_to_mps_preserves_shape() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    assert mps.original_shape == tt.original_shape


def test_tt_ranks_become_mps_bonds() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    assert mps.bond_dimensions == tt.ranks


def test_mps_boundary_bonds_are_one() -> None:
    mps = tt_to_mps(_decomposition())

    assert mps.bond_dimensions[0] == 1

    assert mps.bond_dimensions[-1] == 1


def test_mps_physical_dimensions_match_tensor_shape() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    assert mps.physical_dimensions == tt.original_shape


def test_tt_and_mps_parameter_counts_match() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    assert tt.stored_parameters == mps.stored_parameters


def test_tt_and_mps_reconstruction_match() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    tt_tensor = reconstruct_tt(tt)

    mps_tensor = reconstruct_mps(mps)

    assert torch.allclose(
        tt_tensor,
        mps_tensor,
        atol=1e-6,
        rtol=1e-6,
    )


def test_tt_to_mps_to_tt_round_trip() -> None:
    original = _decomposition()

    mps = tt_to_mps(original)

    restored = mps_to_tt(mps)

    assert restored.original_shape == original.original_shape

    assert restored.ranks == original.ranks

    assert restored.stored_parameters == original.stored_parameters

    for (
        expected,
        actual,
    ) in zip(
        original.cores,
        restored.cores,
        strict=True,
    ):
        assert torch.equal(
            expected,
            actual,
        )


def test_verification_passes() -> None:
    result = verify_tt_mps_equivalence(_decomposition())

    assert result.same_parameter_count
    assert result.same_bond_dimensions
    assert result.reconstruction_equivalent


def test_verification_difference_is_near_zero() -> None:
    result = verify_tt_mps_equivalence(_decomposition())

    assert math.isfinite(result.maximum_absolute_difference)

    assert result.maximum_absolute_difference <= 1e-6

    assert result.relative_difference <= 1e-6


@pytest.mark.parametrize(
    "rank",
    [
        2,
        4,
        8,
    ],
)
def test_equivalence_across_tt_ranks(
    rank: int,
) -> None:
    result = verify_tt_mps_equivalence(_decomposition(rank))

    assert result.reconstruction_equivalent
    assert result.same_parameter_count
    assert result.same_bond_dimensions


def test_mps_order_matches_core_count() -> None:
    mps = tt_to_mps(_decomposition())

    assert mps.order == len(mps.cores)


def test_mapping_clones_cores_by_default() -> None:
    tt = _decomposition()

    mps = tt_to_mps(tt)

    for (
        tt_core,
        mps_core,
    ) in zip(
        tt.cores,
        mps.cores,
        strict=True,
    ):
        assert torch.equal(
            tt_core,
            mps_core,
        )

        assert tt_core.data_ptr() != mps_core.data_ptr()


def test_negative_atol_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="atol must not be negative",
    ):
        verify_tt_mps_equivalence(
            _decomposition(),
            atol=-1.0,
        )


def test_negative_rtol_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="rtol must not be negative",
    ):
        verify_tt_mps_equivalence(
            _decomposition(),
            rtol=-1.0,
        )
