"""Open-boundary Matrix Product State mapping for Tensor Train cores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from q_vla_forge.compression.tensor_train import (
    TTDecomposition,
    reconstruct_tt,
)


@dataclass(frozen=True)
class MPSRepresentation:
    """Open-boundary Matrix Product State representation."""

    original_shape: tuple[int, ...]
    bond_dimensions: tuple[int, ...]
    cores: tuple[torch.Tensor, ...]

    @property
    def order(self) -> int:
        """Return the number of MPS sites."""
        return len(self.cores)

    @property
    def stored_parameters(self) -> int:
        """Return total scalar coefficients stored in MPS cores."""
        return sum(int(core.numel()) for core in self.cores)

    @property
    def physical_dimensions(
        self,
    ) -> tuple[int, ...]:
        """Return the physical dimension at each MPS site."""
        return tuple(int(core.shape[1]) for core in self.cores)


@dataclass(frozen=True)
class TTMPSVerification:
    """Structural and numerical TT/MPS equivalence evidence."""

    order: int
    tensor_shape: tuple[int, ...]

    tt_ranks: tuple[int, ...]
    mps_bond_dimensions: tuple[int, ...]

    tt_parameters: int
    mps_parameters: int

    maximum_absolute_difference: float
    relative_difference: float

    same_parameter_count: bool
    same_bond_dimensions: bool
    reconstruction_equivalent: bool


def _validate_mps(
    representation: MPSRepresentation,
) -> None:
    """Validate open-boundary MPS structure."""
    if not representation.cores:
        raise ValueError("MPS representation must contain cores")

    if len(representation.original_shape) != len(representation.cores):
        raise ValueError("original_shape and core count must match")

    if len(representation.bond_dimensions) != len(representation.cores) + 1:
        raise ValueError("bond_dimensions must contain order + 1 entries")

    if representation.bond_dimensions[0] != 1:
        raise ValueError("left boundary bond must equal one")

    if representation.bond_dimensions[-1] != 1:
        raise ValueError("right boundary bond must equal one")

    for index, core in enumerate(representation.cores):
        if core.ndim != 3:
            raise ValueError("every MPS core must be three-dimensional")

        left_bond = int(core.shape[0])

        physical_dimension = int(core.shape[1])

        right_bond = int(core.shape[2])

        if left_bond != representation.bond_dimensions[index]:
            raise ValueError("core left bond does not match MPS bonds")

        if right_bond != representation.bond_dimensions[index + 1]:
            raise ValueError("core right bond does not match MPS bonds")

        if physical_dimension != representation.original_shape[index]:
            raise ValueError("core physical dimension does not match tensor shape")


def tt_to_mps(
    decomposition: TTDecomposition,
    *,
    clone_cores: bool = True,
) -> MPSRepresentation:
    """
    Map a Tensor Train decomposition to open-boundary MPS form.

    TT core layout:
        [left_rank, physical_mode, right_rank]

    This is the same three-index layout used by an
    open-boundary MPS site tensor.
    """
    if not decomposition.cores:
        raise ValueError("TT decomposition must contain cores")

    if clone_cores:
        cores = tuple(core.detach().clone() for core in decomposition.cores)
    else:
        cores = decomposition.cores

    representation = MPSRepresentation(
        original_shape=(decomposition.original_shape),
        bond_dimensions=(decomposition.ranks),
        cores=cores,
    )

    _validate_mps(representation)

    return representation


def mps_to_tt(
    representation: MPSRepresentation,
    *,
    clone_cores: bool = True,
) -> TTDecomposition:
    """Convert an open-boundary MPS into Tensor Train form."""
    _validate_mps(representation)

    if clone_cores:
        cores = tuple(core.detach().clone() for core in representation.cores)
    else:
        cores = representation.cores

    return TTDecomposition(
        original_shape=(representation.original_shape),
        ranks=(representation.bond_dimensions),
        cores=cores,
    )


def reconstruct_mps(
    representation: MPSRepresentation,
) -> torch.Tensor:
    """Reconstruct the dense tensor represented by an MPS."""
    decomposition = mps_to_tt(
        representation,
        clone_cores=False,
    )

    return reconstruct_tt(decomposition)


def verify_tt_mps_equivalence(
    decomposition: TTDecomposition,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> TTMPSVerification:
    """Verify structural and numerical TT/MPS equivalence."""
    if atol < 0.0:
        raise ValueError("atol must not be negative")

    if rtol < 0.0:
        raise ValueError("rtol must not be negative")

    mps = tt_to_mps(decomposition)

    tt_reconstruction = reconstruct_tt(decomposition)

    mps_reconstruction = reconstruct_mps(mps)

    difference = tt_reconstruction - mps_reconstruction

    maximum_absolute_difference = float(difference.abs().max().item())

    numerator = torch.linalg.norm(difference)

    denominator = torch.linalg.norm(tt_reconstruction)

    denominator_value = float(denominator.item())

    if denominator_value == 0.0:
        if float(numerator.item()) == 0.0:
            relative_difference = 0.0
        else:
            relative_difference = math.inf
    else:
        relative_difference = float((numerator / denominator).item())

    same_parameter_count = decomposition.stored_parameters == mps.stored_parameters

    same_bond_dimensions = decomposition.ranks == mps.bond_dimensions

    reconstruction_equivalent = bool(
        torch.allclose(
            tt_reconstruction,
            mps_reconstruction,
            atol=atol,
            rtol=rtol,
        )
    )

    return TTMPSVerification(
        order=len(decomposition.original_shape),
        tensor_shape=(decomposition.original_shape),
        tt_ranks=(decomposition.ranks),
        mps_bond_dimensions=(mps.bond_dimensions),
        tt_parameters=(decomposition.stored_parameters),
        mps_parameters=(mps.stored_parameters),
        maximum_absolute_difference=(maximum_absolute_difference),
        relative_difference=(relative_difference),
        same_parameter_count=(same_parameter_count),
        same_bond_dimensions=(same_bond_dimensions),
        reconstruction_equivalent=(reconstruction_equivalent),
    )
