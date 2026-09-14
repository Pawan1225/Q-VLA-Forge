"""Deterministic perturbation utilities for Sprint 5 robustness."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

GAUSSIAN_NOISE_LEVELS: tuple[float, ...] = (
    0.00,
    0.01,
    0.05,
    0.10,
)

GAUSSIAN_PRINCIPAL_NOISY_LEVELS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
)

GAUSSIAN_NOISE_NAMESPACE = "q-vla-forge-sprint5.10-gaussian"


@dataclass(frozen=True)
class GaussianObservationPerturbation:
    """Frozen Gaussian observation perturbation definition."""

    sigma: float
    noise_seed: int

    def __post_init__(self) -> None:
        validate_gaussian_sigma(self.sigma)

        if self.noise_seed < 0:
            raise ValueError("noise_seed must be non-negative")


@dataclass(frozen=True)
class GaussianPerturbationResult:
    """One deterministic observation perturbation."""

    true_state: FloatArray
    observed_state: FloatArray
    noise_vector: FloatArray
    sigma: float
    noise_seed: int


def validate_gaussian_sigma(
    sigma: float,
) -> None:
    """Validate Gaussian standard deviation."""

    if not math.isfinite(sigma):
        raise ValueError("sigma must be finite")

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")


def gaussian_sigma_index(
    sigma: float,
) -> int:
    """Return frozen index for a supported sigma."""

    validate_gaussian_sigma(sigma)

    for index, level in enumerate(GAUSSIAN_NOISE_LEVELS):
        if sigma == level:
            return index

    raise ValueError("sigma must be one of " f"{GAUSSIAN_NOISE_LEVELS}")


def gaussian_noise_seed(
    *,
    domain: str,
    principal_seed: int,
    evaluation_seed: int,
    sigma_index: int,
) -> int:
    """Derive a deterministic method-independent RNG seed."""

    if not domain:
        raise ValueError("domain must be non-empty")

    if principal_seed < 0:
        raise ValueError("principal_seed must be non-negative")

    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be non-negative")

    if sigma_index < 0:
        raise ValueError("sigma_index must be non-negative")

    key = (
        f"{GAUSSIAN_NOISE_NAMESPACE}|"
        f"{domain}|"
        f"{principal_seed}|"
        f"{evaluation_seed}|"
        f"{sigma_index}"
    )

    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def make_gaussian_rng(
    *,
    domain: str,
    principal_seed: int,
    evaluation_seed: int,
    sigma: float,
) -> tuple[np.random.Generator, int]:
    """Create the deterministic RNG for one episode."""

    sigma_index = gaussian_sigma_index(sigma)

    seed = gaussian_noise_seed(
        domain=domain,
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        sigma_index=sigma_index,
    )

    return (
        np.random.default_rng(seed),
        seed,
    )


def apply_gaussian_observation_noise(
    *,
    true_state: NDArray[np.floating],
    sigma: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Apply observation-only Gaussian noise.

    The true state is copied, never mutated, and the noisy
    observation is deliberately not clipped.
    """

    validate_gaussian_sigma(sigma)

    state = np.asarray(
        true_state,
        dtype=np.float32,
    )

    if state.ndim != 1:
        raise ValueError("true_state must be one-dimensional")

    if state.size == 0:
        raise ValueError("true_state must not be empty")

    if not np.all(np.isfinite(state)):
        raise ValueError("true_state must contain finite values")

    true_copy = state.copy()

    if sigma == 0.0:
        noise = np.zeros_like(
            true_copy,
            dtype=np.float32,
        )
    else:
        noise = np.asarray(
            rng.normal(
                loc=0.0,
                scale=sigma,
                size=true_copy.shape,
            ),
            dtype=np.float32,
        )

    observed = (true_copy + noise).astype(
        np.float32,
        copy=False,
    )

    return observed, noise


def perturb_gaussian_observation(
    *,
    true_state: NDArray[np.floating],
    domain: str,
    principal_seed: int,
    evaluation_seed: int,
    sigma: float,
) -> GaussianPerturbationResult:
    """Apply the complete Sprint 5.10 noise contract."""

    rng, noise_seed = make_gaussian_rng(
        domain=domain,
        principal_seed=principal_seed,
        evaluation_seed=evaluation_seed,
        sigma=sigma,
    )

    state_copy = np.asarray(
        true_state,
        dtype=np.float32,
    ).copy()

    observed_state, noise_vector = apply_gaussian_observation_noise(
        true_state=state_copy,
        sigma=sigma,
        rng=rng,
    )

    return GaussianPerturbationResult(
        true_state=state_copy,
        observed_state=observed_state,
        noise_vector=noise_vector,
        sigma=float(sigma),
        noise_seed=noise_seed,
    )


# ============================================================
# Sprint 5.11 structured state perturbations
# ============================================================


@dataclass(frozen=True)
class StructuredStatePerturbation:
    """Deterministic sparse perturbation of policy state input."""

    name: str
    family: str
    updates: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("perturbation name must be non-empty")

        if not self.family.strip():
            raise ValueError("perturbation family must be non-empty")

        if not self.updates:
            raise ValueError(
                "structured perturbation must contain " "at least one update"
            )

        indices: set[int] = set()

        for index, delta in self.updates:
            if index < 0:
                raise ValueError("perturbation index must be nonnegative")

            if index in indices:
                raise ValueError("duplicate perturbation index")

            indices.add(index)

            if not np.isfinite(delta):
                raise ValueError("perturbation delta must be finite")


@dataclass(frozen=True)
class StructuredPerturbationResult:
    """Auditable result of deterministic state perturbation."""

    true_state: FloatArray
    observed_state: FloatArray
    perturbation_vector: FloatArray
    perturbation_name: str
    perturbation_family: str


DRIVING_STRUCTURED_PERTURBATIONS = (
    StructuredStatePerturbation(
        name="lane_offset_plus_0p05",
        family="lane_offset",
        updates=((1, 0.05),),
    ),
    StructuredStatePerturbation(
        name="lane_offset_minus_0p05",
        family="lane_offset",
        updates=((1, -0.05),),
    ),
    StructuredStatePerturbation(
        name="lane_offset_plus_0p10",
        family="lane_offset",
        updates=((1, 0.10),),
    ),
    StructuredStatePerturbation(
        name="lane_offset_minus_0p10",
        family="lane_offset",
        updates=((1, -0.10),),
    ),
    StructuredStatePerturbation(
        name="obstacle_distance_plus_0p05",
        family="obstacle_distance",
        updates=((3, 0.05),),
    ),
    StructuredStatePerturbation(
        name="obstacle_distance_plus_0p10",
        family="obstacle_distance",
        updates=((3, 0.10),),
    ),
    StructuredStatePerturbation(
        name="speed_plus_0p05",
        family="speed",
        updates=((0, 0.05),),
    ),
    StructuredStatePerturbation(
        name="speed_minus_0p05",
        family="speed",
        updates=((0, -0.05),),
    ),
    StructuredStatePerturbation(
        name="heading_error_plus_0p05",
        family="heading_error",
        updates=((2, 0.05),),
    ),
    StructuredStatePerturbation(
        name="heading_error_minus_0p05",
        family="heading_error",
        updates=((2, -0.05),),
    ),
)


ROBOTICS_STRUCTURED_PERTURBATIONS = (
    StructuredStatePerturbation(
        name="robot_x_plus_0p05",
        family="robot_position",
        updates=((0, 0.05),),
    ),
    StructuredStatePerturbation(
        name="robot_x_minus_0p05",
        family="robot_position",
        updates=((0, -0.05),),
    ),
    StructuredStatePerturbation(
        name="robot_y_plus_0p05",
        family="robot_position",
        updates=((1, 0.05),),
    ),
    StructuredStatePerturbation(
        name="robot_y_minus_0p05",
        family="robot_position",
        updates=((1, -0.05),),
    ),
    StructuredStatePerturbation(
        name="object_x_plus_0p05",
        family="object_position",
        updates=((2, 0.05),),
    ),
    StructuredStatePerturbation(
        name="object_x_minus_0p05",
        family="object_position",
        updates=((2, -0.05),),
    ),
    StructuredStatePerturbation(
        name="object_y_plus_0p05",
        family="object_position",
        updates=((3, 0.05),),
    ),
    StructuredStatePerturbation(
        name="object_y_minus_0p05",
        family="object_position",
        updates=((3, -0.05),),
    ),
    StructuredStatePerturbation(
        name="target_x_plus_0p05",
        family="target_position",
        updates=((4, 0.05),),
    ),
    StructuredStatePerturbation(
        name="target_x_minus_0p05",
        family="target_position",
        updates=((4, -0.05),),
    ),
    StructuredStatePerturbation(
        name="target_y_plus_0p05",
        family="target_position",
        updates=((5, 0.05),),
    ),
    StructuredStatePerturbation(
        name="target_y_minus_0p05",
        family="target_position",
        updates=((5, -0.05),),
    ),
)


def structured_perturbations_for_domain(
    domain: str,
) -> tuple[StructuredStatePerturbation, ...]:
    """Return the frozen Sprint 5.11 perturbation set."""

    if domain == "autonomous_driving":
        return DRIVING_STRUCTURED_PERTURBATIONS

    if domain == "robotics":
        return ROBOTICS_STRUCTURED_PERTURBATIONS

    raise ValueError(f"unsupported domain: {domain}")


def structured_perturbation_by_name(
    *,
    domain: str,
    name: str,
) -> StructuredStatePerturbation:
    """Resolve one frozen perturbation by exact name."""

    matches = tuple(
        perturbation
        for perturbation in structured_perturbations_for_domain(domain)
        if perturbation.name == name
    )

    if len(matches) != 1:
        raise ValueError(f"unknown structured perturbation " f"for {domain}: {name}")

    return matches[0]


def apply_structured_state_perturbation(
    *,
    true_state: np.ndarray,
    perturbation: StructuredStatePerturbation,
) -> tuple[FloatArray, FloatArray]:
    """Apply a deterministic sparse observation perturbation.

    The true simulator state is never modified.

    No clipping is applied to the observed state.
    """

    state = np.asarray(
        true_state,
        dtype=np.float32,
    )

    if state.ndim != 1:
        raise ValueError("true_state must be one-dimensional")

    if state.size == 0:
        raise ValueError("true_state must be non-empty")

    if not np.all(np.isfinite(state)):
        raise ValueError("true_state must contain only finite values")

    true_copy = state.copy()

    perturbation_vector = np.zeros_like(
        true_copy,
        dtype=np.float32,
    )

    for index, delta in perturbation.updates:
        if index >= true_copy.size:
            raise ValueError(
                "perturbation index out of range: "
                f"{index} for state size {true_copy.size}"
            )

        perturbation_vector[index] = np.float32(delta)

    observed_state = true_copy + perturbation_vector

    return (
        observed_state.astype(
            np.float32,
            copy=False,
        ),
        perturbation_vector,
    )


def perturb_structured_observation(
    *,
    true_state: np.ndarray,
    perturbation: StructuredStatePerturbation,
) -> StructuredPerturbationResult:
    """Return the full structured perturbation audit contract."""

    observed_state, perturbation_vector = apply_structured_state_perturbation(
        true_state=true_state,
        perturbation=perturbation,
    )

    true_copy = np.asarray(
        true_state,
        dtype=np.float32,
    ).copy()

    return StructuredPerturbationResult(
        true_state=true_copy,
        observed_state=observed_state,
        perturbation_vector=perturbation_vector,
        perturbation_name=perturbation.name,
        perturbation_family=perturbation.family,
    )


# ============================================================
# Sprint 5.12 structured action perturbations
# ============================================================


@dataclass(frozen=True)
class StructuredActionPerturbation:
    """Deterministic sparse perturbation of a policy action."""

    name: str
    family: str
    updates: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("perturbation name must be non-empty")

        if not self.family.strip():
            raise ValueError("perturbation family must be non-empty")

        if not self.updates:
            raise ValueError("action perturbation must contain at least one update")

        indices: set[int] = set()

        for index, delta in self.updates:
            if index < 0:
                raise ValueError("perturbation index must be nonnegative")

            if index in indices:
                raise ValueError("duplicate perturbation index")

            indices.add(index)

            if not np.isfinite(delta):
                raise ValueError("perturbation delta must be finite")


@dataclass(frozen=True)
class StructuredActionPerturbationResult:
    """Auditable result of deterministic action perturbation."""

    proposed_action: FloatArray
    perturbed_action: FloatArray
    perturbation_vector: FloatArray
    perturbation_name: str
    perturbation_family: str


DRIVING_ACTION_PERTURBATIONS = (
    StructuredActionPerturbation(
        name="steering_plus_0p10",
        family="steering",
        updates=((0, 0.10),),
    ),
    StructuredActionPerturbation(
        name="steering_minus_0p10",
        family="steering",
        updates=((0, -0.10),),
    ),
    StructuredActionPerturbation(
        name="steering_plus_0p25",
        family="steering",
        updates=((0, 0.25),),
    ),
    StructuredActionPerturbation(
        name="steering_minus_0p25",
        family="steering",
        updates=((0, -0.25),),
    ),
    StructuredActionPerturbation(
        name="acceleration_plus_0p10",
        family="acceleration",
        updates=((1, 0.10),),
    ),
    StructuredActionPerturbation(
        name="acceleration_minus_0p10",
        family="acceleration",
        updates=((1, -0.10),),
    ),
    StructuredActionPerturbation(
        name="acceleration_plus_0p25",
        family="acceleration",
        updates=((1, 0.25),),
    ),
    StructuredActionPerturbation(
        name="acceleration_minus_0p25",
        family="acceleration",
        updates=((1, -0.25),),
    ),
    StructuredActionPerturbation(
        name="braking_plus_0p10",
        family="braking",
        updates=((2, 0.10),),
    ),
    StructuredActionPerturbation(
        name="braking_minus_0p10",
        family="braking",
        updates=((2, -0.10),),
    ),
    StructuredActionPerturbation(
        name="braking_plus_0p25",
        family="braking",
        updates=((2, 0.25),),
    ),
    StructuredActionPerturbation(
        name="braking_minus_0p25",
        family="braking",
        updates=((2, -0.25),),
    ),
)


ROBOTICS_ACTION_PERTURBATIONS = (
    StructuredActionPerturbation(
        name="delta_x_plus_0p10",
        family="delta_x",
        updates=((0, 0.10),),
    ),
    StructuredActionPerturbation(
        name="delta_x_minus_0p10",
        family="delta_x",
        updates=((0, -0.10),),
    ),
    StructuredActionPerturbation(
        name="delta_x_plus_0p25",
        family="delta_x",
        updates=((0, 0.25),),
    ),
    StructuredActionPerturbation(
        name="delta_x_minus_0p25",
        family="delta_x",
        updates=((0, -0.25),),
    ),
    StructuredActionPerturbation(
        name="delta_y_plus_0p10",
        family="delta_y",
        updates=((1, 0.10),),
    ),
    StructuredActionPerturbation(
        name="delta_y_minus_0p10",
        family="delta_y",
        updates=((1, -0.10),),
    ),
    StructuredActionPerturbation(
        name="delta_y_plus_0p25",
        family="delta_y",
        updates=((1, 0.25),),
    ),
    StructuredActionPerturbation(
        name="delta_y_minus_0p25",
        family="delta_y",
        updates=((1, -0.25),),
    ),
    StructuredActionPerturbation(
        name="gripper_plus_0p25",
        family="gripper",
        updates=((2, 0.25),),
    ),
    StructuredActionPerturbation(
        name="gripper_minus_0p25",
        family="gripper",
        updates=((2, -0.25),),
    ),
    StructuredActionPerturbation(
        name="gripper_plus_0p50",
        family="gripper",
        updates=((2, 0.50),),
    ),
    StructuredActionPerturbation(
        name="gripper_minus_0p50",
        family="gripper",
        updates=((2, -0.50),),
    ),
)


def action_perturbations_for_domain(
    domain: str,
) -> tuple[StructuredActionPerturbation, ...]:
    """Return the frozen Sprint 5.12 action perturbation set."""

    if domain == "autonomous_driving":
        return DRIVING_ACTION_PERTURBATIONS

    if domain == "robotics":
        return ROBOTICS_ACTION_PERTURBATIONS

    raise ValueError(f"unsupported domain: {domain}")


def action_perturbation_by_name(
    *,
    domain: str,
    name: str,
) -> StructuredActionPerturbation:
    """Resolve one frozen action perturbation by exact name."""

    matches = tuple(
        perturbation
        for perturbation in action_perturbations_for_domain(domain)
        if perturbation.name == name
    )

    if len(matches) != 1:
        raise ValueError(f"unknown action perturbation for {domain}: {name}")

    return matches[0]


def apply_structured_action_perturbation(
    *,
    proposed_action: np.ndarray,
    perturbation: StructuredActionPerturbation,
) -> tuple[FloatArray, FloatArray]:
    """Apply deterministic action disturbance without clipping."""

    action = np.asarray(
        proposed_action,
        dtype=np.float32,
    )

    if action.ndim != 1:
        raise ValueError("proposed_action must be one-dimensional")

    if action.size == 0:
        raise ValueError("proposed_action must be non-empty")

    if not np.all(np.isfinite(action)):
        raise ValueError("proposed_action must contain only finite values")

    proposed_copy = action.copy()

    perturbation_vector = np.zeros_like(
        proposed_copy,
        dtype=np.float32,
    )

    for index, delta in perturbation.updates:
        if index >= proposed_copy.size:
            raise ValueError(
                "perturbation index out of range: "
                f"{index} for action size {proposed_copy.size}"
            )

        perturbation_vector[index] = np.float32(delta)

    perturbed_action = proposed_copy + perturbation_vector

    return (
        perturbed_action.astype(
            np.float32,
            copy=False,
        ),
        perturbation_vector,
    )


def perturb_structured_action(
    *,
    proposed_action: np.ndarray,
    perturbation: StructuredActionPerturbation,
) -> StructuredActionPerturbationResult:
    """Return the full Sprint 5.12 action audit result."""

    (
        perturbed_action,
        perturbation_vector,
    ) = apply_structured_action_perturbation(
        proposed_action=proposed_action,
        perturbation=perturbation,
    )

    proposed_copy = np.asarray(
        proposed_action,
        dtype=np.float32,
    ).copy()

    return StructuredActionPerturbationResult(
        proposed_action=proposed_copy,
        perturbed_action=perturbed_action,
        perturbation_vector=perturbation_vector,
        perturbation_name=perturbation.name,
        perturbation_family=perturbation.family,
    )
