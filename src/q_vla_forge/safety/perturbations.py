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
