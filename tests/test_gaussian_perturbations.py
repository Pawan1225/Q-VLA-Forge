"""Tests for Sprint 5.10 Gaussian perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.perturbations import (
    GAUSSIAN_NOISE_LEVELS,
    GAUSSIAN_PRINCIPAL_NOISY_LEVELS,
    apply_gaussian_observation_noise,
    gaussian_noise_seed,
    gaussian_sigma_index,
    make_gaussian_rng,
    perturb_gaussian_observation,
    validate_gaussian_sigma,
)


def test_frozen_noise_levels() -> None:
    assert GAUSSIAN_NOISE_LEVELS == (
        0.0,
        0.01,
        0.05,
        0.10,
    )

    assert GAUSSIAN_PRINCIPAL_NOISY_LEVELS == (
        0.01,
        0.05,
        0.10,
    )


@pytest.mark.parametrize(
    "sigma",
    [0.0, 0.01, 0.05, 0.10],
)
def test_valid_sigma(
    sigma: float,
) -> None:
    validate_gaussian_sigma(sigma)


@pytest.mark.parametrize(
    "sigma",
    [
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_sigma(
    sigma: float,
) -> None:
    with pytest.raises(ValueError):
        validate_gaussian_sigma(sigma)


def test_sigma_index() -> None:
    assert gaussian_sigma_index(0.0) == 0
    assert gaussian_sigma_index(0.01) == 1
    assert gaussian_sigma_index(0.05) == 2
    assert gaussian_sigma_index(0.10) == 3

    with pytest.raises(ValueError):
        gaussian_sigma_index(0.02)


def test_noise_seed_deterministic() -> None:
    kwargs = {
        "domain": "autonomous_driving",
        "principal_seed": 42,
        "evaluation_seed": 20000,
        "sigma_index": 3,
    }

    assert gaussian_noise_seed(**kwargs) == gaussian_noise_seed(**kwargs)


def test_noise_seed_changes_with_inputs() -> None:
    base = gaussian_noise_seed(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=20000,
        sigma_index=3,
    )

    assert base != gaussian_noise_seed(
        domain="robotics",
        principal_seed=42,
        evaluation_seed=20000,
        sigma_index=3,
    )

    assert base != gaussian_noise_seed(
        domain="autonomous_driving",
        principal_seed=123,
        evaluation_seed=20000,
        sigma_index=3,
    )

    assert base != gaussian_noise_seed(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=20001,
        sigma_index=3,
    )


def test_sigma_zero_exact_identity() -> None:
    state = np.asarray(
        [0.4, -0.2, 0.1, 0.8],
        dtype=np.float32,
    )

    rng = np.random.default_rng(42)

    observed, noise = apply_gaussian_observation_noise(
        true_state=state,
        sigma=0.0,
        rng=rng,
    )

    np.testing.assert_array_equal(
        observed,
        state,
    )

    np.testing.assert_array_equal(
        noise,
        np.zeros_like(state),
    )


def test_true_state_not_mutated() -> None:
    state = np.asarray(
        [0.4, -0.2, 0.1, 0.8],
        dtype=np.float32,
    )
    original = state.copy()

    rng = np.random.default_rng(42)

    apply_gaussian_observation_noise(
        true_state=state,
        sigma=0.10,
        rng=rng,
    )

    np.testing.assert_array_equal(
        state,
        original,
    )


def test_observed_equals_true_plus_noise() -> None:
    state = np.asarray(
        [0.4, -0.2, 0.1, 0.8],
        dtype=np.float32,
    )

    rng = np.random.default_rng(42)

    observed, noise = apply_gaussian_observation_noise(
        true_state=state,
        sigma=0.10,
        rng=rng,
    )

    np.testing.assert_array_equal(
        observed,
        state + noise,
    )


def test_same_episode_contract_reproduces_noise() -> None:
    state = np.zeros(
        6,
        dtype=np.float32,
    )

    first = perturb_gaussian_observation(
        true_state=state,
        domain="robotics",
        principal_seed=42,
        evaluation_seed=20000,
        sigma=0.10,
    )

    second = perturb_gaussian_observation(
        true_state=state,
        domain="robotics",
        principal_seed=42,
        evaluation_seed=20000,
        sigma=0.10,
    )

    assert first.noise_seed == second.noise_seed

    np.testing.assert_array_equal(
        first.noise_vector,
        second.noise_vector,
    )

    np.testing.assert_array_equal(
        first.observed_state,
        second.observed_state,
    )


def test_rng_reconstruction() -> None:
    first_rng, first_seed = make_gaussian_rng(
        domain="robotics",
        principal_seed=456,
        evaluation_seed=20010,
        sigma=0.05,
    )

    second_rng, second_seed = make_gaussian_rng(
        domain="robotics",
        principal_seed=456,
        evaluation_seed=20010,
        sigma=0.05,
    )

    assert first_seed == second_seed

    np.testing.assert_array_equal(
        first_rng.normal(size=20),
        second_rng.normal(size=20),
    )


def test_no_silent_observation_clipping() -> None:
    state = np.asarray(
        [0.99, -0.99, 0.99, -0.99],
        dtype=np.float32,
    )

    rng = np.random.default_rng(1)

    observed, noise = apply_gaussian_observation_noise(
        true_state=state,
        sigma=1.0,
        rng=rng,
    )

    np.testing.assert_array_equal(
        observed,
        state + noise,
    )

    assert bool(np.any(observed > 1.0) or np.any(observed < -1.0))


@pytest.mark.parametrize(
    "dimension",
    [4, 6],
)
def test_dimension_preserved(
    dimension: int,
) -> None:
    state = np.zeros(
        dimension,
        dtype=np.float32,
    )

    rng = np.random.default_rng(123)

    observed, noise = apply_gaussian_observation_noise(
        true_state=state,
        sigma=0.10,
        rng=rng,
    )

    assert observed.shape == (dimension,)
    assert noise.shape == (dimension,)
