"""Execution-level tests for Sprint 5.10 Gaussian robustness."""

from __future__ import annotations

import numpy as np

from q_vla_forge.safety.perturbations import (
    apply_gaussian_observation_noise,
    make_gaussian_rng,
)


def _noise_sequence(
    *,
    domain: str,
    dimension: int,
    sigma: float,
) -> list[np.ndarray]:
    rng, _ = make_gaussian_rng(
        domain=domain,
        principal_seed=42,
        evaluation_seed=20_000,
        sigma=sigma,
    )

    state = np.zeros(
        dimension,
        dtype=np.float32,
    )

    result: list[np.ndarray] = []

    for _ in range(10):
        _, noise = apply_gaussian_observation_noise(
            true_state=state,
            sigma=sigma,
            rng=rng,
        )

        result.append(noise.copy())

    return result


def test_paired_noise_stream_is_method_independent() -> None:
    none_stream = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.10,
    )

    clipping_stream = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.10,
    )

    lyapunov_stream = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.10,
    )

    for none, clipping, lyapunov in zip(
        none_stream,
        clipping_stream,
        lyapunov_stream,
        strict=True,
    ):
        np.testing.assert_array_equal(
            none,
            clipping,
        )

        np.testing.assert_array_equal(
            none,
            lyapunov,
        )


def test_noise_stream_advances_each_step() -> None:
    sequence = _noise_sequence(
        domain="robotics",
        dimension=6,
        sigma=0.10,
    )

    assert not np.array_equal(
        sequence[0],
        sequence[1],
    )


def test_sigma_zero_stream_is_exactly_zero() -> None:
    sequence = _noise_sequence(
        domain="robotics",
        dimension=6,
        sigma=0.0,
    )

    for noise in sequence:
        np.testing.assert_array_equal(
            noise,
            np.zeros(
                6,
                dtype=np.float32,
            ),
        )


def test_robotics_and_driving_have_distinct_streams() -> None:
    driving = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.10,
    )

    robotics = _noise_sequence(
        domain="robotics",
        dimension=6,
        sigma=0.10,
    )

    assert driving[0].shape == (4,)

    assert robotics[0].shape == (6,)


def test_reconstructed_episode_stream_is_exact() -> None:
    first = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.05,
    )

    second = _noise_sequence(
        domain="autonomous_driving",
        dimension=4,
        sigma=0.05,
    )

    assert len(first) == len(second)

    for left, right in zip(
        first,
        second,
        strict=True,
    ):
        np.testing.assert_array_equal(
            left,
            right,
        )
