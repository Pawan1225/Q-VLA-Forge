"""Execution invariants for Sprint 5.11 structured robustness."""

from __future__ import annotations

import numpy as np

from q_vla_forge.safety.perturbations import (
    apply_structured_state_perturbation,
    structured_perturbation_by_name,
)


def test_same_structured_delta_across_method_reconstruction() -> None:
    state = np.asarray(
        [
            0.2,
            -0.1,
            0.05,
            0.25,
        ],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="autonomous_driving",
        name="obstacle_distance_plus_0p10",
    )

    outputs = []

    for _ in (
        "none",
        "clipping",
        "lyapunov",
    ):
        observed, delta = apply_structured_state_perturbation(
            true_state=state,
            perturbation=spec,
        )

        outputs.append(
            (
                observed,
                delta,
            )
        )

    for (
        observed,
        delta,
    ) in outputs[1:]:
        np.testing.assert_array_equal(
            observed,
            outputs[0][0],
        )

        np.testing.assert_array_equal(
            delta,
            outputs[0][1],
        )


def test_obstacle_overestimate_exact() -> None:
    state = np.asarray(
        [
            0.3,
            0.0,
            0.0,
            0.20,
        ],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="autonomous_driving",
        name="obstacle_distance_plus_0p10",
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    assert delta[3] == np.float32(0.10)

    assert observed[3] == np.float32(0.30)

    np.testing.assert_array_equal(
        state,
        np.asarray(
            [
                0.3,
                0.0,
                0.0,
                0.20,
            ],
            dtype=np.float32,
        ),
    )


def test_robot_position_underestimate_exact() -> None:
    state = np.asarray(
        [
            0.88,
            0.0,
            -0.2,
            0.3,
            0.5,
            -0.4,
        ],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="robotics",
        name="robot_x_minus_0p05",
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    assert delta[0] == np.float32(-0.05)

    assert observed[0] < state[0]

    np.testing.assert_array_equal(
        observed,
        state + delta,
    )


def test_only_declared_dimension_changes() -> None:
    state = np.asarray(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="robotics",
        name="target_y_minus_0p05",
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    changed = np.flatnonzero(delta).tolist()

    assert changed == [5]

    for index in range(5):
        assert observed[index] == state[index]


def test_structured_perturbation_has_no_randomness() -> None:
    state = np.asarray(
        [
            0.4,
            -0.2,
            0.1,
            0.8,
        ],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="autonomous_driving",
        name="lane_offset_plus_0p05",
    )

    results = [
        apply_structured_state_perturbation(
            true_state=state,
            perturbation=spec,
        )
        for _ in range(10)
    ]

    for observed, delta in results[1:]:
        np.testing.assert_array_equal(
            observed,
            results[0][0],
        )

        np.testing.assert_array_equal(
            delta,
            results[0][1],
        )
