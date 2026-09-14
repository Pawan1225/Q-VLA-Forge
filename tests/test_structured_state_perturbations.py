"""Tests for Sprint 5.11 structured state perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.perturbations import (
    DRIVING_STRUCTURED_PERTURBATIONS,
    ROBOTICS_STRUCTURED_PERTURBATIONS,
    StructuredStatePerturbation,
    apply_structured_state_perturbation,
    perturb_structured_observation,
    structured_perturbation_by_name,
    structured_perturbations_for_domain,
)


def test_frozen_condition_counts() -> None:
    assert len(DRIVING_STRUCTURED_PERTURBATIONS) == 10

    assert len(ROBOTICS_STRUCTURED_PERTURBATIONS) == 12

    assert (
        len(DRIVING_STRUCTURED_PERTURBATIONS) + len(ROBOTICS_STRUCTURED_PERTURBATIONS)
        == 22
    )


def test_driving_names_are_unique() -> None:
    names = [item.name for item in DRIVING_STRUCTURED_PERTURBATIONS]

    assert len(names) == len(set(names))


def test_robotics_names_are_unique() -> None:
    names = [item.name for item in ROBOTICS_STRUCTURED_PERTURBATIONS]

    assert len(names) == len(set(names))


def test_positive_single_index_update() -> None:
    state = np.asarray(
        [0.2, -0.3, 0.1, 0.8],
        dtype=np.float32,
    )

    spec = StructuredStatePerturbation(
        name="test",
        family="test",
        updates=((1, 0.05),),
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    expected_delta = np.asarray(
        [0.0, 0.05, 0.0, 0.0],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        delta,
        expected_delta,
    )

    np.testing.assert_array_equal(
        observed,
        state + expected_delta,
    )


def test_negative_update() -> None:
    state = np.asarray(
        [0.2, 0.3],
        dtype=np.float32,
    )

    spec = StructuredStatePerturbation(
        name="test",
        family="test",
        updates=((0, -0.05),),
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    assert delta[0] == pytest.approx(-0.05)

    assert observed[0] == pytest.approx(0.15)


def test_sparse_multiple_updates() -> None:
    state = np.zeros(
        6,
        dtype=np.float32,
    )

    spec = StructuredStatePerturbation(
        name="multi",
        family="test",
        updates=(
            (0, 0.05),
            (4, -0.10),
        ),
    )

    observed, delta = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    expected = np.asarray(
        [
            0.05,
            0.0,
            0.0,
            0.0,
            -0.10,
            0.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        delta,
        expected,
    )

    np.testing.assert_array_equal(
        observed,
        expected,
    )


def test_negative_index_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        StructuredStatePerturbation(
            name="bad",
            family="test",
            updates=((-1, 0.05),),
        )


def test_duplicate_index_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        StructuredStatePerturbation(
            name="bad",
            family="test",
            updates=(
                (1, 0.05),
                (1, -0.05),
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_delta_rejected(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        StructuredStatePerturbation(
            name="bad",
            family="test",
            updates=((0, value),),
        )


def test_out_of_range_index_rejected_at_apply() -> None:
    state = np.zeros(
        4,
        dtype=np.float32,
    )

    spec = StructuredStatePerturbation(
        name="bad",
        family="test",
        updates=((4, 0.05),),
    )

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        apply_structured_state_perturbation(
            true_state=state,
            perturbation=spec,
        )


def test_input_is_not_mutated() -> None:
    state = np.asarray(
        [0.4, -0.2, 0.1, 0.8],
        dtype=np.float32,
    )

    before = state.copy()

    spec = StructuredStatePerturbation(
        name="test",
        family="test",
        updates=((3, 0.10),),
    )

    apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    np.testing.assert_array_equal(
        state,
        before,
    )


def test_no_observation_clipping() -> None:
    state = np.asarray(
        [0.0, 0.0, 0.0, 0.98],
        dtype=np.float32,
    )

    spec = StructuredStatePerturbation(
        name="test",
        family="test",
        updates=((3, 0.10),),
    )

    observed, _ = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    assert observed[3] > 1.0


def test_exact_unaffected_dimensions() -> None:
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

    spec = StructuredStatePerturbation(
        name="test",
        family="test",
        updates=((2, 0.05),),
    )

    observed, _ = apply_structured_state_perturbation(
        true_state=state,
        perturbation=spec,
    )

    for index in (
        0,
        1,
        3,
        4,
        5,
    ):
        assert observed[index] == state[index]


def test_deterministic() -> None:
    state = np.asarray(
        [0.2, -0.1, 0.3, 0.9],
        dtype=np.float32,
    )

    spec = structured_perturbation_by_name(
        domain="autonomous_driving",
        name="obstacle_distance_plus_0p10",
    )

    first = perturb_structured_observation(
        true_state=state,
        perturbation=spec,
    )

    second = perturb_structured_observation(
        true_state=state,
        perturbation=spec,
    )

    np.testing.assert_array_equal(
        first.observed_state,
        second.observed_state,
    )

    np.testing.assert_array_equal(
        first.perturbation_vector,
        second.perturbation_vector,
    )


def test_domain_lookup() -> None:
    assert len(structured_perturbations_for_domain("autonomous_driving")) == 10

    assert len(structured_perturbations_for_domain("robotics")) == 12


def test_unknown_domain_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported domain",
    ):
        structured_perturbations_for_domain("unknown")


def test_unknown_name_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unknown structured perturbation",
    ):
        structured_perturbation_by_name(
            domain="robotics",
            name="unknown",
        )
