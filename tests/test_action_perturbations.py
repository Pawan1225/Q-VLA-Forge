"""Tests for Sprint 5.12 structured action perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.perturbations import (
    DRIVING_ACTION_PERTURBATIONS,
    ROBOTICS_ACTION_PERTURBATIONS,
    StructuredActionPerturbation,
    action_perturbation_by_name,
    action_perturbations_for_domain,
    apply_structured_action_perturbation,
    perturb_structured_action,
)


def test_frozen_condition_counts() -> None:
    assert len(DRIVING_ACTION_PERTURBATIONS) == 12

    assert len(ROBOTICS_ACTION_PERTURBATIONS) == 12

    assert len(DRIVING_ACTION_PERTURBATIONS) + len(ROBOTICS_ACTION_PERTURBATIONS) == 24


def test_positive_action_offset() -> None:
    proposed = np.asarray(
        [
            0.90,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    spec = StructuredActionPerturbation(
        name="test",
        family="test",
        updates=((0, 0.25),),
    )

    perturbed, delta = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=spec,
    )

    assert delta[0] == pytest.approx(0.25)

    assert perturbed[0] == pytest.approx(1.15)


def test_negative_action_offset() -> None:
    proposed = np.asarray(
        [
            0.0,
            0.2,
            0.5,
        ],
        dtype=np.float32,
    )

    spec = StructuredActionPerturbation(
        name="test",
        family="test",
        updates=((2, -0.50),),
    )

    perturbed, delta = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=spec,
    )

    assert delta[2] == pytest.approx(-0.50)

    assert perturbed[2] == pytest.approx(0.0)


def test_no_pre_filter_clipping() -> None:
    proposed = np.asarray(
        [
            0.90,
            0.95,
            0.95,
        ],
        dtype=np.float32,
    )

    spec = StructuredActionPerturbation(
        name="test",
        family="test",
        updates=((0, 0.25),),
    )

    perturbed, _ = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=spec,
    )

    assert perturbed[0] > 1.0


def test_sparse_action_update() -> None:
    proposed = np.asarray(
        [
            0.1,
            0.2,
            0.3,
        ],
        dtype=np.float32,
    )

    spec = StructuredActionPerturbation(
        name="test",
        family="test",
        updates=((1, -0.10),),
    )

    perturbed, delta = apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=spec,
    )

    np.testing.assert_array_equal(
        delta,
        np.asarray(
            [
                0.0,
                -0.10,
                0.0,
            ],
            dtype=np.float32,
        ),
    )

    assert perturbed[0] == proposed[0]
    assert perturbed[2] == proposed[2]


def test_duplicate_index_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        StructuredActionPerturbation(
            name="bad",
            family="test",
            updates=(
                (0, 0.10),
                (0, -0.10),
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
        StructuredActionPerturbation(
            name="bad",
            family="test",
            updates=((0, value),),
        )


def test_out_of_range_index_rejected() -> None:
    proposed = np.zeros(
        3,
        dtype=np.float32,
    )

    spec = StructuredActionPerturbation(
        name="bad",
        family="test",
        updates=((3, 0.10),),
    )

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        apply_structured_action_perturbation(
            proposed_action=proposed,
            perturbation=spec,
        )


def test_input_not_mutated() -> None:
    proposed = np.asarray(
        [
            0.2,
            -0.3,
            0.4,
        ],
        dtype=np.float32,
    )

    before = proposed.copy()

    spec = action_perturbation_by_name(
        domain="robotics",
        name="gripper_plus_0p50",
    )

    apply_structured_action_perturbation(
        proposed_action=proposed,
        perturbation=spec,
    )

    np.testing.assert_array_equal(
        proposed,
        before,
    )


def test_deterministic() -> None:
    proposed = np.asarray(
        [
            0.2,
            -0.3,
            0.4,
        ],
        dtype=np.float32,
    )

    spec = action_perturbation_by_name(
        domain="autonomous_driving",
        name="steering_plus_0p25",
    )

    first = perturb_structured_action(
        proposed_action=proposed,
        perturbation=spec,
    )

    second = perturb_structured_action(
        proposed_action=proposed,
        perturbation=spec,
    )

    np.testing.assert_array_equal(
        first.perturbed_action,
        second.perturbed_action,
    )

    np.testing.assert_array_equal(
        first.perturbation_vector,
        second.perturbation_vector,
    )


def test_domain_lookup() -> None:
    assert len(action_perturbations_for_domain("autonomous_driving")) == 12

    assert len(action_perturbations_for_domain("robotics")) == 12
