from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.safety.lyapunov_candidates import (
    ACTION_DEDUPLICATION_TOLERANCE,
    DRIVING_HARD_GUARD_NAMES,
    MAX_DRIVING_CANDIDATES,
    MAX_ROBOTICS_CANDIDATES,
    ROBOTICS_HARD_GUARD_NAMES,
    evaluate_driving_hard_guards,
    evaluate_hard_guards,
    evaluate_robotics_hard_guards,
    generate_driving_candidates,
    generate_robotics_candidates,
)


def test_candidate_deduplication_tolerance_is_frozen() -> None:
    assert ACTION_DEDUPLICATION_TOLERANCE == 1.0e-12


def test_driving_hard_guard_names_are_exact() -> None:
    assert DRIVING_HARD_GUARD_NAMES == frozenset(
        {
            "steering_risk",
            "acceleration_braking_conflict",
        }
    )


def test_robotics_hard_guard_names_are_exact() -> None:
    assert ROBOTICS_HARD_GUARD_NAMES == frozenset(
        {
            "unsafe_motion",
            "unsafe_gripper_condition",
        }
    )


def test_driving_candidate_order_starts_with_bounded_proposed() -> None:
    state = np.array(
        [
            0.4,
            0.2,
            0.1,
            1.0,
        ],
        dtype=np.float64,
    )

    proposed = np.array(
        [
            2.0,
            -2.0,
            2.0,
        ],
        dtype=np.float64,
    )

    candidates = generate_driving_candidates(
        state=state,
        proposed_action=proposed,
    )

    assert candidates[0].source == "bounded_proposed"

    np.testing.assert_array_equal(
        candidates[0].action,
        np.array(
            [
                1.0,
                -1.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )


def test_driving_candidate_indices_are_contiguous() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.5,
                0.8,
                0.4,
                0.5,
            ]
        ),
        proposed_action=np.array(
            [
                0.7,
                0.4,
                0.2,
            ]
        ),
    )

    assert [candidate.index for candidate in candidates] == list(range(len(candidates)))


def test_driving_candidates_are_unique() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.3,
                0.0,
                0.0,
                1.0,
            ]
        ),
        proposed_action=np.zeros(3),
    )

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            assert not np.all(
                np.abs(left.action - right.action) <= ACTION_DEDUPLICATION_TOLERANCE
            )


def test_driving_candidate_ceiling() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.7,
                0.9,
                0.6,
                0.3,
            ]
        ),
        proposed_action=np.array(
            [
                0.9,
                0.8,
                0.1,
            ]
        ),
    )

    assert len(candidates) <= MAX_DRIVING_CANDIDATES


def test_positive_heading_generates_negative_inward_steering() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.5,
                0.8,
                0.6,
                1.0,
            ]
        ),
        proposed_action=np.array(
            [
                0.5,
                0.2,
                0.0,
            ]
        ),
    )

    inward = next(
        candidate for candidate in candidates if candidate.source == "inward_steer"
    )

    assert inward.action[0] == -0.8


def test_negative_heading_generates_positive_inward_steering() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.5,
                -0.8,
                -0.6,
                1.0,
            ]
        ),
        proposed_action=np.array(
            [
                -0.5,
                0.2,
                0.0,
            ]
        ),
    )

    inward = next(
        candidate for candidate in candidates if candidate.source == "inward_steer"
    )

    assert inward.action[0] == 0.8


def test_driving_candidate_contains_full_brake() -> None:
    candidates = generate_driving_candidates(
        state=np.array(
            [
                0.8,
                0.4,
                0.2,
                0.2,
            ]
        ),
        proposed_action=np.array(
            [
                0.3,
                0.8,
                0.0,
            ]
        ),
    )

    assert any(candidate.action[2] == 1.0 for candidate in candidates)


def test_driving_hard_guard_rejects_steering_risk() -> None:
    result = evaluate_driving_hard_guards(
        state=np.array(
            [
                0.4,
                0.7,
                0.6,
                1.0,
            ]
        ),
        action=np.array(
            [
                0.9,
                0.0,
                0.0,
            ]
        ),
    )

    assert result.passed is False

    assert "steering_risk" in result.violated_guard_names


def test_driving_hard_guard_rejects_accel_brake_conflict() -> None:
    result = evaluate_driving_hard_guards(
        state=np.array(
            [
                0.4,
                0.0,
                0.0,
                1.0,
            ]
        ),
        action=np.array(
            [
                0.0,
                0.5,
                0.5,
            ]
        ),
    )

    assert result.passed is False

    assert "acceleration_braking_conflict" in result.violated_guard_names


def test_driving_state_only_violation_does_not_fail_hard_guard() -> None:
    result = evaluate_driving_hard_guards(
        state=np.array(
            [
                0.5,
                0.9,
                0.0,
                0.1,
            ]
        ),
        action=np.array(
            [
                0.0,
                0.0,
                1.0,
            ]
        ),
    )

    assert result.passed is True

    assert result.violated_guard_names == ()

    assert any(
        record.violated and record.name == "lane_boundary"
        for record in result.violations
    )


def test_robotics_candidate_order_starts_with_bounded_proposed() -> None:
    candidates = generate_robotics_candidates(
        state=np.array(
            [
                0.2,
                -0.3,
                0.5,
                0.5,
                0.0,
                0.0,
            ]
        ),
        proposed_action=np.array(
            [
                2.0,
                -2.0,
                2.0,
            ]
        ),
    )

    assert candidates[0].source == "bounded_proposed"

    np.testing.assert_array_equal(
        candidates[0].action,
        np.array(
            [
                1.0,
                -1.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )


def test_robotics_candidate_indices_are_contiguous() -> None:
    candidates = generate_robotics_candidates(
        state=np.array(
            [
                0.8,
                -0.7,
                0.4,
                0.4,
                0.0,
                0.0,
            ]
        ),
        proposed_action=np.array(
            [
                0.7,
                -0.6,
                0.8,
            ]
        ),
    )

    assert [candidate.index for candidate in candidates] == list(range(len(candidates)))


def test_robotics_candidates_are_unique() -> None:
    candidates = generate_robotics_candidates(
        state=np.zeros(6),
        proposed_action=np.zeros(3),
    )

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            assert not np.all(
                np.abs(left.action - right.action) <= ACTION_DEDUPLICATION_TOLERANCE
            )


def test_robotics_candidate_ceiling() -> None:
    candidates = generate_robotics_candidates(
        state=np.array(
            [
                0.95,
                -0.95,
                0.2,
                0.2,
                0.0,
                0.0,
            ]
        ),
        proposed_action=np.array(
            [
                0.8,
                -0.8,
                0.9,
            ]
        ),
    )

    assert len(candidates) <= MAX_ROBOTICS_CANDIDATES


def test_robotics_positive_position_generates_negative_inward_motion() -> None:
    candidates = generate_robotics_candidates(
        state=np.array(
            [
                0.95,
                0.95,
                0.2,
                0.2,
                0.0,
                0.0,
            ]
        ),
        proposed_action=np.array(
            [
                0.8,
                0.8,
                0.0,
            ]
        ),
    )

    inward = next(
        candidate for candidate in candidates if candidate.source == "inward_motion"
    )

    np.testing.assert_array_equal(
        inward.action[0:2],
        np.array(
            [
                -1.0,
                -1.0,
            ]
        ),
    )


def test_robotics_negative_position_generates_positive_inward_motion() -> None:
    candidates = generate_robotics_candidates(
        state=np.array(
            [
                -0.95,
                -0.95,
                0.2,
                0.2,
                0.0,
                0.0,
            ]
        ),
        proposed_action=np.array(
            [
                -0.8,
                -0.8,
                0.0,
            ]
        ),
    )

    inward = next(
        candidate for candidate in candidates if candidate.source == "inward_motion"
    )

    np.testing.assert_array_equal(
        inward.action[0:2],
        np.array(
            [
                1.0,
                1.0,
            ]
        ),
    )


def test_robotics_hard_guard_rejects_outward_motion() -> None:
    result = evaluate_robotics_hard_guards(
        state=np.array(
            [
                0.89,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
            ]
        ),
        action=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert result.passed is False

    assert "unsafe_motion" in result.violated_guard_names


def test_robotics_hard_guard_allows_inward_recovery() -> None:
    result = evaluate_robotics_hard_guards(
        state=np.array(
            [
                0.95,
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
            ]
        ),
        action=np.array(
            [
                -1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert result.passed is True


def test_robotics_hard_guard_rejects_unsafe_gripper_close() -> None:
    result = evaluate_robotics_hard_guards(
        state=np.array(
            [
                0.0,
                0.0,
                0.8,
                0.8,
                0.0,
                0.0,
            ]
        ),
        action=np.array(
            [
                0.0,
                0.0,
                1.0,
            ]
        ),
    )

    assert result.passed is False

    assert "unsafe_gripper_condition" in result.violated_guard_names


def test_robotics_state_only_violation_does_not_fail_hard_guard() -> None:
    result = evaluate_robotics_hard_guards(
        state=np.array(
            [
                0.95,
                0.0,
                0.95,
                0.0,
                0.0,
                0.0,
            ]
        ),
        action=np.array(
            [
                -1.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert result.passed is True

    assert result.violated_guard_names == ()

    assert any(
        record.violated and record.name == "workspace_boundary"
        for record in result.violations
    )


def test_hard_guard_router_driving() -> None:
    result = evaluate_hard_guards(
        domain="autonomous_driving",
        state=np.array(
            [
                0.3,
                0.0,
                0.0,
                1.0,
            ]
        ),
        action=np.zeros(3),
    )

    assert result.passed is True


def test_hard_guard_router_robotics() -> None:
    result = evaluate_hard_guards(
        domain="robotics",
        state=np.array(
            [
                0.0,
                0.0,
                0.1,
                0.1,
                0.5,
                0.5,
            ]
        ),
        action=np.zeros(3),
    )

    assert result.passed is True


def test_hard_guard_router_rejects_unknown_domain() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported domain",
    ):
        evaluate_hard_guards(
            domain="unknown",
            state=np.zeros(4),
            action=np.zeros(3),
        )


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(3),
        np.array(
            [
                0.0,
                np.nan,
                0.0,
                0.0,
            ]
        ),
    ],
)
def test_driving_generator_rejects_invalid_state(
    state: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        generate_driving_candidates(
            state=state,
            proposed_action=np.zeros(3),
        )


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(5),
        np.array(
            [
                0.0,
                0.0,
                np.inf,
                0.0,
                0.0,
                0.0,
            ]
        ),
    ],
)
def test_robotics_generator_rejects_invalid_state(
    state: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        generate_robotics_candidates(
            state=state,
            proposed_action=np.zeros(3),
        )


def test_driving_generator_is_deterministic() -> None:
    state = np.array(
        [
            0.6,
            0.8,
            0.4,
            0.5,
        ]
    )

    proposed = np.array(
        [
            0.7,
            0.4,
            0.1,
        ]
    )

    first = generate_driving_candidates(
        state=state,
        proposed_action=proposed,
    )

    second = generate_driving_candidates(
        state=state,
        proposed_action=proposed,
    )

    assert [candidate.source for candidate in first] == [
        candidate.source for candidate in second
    ]

    for left, right in zip(
        first,
        second,
        strict=True,
    ):
        np.testing.assert_array_equal(
            left.action,
            right.action,
        )


def test_robotics_generator_is_deterministic() -> None:
    state = np.array(
        [
            0.8,
            -0.8,
            0.2,
            0.2,
            0.0,
            0.0,
        ]
    )

    proposed = np.array(
        [
            0.7,
            -0.6,
            0.8,
        ]
    )

    first = generate_robotics_candidates(
        state=state,
        proposed_action=proposed,
    )

    second = generate_robotics_candidates(
        state=state,
        proposed_action=proposed,
    )

    assert [candidate.source for candidate in first] == [
        candidate.source for candidate in second
    ]

    for left, right in zip(
        first,
        second,
        strict=True,
    ):
        np.testing.assert_array_equal(
            left.action,
            right.action,
        )
