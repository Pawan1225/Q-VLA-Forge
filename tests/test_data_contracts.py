from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.data.contracts import (
    Action,
    Domain,
    Observation,
    SafetyConstraints,
    TaskSample,
    Transition,
)


def make_observation() -> Observation:
    return Observation(
        visual=np.zeros((3, 8, 8), dtype=np.float32),
        state=np.array(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float32,
        ),
        language_goal="maintain the target trajectory",
    )


def make_action() -> Action:
    return Action(
        values=np.array(
            [0.1, 0.2, 0.0],
            dtype=np.float32,
        )
    )


def make_constraints() -> SafetyConstraints:
    return SafetyConstraints(
        lower_bounds=np.array(
            [-1.0, -1.0, 0.0],
            dtype=np.float32,
        ),
        upper_bounds=np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_domain_values() -> None:
    assert Domain.AUTONOMOUS_DRIVING.value == "autonomous_driving"
    assert Domain.ROBOTICS.value == "robotics"


def test_observation_creation() -> None:
    observation = make_observation()

    assert observation.visual.shape == (3, 8, 8)
    assert observation.state.shape == (4,)
    assert observation.language_goal == "maintain the target trajectory"


def test_observation_rejects_empty_language_goal() -> None:
    with pytest.raises(ValueError):
        Observation(
            visual=np.zeros(
                (3, 8, 8),
                dtype=np.float32,
            ),
            state=np.zeros(
                4,
                dtype=np.float32,
            ),
            language_goal="   ",
        )


def test_observation_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError):
        Observation(
            visual=np.zeros(
                (3, 8, 8),
                dtype=np.float64,
            ),
            state=np.zeros(
                4,
                dtype=np.float32,
            ),
            language_goal="test",
        )


def test_action_creation() -> None:
    action = make_action()

    assert action.values.shape == (3,)
    assert action.values.dtype == np.float32


def test_action_rejects_non_vector() -> None:
    with pytest.raises(ValueError):
        Action(
            values=np.zeros(
                (2, 2),
                dtype=np.float32,
            )
        )


def test_safety_constraints_creation() -> None:
    constraints = make_constraints()

    assert constraints.lower_bounds.shape == (3,)
    assert constraints.upper_bounds.shape == (3,)


def test_safety_constraints_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        SafetyConstraints(
            lower_bounds=np.zeros(
                2,
                dtype=np.float32,
            ),
            upper_bounds=np.zeros(
                3,
                dtype=np.float32,
            ),
        )


def test_safety_constraints_reject_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        SafetyConstraints(
            lower_bounds=np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            upper_bounds=np.array(
                [0.0, 1.0],
                dtype=np.float32,
            ),
        )


def test_task_sample_creation() -> None:
    sample = TaskSample(
        sample_id="driving-001",
        domain=Domain.AUTONOMOUS_DRIVING,
        observation=make_observation(),
        target_action=make_action(),
        safety_constraints=make_constraints(),
    )

    assert sample.sample_id == "driving-001"
    assert sample.domain == Domain.AUTONOMOUS_DRIVING


def test_task_sample_rejects_action_constraint_mismatch() -> None:
    with pytest.raises(ValueError):
        TaskSample(
            sample_id="invalid",
            domain=Domain.ROBOTICS,
            observation=make_observation(),
            target_action=Action(
                values=np.zeros(
                    2,
                    dtype=np.float32,
                )
            ),
            safety_constraints=make_constraints(),
        )


def test_transition_creation() -> None:
    transition = Transition(
        domain=Domain.ROBOTICS,
        observation=make_observation(),
        action=make_action(),
        reward=1.0,
        next_observation=make_observation(),
        terminated=False,
        truncated=False,
        safety_constraints=make_constraints(),
    )

    assert transition.reward == 1.0
    assert transition.terminated is False


def test_transition_rejects_non_finite_reward() -> None:
    with pytest.raises(ValueError):
        Transition(
            domain=Domain.ROBOTICS,
            observation=make_observation(),
            action=make_action(),
            reward=float("nan"),
            next_observation=make_observation(),
            terminated=False,
            truncated=False,
            safety_constraints=make_constraints(),
        )
