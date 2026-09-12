from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

import numpy as np

from q_vla_forge.data.contracts import (
    Action,
    Domain,
    Observation,
    SafetyConstraints,
    TaskSample,
)

ROBOTICS_ACTION_DIM = 3
ROBOTICS_STATE_DIM = 6
DEFAULT_ROBOTICS_IMAGE_SIZE = 32

ROBOTICS_GOALS = (
    "move object to target",
    "move left",
    "move right",
    "move up",
    "move down",
    "close gripper",
    "open gripper",
)


def robotics_safety_constraints() -> SafetyConstraints:
    """Return generic continuous action limits for robotics."""
    return SafetyConstraints(
        lower_bounds=np.array(
            [-1.0, -1.0, -1.0],
            dtype=np.float32,
        ),
        upper_bounds=np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
        metadata={
            "action_order": [
                "delta_x",
                "delta_y",
                "gripper",
            ]
        },
    )


def compute_robotics_target_action(
    state: np.ndarray,
    language_goal: str,
) -> Action:
    """Compute a deterministic target robotics action."""
    if state.shape != (ROBOTICS_STATE_DIM,):
        raise ValueError(f"robotics state must have shape ({ROBOTICS_STATE_DIM},)")

    if state.dtype != np.float32:
        raise TypeError("robotics state must use dtype float32")

    robot_x = float(state[0])
    robot_y = float(state[1])
    object_x = float(state[2])
    object_y = float(state[3])
    target_x = float(state[4])
    target_y = float(state[5])

    delta_x = 0.0
    delta_y = 0.0
    gripper = 0.0

    if language_goal == "move object to target":
        object_robot_distance = np.hypot(
            object_x - robot_x,
            object_y - robot_y,
        )

        if object_robot_distance > 0.15:
            delta_x = object_x - robot_x
            delta_y = object_y - robot_y
            gripper = -1.0
        else:
            delta_x = target_x - object_x
            delta_y = target_y - object_y
            gripper = 1.0

    elif language_goal == "move left":
        delta_x = -0.5

    elif language_goal == "move right":
        delta_x = 0.5

    elif language_goal == "move up":
        delta_y = 0.5

    elif language_goal == "move down":
        delta_y = -0.5

    elif language_goal == "close gripper":
        gripper = 1.0

    elif language_goal == "open gripper":
        gripper = -1.0

    else:
        raise ValueError(f"unsupported robotics language goal: {language_goal}")

    delta_x = float(np.clip(delta_x, -1.0, 1.0))
    delta_y = float(np.clip(delta_y, -1.0, 1.0))
    gripper = float(np.clip(gripper, -1.0, 1.0))

    return Action(
        values=np.array(
            [
                delta_x,
                delta_y,
                gripper,
            ],
            dtype=np.float32,
        )
    )


def _coordinate_to_pixel(
    value: float,
    image_size: int,
) -> int:
    """Map a normalized [-1, 1] coordinate into an image index."""
    clipped = float(np.clip(value, -1.0, 1.0))
    normalized = (clipped + 1.0) / 2.0

    return round(normalized * (image_size - 1))


def render_robotics_visual(
    robot_position: tuple[float, float],
    object_position: tuple[float, float],
    target_position: tuple[float, float],
    image_size: int = DEFAULT_ROBOTICS_IMAGE_SIZE,
) -> np.ndarray:
    """Create a simple synthetic robot workspace image."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8")

    image = np.zeros(
        (3, image_size, image_size),
        dtype=np.float32,
    )

    robot_x, robot_y = robot_position
    object_x, object_y = object_position
    target_x, target_y = target_position

    robot_col = _coordinate_to_pixel(robot_x, image_size)
    robot_row = (
        image_size
        - 1
        - _coordinate_to_pixel(
            robot_y,
            image_size,
        )
    )

    object_col = _coordinate_to_pixel(object_x, image_size)
    object_row = (
        image_size
        - 1
        - _coordinate_to_pixel(
            object_y,
            image_size,
        )
    )

    target_col = _coordinate_to_pixel(target_x, image_size)
    target_row = (
        image_size
        - 1
        - _coordinate_to_pixel(
            target_y,
            image_size,
        )
    )

    image[
        0,
        max(0, robot_row - 1) : min(image_size, robot_row + 2),
        max(0, robot_col - 1) : min(image_size, robot_col + 2),
    ] = 1.0

    image[
        1,
        max(0, object_row - 1) : min(image_size, object_row + 2),
        max(0, object_col - 1) : min(image_size, object_col + 2),
    ] = 1.0

    image[
        2,
        max(0, target_row - 1) : min(image_size, target_row + 2),
        max(0, target_col - 1) : min(image_size, target_col + 2),
    ] = 1.0

    return image


def generate_robotics_sample(
    sample_index: int,
    rng: np.random.Generator,
    image_size: int = DEFAULT_ROBOTICS_IMAGE_SIZE,
) -> TaskSample:
    """Generate one deterministic synthetic robotics sample."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")

    robot_x = rng.uniform(-1.0, 1.0)
    robot_y = rng.uniform(-1.0, 1.0)

    object_x = rng.uniform(-1.0, 1.0)
    object_y = rng.uniform(-1.0, 1.0)

    target_x = rng.uniform(-1.0, 1.0)
    target_y = rng.uniform(-1.0, 1.0)

    state = np.array(
        [
            robot_x,
            robot_y,
            object_x,
            object_y,
            target_x,
            target_y,
        ],
        dtype=np.float32,
    )

    language_goal = str(rng.choice(ROBOTICS_GOALS))

    visual = render_robotics_visual(
        robot_position=(
            robot_x,
            robot_y,
        ),
        object_position=(
            object_x,
            object_y,
        ),
        target_position=(
            target_x,
            target_y,
        ),
        image_size=image_size,
    )

    observation = Observation(
        visual=visual,
        state=state,
        language_goal=language_goal,
        metadata={
            "state_order": [
                "robot_x",
                "robot_y",
                "object_x",
                "object_y",
                "target_x",
                "target_y",
            ],
            "synthetic": True,
        },
    )

    target_action = compute_robotics_target_action(
        state=state,
        language_goal=language_goal,
    )

    return TaskSample(
        sample_id=f"robotics-{sample_index:06d}",
        domain=Domain.ROBOTICS,
        observation=observation,
        target_action=target_action,
        safety_constraints=robotics_safety_constraints(),
    )


class SyntheticRoboticsDataset(Sequence[TaskSample]):
    """Deterministic synthetic robotics dataset."""

    def __init__(
        self,
        size: int,
        seed: int = 42,
        image_size: int = DEFAULT_ROBOTICS_IMAGE_SIZE,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be greater than zero")

        if image_size < 8:
            raise ValueError("image_size must be at least 8")

        self._size = size
        self._seed = seed
        self._image_size = image_size

        rng = np.random.default_rng(seed)

        self._samples = tuple(
            generate_robotics_sample(
                sample_index=index,
                rng=rng,
                image_size=image_size,
            )
            for index in range(size)
        )

    @property
    def seed(self) -> int:
        """Return the dataset seed."""
        return self._seed

    @property
    def image_size(self) -> int:
        """Return the visual workspace size."""
        return self._image_size

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(
        self,
        index: int,
    ) -> TaskSample: ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> Sequence[TaskSample]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> TaskSample | Sequence[TaskSample]:
        return self._samples[index]

    def __iter__(self) -> Iterator[TaskSample]:
        return iter(self._samples)