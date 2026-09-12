from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from q_vla_forge.data.contracts import (
    Action,
    Domain,
    Observation,
    SafetyConstraints,
    TaskSample,
)

DRIVING_ACTION_DIM = 3
DRIVING_STATE_DIM = 4
DEFAULT_IMAGE_SIZE = 32

DRIVING_GOALS = (
    "keep lane",
    "slow down",
    "stop safely",
    "move left",
    "move right",
)


def driving_safety_constraints() -> SafetyConstraints:
    """Return the common continuous action limits for driving."""
    return SafetyConstraints(
        lower_bounds=np.array(
            [-1.0, -1.0, 0.0],
            dtype=np.float32,
        ),
        upper_bounds=np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
        metadata={
            "action_order": [
                "steering",
                "acceleration",
                "braking",
            ]
        },
    )


def compute_target_action(
    state: np.ndarray,
    language_goal: str,
) -> Action:
    """Compute a deterministic target driving action."""
    if state.shape != (DRIVING_STATE_DIM,):
        raise ValueError(f"driving state must have shape ({DRIVING_STATE_DIM},)")

    if state.dtype != np.float32:
        raise TypeError("driving state must use dtype float32")

    speed = float(state[0])
    lane_offset = float(state[1])
    heading_error = float(state[2])
    obstacle_distance = float(state[3])

    steering = -(0.75 * lane_offset + 0.50 * heading_error)

    target_speed = 0.65

    if language_goal == "slow down":
        target_speed = 0.30

    elif language_goal == "stop safely":
        target_speed = 0.0

    elif language_goal == "move left":
        steering -= 0.35

    elif language_goal == "move right":
        steering += 0.35

    elif language_goal != "keep lane":
        raise ValueError(f"unsupported driving language goal: {language_goal}")

    speed_error = target_speed - speed

    acceleration = max(0.0, speed_error * 1.5)
    braking = max(0.0, -speed_error * 1.5)

    if obstacle_distance < 0.20:
        acceleration = 0.0
        braking = max(
            braking,
            min(1.0, (0.20 - obstacle_distance) * 5.0),
        )

    steering = float(np.clip(steering, -1.0, 1.0))
    acceleration = float(np.clip(acceleration, 0.0, 1.0))
    braking = float(np.clip(braking, 0.0, 1.0))

    return Action(
        values=np.array(
            [
                steering,
                acceleration,
                braking,
            ],
            dtype=np.float32,
        )
    )


def render_driving_visual(
    lane_offset: float,
    obstacle_distance: float,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Create a small synthetic road representation."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8")

    image = np.zeros(
        (3, image_size, image_size),
        dtype=np.float32,
    )

    center = image_size // 2

    lane_shift = int(
        np.clip(
            lane_offset,
            -1.0,
            1.0,
        )
        * (image_size // 6)
    )

    road_center = int(
        np.clip(
            center - lane_shift,
            2,
            image_size - 3,
        )
    )

    lane_half_width = max(
        2,
        image_size // 6,
    )

    left_lane = max(
        0,
        road_center - lane_half_width,
    )

    right_lane = min(
        image_size - 1,
        road_center + lane_half_width,
    )

    image[0, :, left_lane] = 1.0
    image[0, :, right_lane] = 1.0

    image[1, :, road_center] = 0.5

    obstacle_distance = float(
        np.clip(
            obstacle_distance,
            0.0,
            1.0,
        )
    )

    obstacle_row = int(obstacle_distance * (image_size - 1))

    obstacle_row = image_size - 1 - obstacle_row

    row_start = max(
        0,
        obstacle_row - 1,
    )

    row_end = min(
        image_size,
        obstacle_row + 2,
    )

    column_start = max(
        0,
        road_center - 1,
    )

    column_end = min(
        image_size,
        road_center + 2,
    )

    image[
        2,
        row_start:row_end,
        column_start:column_end,
    ] = 1.0

    return image


def generate_driving_sample(
    sample_index: int,
    rng: np.random.Generator,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> TaskSample:
    """Generate one deterministic synthetic driving sample."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")

    speed = rng.uniform(0.0, 1.0)
    lane_offset = rng.uniform(-0.75, 0.75)
    heading_error = rng.uniform(-0.40, 0.40)
    obstacle_distance = rng.uniform(0.05, 1.0)

    state = np.array(
        [
            speed,
            lane_offset,
            heading_error,
            obstacle_distance,
        ],
        dtype=np.float32,
    )

    language_goal = str(rng.choice(DRIVING_GOALS))

    visual = render_driving_visual(
        lane_offset=lane_offset,
        obstacle_distance=obstacle_distance,
        image_size=image_size,
    )

    observation = Observation(
        visual=visual,
        state=state,
        language_goal=language_goal,
        metadata={
            "state_order": [
                "speed",
                "lane_offset",
                "heading_error",
                "obstacle_distance",
            ],
            "synthetic": True,
        },
    )

    target_action = compute_target_action(
        state=state,
        language_goal=language_goal,
    )

    return TaskSample(
        sample_id=f"driving-{sample_index:06d}",
        domain=Domain.AUTONOMOUS_DRIVING,
        observation=observation,
        target_action=target_action,
        safety_constraints=driving_safety_constraints(),
    )


class SyntheticDrivingDataset(Sequence[TaskSample]):
    """Deterministic synthetic autonomous-driving dataset."""

    def __init__(
        self,
        size: int,
        seed: int = 42,
        image_size: int = DEFAULT_IMAGE_SIZE,
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
            generate_driving_sample(
                sample_index=index,
                rng=rng,
                image_size=image_size,
            )
            for index in range(size)
        )

    @property
    def seed(self) -> int:
        """Return the seed used to generate the dataset."""
        return self._seed

    @property
    def image_size(self) -> int:
        """Return the synthetic image size."""
        return self._image_size

    def __len__(self) -> int:
        return self._size

    def __getitem__(
        self,
        index: int,
    ) -> TaskSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[TaskSample]:
        return iter(self._samples)
