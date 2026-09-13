"""Smoke-test the driving environment with a baseline controller."""

from __future__ import annotations

from q_vla_forge.rl.driving_controller import (
    zero_action_controller,
)
from q_vla_forge.rl.driving_env import DrivingRLEnv


def main() -> None:
    env = DrivingRLEnv()

    observation, _ = env.reset(seed=42)

    total_reward = 0.0

    while True:
        action = zero_action_controller(observation)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(
        "steps=",
        env.step_count,
    )

    print(
        "success=",
        info["success"],
    )

    print(
        "collision=",
        info["collision"],
    )

    print(
        "lane_departure=",
        info["lane_departure"],
    )

    print(
        "total_reward=",
        total_reward,
    )

    print("SPRINT 4.2.18 CONTROLLER SMOKE: PASS")


if __name__ == "__main__":
    main()
