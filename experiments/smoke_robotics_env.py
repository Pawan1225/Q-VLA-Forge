"""Deterministic smoke test for the Sprint 4 robotics proxy."""

from __future__ import annotations

from q_vla_forge.rl.robotics_controller import (
    deterministic_robotics_action,
)
from q_vla_forge.rl.robotics_env import RoboticsRLEnv


def main() -> None:
    """Run one deterministic pick-and-place episode."""
    env = RoboticsRLEnv()

    observation, _ = env.reset(
        seed=42,
    )

    total_reward = 0.0
    terminated = False
    truncated = False
    final_info: dict[str, object] = {}

    while not (terminated or truncated):
        action = deterministic_robotics_action(
            observation,
            object_grasped=env.object_grasped,
            grasp_distance=env.config.grasp_distance,
            target_tolerance=env.config.target_tolerance,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            final_info,
        ) = env.step(action)

        total_reward += reward

    print(
        "steps=",
        env.step_count,
    )

    print(
        "total_reward=",
        total_reward,
    )

    print(
        "terminated=",
        terminated,
    )

    print(
        "truncated=",
        truncated,
    )

    print(
        "success=",
        final_info["success"],
    )

    print(
        "object_grasped=",
        final_info["object_grasped"],
    )

    print(
        "final_robot_position=",
        final_info["robot_position"],
    )

    print(
        "final_object_position=",
        final_info["object_position"],
    )

    print(
        "target_position=",
        final_info["target_position"],
    )

    if final_info["success"] is not True:
        raise RuntimeError("deterministic robotics smoke episode did not succeed")

    if terminated is not True:
        raise RuntimeError("successful robotics episode must terminate")

    if truncated is not False:
        raise RuntimeError("successful robotics episode must not truncate")

    print("SPRINT 4.3.18 ROBOTICS SMOKE: PASS")


if __name__ == "__main__":
    main()
