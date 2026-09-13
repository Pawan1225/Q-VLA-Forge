"""Verify Gymnasium compliance for the driving RL environment."""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from q_vla_forge.rl.driving_env import DrivingRLEnv


def main() -> None:
    env = DrivingRLEnv()

    check_env(
        env,
        skip_render_check=True,
    )

    print("SPRINT 4.2.18 GYMNASIUM CHECK: PASS")


if __name__ == "__main__":
    main()
