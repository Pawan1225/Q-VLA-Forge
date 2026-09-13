"""Gymnasium verification for the Sprint 4 robotics environment."""

from gymnasium.utils.env_checker import check_env

from q_vla_forge.rl.robotics_env import RoboticsRLEnv


def main() -> None:
    """Run Gymnasium compatibility verification."""
    env = RoboticsRLEnv()

    check_env(
        env,
        skip_render_check=True,
    )

    print("SPRINT 4.3.18 GYMNASIUM CHECK: PASS")


if __name__ == "__main__":
    main()
