"""Independently verify the frozen Sprint 4.4 environment audit artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

EXPECTED_EPISODES = 100

EXPECTED_KEYS = (
    "driving_random",
    "driving_degenerate",
    "driving_heuristic",
    "robotics_random",
    "robotics_degenerate",
    "robotics_heuristic",
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def verify_summary(
    name: str,
    summary: dict[str, Any],
) -> None:
    require(
        summary["episodes"] == EXPECTED_EPISODES,
        f"{name}: unexpected episode count",
    )

    require(
        math.isfinite(float(summary["mean_reward"])),
        f"{name}: non-finite mean reward",
    )

    require(
        math.isfinite(float(summary["reward_standard_deviation"])),
        f"{name}: non-finite reward SD",
    )

    require(
        0.0 <= float(summary["success_rate"]) <= 1.0,
        f"{name}: invalid success rate",
    )

    require(
        float(summary["mean_episode_length"]) > 0.0,
        f"{name}: invalid mean episode length",
    )


def verify_episode_records(
    name: str,
    records: list[dict[str, Any]],
) -> None:
    require(
        len(records) == EXPECTED_EPISODES,
        f"{name}: unexpected record count",
    )

    for record in records:
        require(
            math.isfinite(float(record["total_reward"])),
            f"{name}: non-finite episode reward",
        )

        require(
            int(record["episode_length"]) > 0,
            f"{name}: invalid episode length",
        )

        require(
            not (bool(record["terminated"]) and bool(record["truncated"])),
            f"{name}: terminated and truncated both true",
        )


def main() -> None:
    require(
        AUDIT_PATH.exists(),
        f"missing audit artifact: {AUDIT_PATH}",
    )

    payload = json.loads(
        AUDIT_PATH.read_text(
            encoding="utf-8",
        )
    )

    audit = payload["audit"]
    protocol = payload["protocol"]
    summaries = payload["summaries"]
    episode_records = payload["episode_records"]

    require(
        audit["sprint"] == "4.4",
        "unexpected sprint identifier",
    )

    require(
        audit["training_performed"] is False,
        "training must not have been performed",
    )

    require(
        audit["qml_performed"] is False,
        "QML must not have been performed",
    )

    require(
        audit["safety_filter_used"] is False,
        "safety filter must not have been used",
    )

    require(
        protocol["episode_count_per_policy"] == EXPECTED_EPISODES,
        "unexpected audit episode count",
    )

    require(
        protocol["seeds"]
        == list(
            range(
                10_000,
                10_100,
            )
        ),
        "audit seed bank mismatch",
    )

    require(
        protocol["policies"]
        == [
            "random",
            "degenerate",
            "heuristic",
        ],
        "unexpected policy hierarchy",
    )

    for key in EXPECTED_KEYS:
        require(
            key in summaries,
            f"missing summary: {key}",
        )

        require(
            key in episode_records,
            f"missing episode records: {key}",
        )

        verify_summary(
            key,
            summaries[key],
        )

        verify_episode_records(
            key,
            episode_records[key],
        )

    driving_random = summaries["driving_random"]

    driving_degenerate = summaries["driving_degenerate"]

    driving_heuristic = summaries["driving_heuristic"]

    robotics_random = summaries["robotics_random"]

    robotics_degenerate = summaries["robotics_degenerate"]

    robotics_heuristic = summaries["robotics_heuristic"]

    require(
        float(driving_random["success_rate"]) < 1.0,
        "driving random policy is degenerate",
    )

    require(
        float(driving_degenerate["success_rate"]) < 1.0,
        "driving degenerate policy is degenerate",
    )

    require(
        float(robotics_random["success_rate"]) < 1.0,
        "robotics random policy is degenerate",
    )

    require(
        float(robotics_degenerate["success_rate"]) < 1.0,
        "robotics degenerate policy is degenerate",
    )

    require(
        float(driving_heuristic["mean_reward"]) > float(driving_random["mean_reward"]),
        "driving heuristic does not beat random reward",
    )

    require(
        float(robotics_heuristic["mean_reward"])
        > float(robotics_random["mean_reward"]),
        "robotics heuristic does not beat random reward",
    )

    require(
        float(driving_heuristic["success_rate"])
        > float(driving_random["success_rate"]),
        "driving heuristic does not beat random success",
    )

    require(
        float(driving_heuristic["success_rate"])
        > float(driving_degenerate["success_rate"]),
        "driving heuristic does not beat degenerate success",
    )

    require(
        float(robotics_heuristic["success_rate"])
        > float(robotics_random["success_rate"]),
        "robotics heuristic does not beat random success",
    )

    require(
        float(robotics_heuristic["success_rate"])
        > float(robotics_degenerate["success_rate"]),
        "robotics heuristic does not beat degenerate success",
    )

    require(
        float(driving_heuristic["success_rate"]) > 0.0,
        "driving heuristic demonstrates no task success",
    )

    require(
        float(robotics_heuristic["success_rate"]) > 0.0,
        "robotics heuristic demonstrates no task success",
    )

    print(
        "Audit artifact:",
        AUDIT_PATH,
    )

    print(
        "Driving random success:",
        driving_random["success_rate"],
    )

    print(
        "Driving degenerate success:",
        driving_degenerate["success_rate"],
    )

    print(
        "Driving heuristic success:",
        driving_heuristic["success_rate"],
    )

    print(
        "Robotics random success:",
        robotics_random["success_rate"],
    )

    print(
        "Robotics degenerate success:",
        robotics_degenerate["success_rate"],
    )

    print(
        "Robotics heuristic success:",
        robotics_heuristic["success_rate"],
    )

    print("SPRINT 4.4.14 ENVIRONMENT READINESS: PASS")


if __name__ == "__main__":
    main()
