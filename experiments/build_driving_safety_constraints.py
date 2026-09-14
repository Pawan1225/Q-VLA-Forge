"""Build the frozen Sprint 5.2 driving-safety contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from q_vla_forge.safety.driving_constraints import (
    ACCEL_BRAKE_CONFLICT_THRESHOLD,
    ACCEL_BRAKE_CRITICAL_THRESHOLD,
    DRIVING_ACTION_DIM,
    DRIVING_STATE_DIM,
    HEADING_STEERING_CRITICAL_ABS,
    HEADING_STEERING_RISK_ABS,
    LANE_CRITICAL_ABS,
    LANE_STEERING_CRITICAL_ABS,
    LANE_STEERING_RISK_ABS,
    LANE_WARNING_ABS,
    OBSTACLE_CRITICAL_DISTANCE,
    OBSTACLE_WARNING_DISTANCE,
    SAFE_DISTANCE_BASE,
    SAFE_DISTANCE_CRITICAL_FRACTION,
    SAFE_DISTANCE_SPEED_SCALE,
    STEERING_CRITICAL_ABS,
    STEERING_WARNING_ABS,
    VIOLATION_CATEGORIES,
)

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)

DRIVING_ENV_PATH = ROOT / "src" / "q_vla_forge" / "rl" / "driving_env.py"

CONFIG_PATH = ROOT / "configs" / "driving_safety.yaml"

OUTPUT_DIR = ROOT / "results" / "safety" / "driving"

JSON_PATH = OUTPUT_DIR / "sprint5-driving-safety-constraints.json"

MARKDOWN_PATH = OUTPUT_DIR / "sprint5-driving-safety-constraints.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65_536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Sprint 5.1 protocol missing: {PROTOCOL_PATH}")

    if not DRIVING_ENV_PATH.exists():
        raise FileNotFoundError(f"Driving environment missing: {DRIVING_ENV_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    if protocol["protocol_status"] != ("frozen_before_safety_results"):
        raise ValueError("Sprint 5.1 protocol is not frozen.")

    artifact = {
        "artifact": ("sprint5-driving-safety-constraints"),
        "sprint": "5",
        "sub_sprint": "5.2",
        "domain": "autonomous_driving",
        "status": "frozen_before_filter_evaluation",
        "state_contract": {
            "dimension": DRIVING_STATE_DIM,
            "dimensions": [
                "speed",
                "lane_offset",
                "heading_error",
                "obstacle_distance",
            ],
        },
        "action_contract": {
            "dimension": DRIVING_ACTION_DIM,
            "dimensions": [
                "steering",
                "acceleration",
                "braking",
            ],
            "constraint_evaluator_clips_actions": False,
        },
        "violation_categories": list(VIOLATION_CATEGORIES),
        "thresholds": {
            "lane_boundary": {
                "warning_abs": LANE_WARNING_ABS,
                "critical_abs": LANE_CRITICAL_ABS,
            },
            "obstacle_distance": {
                "warning_distance": (OBSTACLE_WARNING_DISTANCE),
                "critical_distance": (OBSTACLE_CRITICAL_DISTANCE),
            },
            "speed_distance_envelope": {
                "base_distance": SAFE_DISTANCE_BASE,
                "speed_scale": (SAFE_DISTANCE_SPEED_SCALE),
                "critical_fraction": (SAFE_DISTANCE_CRITICAL_FRACTION),
                "formula": ("0.15 + 0.35 * max(speed, 0)"),
                "interpretation": ("pilot heuristic safety envelope"),
            },
            "steering_risk": {
                "steering_warning_abs": (STEERING_WARNING_ABS),
                "steering_critical_abs": (STEERING_CRITICAL_ABS),
                "lane_warning_abs": (LANE_STEERING_RISK_ABS),
                "lane_critical_abs": (LANE_STEERING_CRITICAL_ABS),
                "heading_warning_abs": (HEADING_STEERING_RISK_ABS),
                "heading_critical_abs": (HEADING_STEERING_CRITICAL_ABS),
            },
            "acceleration_braking_conflict": {
                "warning_threshold": (ACCEL_BRAKE_CONFLICT_THRESHOLD),
                "critical_threshold": (ACCEL_BRAKE_CRITICAL_THRESHOLD),
            },
        },
        "provenance": {
            "source_protocol": (PROTOCOL_PATH.relative_to(ROOT).as_posix()),
            "source_protocol_sha256": (sha256_file(PROTOCOL_PATH)),
            "driving_environment": (DRIVING_ENV_PATH.relative_to(ROOT).as_posix()),
            "driving_environment_sha256": (sha256_file(DRIVING_ENV_PATH)),
        },
        "claims": config["claims"],
        "results": {
            "safety_results_present": False,
            "safety_filter_evaluated": False,
            "clipping_evaluated": False,
            "lyapunov_evaluated": False,
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown = """# Q-VLA Forge — Sprint 5.2 Driving Safety Constraints

## Status

**FROZEN BEFORE SAFETY-FILTER EVALUATION**

## Domain

Autonomous-driving synthetic proxy.

## State

1. speed
2. lane offset
3. heading error
4. obstacle distance

## Action

1. steering
2. acceleration
3. braking

## Violation categories

1. lane boundary
2. unsafe obstacle distance
3. unsafe speed condition
4. contextual steering risk
5. acceleration/braking conflict

## Lane boundary

Warning violation:

`abs(lane_offset) > 0.75`

Critical:

`abs(lane_offset) > 0.95`

## Obstacle distance

Warning violation:

`obstacle_distance < 0.30`

Critical:

`obstacle_distance < 0.15`

## Pilot speed-distance envelope

`required_safe_distance = 0.15 + 0.35 * max(speed, 0)`

This is a pilot heuristic safety envelope.

It is not a physical stopping-distance model.

## Steering risk

Large steering is only considered a violation when combined
with elevated lane offset or heading error.

## Acceleration/braking conflict

Warning:

`acceleration > 0.25 AND braking > 0.25`

Critical:

`acceleration > 0.60 AND braking > 0.60`

## Scientific boundary

No safety filter has been evaluated in Sprint 5.2.

No safety improvement, collision reduction, certification,
production-vehicle safety, or ISO-compliance claim is made.
"""

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print("=" * 58)
    print(" Q-VLA FORGE — SPRINT 5.2 DRIVING SAFETY")
    print("=" * 58)
    print()
    print("Violation categories: 5")
    print("Lane warning: 0.75")
    print("Lane critical: 0.95")
    print("Obstacle warning: 0.30")
    print("Obstacle critical: 0.15")
    print("Speed-distance envelope: " "0.15 + 0.35 * max(speed, 0)")
    print("Steering risk: ENABLED")
    print("Acceleration/braking conflict: ENABLED")
    print("Safety evaluation performed: NO")
    print()
    print("SPRINT 5.2 DRIVING SAFETY CONTRACT BUILT")


if __name__ == "__main__":
    main()
