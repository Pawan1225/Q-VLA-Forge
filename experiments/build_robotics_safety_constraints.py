"""Build the frozen Sprint 5.3 robotics-safety contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from q_vla_forge.safety.robotics_constraints import (
    GRASP_SAFE_DISTANCE,
    GRIPPER_CLOSE_THRESHOLD,
    GRIPPER_CRITICAL_DISTANCE,
    MOVEMENT_SCALE,
    OBJECT_BOUNDARY_CRITICAL_ABS,
    OBJECT_BOUNDARY_WARNING_ABS,
    OBJECT_TARGET_CRITICAL_DISTANCE,
    OBJECT_TARGET_WARNING_DISTANCE,
    ROBOTICS_ACTION_DIM,
    ROBOTICS_STATE_DIM,
    VIOLATION_CATEGORIES,
    WORKSPACE_CRITICAL_ABS,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    WORKSPACE_WARNING_ABS,
)

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)

ROBOTICS_ENV_PATH = ROOT / "src" / "q_vla_forge" / "rl" / "robotics_env.py"

CONFIG_PATH = ROOT / "configs" / "robotics_safety.yaml"

OUTPUT_DIR = ROOT / "results" / "safety" / "robotics"

JSON_PATH = OUTPUT_DIR / "sprint5-robotics-safety-constraints.json"

MARKDOWN_PATH = OUTPUT_DIR / "sprint5-robotics-safety-constraints.md"


def sha256_file(
    path: Path,
) -> str:
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

    if not ROBOTICS_ENV_PATH.exists():
        raise FileNotFoundError(f"Robotics environment missing: {ROBOTICS_ENV_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Robotics safety config missing: {CONFIG_PATH}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if protocol["protocol_status"] != ("frozen_before_safety_results"):
        raise ValueError("Sprint 5.1 protocol is not frozen.")

    artifact = {
        "artifact": ("sprint5-robotics-safety-constraints"),
        "sprint": "5",
        "sub_sprint": "5.3",
        "domain": "robotics",
        "status": ("frozen_before_filter_evaluation"),
        "state_contract": {
            "dimension": ROBOTICS_STATE_DIM,
            "dimensions": [
                "robot_x",
                "robot_y",
                "object_x",
                "object_y",
                "target_x",
                "target_y",
            ],
        },
        "action_contract": {
            "dimension": ROBOTICS_ACTION_DIM,
            "dimensions": [
                "delta_x",
                "delta_y",
                "gripper",
            ],
            "constraint_evaluator_clips_actions": False,
        },
        "violation_categories": list(VIOLATION_CATEGORIES),
        "environment_constants": {
            "workspace_min": WORKSPACE_MIN,
            "workspace_max": WORKSPACE_MAX,
            "movement_scale": MOVEMENT_SCALE,
            "grasp_distance": GRASP_SAFE_DISTANCE,
            "close_threshold": (GRIPPER_CLOSE_THRESHOLD),
        },
        "thresholds": {
            "workspace_boundary": {
                "warning_abs": (WORKSPACE_WARNING_ABS),
                "critical_abs": (WORKSPACE_CRITICAL_ABS),
            },
            "unsafe_motion": {
                "warning_abs": (WORKSPACE_WARNING_ABS),
                "critical_abs": (WORKSPACE_CRITICAL_ABS),
                "outward_motion_only": True,
                "prediction": ("robot_position + " "movement_scale * action_delta"),
            },
            "gripper": {
                "close_threshold": (GRIPPER_CLOSE_THRESHOLD),
                "safe_grasp_distance": (GRASP_SAFE_DISTANCE),
                "critical_distance": (GRIPPER_CRITICAL_DISTANCE),
            },
            "object_boundary": {
                "warning_abs": (OBJECT_BOUNDARY_WARNING_ABS),
                "critical_abs": (OBJECT_BOUNDARY_CRITICAL_ABS),
            },
            "object_target_interaction": {
                "warning_distance": (OBJECT_TARGET_WARNING_DISTANCE),
                "critical_distance": (OBJECT_TARGET_CRITICAL_DISTANCE),
                "target_outside_workspace_is_critical": True,
                "interpretation": ("pilot object-target safety heuristic"),
            },
        },
        "provenance": {
            "source_protocol": (PROTOCOL_PATH.relative_to(ROOT).as_posix()),
            "source_protocol_sha256": (sha256_file(PROTOCOL_PATH)),
            "robotics_environment": (ROBOTICS_ENV_PATH.relative_to(ROOT).as_posix()),
            "robotics_environment_sha256": (sha256_file(ROBOTICS_ENV_PATH)),
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

    markdown = """# Q-VLA Forge — Sprint 5.3 Robotics Safety Constraints

## Status

**FROZEN BEFORE SAFETY-FILTER EVALUATION**

## Domain

Synthetic 2-D robotics pick-and-place proxy.

## State

1. robot x
2. robot y
3. object x
4. object y
5. target x
6. target y

## Action

1. delta x
2. delta y
3. gripper

## Violation categories

1. workspace boundary
2. unsafe outward motion
3. unsafe gripper condition
4. object boundary
5. unsafe object/target interaction

## Workspace boundary

Warning:

`abs(coordinate) > 0.90`

Critical:

`abs(coordinate) > 1.00`

## Unsafe motion

Predicted robot position:

`robot_position + 0.08 * action_delta`

Only outward motion into the warning region is considered unsafe.

Inward recovery motion is preserved.

## Gripper safety

Closing command:

`gripper > 0.50`

Unsafe if the robot-object distance exceeds:

`0.12`

Critical if the distance exceeds:

`0.30`

## Object boundary

Warning:

`abs(object coordinate) > 0.90`

Critical:

`abs(object coordinate) > 1.00`

## Object/target interaction

This is a pilot safety heuristic.

A distant target inside the workspace is not automatically unsafe.

Target positions outside the workspace are treated as critical.

## Scientific boundary

No clipping or Lyapunov safety filter has been evaluated.

No safety improvement, industrial robot safety,
production safety, ISO certification, IEC certification,
or formal safety guarantee is claimed.
"""

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print("=" * 58)
    print(" Q-VLA FORGE — SPRINT 5.3 ROBOTICS SAFETY")
    print("=" * 58)
    print()
    print("State dimensions: 6")
    print("Action dimensions: 3")
    print("Violation categories: 5")
    print("Workspace warning: 0.90")
    print("Workspace critical: 1.00")
    print("Movement scale: 0.08")
    print("Safe grasp distance: 0.12")
    print("Gripper close threshold: 0.50")
    print("Object boundary: ENABLED")
    print("Object/target safety heuristic: ENABLED")
    print("Safety evaluation performed: NO")
    print()
    print("SPRINT 5.3 ROBOTICS SAFETY CONTRACT BUILT")


if __name__ == "__main__":
    main()
