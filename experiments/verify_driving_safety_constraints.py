"""Independent verification for Sprint 5.2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from q_vla_forge.safety.contracts import (
    ViolationSeverity,
)
from q_vla_forge.safety.driving_constraints import (
    ACCEL_BRAKE_CONFLICT_THRESHOLD,
    ACCEL_BRAKE_CRITICAL_THRESHOLD,
    LANE_CRITICAL_ABS,
    LANE_WARNING_ABS,
    OBSTACLE_CRITICAL_DISTANCE,
    OBSTACLE_WARNING_DISTANCE,
    SAFE_DISTANCE_BASE,
    SAFE_DISTANCE_SPEED_SCALE,
    STEERING_CRITICAL_ABS,
    STEERING_WARNING_ABS,
    VIOLATION_CATEGORIES,
    count_driving_violations,
    evaluate_driving_violations,
    required_safe_distance,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "driving_safety.yaml"

ARTIFACT_PATH = (
    ROOT / "results" / "safety" / "driving" / "sprint5-driving-safety-constraints.json"
)

PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def by_name(
    state: np.ndarray,
    action: np.ndarray,
):
    return {
        item.name: item
        for item in evaluate_driving_violations(
            state,
            action,
        )
    }


def main() -> None:
    require(
        PROTOCOL_PATH.exists(),
        "Sprint 5.1 protocol exists",
    )
    require(
        CONFIG_PATH.exists(),
        "Driving safety config exists",
    )
    require(
        ARTIFACT_PATH.exists(),
        "Driving safety artifact exists",
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    require(
        tuple(artifact["violation_categories"]) == VIOLATION_CATEGORIES,
        "Exactly five violation categories are frozen",
    )

    require(
        artifact["state_contract"]["dimension"] == 4,
        "Driving state dimension is 4",
    )

    require(
        artifact["action_contract"]["dimension"] == 3,
        "Driving action dimension is 3",
    )

    thresholds = artifact["thresholds"]

    require(
        thresholds["lane_boundary"]["warning_abs"] == LANE_WARNING_ABS == 0.75,
        "Lane warning threshold is frozen",
    )

    require(
        thresholds["lane_boundary"]["critical_abs"] == LANE_CRITICAL_ABS == 0.95,
        "Lane critical threshold is frozen",
    )

    require(
        thresholds["obstacle_distance"]["warning_distance"]
        == OBSTACLE_WARNING_DISTANCE
        == 0.30,
        "Obstacle warning threshold is frozen",
    )

    require(
        thresholds["obstacle_distance"]["critical_distance"]
        == OBSTACLE_CRITICAL_DISTANCE
        == 0.15,
        "Obstacle critical threshold is frozen",
    )

    require(
        STEERING_WARNING_ABS == 0.80,
        "Steering warning threshold is frozen",
    )

    require(
        STEERING_CRITICAL_ABS == 0.95,
        "Steering critical threshold is frozen",
    )

    require(
        ACCEL_BRAKE_CONFLICT_THRESHOLD == 0.25,
        "Acceleration/braking warning is frozen",
    )

    require(
        ACCEL_BRAKE_CRITICAL_THRESHOLD == 0.60,
        "Acceleration/braking critical is frozen",
    )

    require(
        SAFE_DISTANCE_BASE == 0.15,
        "Speed-distance base is frozen",
    )

    require(
        SAFE_DISTANCE_SPEED_SCALE == 0.35,
        "Speed-distance scale is frozen",
    )

    expected_distances = {
        0.0: 0.15,
        0.5: 0.325,
        1.0: 0.50,
    }

    for speed, expected in expected_distances.items():
        require(
            np.isclose(
                required_safe_distance(speed),
                expected,
            ),
            ("Speed-distance envelope matches " f"at speed {speed}"),
        )

    sequence = [
        required_safe_distance(speed)
        for speed in np.linspace(
            0.0,
            1.0,
            11,
        )
    ]

    require(
        sequence == sorted(sequence),
        "Speed-distance envelope is monotonic",
    )

    safe_state = np.array(
        [0.2, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    safe_action = np.array(
        [0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    require(
        count_driving_violations(
            safe_state,
            safe_action,
        )
        == 0,
        "Nominal safe case has zero violations",
    )

    lane_state = np.array(
        [0.2, 0.98, 0.0, 1.0],
        dtype=np.float32,
    )

    lane_record = by_name(
        lane_state,
        safe_action,
    )["lane_boundary"]

    require(
        lane_record.violated and lane_record.severity == ViolationSeverity.CRITICAL,
        "Known lane-critical case verified",
    )

    multiple_state = np.array(
        [1.0, 0.90, 0.0, 0.10],
        dtype=np.float32,
    )

    multiple_action = np.array(
        [0.99, 0.80, 0.80],
        dtype=np.float32,
    )

    require(
        count_driving_violations(
            multiple_state,
            multiple_action,
        )
        >= 4,
        "Multiple violations can occur in one step",
    )

    positive = by_name(
        np.array(
            [0.2, 0.80, 0.0, 1.0],
            dtype=np.float32,
        ),
        safe_action,
    )["lane_boundary"]

    negative = by_name(
        np.array(
            [0.2, -0.80, 0.0, 1.0],
            dtype=np.float32,
        ),
        safe_action,
    )["lane_boundary"]

    require(
        positive.violated == negative.violated,
        "Lane constraint is sign symmetric",
    )

    original_state = multiple_state.copy()
    original_action = multiple_action.copy()

    evaluate_driving_violations(
        multiple_state,
        multiple_action,
    )

    require(
        np.array_equal(
            multiple_state,
            original_state,
        )
        and np.array_equal(
            multiple_action,
            original_action,
        ),
        "Constraint evaluator does not mutate inputs",
    )

    require(
        (artifact["action_contract"]["constraint_evaluator_clips_actions"] is False),
        "Constraint evaluator does not clip actions",
    )

    require(
        config["claims"]["safety_improvement_demonstrated"] is False,
        "No driving safety-improvement claim exists",
    )

    require(
        config["claims"]["physical_stopping_distance_model"] is False,
        "No physical stopping-distance claim exists",
    )

    require(
        config["claims"]["certification_claim"] is False,
        "No certification claim exists",
    )

    results = artifact["results"]

    require(
        results["safety_results_present"] is False,
        "No safety results exist",
    )

    require(
        results["safety_filter_evaluated"] is False,
        "No safety filter has been evaluated",
    )

    serialized = json.dumps(artifact).lower()

    require(
        "best_method" not in serialized,
        "No winner field exists",
    )

    print()
    print("=" * 62)
    print(" SPRINT 5.2 — DRIVING SAFETY CONSTRAINTS")
    print("=" * 62)
    print()
    print("Constraint taxonomy: PASS")
    print("Threshold freeze: PASS")
    print("State/action contract: PASS")
    print("Speed-distance envelope: PASS")
    print("Boundary semantics: PASS")
    print("Symmetry: PASS")
    print("Purity: PASS")
    print("Claim controls: PASS")
    print("Premature-result protection: PASS")
    print()
    print("SPRINT 5.2 DRIVING SAFETY CONSTRAINTS: PASS")


if __name__ == "__main__":
    main()
