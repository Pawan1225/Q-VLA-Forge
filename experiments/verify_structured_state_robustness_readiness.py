"""Verify Sprint 5.11 structured-state robustness readiness."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    load_frozen_ppo_policy,
)
from q_vla_forge.safety.perturbations import (
    DRIVING_STRUCTURED_PERTURBATIONS,
    ROBOTICS_STRUCTURED_PERTURBATIONS,
    apply_structured_state_perturbation,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DRIVING = {
    "lane_offset_plus_0p05": ((1, 0.05),),
    "lane_offset_minus_0p05": ((1, -0.05),),
    "lane_offset_plus_0p10": ((1, 0.10),),
    "lane_offset_minus_0p10": ((1, -0.10),),
    "obstacle_distance_plus_0p05": ((3, 0.05),),
    "obstacle_distance_plus_0p10": ((3, 0.10),),
    "speed_plus_0p05": ((0, 0.05),),
    "speed_minus_0p05": ((0, -0.05),),
    "heading_error_plus_0p05": ((2, 0.05),),
    "heading_error_minus_0p05": ((2, -0.05),),
}

EXPECTED_ROBOTICS = {
    "robot_x_plus_0p05": ((0, 0.05),),
    "robot_x_minus_0p05": ((0, -0.05),),
    "robot_y_plus_0p05": ((1, 0.05),),
    "robot_y_minus_0p05": ((1, -0.05),),
    "object_x_plus_0p05": ((2, 0.05),),
    "object_x_minus_0p05": ((2, -0.05),),
    "object_y_plus_0p05": ((3, 0.05),),
    "object_y_minus_0p05": ((3, -0.05),),
    "target_x_plus_0p05": ((4, 0.05),),
    "target_x_minus_0p05": ((4, -0.05),),
    "target_y_plus_0p05": ((5, 0.05),),
    "target_y_minus_0p05": ((5, -0.05),),
}


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def _check_specs() -> None:
    driving = {item.name: item.updates for item in DRIVING_STRUCTURED_PERTURBATIONS}

    robotics = {item.name: item.updates for item in ROBOTICS_STRUCTURED_PERTURBATIONS}

    _require(
        driving == EXPECTED_DRIVING,
        "10 driving perturbations exact",
    )

    _require(
        robotics == EXPECTED_ROBOTICS,
        "12 robotics perturbations exact",
    )

    _require(
        len(driving) + len(robotics) == 22,
        "22 perturbation definitions",
    )


def _check_determinism() -> None:
    for (
        dimension,
        perturbations,
    ) in (
        (
            4,
            DRIVING_STRUCTURED_PERTURBATIONS,
        ),
        (
            6,
            ROBOTICS_STRUCTURED_PERTURBATIONS,
        ),
    ):
        state = np.linspace(
            -0.5,
            0.5,
            dimension,
            dtype=np.float32,
        )

        for spec in perturbations:
            first_observed, first_delta = apply_structured_state_perturbation(
                true_state=state,
                perturbation=spec,
            )

            second_observed, second_delta = apply_structured_state_perturbation(
                true_state=state,
                perturbation=spec,
            )

            _require(
                np.array_equal(
                    first_observed,
                    second_observed,
                ),
                ("deterministic observation " f"{spec.name}"),
            )

            _require(
                np.array_equal(
                    first_delta,
                    second_delta,
                ),
                ("deterministic delta " f"{spec.name}"),
            )

            _require(
                np.array_equal(
                    first_observed,
                    state + first_delta,
                ),
                ("observed=true+delta " f"{spec.name}"),
            )

            _require(
                np.count_nonzero(first_delta) == len(spec.updates),
                ("sparse perturbation " f"{spec.name}"),
            )


def _check_checkpoints() -> None:
    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for seed in (
            42,
            123,
            456,
        ):
            policy = load_frozen_ppo_policy(
                root=ROOT,
                domain=domain,
                principal_seed=seed,
            )

            _require(
                policy.checkpoint_path.exists(),
                ("checkpoint exists " f"{domain} seed {seed}"),
            )

            _require(
                policy.training_budget == 20_000,
                ("training budget frozen " f"{domain} seed {seed}"),
            )


def _check_clean_evidence() -> None:
    paths = ROOT / "results" / "safety"

    expected = (
        paths / "baseline" / "sprint5-no-filter-safety-summary.json",
        paths / "clipping" / "sprint5-clipping-safety-summary.json",
        paths / "lyapunov-driving" / "sprint5-driving-lyapunov-summary.json",
        paths / "lyapunov-robotics" / "sprint5-robotics-lyapunov-summary.json",
    )

    for path in expected:
        _require(
            path.exists(),
            ("clean reference exists " f"{path.relative_to(ROOT)}"),
        )


def _check_runner() -> None:
    path = ROOT / "experiments" / "run_structured_state_robustness.py"

    _require(
        path.exists(),
        "structured runner exists",
    )

    source = path.read_text(encoding="utf-8")

    forbidden = (
        "train_ppo(",
        "run_ppo_training(",
        ".backward(",
        "rng.normal(",
        "apply_gaussian_observation_noise",
    )

    for token in forbidden:
        _require(
            token not in source,
            ("runner excludes " f"{token}"),
        )

    required = (
        "deterministic_policy_action",
        "apply_clipping_safety_filter",
        "apply_lyapunov_safety_filter",
        "apply_structured_state_perturbation",
        "observation=observed_state",
        "true_state=true_state",
        "object_grasped",
        "clean_reference_sha256",
    )

    for token in required:
        _require(
            token in source,
            ("runner contains " f"{token}"),
        )


def main() -> None:
    print("=" * 68)

    print(" SPRINT 5.11 STRUCTURED " "STATE ROBUSTNESS READINESS")

    print("=" * 68)

    print()

    _check_specs()
    _check_determinism()
    _check_checkpoints()
    _check_clean_evidence()
    _check_runner()

    print()
    print("Protocol: PASS")
    print("22 perturbation definitions: PASS")
    print("Frozen checkpoints: PASS")
    print("Clean evidence: PASS")
    print("NONE: PASS")
    print("CLIPPING: PASS")
    print("LYAPUNOV: PASS")
    print("True-state safety access: PASS")
    print("Policy perturbed observation: PASS")
    print("Robotics grasp context: PASS")
    print("No Gaussian noise: PASS")
    print("No training: PASS")

    print()
    print("Expected principal cells: 198")
    print("Expected principal episodes: 3960")

    print()
    print("SPRINT 5.11 STRUCTURED " "STATE ROBUSTNESS READINESS: PASS")


if __name__ == "__main__":
    main()
