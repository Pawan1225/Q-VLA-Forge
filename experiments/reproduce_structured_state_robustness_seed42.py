"""Reproduce frozen Sprint 5.11 representative seed-42 cases."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_structured_state_robustness import (
    run_structured_episode,
)

from q_vla_forge.safety.perturbations import (
    structured_perturbation_by_name,
)

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "safety" / "structured-state-robustness" / "runs"

CASES = (
    (
        "autonomous_driving",
        "obstacle_distance_plus_0p10",
    ),
    (
        "robotics",
        "robot_x_minus_0p05",
    ),
)

METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)


def _slug(
    value: str,
) -> str:
    return value.replace(
        "_",
        "-",
    )


def _compare(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    if isinstance(
        actual,
        float,
    ) or isinstance(
        expected,
        float,
    ):
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise AssertionError(f"{path}: " f"{actual!r} != {expected!r}")

        return

    if isinstance(
        actual,
        dict,
    ):
        if not isinstance(
            expected,
            dict,
        ):
            raise TypeError(f"{path}: expected dict")

        if set(actual) != set(expected):
            raise AssertionError(f"{path}: key mismatch")

        for key in actual:
            _compare(
                actual[key],
                expected[key],
                path=(f"{path}.{key}"),
            )

        return

    if isinstance(
        actual,
        list,
    ):
        if not isinstance(
            expected,
            list,
        ):
            raise TypeError(f"{path}: expected list")

        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length mismatch")

        for index, (
            left,
            right,
        ) in enumerate(
            zip(
                actual,
                expected,
                strict=True,
            )
        ):
            _compare(
                left,
                right,
                path=(f"{path}[{index}]"),
            )

        return

    if actual != expected:
        raise AssertionError(f"{path}: " f"{actual!r} != {expected!r}")


def main() -> None:
    reproduced = 0

    print("=" * 72)
    print(" Q-VLA FORGE - SPRINT 5.11 " "SEED-42 REPRODUCTION")
    print("=" * 72)
    print()

    for (
        domain,
        perturbation_name,
    ) in CASES:
        perturbation = structured_perturbation_by_name(
            domain=domain,
            name=perturbation_name,
        )

        for method in METHODS:
            path = RUNS / (
                f"{_slug(domain)}-"
                f"{_slug(perturbation_name)}-"
                f"{method}-"
                "seed-42.json"
            )

            reference = json.loads(path.read_text(encoding="utf-8"))

            for index, evaluation_seed in enumerate(EVALUATION_SEEDS):
                (
                    evidence,
                    audits,
                    checkpoint_sha,
                ) = run_structured_episode(
                    domain=domain,
                    perturbation=perturbation,
                    method=method,
                    principal_seed=42,
                    evaluation_seed=evaluation_seed,
                )

                expected = reference["episodes"][index]

                actual_core = asdict(evidence)

                expected_core = {
                    key: value
                    for key, value in expected.items()
                    if key != "audit_snapshots"
                }

                _compare(
                    actual_core,
                    expected_core,
                    path=(
                        f"{domain}."
                        f"{perturbation_name}."
                        f"{method}."
                        f"{evaluation_seed}"
                    ),
                )

                _compare(
                    audits,
                    expected["audit_snapshots"],
                    path=(
                        f"{domain}."
                        f"{perturbation_name}."
                        f"{method}."
                        f"{evaluation_seed}."
                        "audit"
                    ),
                )

                if checkpoint_sha != reference["checkpoint_sha256"]:
                    raise AssertionError("checkpoint SHA mismatch")

                reproduced += 1

            print(f"[PASS] {domain} " f"{perturbation_name} " f"{method}: 20 / 20")

    if reproduced != 120:
        raise AssertionError(f"expected 120 episodes, " f"got {reproduced}")

    print()
    print("120 / 120 reproduction episodes")
    print("Structured perturbation equality: PASS")
    print("Safety accounting: PASS")
    print("Mechanism accounting: PASS")
    print("Grasp accounting: PASS")
    print()
    print("SPRINT 5.11 SEED-42 " "STRUCTURED REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
