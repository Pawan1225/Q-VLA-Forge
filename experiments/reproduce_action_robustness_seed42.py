"""Reproduce frozen Sprint 5.12 seed-42 action-robustness cases."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_action_robustness import (
    run_action_episode,
)

from q_vla_forge.safety.perturbations import (
    action_perturbation_by_name,
)

ROOT = Path(__file__).resolve().parents[1]

RUN_ROOT = ROOT / "results" / "safety" / "action-robustness" / "runs"

CASES = (
    (
        "autonomous_driving",
        "steering_plus_0p25",
    ),
    (
        "robotics",
        "gripper_plus_0p50",
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


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    path: str,
    atol: float = 1.0e-10,
) -> None:
    if isinstance(
        expected,
        float,
    ):
        if abs(float(actual) - expected) > atol:
            raise AssertionError(f"{path}: {actual} != {expected}")

        return

    if isinstance(
        expected,
        dict,
    ):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: dict keys differ")

        for key in expected:
            _assert_close(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                atol=atol,
            )

        return

    if isinstance(
        expected,
        list,
    ):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence length differs")

        for index, (
            actual_item,
            expected_item,
        ) in enumerate(
            zip(
                actual,
                expected,
                strict=True,
            )
        ):
            _assert_close(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                atol=atol,
            )

        return

    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def main() -> None:
    print("=" * 72)
    print(" Q-VLA FORGE - SPRINT 5.12 SEED-42 REPRODUCTION")
    print("=" * 72)
    print()

    reproduced = 0

    for (
        domain,
        perturbation_name,
    ) in CASES:
        perturbation = action_perturbation_by_name(
            domain=domain,
            name=perturbation_name,
        )

        for method in METHODS:
            path = RUN_ROOT / (
                f"{_slug(domain)}-"
                f"{_slug(perturbation_name)}-"
                f"{method}-seed-42.json"
            )

            reference = json.loads(path.read_text(encoding="utf-8"))

            expected_episodes = {
                int(episode["evaluation_seed"]): episode
                for episode in reference["episodes"]
            }

            matched = 0

            for evaluation_seed in EVALUATION_SEEDS:
                (
                    evidence,
                    audits,
                    checkpoint_sha,
                ) = run_action_episode(
                    domain=domain,
                    perturbation=perturbation,
                    method=method,
                    principal_seed=42,
                    evaluation_seed=evaluation_seed,
                )

                if checkpoint_sha != reference["checkpoint_sha256"]:
                    raise AssertionError("checkpoint SHA mismatch")

                actual = asdict(evidence)

                expected = dict(expected_episodes[evaluation_seed])

                expected.pop(
                    "audit_snapshots",
                    None,
                )

                _assert_close(
                    actual,
                    expected,
                    path=(
                        f"{domain}/"
                        f"{perturbation_name}/"
                        f"{method}/"
                        f"{evaluation_seed}"
                    ),
                )

                reference_audits = expected_episodes[evaluation_seed]["audit_snapshots"]

                _assert_close(
                    audits,
                    reference_audits,
                    path=(
                        f"{domain}/"
                        f"{perturbation_name}/"
                        f"{method}/"
                        f"{evaluation_seed}/audits"
                    ),
                    atol=1.0e-9,
                )

                matched += 1
                reproduced += 1

            print(
                "[PASS]",
                domain,
                perturbation_name,
                method,
                f"{matched} / 20",
            )

    if reproduced != 120:
        raise AssertionError(f"expected 120 episodes, got {reproduced}")

    print()
    print("120 / 120 reproduction episodes")

    print("Action perturbation equality: PASS")

    print("Safety recovery accounting: PASS")

    print("Environment-interface accounting: PASS")

    print("Lyapunov mechanism accounting: PASS")

    print("Gripper semantic accounting: PASS")

    print("Grasp accounting: PASS")

    print()
    print("SPRINT 5.12 SEED-42 ACTION REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
