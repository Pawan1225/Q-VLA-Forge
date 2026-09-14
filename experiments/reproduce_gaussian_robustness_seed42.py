"""Reproduce Sprint 5.10 seed-42 sigma-0.10 principal evidence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_gaussian_robustness import (
    run_gaussian_episode,
)

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "safety" / "gaussian-robustness" / "runs"

DOMAINS = (
    "autonomous_driving",
    "robotics",
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

SIGMA = 0.10

FLOAT_ATOL = 1.0e-10


def _slug(
    domain: str,
) -> str:
    return domain.replace(
        "_",
        "-",
    )


def _close(
    left: float,
    right: float,
    *,
    name: str,
) -> None:
    if not math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=FLOAT_ATOL,
    ):
        raise AssertionError(f"{name}: " f"{left!r} != {right!r}")


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
        _close(
            float(actual),
            float(expected),
            name=path,
        )

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
                path=f"{path}.{key}",
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
    print("=" * 72)
    print(" Q-VLA FORGE - SPRINT 5.10 " "SEED-42 SIGMA-0.10 REPRODUCTION")
    print("=" * 72)
    print()

    reproduced = 0

    for domain in DOMAINS:
        for method in METHODS:
            path = RUNS / (
                f"{_slug(domain)}-" f"{method}-" "seed-42-" "sigma-0p10.json"
            )

            reference = json.loads(path.read_text(encoding="utf-8"))

            assert reference["principal_seed"] == 42

            assert float(reference["sigma"]) == SIGMA

            for index, evaluation_seed in enumerate(EVALUATION_SEEDS):
                evidence, audits, checkpoint_sha = run_gaussian_episode(
                    domain=domain,
                    method=method,
                    principal_seed=42,
                    evaluation_seed=evaluation_seed,
                    sigma=SIGMA,
                )

                expected = reference["episodes"][index]

                expected_core = {
                    key: value
                    for key, value in expected.items()
                    if key != "audit_snapshots"
                }

                actual_core = asdict(evidence)

                _compare(
                    actual_core,
                    expected_core,
                    path=(f"{domain}." f"{method}." f"{evaluation_seed}"),
                )

                _compare(
                    audits,
                    expected["audit_snapshots"],
                    path=(f"{domain}." f"{method}." f"{evaluation_seed}." "audit"),
                )

                if checkpoint_sha != reference["checkpoint_sha256"]:
                    raise AssertionError("checkpoint SHA mismatch")

                reproduced += 1

            print(f"[PASS] {domain} " f"{method}: 20 / 20")

    if reproduced != 120:
        raise AssertionError("expected 120 reproduced episodes")

    print()
    print("120 / 120 reproduction episodes")
    print("Noise equality: PASS")
    print("Safety accounting: PASS")
    print("Mechanism accounting: PASS")
    print("Grasp accounting: PASS")
    print()
    print("SPRINT 5.10 SEED-42 " "SIGMA-0.10 REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
