"""Audit environment handling of out-of-range actions for Sprint 5.12."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from q_vla_forge.evaluation.safety_baseline import (
    environment_for_domain,
)

ROOT = Path(__file__).resolve().parents[1]


def _probe(
    *,
    domain: str,
    action: np.ndarray,
) -> None:
    env = environment_for_domain(domain)

    env.reset(seed=20_000)

    before = np.asarray(
        env.state,
        dtype=np.float32,
    ).copy()

    print()
    print(
        f"{domain} probe action:",
        action.tolist(),
    )

    print(
        "state before:",
        before.tolist(),
    )

    try:
        result = env.step(action.copy())

    except (
        AssertionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print("RESULT: REJECTED")

        print(
            "exception:",
            type(exc).__name__,
            str(exc),
        )

        return

    after = np.asarray(
        env.state,
        dtype=np.float32,
    ).copy()

    print("RESULT: ACCEPTED")

    print(
        "state after:",
        after.tolist(),
    )

    print(
        "reward:",
        float(result[1]),
    )

    print(
        "terminated:",
        bool(result[2]),
    )

    print(
        "truncated:",
        bool(result[3]),
    )


def main() -> None:
    print("=" * 72)
    print(" SPRINT 5.12 ENVIRONMENT ACTION-BOUND AUDIT")
    print("=" * 72)

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        env = environment_for_domain(domain)

        print()
        print("=" * 72)
        print(domain.upper())
        print("=" * 72)

        print()
        print("ACTION SPACE:")
        print(env.action_space)

        print()
        print("STEP SOURCE:")

        try:
            print(inspect.getsource(type(env).step))
        except (
            OSError,
            TypeError,
        ) as exc:
            print(
                "SOURCE UNAVAILABLE:",
                exc,
            )

    _probe(
        domain="autonomous_driving",
        action=np.asarray(
            [
                1.25,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    )

    _probe(
        domain="autonomous_driving",
        action=np.asarray(
            [
                0.0,
                1.25,
                -0.25,
            ],
            dtype=np.float32,
        ),
    )

    _probe(
        domain="robotics",
        action=np.asarray(
            [
                1.25,
                0.0,
                1.50,
            ],
            dtype=np.float32,
        ),
    )

    _probe(
        domain="robotics",
        action=np.asarray(
            [
                -1.25,
                0.0,
                -1.50,
            ],
            dtype=np.float32,
        ),
    )

    print()
    print("SPRINT 5.12 ENVIRONMENT ACTION-BOUND AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
