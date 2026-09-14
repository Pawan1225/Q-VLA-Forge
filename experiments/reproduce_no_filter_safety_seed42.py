"""Reproduce Sprint 5.4 seed-42 no-filter safety evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    episode_evidence_to_dict,
    load_frozen_ppo_policy,
    run_no_filter_episode,
)

ROOT = Path(__file__).resolve().parents[1]

RUN_DIRECTORY = ROOT / "results" / "safety" / "baseline" / "runs"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

PRINCIPAL_SEED = 42

FLOAT_TOLERANCE = 1e-12


def require_equal(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def require_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def compare_episode(
    *,
    reproduced: dict[str, Any],
    stored: dict[str, Any],
    label: str,
) -> None:
    exact_fields = (
        "domain",
        "principal_seed",
        "evaluation_seed",
        "method",
        "robustness_condition",
        "success",
        "episode_length",
        "violation_step_count",
        "constraint_violation_count",
        "critical_violation_step_count",
        "intervention_count",
        "policy_action_out_of_bounds_count",
        "category_violation_counts",
    )

    for field in exact_fields:
        require_equal(
            reproduced[field],
            stored[field],
            label=f"{label}.{field}",
        )

    float_fields = (
        "reward",
        "total_action_correction_l2",
        "max_action_correction_l2",
    )

    for field in float_fields:
        require_close(
            reproduced[field],
            stored[field],
            label=f"{label}.{field}",
        )


def reproduce_domain(
    domain: str,
) -> None:
    run_path = RUN_DIRECTORY / f"{domain}-seed-{PRINCIPAL_SEED}.json"

    stored = json.loads(run_path.read_text(encoding="utf-8"))

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain=domain,
        principal_seed=PRINCIPAL_SEED,
    )

    stored_episodes = stored["episodes"]

    require_equal(
        len(stored_episodes),
        20,
        label=f"{domain}.stored_episode_count",
    )

    for index, evaluation_seed in enumerate(SAFETY_EVALUATION_SEEDS):
        reproduced_episode = run_no_filter_episode(
            policy=policy,
            evaluation_seed=evaluation_seed,
        )

        reproduced = episode_evidence_to_dict(reproduced_episode)

        compare_episode(
            reproduced=reproduced,
            stored=stored_episodes[index],
            label=(f"{domain}.episode[{index}]"),
        )

    print(f"[PASS] {domain} seed 42: " "20 / 20 episodes reproduced")


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.4 SEED-42 REPRODUCTION")
    print("=" * 68)
    print()

    for domain in DOMAINS:
        reproduce_domain(domain)

    print()
    print("Reproduced episodes: 40 / 40")
    print("Artifact modification: NONE")
    print()
    print("SPRINT 5.4 SEED-42 REPRODUCTION: PASS")


if __name__ == "__main__":
    main()
