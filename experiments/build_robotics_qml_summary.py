"""Build Sprint 4.9 robotics PPO-vs-QML summary."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

DIRECTORY = Path("results") / "rl" / "qml"

OUTPUT = DIRECTORY / "sprint4-robotics-qml-summary.json"

SEEDS = (
    42,
    123,
    456,
)


def load(
    seed: int,
) -> dict[str, Any]:
    """Load one principal robotics QML artifact."""
    path = DIRECTORY / f"robotics-seed-{seed}.json"

    if not path.exists():
        raise FileNotFoundError(f"missing robotics QML artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(
    values: list[float],
) -> dict[str, float]:
    """Return mean and sample standard deviation."""
    return {
        "mean": float(statistics.mean(values)),
        "sample_standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def main() -> None:
    """Build and write the Sprint 4.9 robotics summary."""
    runs = [load(seed) for seed in SEEDS]

    for seed, run in zip(
        SEEDS,
        runs,
        strict=True,
    ):
        if run["domain"] != "robotics":
            raise RuntimeError(f"seed {seed} has wrong domain")

        if run["seed"] != seed:
            raise RuntimeError(f"seed artifact mismatch for {seed}")

        if run["total_environment_steps"] != 20_000:
            raise RuntimeError(f"seed {seed} has wrong training budget")

    best_rewards = [
        max(evaluation["mean_reward"] for evaluation in run["evaluations"])
        for run in runs
    ]

    final_rewards = [run["evaluations"][-1]["mean_reward"] for run in runs]

    final_success = [run["evaluations"][-1]["success_rate"] for run in runs]

    qml_steps = [run["comparison"]["qml_environment_steps_to_target"] for run in runs]

    improvements = [
        run["comparison"]["sample_efficiency_improvement_percent"] for run in runs
    ]

    valid_steps = [float(value) for value in qml_steps if value is not None]

    valid_improvements = [float(value) for value in improvements if value is not None]

    target_reach_count = sum(value is not None for value in qml_steps)

    robust_10_percent = bool(
        target_reach_count == 3
        and len(valid_improvements) == 3
        and statistics.mean(valid_improvements) >= 10.0
    )

    payload = {
        "sprint": "4.9",
        "domain": "robotics",
        "method": ("hybrid_quantum_classical_ppo"),
        "seeds": list(SEEDS),
        "run_count": 3,
        "budget_environment_steps_per_run": 20_000,
        "best_reward": mean_sd(best_rewards),
        "final_reward": mean_sd(final_rewards),
        "final_success_rate": mean_sd(final_success),
        "qml_target_reach_count": (target_reach_count),
        "qml_steps_to_target": (mean_sd(valid_steps) if valid_steps else None),
        "sample_efficiency_improvement_percent": (
            mean_sd(valid_improvements) if valid_improvements else None
        ),
        "robotics_robust_10_percent_criterion": (robust_10_percent),
        "robust_rule": (
            "all 3 QML seeds reach paired targets "
            "and mean environment-step reduction >= 10%"
        ),
        "parameter_accounting": {
            "classical_actor_parameters": 1382,
            "hybrid_actor_parameters": 62,
            "hybrid_quantum_parameters": 16,
        },
        "scientific_boundary": {
            "quantum_hardware_used": False,
            "quantum_speedup_claimed": False,
            "wall_clock_speedup_claimed": False,
            "cross_domain_claim_deferred": True,
        },
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=========================================")

    print(" SPRINT 4.9 ROBOTICS PPO VS HYBRID QML")

    print("=========================================")

    print(
        "Target reach:",
        target_reach_count,
        "/ 3",
    )

    print(
        "Best reward:",
        payload["best_reward"],
    )

    print(
        "Final reward:",
        payload["final_reward"],
    )

    print(
        "Final success:",
        payload["final_success_rate"],
    )

    print(
        "QML steps to target:",
        payload["qml_steps_to_target"],
    )

    print(
        "Sample-efficiency improvement:",
        payload["sample_efficiency_improvement_percent"],
    )

    print(
        "Robust >=10% criterion:",
        robust_10_percent,
    )

    print(
        "Artifact:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
