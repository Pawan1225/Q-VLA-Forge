"""Independent verification for Sprint 5.4 no-filter safety baseline."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIRECTORY = ROOT / "results" / "safety" / "baseline"

RUN_DIRECTORY = BASELINE_DIRECTORY / "runs"

PPO_DIRECTORY = ROOT / "results" / "rl" / "ppo"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

EVALUATION_SEEDS = list(
    range(
        20_000,
        20_020,
    )
)

FLOAT_TOLERANCE = 1e-12


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


def require(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def require_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        label,
    )


def sample_sd(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def verify_run(
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    path = RUN_DIRECTORY / f"{domain}-seed-{seed}.json"

    require(
        path.exists(),
        f"{domain} seed {seed} run exists",
    )

    run = json.loads(path.read_text(encoding="utf-8"))

    require(
        run["domain"] == domain,
        f"{domain} seed {seed} domain",
    )

    require(
        run["principal_seed"] == seed,
        f"{domain} seed {seed} principal seed",
    )

    require(
        run["safety_method"] == "none",
        f"{domain} seed {seed} method NONE",
    )

    require(
        run["robustness_condition"] == "clean",
        f"{domain} seed {seed} condition CLEAN",
    )

    require(
        run["new_training_performed"] is False,
        f"{domain} seed {seed} no new training",
    )

    require(
        run["safety_filter_applied"] is False,
        f"{domain} seed {seed} no safety filter",
    )

    require(
        run["evaluation_seeds"] == EVALUATION_SEEDS,
        f"{domain} seed {seed} evaluation seeds",
    )

    require(
        run["episode_count"] == 20,
        f"{domain} seed {seed} 20 episodes",
    )

    checkpoint_path = ROOT / run["policy_checkpoint"]

    require(
        checkpoint_path.exists(),
        f"{domain} seed {seed} checkpoint exists",
    )

    require(
        sha256_file(checkpoint_path) == run["policy_checkpoint_sha256"],
        f"{domain} seed {seed} checkpoint SHA256",
    )

    recovery_path = ROOT / run["checkpoint_recovery_record"]

    require(
        recovery_path.exists(),
        f"{domain} seed {seed} recovery record exists",
    )

    require(
        sha256_file(recovery_path) == run["checkpoint_recovery_record_sha256"],
        f"{domain} seed {seed} recovery SHA256",
    )

    episodes = run["episodes"]

    require(
        [episode["evaluation_seed"] for episode in episodes] == EVALUATION_SEEDS,
        f"{domain} seed {seed} episode ordering",
    )

    total_steps = 0
    violation_steps = 0
    constraint_violations = 0
    critical_steps = 0
    interventions = 0
    out_of_bounds = 0

    rewards = []
    successes = []
    lengths = []

    category_counts: dict[
        str,
        int,
    ] = {}

    for episode in episodes:
        require(
            episode["method"] == "none",
            (f"{domain} seed {seed} " f"episode {episode['evaluation_seed']} method"),
        )

        require(
            episode["robustness_condition"] == "clean",
            (
                f"{domain} seed {seed} "
                f"episode {episode['evaluation_seed']} condition"
            ),
        )

        require(
            episode["intervention_count"] == 0,
            (
                f"{domain} seed {seed} "
                f"episode {episode['evaluation_seed']} no intervention"
            ),
        )

        require_close(
            episode["total_action_correction_l2"],
            0.0,
            label=(
                f"{domain} seed {seed} "
                f"episode {episode['evaluation_seed']} correction"
            ),
        )

        require_close(
            episode["max_action_correction_l2"],
            0.0,
            label=(
                f"{domain} seed {seed} "
                f"episode {episode['evaluation_seed']} max correction"
            ),
        )

        require(
            episode["policy_action_out_of_bounds_count"] == 0,
            (
                f"{domain} seed {seed} "
                f"episode {episode['evaluation_seed']} bounded policy"
            ),
        )

        total_steps += int(episode["episode_length"])

        violation_steps += int(episode["violation_step_count"])

        constraint_violations += int(episode["constraint_violation_count"])

        critical_steps += int(episode["critical_violation_step_count"])

        interventions += int(episode["intervention_count"])

        out_of_bounds += int(episode["policy_action_out_of_bounds_count"])

        rewards.append(float(episode["reward"]))

        successes.append(float(episode["success"]))

        lengths.append(float(episode["episode_length"]))

        for (
            category,
            count,
        ) in episode["category_violation_counts"].items():
            category_counts[category] = category_counts.get(
                category,
                0,
            ) + int(count)

    summary = run["summary"]

    require(
        summary["total_environment_steps"] == total_steps,
        f"{domain} seed {seed} total steps",
    )

    require(
        summary["violation_step_count"] == violation_steps,
        f"{domain} seed {seed} violation count",
    )

    require_close(
        summary["violation_step_rate"],
        violation_steps / total_steps,
        label=(f"{domain} seed {seed} violation-step rate"),
    )

    require(
        summary["constraint_violation_count"] == constraint_violations,
        f"{domain} seed {seed} constraint count",
    )

    require_close(
        summary["constraint_violation_rate"],
        constraint_violations / total_steps,
        label=(f"{domain} seed {seed} constraint rate"),
    )

    require(
        summary["critical_violation_step_count"] == critical_steps,
        f"{domain} seed {seed} critical count",
    )

    require_close(
        summary["critical_violation_step_rate"],
        critical_steps / total_steps,
        label=(f"{domain} seed {seed} critical rate"),
    )

    require(
        summary["intervention_count"] == interventions == 0,
        f"{domain} seed {seed} zero interventions",
    )

    require(
        summary["policy_action_out_of_bounds_count"] == out_of_bounds == 0,
        f"{domain} seed {seed} zero OOB actions",
    )

    require_close(
        summary["mean_reward"],
        statistics.mean(rewards),
        label=(f"{domain} seed {seed} mean reward"),
    )

    require_close(
        summary["reward_sample_sd"],
        sample_sd(rewards),
        label=(f"{domain} seed {seed} reward sample SD"),
    )

    require_close(
        summary["success_rate"],
        statistics.mean(successes),
        label=(f"{domain} seed {seed} success rate"),
    )

    require_close(
        summary["mean_episode_length"],
        statistics.mean(lengths),
        label=(f"{domain} seed {seed} mean episode length"),
    )

    require(
        summary["category_violation_counts"] == dict(sorted(category_counts.items())),
        f"{domain} seed {seed} category counts",
    )

    historical_path = PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

    historical = json.loads(historical_path.read_text(encoding="utf-8"))

    final_evaluation = historical["evaluations"][-1]

    require_close(
        summary["mean_reward"],
        final_evaluation["mean_reward"],
        label=(
            f"{domain} seed {seed} " "reward matches frozen Sprint 4 final evaluation"
        ),
    )

    require_close(
        summary["success_rate"],
        final_evaluation["success_rate"],
        label=(
            f"{domain} seed {seed} " "success matches frozen Sprint 4 final evaluation"
        ),
    )

    return run


def verify_aggregate(
    runs: dict[
        str,
        list[dict[str, Any]],
    ],
) -> None:
    summary_path = BASELINE_DIRECTORY / "sprint5-no-filter-safety-summary.json"

    csv_path = BASELINE_DIRECTORY / "sprint5-no-filter-safety-summary.csv"

    markdown_path = BASELINE_DIRECTORY / "sprint5-no-filter-safety-summary.md"

    require(
        summary_path.exists(),
        "aggregate JSON exists",
    )

    require(
        csv_path.exists(),
        "aggregate CSV exists",
    )

    require(
        markdown_path.exists(),
        "aggregate Markdown exists",
    )

    aggregate = json.loads(summary_path.read_text(encoding="utf-8"))

    require(
        aggregate["method"] == "none",
        "aggregate method NONE",
    )

    require(
        aggregate["condition"] == "clean",
        "aggregate condition CLEAN",
    )

    require(
        aggregate["new_training_performed"] is False,
        "aggregate no new training",
    )

    for domain in DOMAINS:
        seed_summaries = [run["summary"] for run in runs[domain]]

        rates = [float(item["violation_step_rate"]) for item in seed_summaries]

        domain_summary = aggregate["domains"][domain]

        require_close(
            domain_summary["violation_step_rate"]["mean"],
            statistics.mean(rates),
            label=(f"{domain} aggregate violation mean"),
        )

        require_close(
            domain_summary["violation_step_rate"]["sample_sd"],
            sample_sd(rates),
            label=(f"{domain} aggregate violation SD"),
        )

        require(
            domain_summary["baseline_has_observed_violations"]
            is any(rate > 0.0 for rate in rates),
            (f"{domain} observed-violation flag"),
        )

    require(
        aggregate["claims"]["clipping_improvement"] == "NOT_YET_TESTED",
        "clipping claim remains NOT_YET_TESTED",
    )

    require(
        aggregate["claims"]["lyapunov_improvement"] == "NOT_YET_TESTED",
        "Lyapunov claim remains NOT_YET_TESTED",
    )

    require(
        aggregate["claims"]["robustness_improvement"] == "NOT_YET_TESTED",
        "robustness claim remains NOT_YET_TESTED",
    )


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.4 FINAL VERIFICATION")
    print("=" * 68)
    print()

    run_map: dict[
        str,
        list[dict[str, Any]],
    ] = {domain: [] for domain in DOMAINS}

    for domain in DOMAINS:
        for seed in SEEDS:
            run_map[domain].append(
                verify_run(
                    domain=domain,
                    seed=seed,
                )
            )

    verify_aggregate(run_map)

    total_episodes = sum(
        run["episode_count"] for domain_runs in run_map.values() for run in domain_runs
    )

    require(
        total_episodes == 120,
        "total principal episodes = 120",
    )

    print()
    print("Principal runs: 6 / 6")
    print("Principal episodes: 120 / 120")
    print("Policy action bounds: PASS")
    print("Sprint 4 reward/success regression: PASS")
    print("NONE intervention invariants: PASS")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.4 NO-FILTER SAFETY BASELINE: PASS")


if __name__ == "__main__":
    main()
