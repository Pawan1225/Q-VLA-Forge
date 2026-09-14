"""Independent verification for Sprint 5.5 heuristic clipping."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

NONE_DIRECTORY = ROOT / "results" / "safety" / "baseline" / "runs"

CLIPPING_DIRECTORY = ROOT / "results" / "safety" / "clipping" / "runs"

SUMMARY_PATH = (
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json"
)

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


def load_json(
    path: Path,
) -> dict[str, Any]:
    require(
        path.exists(),
        f"artifact exists: {path.name}",
    )

    return json.loads(path.read_text(encoding="utf-8"))


def verify_pair(
    *,
    domain: str,
    seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    none_path = NONE_DIRECTORY / f"{domain}-seed-{seed}.json"

    clipping_path = CLIPPING_DIRECTORY / f"{domain}-seed-{seed}.json"

    none = load_json(none_path)

    clipping = load_json(clipping_path)

    require(
        none["domain"] == clipping["domain"] == domain,
        f"{domain} seed {seed} domain equality",
    )

    require(
        none["principal_seed"] == clipping["principal_seed"] == seed,
        f"{domain} seed {seed} principal-seed equality",
    )

    require(
        none["policy_checkpoint_sha256"] == clipping["policy_checkpoint_sha256"],
        f"{domain} seed {seed} checkpoint SHA equality",
    )

    require(
        none["checkpoint_recovery_record_sha256"]
        == clipping["checkpoint_recovery_record_sha256"],
        f"{domain} seed {seed} recovery-record SHA equality",
    )

    require(
        none["training_budget"] == clipping["training_budget"] == 20_000,
        f"{domain} seed {seed} training budget equality",
    )

    require(
        none["checkpoint_selection"] == clipping["checkpoint_selection"],
        f"{domain} seed {seed} checkpoint selection equality",
    )

    require(
        none["evaluation_seeds"] == clipping["evaluation_seeds"] == EVALUATION_SEEDS,
        f"{domain} seed {seed} evaluation seeds equality",
    )

    require(
        none["episode_count"] == clipping["episode_count"] == 20,
        f"{domain} seed {seed} 20 paired episodes",
    )

    require(
        none["safety_method"] == "none",
        f"{domain} seed {seed} NONE control",
    )

    require(
        clipping["safety_method"] == "clipping",
        f"{domain} seed {seed} CLIPPING intervention",
    )

    require(
        clipping["robustness_condition"] == "clean",
        f"{domain} seed {seed} CLEAN condition",
    )

    require(
        clipping["new_training_performed"] is False,
        f"{domain} seed {seed} no new training",
    )

    summary = clipping["summary"]

    episodes = clipping["episodes"]

    total_steps = 0

    proposed_violation_steps = 0
    executed_violation_steps = 0

    proposed_constraints = 0
    executed_constraints = 0

    critical_steps = 0
    interventions = 0

    corrections: list[float] = []

    policy_oob = 0
    executed_oob = 0

    proposed_categories: dict[
        str,
        int,
    ] = {}

    executed_categories: dict[
        str,
        int,
    ] = {}

    rule_counts: dict[
        str,
        int,
    ] = {}

    for expected_seed, episode in zip(
        EVALUATION_SEEDS,
        episodes,
        strict=True,
    ):
        require(
            episode["evaluation_seed"] == expected_seed,
            (f"{domain} seed {seed} " f"episode {expected_seed} ordering"),
        )

        require(
            episode["method"] == "clipping",
            (f"{domain} seed {seed} " f"episode {expected_seed} method"),
        )

        require(
            episode["robustness_condition"] == "clean",
            (f"{domain} seed {seed} " f"episode {expected_seed} condition"),
        )

        step_corrections = [float(value) for value in episode["action_corrections_l2"]]

        require(
            len(step_corrections) == episode["episode_length"],
            (f"{domain} seed {seed} " f"episode {expected_seed} correction length"),
        )

        reconstructed_interventions = sum(value > 1e-8 for value in step_corrections)

        require(
            reconstructed_interventions == episode["intervention_count"],
            (
                f"{domain} seed {seed} "
                f"episode {expected_seed} intervention accounting"
            ),
        )

        require_close(
            sum(step_corrections),
            episode["total_action_correction_l2"],
            label=(
                f"{domain} seed {seed} " f"episode {expected_seed} correction total"
            ),
        )

        require_close(
            max(
                step_corrections,
                default=0.0,
            ),
            episode["max_action_correction_l2"],
            label=(
                f"{domain} seed {seed} " f"episode {expected_seed} correction maximum"
            ),
        )

        require(
            episode["policy_action_out_of_bounds_count"] == 0,
            (f"{domain} seed {seed} " f"episode {expected_seed} policy bounds"),
        )

        require(
            episode["executed_action_out_of_bounds_count"] == 0,
            (f"{domain} seed {seed} " f"episode {expected_seed} executed bounds"),
        )

        total_steps += int(episode["episode_length"])

        proposed_violation_steps += int(episode["proposed_violation_step_count"])

        executed_violation_steps += int(episode["executed_violation_step_count"])

        proposed_constraints += int(episode["proposed_constraint_violation_count"])

        executed_constraints += int(episode["executed_constraint_violation_count"])

        critical_steps += int(episode["critical_violation_step_count"])

        interventions += int(episode["intervention_count"])

        policy_oob += int(episode["policy_action_out_of_bounds_count"])

        executed_oob += int(episode["executed_action_out_of_bounds_count"])

        corrections.extend(step_corrections)

        for category, count in episode["proposed_category_violation_counts"].items():
            proposed_categories[category] = proposed_categories.get(
                category,
                0,
            ) + int(count)

        for category, count in episode["executed_category_violation_counts"].items():
            executed_categories[category] = executed_categories.get(
                category,
                0,
            ) + int(count)

        for rule, count in episode["triggered_rule_counts"].items():
            rule_counts[rule] = rule_counts.get(
                rule,
                0,
            ) + int(count)

    require(
        summary["total_environment_steps"] == total_steps,
        f"{domain} seed {seed} total steps",
    )

    require(
        summary["proposed_violation_step_count"] == proposed_violation_steps,
        f"{domain} seed {seed} proposed violation count",
    )

    require(
        summary["executed_violation_step_count"] == executed_violation_steps,
        f"{domain} seed {seed} executed violation count",
    )

    require_close(
        summary["proposed_violation_step_rate"],
        proposed_violation_steps / total_steps,
        label=(f"{domain} seed {seed} proposed violation rate"),
    )

    require_close(
        summary["executed_violation_step_rate"],
        executed_violation_steps / total_steps,
        label=(f"{domain} seed {seed} executed violation rate"),
    )

    require(
        summary["proposed_constraint_violation_count"] == proposed_constraints,
        f"{domain} seed {seed} proposed constraints",
    )

    require(
        summary["executed_constraint_violation_count"] == executed_constraints,
        f"{domain} seed {seed} executed constraints",
    )

    require(
        summary["critical_violation_step_count"] == critical_steps,
        f"{domain} seed {seed} critical steps",
    )

    require(
        summary["intervention_count"] == interventions,
        f"{domain} seed {seed} interventions",
    )

    require_close(
        summary["intervention_rate"],
        interventions / total_steps,
        label=(f"{domain} seed {seed} intervention rate"),
    )

    require_close(
        summary["mean_action_correction_l2"],
        statistics.mean(corrections),
        label=(f"{domain} seed {seed} mean correction"),
    )

    sorted_corrections = sorted(corrections)

    percentile_index = 0.95 * (len(sorted_corrections) - 1)

    lower = math.floor(percentile_index)

    upper = math.ceil(percentile_index)

    if lower == upper:
        p95 = sorted_corrections[lower]
    else:
        fraction = percentile_index - lower

        p95 = (
            sorted_corrections[lower] * (1.0 - fraction)
            + sorted_corrections[upper] * fraction
        )

    require_close(
        summary["p95_action_correction_l2"],
        p95,
        label=(f"{domain} seed {seed} P95 correction"),
    )

    require_close(
        summary["max_action_correction_l2"],
        max(
            corrections,
            default=0.0,
        ),
        label=(f"{domain} seed {seed} max correction"),
    )

    require(
        summary["policy_action_out_of_bounds_count"] == policy_oob == 0,
        f"{domain} seed {seed} zero policy OOB",
    )

    require(
        summary["executed_action_out_of_bounds_count"] == executed_oob == 0,
        f"{domain} seed {seed} zero executed OOB",
    )

    require(
        summary["proposed_category_violation_counts"]
        == dict(sorted(proposed_categories.items())),
        f"{domain} seed {seed} proposed category accounting",
    )

    require(
        summary["executed_category_violation_counts"]
        == dict(sorted(executed_categories.items())),
        f"{domain} seed {seed} executed category accounting",
    )

    require(
        summary["triggered_rule_counts"] == dict(sorted(rule_counts.items())),
        f"{domain} seed {seed} rule accounting",
    )

    return (
        none,
        clipping,
    )


def verify_summary(
    pairs: dict[
        str,
        list[
            tuple[
                dict[str, Any],
                dict[str, Any],
            ]
        ],
    ],
) -> None:
    summary = load_json(SUMMARY_PATH)

    require(
        summary["comparison"] == "none_vs_clipping",
        "summary comparison",
    )

    require(
        summary["method"] == "clipping",
        "summary method",
    )

    require(
        summary["condition"] == "clean",
        "summary condition",
    )

    require(
        summary["new_training_performed"] is False,
        "summary no new training",
    )

    for domain in DOMAINS:
        none_rates = [
            float(none["summary"]["violation_step_rate"]) for none, _ in pairs[domain]
        ]

        clipping_rates = [
            float(clipping["summary"]["executed_violation_step_rate"])
            for _, clipping in pairs[domain]
        ]

        none_mean = float(statistics.mean(none_rates))

        clipping_mean = float(statistics.mean(clipping_rates))

        domain_summary = summary["domains"][domain]

        require_close(
            domain_summary["none_violation_step_rate"]["mean"],
            none_mean,
            label=(f"{domain} summary NONE mean"),
        )

        require_close(
            domain_summary["clipping_violation_step_rate"]["mean"],
            clipping_mean,
            label=(f"{domain} summary CLIPPING mean"),
        )

        expected_reduction = (
            (none_mean - clipping_mean) / none_mean if none_mean > 0.0 else None
        )

        actual_reduction = domain_summary["aggregate_violation_reduction_fraction"]

        if expected_reduction is None:
            require(
                actual_reduction is None,
                f"{domain} undefined reduction handling",
            )
        else:
            require_close(
                actual_reduction,
                expected_reduction,
                label=(f"{domain} aggregate reduction"),
            )

        seed_requirement = True

        for (
            none_rate,
            clipping_rate,
        ) in zip(
            none_rates,
            clipping_rates,
            strict=True,
        ):
            if none_rate > 0.0:
                seed_requirement &= clipping_rate < none_rate
            else:
                seed_requirement &= clipping_rate <= none_rate

        require(
            domain_summary["pilot_criterion"]["seed_level_safety_requirement_pass"]
            == seed_requirement,
            f"{domain} seed criterion reconstruction",
        )

        require(
            domain_summary["pilot_criterion"][
                "mean_violation_reduction_at_least_20_percent"
            ]
            == (expected_reduction is not None and expected_reduction >= 0.20),
            f"{domain} 20-percent criterion reconstruction",
        )

        require(
            domain_summary["pilot_criterion"]["overall_pass"] is True,
            f"{domain} empirical effectiveness SUPPORTED",
        )

    require(
        summary["claims"]["lyapunov_improvement"] == "NOT_YET_TESTED",
        "Lyapunov claim remains NOT_YET_TESTED",
    )

    require(
        summary["claims"]["robustness_improvement"] == "NOT_YET_TESTED",
        "robustness claim remains NOT_YET_TESTED",
    )

    require(
        summary["claims"]["formal_safety_guarantee"] is False,
        "no formal safety guarantee",
    )

    require(
        summary["claims"]["production_safety_claim"] is False,
        "no production safety claim",
    )

    require(
        summary["claims"]["quantum_clipping_claim"] is False,
        "no quantum clipping claim",
    )


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.5 FINAL VERIFICATION")
    print("=" * 68)
    print()

    pairs: dict[
        str,
        list[
            tuple[
                dict[str, Any],
                dict[str, Any],
            ]
        ],
    ] = {domain: [] for domain in DOMAINS}

    for domain in DOMAINS:
        for seed in SEEDS:
            pairs[domain].append(
                verify_pair(
                    domain=domain,
                    seed=seed,
                )
            )

    verify_summary(pairs)

    print()
    print("Principal clipping runs: 6 / 6")
    print("Principal clipping episodes: 120 / 120")
    print("Frozen checkpoint equality: PASS")
    print("Evaluation-seed equality: PASS")
    print("Executed action bounds: PASS")
    print("Correction accounting: PASS")
    print("NONE comparison: PASS")
    print("Driving empirical effectiveness: SUPPORTED")
    print("Robotics empirical effectiveness: SUPPORTED")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.5 HEURISTIC CLIPPING: PASS")


if __name__ == "__main__":
    main()
