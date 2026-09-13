"""Build Sprint 4.10 three-seed RL validation evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_three_seed_validation import (
    PRINCIPAL_SEEDS,
    build_seed_validation_record,
    domain_summary_to_dict,
    seed_record_to_dict,
    summarize_domain,
)

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

QML_DIRECTORY = Path("results") / "rl" / "qml"

OUTPUT_DIRECTORY = Path("results") / "rl" / "validation"

JSON_OUTPUT = OUTPUT_DIRECTORY / "sprint4-three-seed-validation.json"

CSV_OUTPUT = OUTPUT_DIRECTORY / "sprint4-three-seed-validation.csv"

MARKDOWN_OUTPUT = OUTPUT_DIRECTORY / "sprint4-three-seed-validation.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def target_for_seed(
    *,
    targets: dict[str, Any],
    domain: str,
    seed: int,
) -> float:
    """Load one frozen target reward."""

    records = targets["domains"][domain]["seeds"]

    matches = [record for record in records if int(record["seed"]) == seed]

    if len(matches) != 1:
        raise RuntimeError("expected exactly one frozen target record")

    return float(matches[0]["target_evaluation_reward"])


def build_records(
    *,
    targets: dict[str, Any],
) -> dict[str, list[Any]]:
    """Build all paired three-seed records."""

    result: dict[
        str,
        list[Any],
    ] = {}

    for domain in DOMAINS:
        domain_records = []

        for seed in PRINCIPAL_SEEDS:
            classical_path = PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

            qml_path = QML_DIRECTORY / f"{domain}-seed-{seed}.json"

            classical_run = load_json(classical_path)

            qml_run = load_json(qml_path)

            target_reward = target_for_seed(
                targets=targets,
                domain=domain,
                seed=seed,
            )

            record = build_seed_validation_record(
                domain=domain,
                seed=seed,
                target_reward=target_reward,
                classical_run=classical_run,
                qml_run=qml_run,
            )

            domain_records.append(record)

        result[domain] = domain_records

    return result


def write_csv(
    records_by_domain: dict[
        str,
        list[Any],
    ],
) -> None:
    """Write paired validation rows as CSV."""

    fieldnames = [
        "domain",
        "seed",
        "target_reward",
        "classical_best_reward",
        "classical_final_reward",
        "classical_final_success_rate",
        "classical_steps_to_target",
        "qml_best_reward",
        "qml_final_reward",
        "qml_final_success_rate",
        "qml_steps_to_target",
        "qml_target_reached",
        "best_reward_target_gap",
        "sample_efficiency_improvement_percent",
    ]

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for domain in DOMAINS:
            for record in records_by_domain[domain]:
                writer.writerow(seed_record_to_dict(record))


def write_markdown(
    *,
    records_by_domain: dict[
        str,
        list[Any],
    ],
    summaries: dict[str, Any],
) -> None:
    """Write proposal-readable Markdown validation evidence."""

    lines = [
        "# Sprint 4.10 — Three-Seed RL Validation",
        "",
        "No new principal RL training was performed.",
        "",
        "## Paired Seed Validation",
        "",
        (
            "| Domain | Seed | Target | "
            "PPO Steps | QML Steps | "
            "QML Best | Gap to Target | "
            "QML Reached |"
        ),
        ("|---|---:|---:|---:|---:|" "---:|---:|---|"),
    ]

    for domain in DOMAINS:
        for record in records_by_domain[domain]:
            qml_steps = (
                str(record.qml_steps_to_target)
                if (record.qml_steps_to_target is not None)
                else "None"
            )

            lines.append(
                "| "
                f"{domain} | "
                f"{record.seed} | "
                f"{record.target_reward:.6f} | "
                f"{record.classical_steps_to_target} | "
                f"{qml_steps} | "
                f"{record.qml_best_reward:.6f} | "
                f"{record.best_reward_target_gap:.6f} | "
                f"{record.qml_target_reached} |"
            )

    lines.extend(
        [
            "",
            "## Domain Summary",
            "",
        ]
    )

    for domain in DOMAINS:
        summary = summaries[domain]

        lines.extend(
            [
                f"### {domain}",
                "",
                (
                    "Classical target reach: "
                    f"{summary.classical_target_reach_count}/3"
                ),
                "",
                ("Hybrid QML target reach: " f"{summary.qml_target_reach_count}/3"),
                "",
                ("Robust >=10% criterion: " f"{summary.robust_10_percent_criterion}"),
                "",
                (
                    "QML best reward: "
                    f"{summary.qml_best_reward_mean:.6f} "
                    f"± {summary.qml_best_reward_sd:.6f}"
                ),
                "",
                (
                    "QML final reward: "
                    f"{summary.qml_final_reward_mean:.6f} "
                    f"± {summary.qml_final_reward_sd:.6f}"
                ),
                "",
                (
                    "QML final success: "
                    f"{summary.qml_final_success_mean:.6f} "
                    f"± {summary.qml_final_success_sd:.6f}"
                ),
                "",
                (
                    "QML best-reward target gap: "
                    f"{summary.qml_best_reward_target_gap_mean:.6f} "
                    f"± {summary.qml_best_reward_target_gap_sd:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Scientific Boundary",
            "",
            (
                "Target gaps are descriptive and are not "
                "sample-efficiency measurements."
            ),
            "",
            (
                "Reward magnitudes and target gaps are not "
                "compared directly across domains because "
                "the reward scales differ."
            ),
            "",
            ("No quantum speedup or hardware quantum " "advantage is claimed."),
            "",
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build all Sprint 4.10 validation artifacts."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = load_json(PPO_DIRECTORY / "sprint4-ppo-targets.json")

    records_by_domain = build_records(targets=targets)

    summaries = {
        domain: summarize_domain(
            domain=domain,
            records=records_by_domain[domain],
        )
        for domain in DOMAINS
    }

    total_qml_target_reaches = sum(
        summary.qml_target_reach_count for summary in summaries.values()
    )

    payload = {
        "sprint": "4.10",
        "title": ("Three-Seed RL Validation"),
        "principal_training_performed": False,
        "principal_seed_count_per_domain": 3,
        "principal_seeds": list(PRINCIPAL_SEEDS),
        "domains": {
            domain: {
                "records": [
                    seed_record_to_dict(record) for record in records_by_domain[domain]
                ],
                "summary": (domain_summary_to_dict(summaries[domain])),
            }
            for domain in DOMAINS
        },
        "global_validation": {
            "principal_classical_runs": 6,
            "principal_qml_runs": 6,
            "principal_runs_total": 12,
            "qml_targets_reached": (total_qml_target_reaches),
            "qml_targets_available": 6,
            "driving_seed42_reproduction_exists": (
                QML_DIRECTORY / "autonomous_driving-seed-42-repro.json"
            ).exists(),
            "robotics_seed42_reproduction_exists": (
                QML_DIRECTORY / "robotics-seed-42-repro.json"
            ).exists(),
        },
        "scientific_boundary": {
            "new_training_performed": False,
            "hyperparameter_tuning_performed": False,
            "targets_modified": False,
            "environments_modified": False,
            "target_gap_is_descriptive_only": True,
            "cross_domain_reward_scale_comparison_allowed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_advantage_claimed": False,
        },
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(records_by_domain)

    write_markdown(
        records_by_domain=records_by_domain,
        summaries=summaries,
    )

    print()
    print("============================================")
    print(" Q-VLA FORGE — SPRINT 4.10 THREE-SEED VALIDATION")
    print("============================================")

    for domain in DOMAINS:
        summary = summaries[domain]

        print()
        print(domain)

        print(
            " Classical target reach:",
            summary.classical_target_reach_count,
            "/ 3",
        )

        print(
            " QML target reach:",
            summary.qml_target_reach_count,
            "/ 3",
        )

        print(
            " QML best reward:",
            f"{summary.qml_best_reward_mean:.6f}",
            "±",
            f"{summary.qml_best_reward_sd:.6f}",
        )

        print(
            " QML target-gap:",
            f"{summary.qml_best_reward_target_gap_mean:.6f}",
            "±",
            f"{summary.qml_best_reward_target_gap_sd:.6f}",
        )

        print(
            " Robust >=10%:",
            summary.robust_10_percent_criterion,
        )

    print()

    print(
        "QML targets reached:",
        total_qml_target_reaches,
        "/ 6",
    )

    print(
        "New training performed:",
        False,
    )

    print()

    print("SPRINT 4.10 VALIDATION ARTIFACT: BUILT")

    print(
        "JSON:",
        JSON_OUTPUT,
    )

    print(
        "CSV:",
        CSV_OUTPUT,
    )

    print(
        "Markdown:",
        MARKDOWN_OUTPUT,
    )


if __name__ == "__main__":
    main()
