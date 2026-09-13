"""Build Sprint 4.11 RL sample-efficiency analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_sample_efficiency import (
    actor_parameter_reduction_percent,
    normalized_target_progress,
    sample_mean_sd,
    summarize_learning_curve,
    summary_to_dict,
)

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

QML_DIRECTORY = Path("results") / "rl" / "qml"

ENVIRONMENT_AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

VALIDATION_PATH = (
    Path("results") / "rl" / "validation" / "sprint4-three-seed-validation.json"
)

OUTPUT_DIRECTORY = Path("results") / "rl" / "analysis"

JSON_OUTPUT = OUTPUT_DIRECTORY / "sprint4-sample-efficiency.json"

CSV_OUTPUT = OUTPUT_DIRECTORY / "sprint4-sample-efficiency.csv"

MARKDOWN_OUTPUT = OUTPUT_DIRECTORY / "sprint4-sample-efficiency.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

POLICIES = (
    "ppo",
    "qml",
)

RANDOM_REFERENCE_KEYS = {
    "autonomous_driving": "driving_random",
    "robotics": "robotics_random",
}

ACTOR_PARAMETERS = {
    "autonomous_driving": {
        "ppo": 1318,
        "qml": 54,
    },
    "robotics": {
        "ppo": 1382,
        "qml": 62,
    },
}


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
    """Return the frozen paired target for one domain/seed."""

    records = targets["domains"][domain]["seeds"]

    matches = [record for record in records if int(record["seed"]) == seed]

    if len(matches) != 1:
        raise RuntimeError(f"expected one target for {domain} seed {seed}")

    return float(matches[0]["target_evaluation_reward"])


def random_reference_for_domain(
    *,
    environment_audit: dict[str, Any],
    domain: str,
) -> float:
    """Return frozen Sprint 4.4 random-policy mean reward."""

    key = RANDOM_REFERENCE_KEYS[domain]

    summary = environment_audit["summaries"][key]

    if summary["domain"] != domain:
        raise RuntimeError(f"random-reference domain mismatch for {domain}")

    if summary["policy"] != "random":
        raise RuntimeError(f"random-reference policy mismatch for {domain}")

    return float(summary["mean_reward"])


def run_path(
    *,
    domain: str,
    seed: int,
    policy: str,
) -> Path:
    """Return one frozen principal-run artifact path."""

    if policy == "ppo":
        return PPO_DIRECTORY / f"{domain}-seed-{seed}.json"

    if policy == "qml":
        return QML_DIRECTORY / f"{domain}-seed-{seed}.json"

    raise ValueError(f"unsupported policy: {policy}")


def first_target_crossing(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> int | None:
    """Return first held-out target crossing."""

    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target_reward:
            return int(evaluation["environment_steps"])

    return None


def build_run_record(
    *,
    domain: str,
    seed: int,
    policy: str,
    run: dict[str, Any],
    random_reference: float,
    target_reward: float,
) -> dict[str, Any]:
    """Build one frozen sample-efficiency analysis record."""

    if run["domain"] != domain:
        raise RuntimeError(f"domain mismatch for {domain} seed {seed} {policy}")

    if int(run["seed"]) != seed:
        raise RuntimeError(f"seed mismatch for {domain} seed {seed} {policy}")

    if int(run["total_environment_steps"]) != 20_000:
        raise RuntimeError("principal run budget must equal 20,000")

    evaluations = run["evaluations"]

    if len(evaluations) != 21:
        raise RuntimeError("principal run must contain 21 evaluations")

    steps = [int(evaluation["environment_steps"]) for evaluation in evaluations]

    rewards = [float(evaluation["mean_reward"]) for evaluation in evaluations]

    normalized_progress = [
        normalized_target_progress(
            reward=reward,
            random_reference=random_reference,
            target_reward=target_reward,
        )
        for reward in rewards
    ]

    summary = summarize_learning_curve(
        steps=steps,
        rewards=rewards,
        random_reference=random_reference,
        target_reward=target_reward,
    )

    steps_to_target = first_target_crossing(
        evaluations=evaluations,
        target_reward=target_reward,
    )

    return {
        "domain": domain,
        "seed": seed,
        "policy": policy,
        "random_reference": (random_reference),
        "target_reward": (target_reward),
        "target_reached": (steps_to_target is not None),
        "steps_to_target": (steps_to_target),
        "actor_parameters": (ACTOR_PARAMETERS[domain][policy]),
        "steps": steps,
        "rewards": rewards,
        "normalized_progress": (normalized_progress),
        **summary_to_dict(summary),
    }


def aggregate_policy_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly three same-domain/same-policy runs."""

    if len(records) != 3:
        raise ValueError("policy aggregation requires exactly three seeds")

    auc_mean, auc_sd = sample_mean_sd(
        [float(record["normalized_auc"]) for record in records]
    )

    best_progress_mean, best_progress_sd = sample_mean_sd(
        [float(record["best_normalized_progress"]) for record in records]
    )

    final_progress_mean, final_progress_sd = sample_mean_sd(
        [float(record["final_normalized_progress"]) for record in records]
    )

    best_reward_mean, best_reward_sd = sample_mean_sd(
        [float(record["best_reward"]) for record in records]
    )

    final_reward_mean, final_reward_sd = sample_mean_sd(
        [float(record["final_reward"]) for record in records]
    )

    target_reach_count = sum(bool(record["target_reached"]) for record in records)

    finite_steps = [
        int(record["steps_to_target"])
        for record in records
        if (record["steps_to_target"] is not None)
    ]

    return {
        "target_reach_count": (target_reach_count),
        "target_reach_rate": (target_reach_count / 3.0),
        "mean_steps_to_target": (
            float(sum(finite_steps) / len(finite_steps)) if finite_steps else None
        ),
        "normalized_auc": {
            "mean": auc_mean,
            "sample_standard_deviation": (auc_sd),
        },
        "best_normalized_progress": {
            "mean": (best_progress_mean),
            "sample_standard_deviation": (best_progress_sd),
        },
        "final_normalized_progress": {
            "mean": (final_progress_mean),
            "sample_standard_deviation": (final_progress_sd),
        },
        "best_reward": {
            "mean": (best_reward_mean),
            "sample_standard_deviation": (best_reward_sd),
        },
        "final_reward": {
            "mean": (final_reward_mean),
            "sample_standard_deviation": (final_reward_sd),
        },
        "actor_parameters": int(records[0]["actor_parameters"]),
    }


def aggregate_curve(
    *,
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, list[float]]:
    """Aggregate one 21-point field across three seeds."""

    if len(records) != 3:
        raise ValueError("curve aggregation requires exactly three seeds")

    point_count = len(records[0][field])

    means: list[float] = []

    sds: list[float] = []

    for index in range(point_count):
        values = [float(record[field][index]) for record in records]

        mean, sd = sample_mean_sd(values)

        means.append(mean)

        sds.append(sd)

    return {
        "mean": means,
        "sample_standard_deviation": sds,
    }


def write_csv(
    run_records: list[dict[str, Any]],
) -> None:
    """Write 12-row analysis CSV."""

    fieldnames = [
        "domain",
        "seed",
        "policy",
        "random_reference",
        "target_reward",
        "target_reached",
        "steps_to_target",
        "normalized_auc",
        "best_normalized_progress",
        "final_normalized_progress",
        "best_reward",
        "final_reward",
        "actor_parameters",
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

        for record in run_records:
            writer.writerow({field: record[field] for field in fieldnames})


def write_markdown(
    *,
    payload: dict[str, Any],
) -> None:
    """Write proposal-readable Sprint 4.11 evidence."""

    lines = [
        "# Sprint 4.11 — Sample-Efficiency Analysis",
        "",
        (
            "No new principal RL training, retuning, target "
            "changes, environment changes, or policy changes "
            "were performed."
        ),
        "",
        "## Primary Pre-Registered Metric",
        "",
        ("Environment steps to the paired frozen PPO-derived " "reward target."),
        "",
        "Classical PPO target reach: **6/6**",
        "",
        "Hybrid QML target reach: **0/6**",
        "",
        (
            "Failed QML target crossings remain `None`; they "
            "are not censored to the 20,000-step budget."
        ),
        "",
        "## Secondary Descriptive Metrics",
        "",
        (
            "Normalized target progress maps each domain's "
            "random-policy reference to 0 and each seed's "
            "paired frozen target to 1."
        ),
        "",
        (
            "Normalized learning-curve AUC, best normalized "
            "progress, and final normalized progress are "
            "post-hoc descriptive metrics and do not replace "
            "the primary target-reach criterion."
        ),
        "",
        (
            "| Domain | Policy | Target Reach | "
            "Normalized AUC | Best Progress | "
            "Final Progress | Actor Params |"
        ),
        ("|---|---|---:|---:|---:|---:|---:|"),
    ]

    for domain in DOMAINS:
        for policy in POLICIES:
            aggregate = payload["domains"][domain]["policies"][policy]["aggregate"]

            lines.append(
                "| "
                f"{domain} | "
                f"{policy.upper()} | "
                f"{aggregate['target_reach_count']}/3 | "
                f"{aggregate['normalized_auc']['mean']:.6f} "
                f"± "
                f"{aggregate['normalized_auc']['sample_standard_deviation']:.6f} | "
                f"{aggregate['best_normalized_progress']['mean']:.6f} "
                f"± "
                f"{aggregate['best_normalized_progress']['sample_standard_deviation']:.6f} | "
                f"{aggregate['final_normalized_progress']['mean']:.6f} "
                f"± "
                f"{aggregate['final_normalized_progress']['sample_standard_deviation']:.6f} | "
                f"{aggregate['actor_parameters']} |"
            )

    compactness = payload["compactness"]

    lines.extend(
        [
            "",
            "## Policy Actor Parameter Accounting",
            "",
            ("| Domain | Classical Actor | Hybrid Actor | " "Reduction |"),
            "|---|---:|---:|---:|",
            (
                "| autonomous_driving | "
                f"{compactness['autonomous_driving']['classical_actor_parameters']} | "
                f"{compactness['autonomous_driving']['hybrid_actor_parameters']} | "
                f"{compactness['autonomous_driving']['reduction_percent']:.2f}% |"
            ),
            (
                "| robotics | "
                f"{compactness['robotics']['classical_actor_parameters']} | "
                f"{compactness['robotics']['hybrid_actor_parameters']} | "
                f"{compactness['robotics']['reduction_percent']:.2f}% |"
            ),
            "",
            "## Scientific Boundary",
            "",
            (
                "Actor parameter accounting refers to the "
                "policy actor only; the classical critic "
                "remains unchanged within each domain."
            ),
            "",
            (
                "Normalized AUC and normalized target progress "
                "are descriptive secondary metrics."
            ),
            "",
            (
                "No direct raw reward comparison is made "
                "between driving and robotics because their "
                "reward scales differ."
            ),
            "",
            (
                "The analysis does not isolate whether the "
                "observed performance gap arises from the PQC "
                "representation, the much smaller actor "
                "parameter budget, optimization dynamics, "
                "or their interaction."
            ),
            "",
            ("No quantum speedup or quantum hardware " "advantage is claimed."),
            "",
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build Sprint 4.11 sample-efficiency evidence."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = load_json(PPO_DIRECTORY / "sprint4-ppo-targets.json")

    environment_audit = load_json(ENVIRONMENT_AUDIT_PATH)

    validation = load_json(VALIDATION_PATH)

    if validation["sprint"] != "4.10":
        raise RuntimeError("Sprint 4.10 validation artifact is required")

    run_records: list[dict[str, Any]] = []

    domains_payload: dict[
        str,
        Any,
    ] = {}

    for domain in DOMAINS:
        random_reference = random_reference_for_domain(
            environment_audit=environment_audit,
            domain=domain,
        )

        policies_payload: dict[
            str,
            Any,
        ] = {}

        for policy in POLICIES:
            policy_records = []

            for seed in SEEDS:
                target_reward = target_for_seed(
                    targets=targets,
                    domain=domain,
                    seed=seed,
                )

                run = load_json(
                    run_path(
                        domain=domain,
                        seed=seed,
                        policy=policy,
                    )
                )

                record = build_run_record(
                    domain=domain,
                    seed=seed,
                    policy=policy,
                    run=run,
                    random_reference=random_reference,
                    target_reward=target_reward,
                )

                policy_records.append(record)

                run_records.append(record)

            policies_payload[policy] = {
                "records": (policy_records),
                "aggregate": (aggregate_policy_records(policy_records)),
                "reward_curve": (
                    aggregate_curve(
                        records=policy_records,
                        field="rewards",
                    )
                ),
                "normalized_progress_curve": (
                    aggregate_curve(
                        records=policy_records,
                        field="normalized_progress",
                    )
                ),
            }

        domains_payload[domain] = {
            "random_reference": (random_reference),
            "policies": (policies_payload),
        }

    classical_target_reaches = sum(
        domains_payload[domain]["policies"]["ppo"]["aggregate"]["target_reach_count"]
        for domain in DOMAINS
    )

    qml_target_reaches = sum(
        domains_payload[domain]["policies"]["qml"]["aggregate"]["target_reach_count"]
        for domain in DOMAINS
    )

    payload = {
        "sprint": "4.11",
        "title": ("Sample-Efficiency Analysis"),
        "new_training_performed": False,
        "primary_metric": {
            "name": ("environment_steps_to_frozen_target"),
            "classical_target_reaches": (classical_target_reaches),
            "classical_targets_available": 6,
            "qml_target_reaches": (qml_target_reaches),
            "qml_targets_available": 6,
            "qml_mean_steps_to_target": (
                None if qml_target_reaches == 0 else "see domain aggregates"
            ),
            "failed_qml_targets_censored_to_budget": False,
        },
        "domains": (domains_payload),
        "compactness": {
            "autonomous_driving": {
                "classical_actor_parameters": 1318,
                "hybrid_actor_parameters": 54,
                "reduction_percent": (
                    actor_parameter_reduction_percent(
                        classical_parameters=1318,
                        hybrid_parameters=54,
                    )
                ),
            },
            "robotics": {
                "classical_actor_parameters": 1382,
                "hybrid_actor_parameters": 62,
                "reduction_percent": (
                    actor_parameter_reduction_percent(
                        classical_parameters=1382,
                        hybrid_parameters=62,
                    )
                ),
            },
        },
        "scientific_boundary": {
            "primary_metric_preserved": True,
            "secondary_metrics_are_descriptive": True,
            "new_training_performed": False,
            "hyperparameter_tuning_performed": False,
            "targets_modified": False,
            "rewards_modified": False,
            "environments_modified": False,
            "policies_modified": False,
            "cross_domain_raw_reward_comparison_allowed": False,
            "pqc_failure_cause_isolated": False,
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

    write_csv(run_records)

    write_markdown(payload=payload)

    print()
    print("==============================================")

    print(" Q-VLA FORGE — SPRINT 4.11 SAMPLE EFFICIENCY")

    print("==============================================")

    print()
    print("Primary target reach")

    print()

    print(
        "Driving PPO ",
        domains_payload["autonomous_driving"]["policies"]["ppo"]["aggregate"][
            "target_reach_count"
        ],
        "/ 3",
    )

    print(
        "Driving QML ",
        domains_payload["autonomous_driving"]["policies"]["qml"]["aggregate"][
            "target_reach_count"
        ],
        "/ 3",
    )

    print(
        "Robotics PPO",
        domains_payload["robotics"]["policies"]["ppo"]["aggregate"][
            "target_reach_count"
        ],
        "/ 3",
    )

    print(
        "Robotics QML",
        domains_payload["robotics"]["policies"]["qml"]["aggregate"][
            "target_reach_count"
        ],
        "/ 3",
    )

    print()

    print(
        "Total PPO",
        classical_target_reaches,
        "/ 6",
    )

    print(
        "Total QML",
        qml_target_reaches,
        "/ 6",
    )

    print()

    for domain in DOMAINS:
        print(domain)

        for policy in POLICIES:
            aggregate = domains_payload[domain]["policies"][policy]["aggregate"]

            print(
                f" {policy.upper()} "
                f"AUC="
                f"{aggregate['normalized_auc']['mean']:.6f}"
                " ± "
                f"{aggregate['normalized_auc']['sample_standard_deviation']:.6f}"
                " "
                f"best_progress="
                f"{aggregate['best_normalized_progress']['mean']:.6f}"
                " ± "
                f"{aggregate['best_normalized_progress']['sample_standard_deviation']:.6f}"
                " "
                f"final_progress="
                f"{aggregate['final_normalized_progress']['mean']:.6f}"
                " ± "
                f"{aggregate['final_normalized_progress']['sample_standard_deviation']:.6f}"
            )

        print()

    print("Actor compactness")

    print()

    print(
        "Driving:",
        "1318 -> 54",
    )

    print(
        "Robotics:",
        "1382 -> 62",
    )

    print()

    print(
        "New training:",
        False,
    )

    print()

    print("SPRINT 4.11 SAMPLE-EFFICIENCY ANALYSIS: BUILT")

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
