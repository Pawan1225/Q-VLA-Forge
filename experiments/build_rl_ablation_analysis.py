"""Build Sprint 4.12 matched-budget classical-vs-QML ablation analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_ablation import (
    PairedAblationRecord,
    paired_delta,
    paired_win_counts,
    record_to_dict,
    sample_mean_sd,
    validate_parameter_match,
)
from q_vla_forge.evaluation.rl_sample_efficiency import (
    normalized_target_progress,
    summarize_learning_curve,
    summary_to_dict,
)

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

QML_DIRECTORY = Path("results") / "rl" / "qml"

ABLATION_DIRECTORY = Path("results") / "rl" / "ablation"

ENVIRONMENT_AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

TARGET_PATH = PPO_DIRECTORY / "sprint4-ppo-targets.json"

SPRINT411_ANALYSIS_PATH = (
    Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.json"
)

JSON_OUTPUT = ABLATION_DIRECTORY / "sprint4-classical-vs-qml-ablation.json"

CSV_OUTPUT = ABLATION_DIRECTORY / "sprint4-classical-vs-qml-ablation.csv"

MARKDOWN_OUTPUT = ABLATION_DIRECTORY / "sprint4-classical-vs-qml-ablation.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)

EXPECTED_STEPS = list(
    range(
        0,
        20_001,
        1_000,
    )
)

RANDOM_REFERENCE_KEYS = {
    "autonomous_driving": "driving_random",
    "robotics": "robotics_random",
}

EXPECTED_PARAMETER_COUNTS = {
    "autonomous_driving": {
        "matched": {
            "actor": 54,
            "critic": 1249,
            "total": 1303,
        },
        "qml": {
            "actor": 54,
            "critic": 1249,
            "total": 1303,
        },
        "full_ppo_actor": 1318,
    },
    "robotics": {
        "matched": {
            "actor": 62,
            "critic": 1313,
            "total": 1375,
        },
        "qml": {
            "actor": 62,
            "critic": 1313,
            "total": 1375,
        },
        "full_ppo_actor": 1382,
    },
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def target_for_seed(
    *,
    targets: dict[str, Any],
    domain: str,
    seed: int,
) -> float:
    """Return one frozen Sprint 4.5 paired PPO target."""

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
    """Return frozen Sprint 4.4 random-policy reward reference."""

    key = RANDOM_REFERENCE_KEYS[domain]

    summary = environment_audit["summaries"][key]

    if summary["domain"] != domain:
        raise RuntimeError(f"random-reference domain mismatch for {domain}")

    if summary["policy"] != "random":
        raise RuntimeError(f"random-reference policy mismatch for {domain}")

    return float(summary["mean_reward"])


def matched_run_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return one matched-classical principal run."""

    return ABLATION_DIRECTORY / f"matched-classical-{domain}-seed-{seed}.json"


def qml_run_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return one frozen QML principal run."""

    return QML_DIRECTORY / f"{domain}-seed-{seed}.json"


def first_target_crossing(
    *,
    evaluations: list[dict[str, Any]],
    target_reward: float,
) -> int | None:
    """Return first held-out evaluation reaching target."""

    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target_reward:
            return int(evaluation["environment_steps"])

    return None


def validate_evaluations(
    *,
    evaluations: list[dict[str, Any]],
    label: str,
) -> None:
    """Validate frozen 21-point evaluation schedule."""

    if len(evaluations) != 21:
        raise RuntimeError(f"{label} must contain 21 evaluations")

    steps = [int(evaluation["environment_steps"]) for evaluation in evaluations]

    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"{label} evaluation schedule mismatch")


def build_learning_record(
    *,
    domain: str,
    seed: int,
    policy: str,
    evaluations: list[dict[str, Any]],
    random_reference: float,
    target_reward: float,
    actor_parameters: int,
) -> dict[str, Any]:
    """Build one Sprint 4.11-compatible learning-curve record."""

    validate_evaluations(
        evaluations=evaluations,
        label=f"{domain} seed {seed} {policy}",
    )

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
        "actor_parameters": (actor_parameters),
        "steps": steps,
        "rewards": rewards,
        "normalized_progress": (normalized_progress),
        **summary_to_dict(summary),
    }


def matched_parameter_counts(
    run: dict[str, Any],
) -> tuple[
    int,
    int,
    int,
]:
    """Return actor, critic, and total matched-classical counts."""

    counts = run["parameter_counts"]

    return (
        int(counts["actor_parameters"]),
        int(counts["critic_parameters"]),
        int(counts["total_parameters"]),
    )


def qml_parameter_counts(
    run: dict[str, Any],
) -> tuple[
    int,
    int,
    int,
]:
    """Return actor, critic, and total QML counts."""

    return (
        int(run["actor_parameters"]),
        int(run["critic_parameters"]),
        int(run["total_parameters"]),
    )


def validate_exact_parameter_match(
    *,
    domain: str,
    matched_run: dict[str, Any],
    qml_run: dict[str, Any],
) -> None:
    """Enforce exact actor, critic, and total matching."""

    (
        matched_actor,
        matched_critic,
        matched_total,
    ) = matched_parameter_counts(matched_run)

    (
        qml_actor,
        qml_critic,
        qml_total,
    ) = qml_parameter_counts(qml_run)

    validate_parameter_match(
        matched_actor_parameters=(matched_actor),
        qml_actor_parameters=(qml_actor),
    )

    expected = EXPECTED_PARAMETER_COUNTS[domain]

    if matched_actor != expected["matched"]["actor"]:
        raise RuntimeError(f"{domain} matched actor count mismatch")

    if qml_actor != expected["qml"]["actor"]:
        raise RuntimeError(f"{domain} QML actor count mismatch")

    if matched_critic != qml_critic:
        raise RuntimeError(f"{domain} critic parameter mismatch")

    if matched_total != qml_total:
        raise RuntimeError(f"{domain} total parameter mismatch")

    if matched_critic != expected["matched"]["critic"]:
        raise RuntimeError(f"{domain} matched critic count mismatch")

    if matched_total != expected["matched"]["total"]:
        raise RuntimeError(f"{domain} matched total count mismatch")


def build_paired_record(
    *,
    domain: str,
    seed: int,
    matched_run: dict[str, Any],
    qml_run: dict[str, Any],
    random_reference: float,
    target_reward: float,
) -> tuple[
    PairedAblationRecord,
    dict[str, Any],
    dict[str, Any],
]:
    """Build one parameter-matched classical-vs-QML comparison."""

    if matched_run["domain"] != domain:
        raise RuntimeError("matched-classical domain mismatch")

    if int(matched_run["seed"]) != seed:
        raise RuntimeError("matched-classical seed mismatch")

    if qml_run["domain"] != domain:
        raise RuntimeError("QML domain mismatch")

    if int(qml_run["seed"]) != seed:
        raise RuntimeError("QML seed mismatch")

    if matched_run["policy"] != "matched_classical":
        raise RuntimeError("unexpected matched-classical policy label")

    matched_steps = int(matched_run["summary"]["total_environment_steps"])

    qml_steps = int(qml_run["total_environment_steps"])

    if matched_steps != 20_000 or qml_steps != 20_000:
        raise RuntimeError("principal ablation runs must use 20,000 steps")

    validate_exact_parameter_match(
        domain=domain,
        matched_run=matched_run,
        qml_run=qml_run,
    )

    (
        matched_actor,
        _,
        _,
    ) = matched_parameter_counts(matched_run)

    (
        qml_actor,
        _,
        _,
    ) = qml_parameter_counts(qml_run)

    matched_record = build_learning_record(
        domain=domain,
        seed=seed,
        policy="matched_classical",
        evaluations=matched_run["evaluations"],
        random_reference=(random_reference),
        target_reward=(target_reward),
        actor_parameters=(matched_actor),
    )

    qml_record = build_learning_record(
        domain=domain,
        seed=seed,
        policy="qml",
        evaluations=qml_run["evaluations"],
        random_reference=(random_reference),
        target_reward=(target_reward),
        actor_parameters=(qml_actor),
    )

    paired = PairedAblationRecord(
        domain=domain,
        seed=seed,
        matched_actor_parameters=(matched_actor),
        qml_actor_parameters=(qml_actor),
        matched_target_reached=bool(matched_record["target_reached"]),
        qml_target_reached=bool(qml_record["target_reached"]),
        matched_normalized_auc=float(matched_record["normalized_auc"]),
        qml_normalized_auc=float(qml_record["normalized_auc"]),
        normalized_auc_delta=(
            paired_delta(
                matched_value=float(matched_record["normalized_auc"]),
                qml_value=float(qml_record["normalized_auc"]),
            )
        ),
        matched_best_progress=float(matched_record["best_normalized_progress"]),
        qml_best_progress=float(qml_record["best_normalized_progress"]),
        best_progress_delta=(
            paired_delta(
                matched_value=float(matched_record["best_normalized_progress"]),
                qml_value=float(qml_record["best_normalized_progress"]),
            )
        ),
        matched_final_progress=float(matched_record["final_normalized_progress"]),
        qml_final_progress=float(qml_record["final_normalized_progress"]),
        final_progress_delta=(
            paired_delta(
                matched_value=float(matched_record["final_normalized_progress"]),
                qml_value=float(qml_record["final_normalized_progress"]),
            )
        ),
    )

    return (
        paired,
        matched_record,
        qml_record,
    )


def aggregate_learning_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly three same-domain policy runs."""

    if len(records) != 3:
        raise ValueError("learning aggregation requires exactly three seeds")

    auc_mean, auc_sd = sample_mean_sd(
        [float(record["normalized_auc"]) for record in records]
    )

    best_mean, best_sd = sample_mean_sd(
        [float(record["best_normalized_progress"]) for record in records]
    )

    final_mean, final_sd = sample_mean_sd(
        [float(record["final_normalized_progress"]) for record in records]
    )

    target_reach_count = sum(bool(record["target_reached"]) for record in records)

    return {
        "target_reach_count": (target_reach_count),
        "target_reach_rate": (target_reach_count / 3.0),
        "normalized_auc": {
            "mean": auc_mean,
            "sample_standard_deviation": (auc_sd),
        },
        "best_normalized_progress": {
            "mean": best_mean,
            "sample_standard_deviation": (best_sd),
        },
        "final_normalized_progress": {
            "mean": final_mean,
            "sample_standard_deviation": (final_sd),
        },
        "actor_parameters": int(records[0]["actor_parameters"]),
    }


def aggregate_paired_records(
    records: list[PairedAblationRecord],
) -> dict[str, Any]:
    """Aggregate paired deltas across three seeds."""

    if len(records) != 3:
        raise ValueError("paired aggregation requires exactly three seeds")

    auc_deltas = [record.normalized_auc_delta for record in records]

    best_deltas = [record.best_progress_delta for record in records]

    final_deltas = [record.final_progress_delta for record in records]

    auc_mean, auc_sd = sample_mean_sd(auc_deltas)

    best_mean, best_sd = sample_mean_sd(best_deltas)

    final_mean, final_sd = sample_mean_sd(final_deltas)

    return {
        "normalized_auc_delta": {
            "mean": auc_mean,
            "sample_standard_deviation": (auc_sd),
            "win_counts": (paired_win_counts(auc_deltas)),
        },
        "best_progress_delta": {
            "mean": best_mean,
            "sample_standard_deviation": (best_sd),
            "win_counts": (paired_win_counts(best_deltas)),
        },
        "final_progress_delta": {
            "mean": final_mean,
            "sample_standard_deviation": (final_sd),
            "win_counts": (paired_win_counts(final_deltas)),
        },
    }


def write_csv(
    paired_records: list[PairedAblationRecord],
) -> None:
    """Write six-row matched-budget ablation table."""

    fieldnames = [
        "domain",
        "seed",
        "matched_actor_parameters",
        "qml_actor_parameters",
        "matched_target_reached",
        "qml_target_reached",
        "matched_normalized_auc",
        "qml_normalized_auc",
        "normalized_auc_delta",
        "matched_best_progress",
        "qml_best_progress",
        "best_progress_delta",
        "matched_final_progress",
        "qml_final_progress",
        "final_progress_delta",
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

        for record in paired_records:
            payload = record_to_dict(record)

            writer.writerow({field: payload[field] for field in fieldnames})


def write_markdown(
    *,
    payload: dict[str, Any],
) -> None:
    """Write proposal-readable Sprint 4.12 evidence."""

    lines = [
        "# Sprint 4.12 — Classical vs QML Matched-Budget Ablation",
        "",
        (
            "Sprint 4.12 introduces a new parameter-matched "
            "classical PPO control while reusing the frozen "
            "hybrid QML runs."
        ),
        "",
        "## Experimental Control",
        "",
        (
            "Driving matched classical and hybrid QML actors "
            "both contain 54 trainable actor parameters."
        ),
        "",
        (
            "Robotics matched classical and hybrid QML actors "
            "both contain 62 trainable actor parameters."
        ),
        "",
        (
            "Critic and total trainable parameter counts are "
            "also exactly matched within each domain."
        ),
        "",
        "## Paired Seed-Level Results",
        "",
        (
            "| Domain | Seed | Matched Params | QML Params | "
            "Matched Reach | QML Reach | Matched AUC | "
            "QML AUC | Delta AUC | Delta Best | Delta Final |"
        ),
        ("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|"),
    ]

    for record in payload["paired_records"]:
        lines.append(
            "| "
            f"{record['domain']} | "
            f"{record['seed']} | "
            f"{record['matched_actor_parameters']} | "
            f"{record['qml_actor_parameters']} | "
            f"{record['matched_target_reached']} | "
            f"{record['qml_target_reached']} | "
            f"{record['matched_normalized_auc']:.6f} | "
            f"{record['qml_normalized_auc']:.6f} | "
            f"{record['normalized_auc_delta']:.6f} | "
            f"{record['best_progress_delta']:.6f} | "
            f"{record['final_progress_delta']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Matched-Budget Results",
            "",
            (
                "| Domain | Method | Actor Params | "
                "Target Reach | Normalized AUC | "
                "Best Progress | Final Progress |"
            ),
            ("|---|---|---:|---:|---:|---:|---:|"),
        ]
    )

    for domain in DOMAINS:
        domain_payload = payload["domains"][domain]

        for method in (
            "full_ppo",
            "matched_classical",
            "qml",
        ):
            aggregate = domain_payload["methods"][method]

            lines.append(
                "| "
                f"{domain} | "
                f"{method} | "
                f"{aggregate['actor_parameters']} | "
                f"{aggregate['target_reach_count']}/3 | "
                f"{aggregate['normalized_auc']['mean']:.6f} ± "
                f"{aggregate['normalized_auc']['sample_standard_deviation']:.6f} | "
                f"{aggregate['best_normalized_progress']['mean']:.6f} ± "
                f"{aggregate['best_normalized_progress']['sample_standard_deviation']:.6f} | "
                f"{aggregate['final_normalized_progress']['mean']:.6f} ± "
                f"{aggregate['final_normalized_progress']['sample_standard_deviation']:.6f} |"
            )

        delta = domain_payload["paired_delta_summary"]

        lines.extend(
            [
                "",
                f"### {domain} paired deltas",
                "",
                ("Delta is defined as matched classical minus hybrid QML."),
                "",
                (
                    "Normalized AUC delta: "
                    f"{delta['normalized_auc_delta']['mean']:.6f} ± "
                    f"{delta['normalized_auc_delta']['sample_standard_deviation']:.6f}"
                ),
                "",
                (
                    "Best-progress delta: "
                    f"{delta['best_progress_delta']['mean']:.6f} ± "
                    f"{delta['best_progress_delta']['sample_standard_deviation']:.6f}"
                ),
                "",
                (
                    "Final-progress delta: "
                    f"{delta['final_progress_delta']['mean']:.6f} ± "
                    f"{delta['final_progress_delta']['sample_standard_deviation']:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Scientific Boundary",
            "",
            (
                "This is a matched-parameter representation ablation, "
                "not a pure test of quantum mechanics."
            ),
            "",
            (
                "The classical and PQC cores have equal trainable "
                "parameter counts, but their parameterizations and "
                "optimization landscapes differ."
            ),
            "",
            (
                "Normalized AUC, best normalized progress, and final "
                "normalized progress remain descriptive secondary metrics."
            ),
            "",
            (
                "No formal statistical significance test is performed "
                "because each domain contains only three principal seeds."
            ),
            "",
            (
                "No quantum speedup, quantum hardware advantage, or "
                "universal claim about quantum versus classical policies "
                "is made."
            ),
            "",
            (
                "The six hybrid QML runs are reused from the frozen "
                "Sprint 4.8/4.9 evidence and were not retrained."
            ),
            "",
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build Sprint 4.12 ablation evidence."""

    ABLATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = load_json(TARGET_PATH)

    environment_audit = load_json(ENVIRONMENT_AUDIT_PATH)

    sprint411 = load_json(SPRINT411_ANALYSIS_PATH)

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("historical frozen PPO target artifact is invalid")

    if sprint411["new_training_performed"] is not False:
        raise RuntimeError("Sprint 4.11 analysis boundary is invalid")

    paired_records: list[PairedAblationRecord] = []

    matched_records_by_domain: dict[
        str,
        list[dict[str, Any]],
    ] = {domain: [] for domain in DOMAINS}

    qml_records_by_domain: dict[
        str,
        list[dict[str, Any]],
    ] = {domain: [] for domain in DOMAINS}

    paired_by_domain: dict[
        str,
        list[PairedAblationRecord],
    ] = {domain: [] for domain in DOMAINS}

    for domain in DOMAINS:
        random_reference = random_reference_for_domain(
            environment_audit=(environment_audit),
            domain=domain,
        )

        for seed in SEEDS:
            target_reward = target_for_seed(
                targets=targets,
                domain=domain,
                seed=seed,
            )

            matched_run = load_json(
                matched_run_path(
                    domain=domain,
                    seed=seed,
                )
            )

            qml_run = load_json(
                qml_run_path(
                    domain=domain,
                    seed=seed,
                )
            )

            (
                paired_record,
                matched_record,
                qml_record,
            ) = build_paired_record(
                domain=domain,
                seed=seed,
                matched_run=matched_run,
                qml_run=qml_run,
                random_reference=(random_reference),
                target_reward=(target_reward),
            )

            paired_records.append(paired_record)

            paired_by_domain[domain].append(paired_record)

            matched_records_by_domain[domain].append(matched_record)

            qml_records_by_domain[domain].append(qml_record)

    domains_payload: dict[
        str,
        dict[str, Any],
    ] = {}

    total_matched_reaches = 0

    total_qml_reaches = 0

    for domain in DOMAINS:
        matched_aggregate = aggregate_learning_records(
            matched_records_by_domain[domain]
        )

        qml_aggregate = aggregate_learning_records(qml_records_by_domain[domain])

        sprint411_ppo = sprint411["domains"][domain]["policies"]["ppo"]["aggregate"]

        full_ppo = {
            "target_reach_count": int(sprint411_ppo["target_reach_count"]),
            "target_reach_rate": float(sprint411_ppo["target_reach_rate"]),
            "normalized_auc": (sprint411_ppo["normalized_auc"]),
            "best_normalized_progress": (sprint411_ppo["best_normalized_progress"]),
            "final_normalized_progress": (sprint411_ppo["final_normalized_progress"]),
            "actor_parameters": (EXPECTED_PARAMETER_COUNTS[domain]["full_ppo_actor"]),
        }

        total_matched_reaches += int(matched_aggregate["target_reach_count"])

        total_qml_reaches += int(qml_aggregate["target_reach_count"])

        domains_payload[domain] = {
            "methods": {
                "full_ppo": full_ppo,
                "matched_classical": (matched_aggregate),
                "qml": (qml_aggregate),
            },
            "paired_delta_summary": (
                aggregate_paired_records(paired_by_domain[domain])
            ),
        }

    payload = {
        "sprint": "4.12",
        "title": ("Classical vs QML Matched-Budget Ablation"),
        "principal_seeds": list(SEEDS),
        "new_training_performed": True,
        "new_qml_training_performed": False,
        "matched_classical_training_performed": True,
        "principal_matched_classical_runs": 6,
        "reused_principal_qml_runs": 6,
        "parameter_matched_ablation": True,
        "primary_target_reach": {
            "matched_classical": {
                "reached": (total_matched_reaches),
                "available": 6,
            },
            "qml": {
                "reached": (total_qml_reaches),
                "available": 6,
            },
        },
        "paired_records": [record_to_dict(record) for record in paired_records],
        "domains": (domains_payload),
        "parameter_matching": {
            "autonomous_driving": {
                "matched_actor": 54,
                "qml_actor": 54,
                "matched_critic": 1249,
                "qml_critic": 1249,
                "matched_total": 1303,
                "qml_total": 1303,
            },
            "robotics": {
                "matched_actor": 62,
                "qml_actor": 62,
                "matched_critic": 1313,
                "qml_critic": 1313,
                "matched_total": 1375,
                "qml_total": 1375,
            },
        },
        "scientific_boundary": {
            "qml_architecture_modified": False,
            "ppo_protocol_modified": False,
            "targets_modified": False,
            "environments_modified": False,
            "qml_retrained": False,
            "matched_classical_new_training": True,
            "parameter_matched_representation_ablation": True,
            "formal_significance_test_performed": False,
            "statistical_significance_claimed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_advantage_claimed": False,
            "universal_quantum_claimed": False,
        },
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(paired_records)

    write_markdown(payload=payload)

    print()
    print("=" * 58)
    print(" Q-VLA FORGE — SPRINT 4.12 MATCHED-BUDGET ABLATION")
    print("=" * 58)
    print()

    for domain in DOMAINS:
        methods = payload["domains"][domain]["methods"]

        delta = payload["domains"][domain]["paired_delta_summary"]

        print(domain)

        print(
            "  Full PPO target reach:",
            methods["full_ppo"]["target_reach_count"],
            "/ 3",
        )

        print(
            "  Matched target reach:",
            methods["matched_classical"]["target_reach_count"],
            "/ 3",
        )

        print(
            "  QML target reach:",
            methods["qml"]["target_reach_count"],
            "/ 3",
        )

        print(
            "  Matched normalized AUC:",
            methods["matched_classical"]["normalized_auc"]["mean"],
            "±",
            methods["matched_classical"]["normalized_auc"]["sample_standard_deviation"],
        )

        print(
            "  QML normalized AUC:",
            methods["qml"]["normalized_auc"]["mean"],
            "±",
            methods["qml"]["normalized_auc"]["sample_standard_deviation"],
        )

        print(
            "  Mean Delta AUC:",
            delta["normalized_auc_delta"]["mean"],
            "±",
            delta["normalized_auc_delta"]["sample_standard_deviation"],
        )

        print(
            "  AUC wins:",
            delta["normalized_auc_delta"]["win_counts"],
        )

        print()

    print(
        "Matched target reaches:",
        total_matched_reaches,
        "/ 6",
    )

    print(
        "QML target reaches:",
        total_qml_reaches,
        "/ 6",
    )

    print(
        "New matched-classical training:",
        True,
    )

    print(
        "New QML training:",
        False,
    )

    print()
    print("SPRINT 4.12 ABLATION ANALYSIS: BUILT")

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
