"""Build Sprint 7.4 final QML ablation artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.qml_ablation import (
    CLAIM_CONTROLS,
    DOMAINS,
    EXECUTION_BACKEND,
    POLICIES,
    QUANTUM_HARDWARE_USED,
    REQUIRED_SEEDS,
    load_canonical_qml_ablation,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "qml-ablation"


def _mean_sample_std(
    values: list[float],
) -> dict[str, float | None]:
    """Return arithmetic mean and sample SD."""

    if not values:
        return {
            "mean": None,
            "sample_standard_deviation": None,
        }

    mean = statistics.fmean(values)

    sample_std = statistics.stdev(values) if len(values) >= 2 else None

    return {
        "mean": mean,
        "sample_standard_deviation": sample_std,
    }


def _policy_label(
    policy: str,
) -> str:
    if policy == "ppo_mlp":
        return "PPO + MLP"

    if policy == "ppo_pqc":
        return "PPO + PQC"

    raise ValueError(f"Unsupported policy: {policy}")


def _aggregate_primary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate primary PPO/MLP vs PPO/PQC records."""

    output: dict[str, Any] = {}

    for domain in DOMAINS:
        output[domain] = {}

        for policy in POLICIES:
            selected = [
                record
                for record in records
                if (record["domain"] == domain and record["policy"] == policy)
            ]

            if len(selected) != 3:
                raise ValueError(f"Expected three records for " f"{domain}/{policy}.")

            reached = [record for record in selected if record["target_reached"]]

            steps = [
                float(record["environment_steps_to_target"])
                for record in reached
                if record["environment_steps_to_target"] is not None
            ]

            episodes = [
                float(record["episodes_to_target"])
                for record in reached
                if record["episodes_to_target"] is not None
            ]

            rewards = [
                float(record["final_evaluation_reward"])
                for record in selected
                if record["final_evaluation_reward"] is not None
            ]

            success_rates = [
                float(record["final_success_rate"])
                for record in selected
                if record["final_success_rate"] is not None
            ]

            training_seconds = [
                float(record["training_seconds"])
                for record in selected
                if record["training_seconds"] is not None
            ]

            actor_parameters = {int(record["actor_parameters"]) for record in selected}

            if len(actor_parameters) != 1:
                raise ValueError(
                    f"Actor parameter count changed "
                    f"across seeds for {domain}/{policy}."
                )

            output[domain][policy] = {
                "target_reach_count": len(reached),
                "target_reach_total": len(selected),
                "all_seeds_reached_target": (len(reached) == len(selected)),
                "environment_steps_to_target": (_mean_sample_std(steps)),
                "episodes_to_target": (_mean_sample_std(episodes)),
                "final_evaluation_reward": (_mean_sample_std(rewards)),
                "final_success_rate": (_mean_sample_std(success_rates)),
                "training_seconds": (_mean_sample_std(training_seconds)),
                "actor_parameters": next(iter(actor_parameters)),
            }

    return output


def _aggregate_controls(
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate parameter-matched classical controls."""

    output: dict[str, Any] = {}

    for domain in DOMAINS:
        selected = [record for record in controls if record["domain"] == domain]

        if len(selected) != 3:
            raise ValueError(f"Expected three matched controls " f"for {domain}.")

        reached = [record for record in selected if record["target_reached"]]

        steps = [
            float(record["environment_steps_to_target"])
            for record in reached
            if record["environment_steps_to_target"] is not None
        ]

        episodes = [
            float(record["episodes_to_target"])
            for record in reached
            if record["episodes_to_target"] is not None
        ]

        rewards = [float(record["final_evaluation_reward"]) for record in selected]

        success_rates = [float(record["final_success_rate"]) for record in selected]

        training_seconds = [float(record["training_seconds"]) for record in selected]

        actor_parameters = {int(record["actor_parameters"]) for record in selected}

        if len(actor_parameters) != 1:
            raise ValueError(
                f"Matched actor parameter count changed " f"across seeds for {domain}."
            )

        output[domain] = {
            "target_reach_count": len(reached),
            "target_reach_total": len(selected),
            "environment_steps_to_target": (_mean_sample_std(steps)),
            "episodes_to_target": (_mean_sample_std(episodes)),
            "final_evaluation_reward": (_mean_sample_std(rewards)),
            "final_success_rate": (_mean_sample_std(success_rates)),
            "training_seconds": (_mean_sample_std(training_seconds)),
            "actor_parameters": next(iter(actor_parameters)),
        }

    return output


def _build_payload() -> dict[str, Any]:
    evidence = load_canonical_qml_ablation(ROOT)

    primary_records = [asdict(record) for record in evidence.primary_records]

    matched_controls = [
        asdict(record) for record in evidence.matched_classical_controls
    ]

    compactness = [dict(record) for record in evidence.compactness]

    primary_aggregate = _aggregate_primary(primary_records)

    control_aggregate = _aggregate_controls(matched_controls)

    return {
        "sprint": "7.4",
        "title": "Final QML Ablation",
        "new_training": False,
        "new_qml_training": False,
        "new_experiments": False,
        "required_seeds": list(REQUIRED_SEEDS),
        "protocol": {
            "domains": list(DOMAINS),
            "primary_policies": list(POLICIES),
            "primary_metric": ("environment_steps_to_target"),
            "secondary_metrics": [
                "episodes_to_target",
                "final_evaluation_reward",
                "final_success_rate",
                "training_seconds",
                "policy_parameters",
            ],
            "execution_backend": (EXECUTION_BACKEND),
            "quantum_hardware_used": (QUANTUM_HARDWARE_USED),
            "non_attainment_rule": (
                "target_reached=false implies "
                "environment_steps_to_target=null "
                "and episodes_to_target=null"
            ),
            "maximum_environment_steps_is_not_convergence": True,
        },
        "primary_records": (primary_records),
        "matched_classical_controls": (matched_controls),
        "primary_aggregate": (primary_aggregate),
        "matched_classical_aggregate": (control_aggregate),
        "compactness": compactness,
        "cross_domain_summary": {
            "classical_ppo_target_reach": {
                "reached": sum(
                    record["target_reached"]
                    for record in primary_records
                    if record["policy"] == "ppo_mlp"
                ),
                "available": 6,
            },
            "qml_pqc_target_reach": {
                "reached": sum(
                    record["target_reached"]
                    for record in primary_records
                    if record["policy"] == "ppo_pqc"
                ),
                "available": 6,
            },
            "matched_classical_target_reach": {
                "reached": sum(record["target_reached"] for record in matched_controls),
                "available": 6,
            },
        },
        "scientific_conclusion": (
            "The evaluated PQC policy used substantially fewer "
            "trainable actor parameters than the classical PPO "
            "actor, but did not demonstrate a sample-efficiency "
            "or performance advantage over the classical PPO "
            "reference under the frozen Phase 1 protocol."
        ),
        "claim_controls": dict(CLAIM_CONTROLS),
        "source_provenance": sorted(
            {
                source
                for record in (primary_records + matched_controls)
                for source in record["source_artifacts"]
            }
        ),
    }


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_primary_csv(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "domain",
        "policy",
        "seed",
        "target_reached",
        "environment_steps_to_target",
        "episodes_to_target",
        "final_evaluation_reward",
        "final_success_rate",
        "training_seconds",
        "actor_parameters",
        "execution_backend",
        "quantum_hardware_used",
        "source_artifacts",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in payload["primary_records"]:
            row = dict(record)

            row["source_artifacts"] = ";".join(row["source_artifacts"])

            writer.writerow(row)


def _write_plot_data(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "record_type",
        "domain",
        "policy",
        "seed",
        "target_reached",
        "environment_steps_to_target",
        "final_evaluation_reward",
        "final_success_rate",
        "actor_parameters",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in payload["primary_records"]:
            writer.writerow(
                {
                    "record_type": "primary",
                    "domain": record["domain"],
                    "policy": record["policy"],
                    "seed": record["seed"],
                    "target_reached": (record["target_reached"]),
                    "environment_steps_to_target": (
                        record["environment_steps_to_target"]
                    ),
                    "final_evaluation_reward": (record["final_evaluation_reward"]),
                    "final_success_rate": (record["final_success_rate"]),
                    "actor_parameters": (record["actor_parameters"]),
                }
            )

        for record in payload["matched_classical_controls"]:
            writer.writerow(
                {
                    "record_type": "control",
                    "domain": record["domain"],
                    "policy": ("matched_classical"),
                    "seed": record["seed"],
                    "target_reached": (record["target_reached"]),
                    "environment_steps_to_target": (
                        record["environment_steps_to_target"]
                    ),
                    "final_evaluation_reward": (record["final_evaluation_reward"]),
                    "final_success_rate": (record["final_success_rate"]),
                    "actor_parameters": (record["actor_parameters"]),
                }
            )


def _format_mean_sd(
    payload: dict[str, float | None],
    *,
    decimals: int = 3,
) -> str:
    """Format a mean/sample-SD pair for Markdown output."""

    mean = payload["mean"]

    sample_std = payload["sample_standard_deviation"]

    if mean is None:
        return "Not reached"

    if sample_std is None:
        return f"{float(mean):.{decimals}f}"

    return f"{float(mean):.{decimals}f} +/- " f"{float(sample_std):.{decimals}f}"


def _build_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge - Sprint 7.4 Final QML Ablation",
        "",
        (
            "Frozen Sprint 4 evidence only. "
            "No new PPO training, QML training, circuit tuning, "
            "target modification, or new experiments."
        ),
        "",
        (
            "The primary comparison is PPO + MLP versus PPO + PQC. "
            "The matched-classical actor is retained separately as "
            "a parameter-matched control."
        ),
        "",
    ]

    for domain in DOMAINS:
        heading = "Autonomous Driving" if domain == "autonomous_driving" else "Robotics"

        lines.extend(
            [
                f"## {heading}",
                "",
                (
                    "| Policy | Target Reach | "
                    "Steps to Target | Final Reward | "
                    "Final Success | Actor Params |"
                ),
                ("|---|---:|---:|---:|---:|---:|"),
            ]
        )

        for policy in POLICIES:
            aggregate = payload["primary_aggregate"][domain][policy]

            reach = (
                f"{aggregate['target_reach_count']}/"
                f"{aggregate['target_reach_total']}"
            )

            steps = _format_mean_sd(
                aggregate["environment_steps_to_target"],
                decimals=1,
            )

            reward = _format_mean_sd(aggregate["final_evaluation_reward"])

            success = _format_mean_sd(aggregate["final_success_rate"])

            lines.append(
                f"| {_policy_label(policy)} | "
                f"{reach} | "
                f"{steps} | "
                f"{reward} | "
                f"{success} | "
                f"{aggregate['actor_parameters']} |"
            )

        lines.extend(
            [
                "",
                "### Parameter-Matched Classical Control",
                "",
            ]
        )

        control = payload["matched_classical_aggregate"][domain]

        lines.extend(
            [
                (
                    f"- Target reach: "
                    f"{control['target_reach_count']}/"
                    f"{control['target_reach_total']}"
                ),
                (
                    "- Steps to target: "
                    + _format_mean_sd(
                        control["environment_steps_to_target"],
                        decimals=1,
                    )
                ),
                (f"- Actor parameters: " f"{control['actor_parameters']}"),
                "",
            ]
        )

        compactness = next(
            record for record in payload["compactness"] if record["domain"] == domain
        )

        lines.extend(
            [
                "### Actor Compactness",
                "",
                (
                    f"- Classical PPO actor: "
                    f"{compactness['classical_actor_parameters']} "
                    "parameters"
                ),
                (
                    f"- PQC actor: "
                    f"{compactness['qml_actor_parameters']} "
                    "parameters"
                ),
                (
                    f"- Parameter reduction: "
                    f"{compactness['parameter_reduction_percent']:.2f}%"
                ),
                "",
            ]
        )

    cross_domain = payload["cross_domain_summary"]

    lines.extend(
        [
            "## Cross-Domain Result",
            "",
            (
                "- Classical PPO target attainment: "
                f"{cross_domain['classical_ppo_target_reach']['reached']}/"
                f"{cross_domain['classical_ppo_target_reach']['available']}"
            ),
            (
                "- PPO + PQC target attainment: "
                f"{cross_domain['qml_pqc_target_reach']['reached']}/"
                f"{cross_domain['qml_pqc_target_reach']['available']}"
            ),
            (
                "- Matched-classical control target attainment: "
                f"{cross_domain['matched_classical_target_reach']['reached']}/"
                f"{cross_domain['matched_classical_target_reach']['available']}"
            ),
            "",
            "## Scientific Conclusion",
            "",
            payload["scientific_conclusion"],
            "",
            (
                "Failure to reach the frozen performance target is "
                "represented as a null steps-to-target value. The "
                "20,000-step training budget is not treated as a "
                "measured convergence value."
            ),
            "",
            (
                "Actor compactness is reported as an architecture "
                "observation and is not treated as evidence of "
                "sample-efficiency or performance superiority."
            ),
            "",
            "## Claim Controls",
            "",
            "- Quantum advantage: blocked",
            "- Quantum speedup: blocked",
            "- QML sample-efficiency advantage: blocked",
            "- QML performance superiority: blocked",
            "- QML convergence superiority: blocked",
            "- QPU advantage: blocked",
            "- Production VLA superiority: blocked",
            "",
            "## Execution Boundary",
            "",
            "- Quantum hardware used: false",
            "- Execution backend: simulator",
            "",
        ]
    )

    return "\n".join(lines)


def _validate_payload(
    payload: dict[str, Any],
) -> None:
    """Builder-side final integrity check."""

    if len(payload["primary_records"]) != 12:
        raise ValueError("Expected 12 primary records.")

    if len(payload["matched_classical_controls"]) != 6:
        raise ValueError("Expected six matched controls.")

    cross_domain = payload["cross_domain_summary"]

    if cross_domain["classical_ppo_target_reach"]["reached"] != 6:
        raise ValueError("Frozen PPO reach must remain 6/6.")

    if cross_domain["qml_pqc_target_reach"]["reached"] != 0:
        raise ValueError("Frozen QML reach must remain 0/6.")

    if cross_domain["matched_classical_target_reach"]["reached"] != 1:
        raise ValueError("Frozen matched-classical reach must remain 1/6.")

    for record in payload["primary_records"]:
        if (
            not record["target_reached"]
            and record["environment_steps_to_target"] is not None
        ):
            raise ValueError(
                "Non-reaching primary record " "has a steps-to-target value."
            )

        if (
            record["policy"] == "ppo_pqc"
            and record["environment_steps_to_target"] == 20_000
        ):
            raise ValueError(
                "QML training budget cannot be " "substituted as steps-to-target."
            )

    for control in payload["matched_classical_controls"]:
        if (
            not control["target_reached"]
            and control["environment_steps_to_target"] is not None
        ):
            raise ValueError(
                "Non-reaching matched control " "has a steps-to-target value."
            )

    for value in payload["claim_controls"].values():
        if value is not False:
            raise ValueError(
                "All Sprint 7.4 superiority/advantage "
                "claim controls must remain blocked."
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = _build_payload()

    _validate_payload(payload)

    json_path = OUTPUT_DIR / "qml-ablation.json"

    csv_path = OUTPUT_DIR / "qml-ablation.csv"

    markdown_path = OUTPUT_DIR / "qml-ablation.md"

    plot_path = OUTPUT_DIR / "qml-ablation-plot-data.csv"

    _write_json(
        json_path,
        payload,
    )

    _write_primary_csv(
        csv_path,
        payload,
    )

    _write_plot_data(
        plot_path,
        payload,
    )

    markdown_path.write_text(
        _build_markdown(payload),
        encoding="utf-8",
    )

    print("=" * 64)
    print(" SPRINT 7.4 QML ABLATION BUILD")
    print("=" * 64)
    print()

    print("Artifacts:")
    print("  qml-ablation.json             PASS")
    print("  qml-ablation.csv              PASS")
    print("  qml-ablation.md               PASS")
    print("  qml-ablation-plot-data.csv    PASS")
    print()

    print("Target attainment:")
    print("  Classical PPO / MLP           6/6")
    print("  QML / PQC                     0/6")
    print("  Matched classical control     1/6")
    print()

    for record in payload["compactness"]:
        print(
            f"Compactness {record['domain']}: "
            f"{record['parameter_reduction_percent']:.2f}%"
        )

    print()
    print("SPRINT 7.4 QML ABLATION BUILD: PASS")


if __name__ == "__main__":
    main()
