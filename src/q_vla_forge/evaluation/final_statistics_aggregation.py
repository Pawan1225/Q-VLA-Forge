"""Sprint 7.2 canonical Phase 1 statistical aggregation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_statistics import (
    LOCKED_SEEDS,
    SeedMetric,
    summarize_metric,
)

BASELINE_SOURCE = "results/baseline-validation-summary.json"

COMPRESSION_SOURCE = "results/compression/compression-validation-summary.json"

TRAINING_SOURCE = "results/training/training-validation-summary.json"

RL_QML_SOURCE = "results/rl/ablation/sprint4-classical-vs-qml-ablation.json"

SAFETY_CLEAN_SOURCE = (
    "results/safety/consolidated/sprint5-clean-three-seed-summary.json"
)

ROBUSTNESS_SOURCES = (
    (
        "gaussian",
        "results/safety/consolidated/sprint5-gaussian-three-seed-summary.json",
    ),
    (
        "structured_state",
        (
            "results/safety/consolidated/"
            "sprint5-structured-state-three-seed-summary.json"
        ),
    ),
    (
        "action",
        "results/safety/consolidated/sprint5-action-three-seed-summary.json",
    ),
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def make_statistical_record(
    *,
    domain: str,
    experiment: str,
    method: str,
    metric: str,
    seeds: list[int],
    values: list[float],
    source_artifacts: list[str],
    unit: str | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """Build one recomputed three-seed statistical record."""

    if tuple(sorted(seeds)) != LOCKED_SEEDS:
        raise ValueError(
            "Expected locked seeds "
            f"{LOCKED_SEEDS}; observed "
            f"{tuple(sorted(seeds))}"
        )

    seed_values = [
        SeedMetric(
            seed=seed,
            value=float(value),
        )
        for seed, value in zip(
            seeds,
            values,
            strict=True,
        )
    ]

    summary = summarize_metric(
        metric,
        seed_values,
    )

    return {
        "domain": domain,
        "experiment": experiment,
        "method": method,
        "metric": metric,
        "condition": condition,
        "n": summary.n,
        "seeds": [item.seed for item in summary.values],
        "values": [item.value for item in summary.values],
        "mean": summary.mean,
        "sample_std": summary.sample_std,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
        "formatted": summary.formatted,
        "unit": unit,
        "aggregation_mode": "recomputed_from_seed_values",
        "source_artifacts": source_artifacts,
    }


def make_canonical_summary_record(
    *,
    domain: str,
    experiment: str,
    method: str,
    metric: str,
    payload: dict[str, Any],
    source_artifacts: list[str],
    unit: str | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """Preserve a frozen n=3 canonical mean/sample-SD summary."""

    n = payload.get("n")

    if n != 3:
        raise ValueError(f"{metric}: expected n=3, observed {n}")

    mean_value = payload.get("mean")

    std_value = payload.get(
        "std",
        payload.get(
            "sample_std",
            payload.get("sample_standard_deviation"),
        ),
    )

    if mean_value is None:
        raise ValueError(f"{metric}: missing mean")

    if std_value is None:
        raise ValueError(f"{metric}: missing sample standard deviation")

    mean_float = float(mean_value)

    std_float = float(std_value)

    return {
        "domain": domain,
        "experiment": experiment,
        "method": method,
        "metric": metric,
        "condition": condition,
        "n": 3,
        "seeds": list(LOCKED_SEEDS),
        "values": None,
        "mean": mean_float,
        "sample_std": std_float,
        "minimum": None,
        "maximum": None,
        "formatted": (f"{mean_float:.6f} \u00b1 {std_float:.6f}"),
        "unit": unit,
        "aggregation_mode": "validated_canonical_summary",
        "source_artifacts": source_artifacts,
    }


def fixed_record(
    *,
    domain: str,
    experiment: str,
    method: str,
    quantity: str,
    value: Any,
    source_artifacts: list[str],
    unit: str | None = None,
) -> dict[str, Any]:
    """Record a deterministic/fixed quantity without fake SD."""

    return {
        "domain": domain,
        "experiment": experiment,
        "method": method,
        "quantity": quantity,
        "value": value,
        "unit": unit,
        "reporting_mode": "fixed_value",
        "source_artifacts": source_artifacts,
    }


def aggregate_baseline(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Aggregate Sprint 1 baseline statistics."""

    source_path = root / BASELINE_SOURCE

    source = load_json(source_path)

    records: list[dict[str, Any]] = []

    fixed: list[dict[str, Any]] = []

    metric_units = {
        "test_mse": None,
        "test_mae": None,
        "initial_validation_mse": None,
        "best_validation_mse": None,
        "final_validation_mse": None,
        "mean_latency_ms": "ms",
        "p95_latency_ms": "ms",
    }

    for domain_key in (
        "driving",
        "robotics",
    ):
        domain = source[domain_key]

        domain_name = domain["domain"]

        seeds = list(domain["seeds"])

        for metric, unit in metric_units.items():
            metric_payload = domain[metric]

            records.append(
                make_statistical_record(
                    domain=domain_name,
                    experiment="baseline",
                    method=source["method"],
                    metric=metric,
                    seeds=seeds,
                    values=list(metric_payload["values"]),
                    unit=unit,
                    source_artifacts=[BASELINE_SOURCE],
                )
            )

        for action_metric, metric_payload in domain.get(
            "action_mae",
            {},
        ).items():
            records.append(
                make_statistical_record(
                    domain=domain_name,
                    experiment="baseline",
                    method=source["method"],
                    metric=action_metric,
                    seeds=seeds,
                    values=list(metric_payload["values"]),
                    source_artifacts=[BASELINE_SOURCE],
                )
            )

        fixed.extend(
            [
                fixed_record(
                    domain=domain_name,
                    experiment="baseline",
                    method=source["method"],
                    quantity="parameters",
                    value=domain["parameters"],
                    source_artifacts=[BASELINE_SOURCE],
                ),
                fixed_record(
                    domain=domain_name,
                    experiment="baseline",
                    method=source["method"],
                    quantity="fp32_model_size_bytes",
                    value=domain["fp32_model_size_bytes"],
                    unit="bytes",
                    source_artifacts=[BASELINE_SOURCE],
                ),
            ]
        )

    return records, fixed


def aggregate_compression(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Aggregate Sprint 2 compression statistics."""

    source = load_json(root / COMPRESSION_SOURCE)

    records: list[dict[str, Any]] = []

    fixed: list[dict[str, Any]] = []

    statistical_metrics = {
        "baseline_test_mse": None,
        "compressed_test_mse": None,
        "mse_change_percent": "%",
        "baseline_test_mae": None,
        "compressed_test_mae": None,
        "mae_change_percent": "%",
        "baseline_mean_latency_ms": "ms",
        "compressed_mean_latency_ms": "ms",
        "latency_change_percent": "%",
    }

    fixed_metrics = {
        "compression_ratio": "x",
        "storage_reduction_percent": "%",
        "parameter_reduction_percent": "%",
    }

    for domain_key in (
        "driving",
        "robotics",
    ):
        domain = source[domain_key]

        domain_name = domain["domain"]

        for method_key in (
            "int8",
            "svd",
            "tensor_network",
        ):
            method = domain[method_key]

            seeds = list(method["seeds"])

            method_name = method["configuration"]

            for metric, unit in statistical_metrics.items():
                payload = method[metric]

                records.append(
                    make_statistical_record(
                        domain=domain_name,
                        experiment="compression",
                        method=method_name,
                        metric=metric,
                        seeds=seeds,
                        values=list(payload["values"]),
                        unit=unit,
                        source_artifacts=[COMPRESSION_SOURCE],
                    )
                )

            for metric, unit in fixed_metrics.items():
                payload = method[metric]

                values = list(payload["values"])

                if len(set(values)) != 1:
                    raise ValueError(
                        f"{method_name} {metric} is not " "invariant across seeds."
                    )

                fixed.append(
                    fixed_record(
                        domain=domain_name,
                        experiment="compression",
                        method=method_name,
                        quantity=metric,
                        value=values[0],
                        unit=unit,
                        source_artifacts=[COMPRESSION_SOURCE],
                    )
                )

            fixed.append(
                fixed_record(
                    domain=domain_name,
                    experiment="compression",
                    method=method_name,
                    quantity="all_runs_pilot_feasible",
                    value=method["all_runs_pilot_feasible"],
                    source_artifacts=[COMPRESSION_SOURCE],
                )
            )

    return records, fixed


def aggregate_training(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Aggregate Sprint 3 training-efficiency statistics."""

    source = load_json(root / TRAINING_SOURCE)

    if tuple(source["seeds"]) != LOCKED_SEEDS:
        raise ValueError("Training summary seed contract mismatch.")

    records: list[dict[str, Any]] = []

    fixed: list[dict[str, Any]] = []

    units = {
        "best_validation_loss": None,
        "final_validation_loss": None,
        "test_mse": None,
        "test_mae": None,
        "epoch_to_target": "epochs",
        "steps_to_target": "optimizer_steps",
        "samples_to_target": "samples",
        "seconds_to_target": "s",
        "epoch_reduction_percent": "%",
        "step_reduction_percent": "%",
        "sample_reduction_percent": "%",
        "wall_time_reduction_percent": "%",
    }

    for method in source["methods"]:
        domain = method["domain"]

        method_name = method["method"]

        if tuple(method["seeds"]) != LOCKED_SEEDS:
            raise ValueError(
                "Training method seed contract mismatch: " f"{domain}/{method_name}"
            )

        for metric, unit in units.items():
            payload = method.get(metric)

            if payload is None:
                continue

            records.append(
                make_canonical_summary_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    metric=metric,
                    payload=payload,
                    unit=unit,
                    source_artifacts=[TRAINING_SOURCE],
                )
            )

        fixed.extend(
            [
                fixed_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    quantity="target_reach_count",
                    value=method["target_reach"]["reached"],
                    source_artifacts=[TRAINING_SOURCE],
                ),
                fixed_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    quantity="target_reach_total",
                    value=method["target_reach"]["total"],
                    source_artifacts=[TRAINING_SOURCE],
                ),
                fixed_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    quantity="trainable_parameters",
                    value=method["trainable_parameters"]["mean"],
                    source_artifacts=[TRAINING_SOURCE],
                ),
                fixed_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    quantity="effective_parameters",
                    value=method["effective_parameters"]["mean"],
                    source_artifacts=[TRAINING_SOURCE],
                ),
                fixed_record(
                    domain=domain,
                    experiment="training_efficiency",
                    method=method_name,
                    quantity="robust_ten_percent_step_efficiency",
                    value=method["robust_ten_percent_step_efficiency"],
                    source_artifacts=[TRAINING_SOURCE],
                ),
            ]
        )

    return records, fixed


def aggregate_rl_qml(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Aggregate Sprint 4 PPO / matched-classical / QML statistics."""

    source = load_json(root / RL_QML_SOURCE)

    if tuple(source["principal_seeds"]) != LOCKED_SEEDS:
        raise ValueError("RL/QML seed contract mismatch.")

    records: list[dict[str, Any]] = []

    fixed: list[dict[str, Any]] = []

    paired = source["paired_records"]

    paired_metrics = {
        "matched_classical": {
            "normalized_auc": "matched_normalized_auc",
            "best_normalized_progress": "matched_best_progress",
            "final_normalized_progress": "matched_final_progress",
        },
        "qml": {
            "normalized_auc": "qml_normalized_auc",
            "best_normalized_progress": "qml_best_progress",
            "final_normalized_progress": "qml_final_progress",
        },
    }

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        domain_rows = [row for row in paired if row["domain"] == domain]

        for method, metrics in paired_metrics.items():
            for metric, key in metrics.items():
                rows = sorted(
                    domain_rows,
                    key=lambda row: row["seed"],
                )

                records.append(
                    make_statistical_record(
                        domain=domain,
                        experiment="rl_qml",
                        method=method,
                        metric=metric,
                        seeds=[int(row["seed"]) for row in rows],
                        values=[float(row[key]) for row in rows],
                        source_artifacts=[RL_QML_SOURCE],
                    )
                )

        methods = source["domains"][domain]["methods"]

        ppo = methods["full_ppo"]

        for metric in (
            "normalized_auc",
            "best_normalized_progress",
            "final_normalized_progress",
        ):
            payload = ppo[metric]

            canonical = {
                "mean": payload["mean"],
                "sample_standard_deviation": payload["sample_standard_deviation"],
                "n": 3,
            }

            records.append(
                make_canonical_summary_record(
                    domain=domain,
                    experiment="rl_qml",
                    method="full_ppo",
                    metric=metric,
                    payload=canonical,
                    source_artifacts=[RL_QML_SOURCE],
                )
            )

        for method_name, method in methods.items():
            fixed.extend(
                [
                    fixed_record(
                        domain=domain,
                        experiment="rl_qml",
                        method=method_name,
                        quantity="target_reach_count",
                        value=method["target_reach_count"],
                        source_artifacts=[RL_QML_SOURCE],
                    ),
                    fixed_record(
                        domain=domain,
                        experiment="rl_qml",
                        method=method_name,
                        quantity="target_reach_rate",
                        value=method["target_reach_rate"],
                        source_artifacts=[RL_QML_SOURCE],
                    ),
                    fixed_record(
                        domain=domain,
                        experiment="rl_qml",
                        method=method_name,
                        quantity="actor_parameters",
                        value=method["actor_parameters"],
                        source_artifacts=[RL_QML_SOURCE],
                    ),
                ]
            )

    return records, fixed


def aggregate_seed_rows(
    *,
    rows: list[dict[str, Any]],
    experiment: str,
    source_artifact: str,
    group_keys: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate generic frozen seed-row evidence."""

    grouped: defaultdict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        group = tuple(row.get(key) for key in group_keys)

        grouped[group].append(row)

    records: list[dict[str, Any]] = []

    for group, group_rows in grouped.items():
        domain = str(group[0])

        method = str(group[1])

        condition_parts = [str(value) for value in group[2:] if value is not None]

        condition = " / ".join(condition_parts) if condition_parts else None

        for metric in metric_names:
            usable = [row for row in group_rows if row.get(metric) is not None]

            seeds = [
                int(row["principal_seed"])
                for row in usable
                if row.get("principal_seed") is not None
            ]

            if tuple(sorted(seeds)) != LOCKED_SEEDS:
                continue

            values = [float(row[metric]) for row in usable]

            records.append(
                make_statistical_record(
                    domain=domain,
                    experiment=experiment,
                    method=method,
                    metric=metric,
                    seeds=seeds,
                    values=values,
                    condition=condition,
                    source_artifacts=[source_artifact],
                )
            )

    return records


def aggregate_safety_clean(
    root: Path,
) -> list[dict[str, Any]]:
    """Aggregate Sprint 5 clean safety seed rows."""

    source = load_json(root / SAFETY_CLEAN_SOURCE)

    if tuple(source["principal_seeds"]) != LOCKED_SEEDS:
        raise ValueError("Clean safety seed contract mismatch.")

    metrics = (
        "reward",
        "success_rate",
        "violation_step_rate",
        "intervention_rate",
        "mean_correction_l2",
    )

    return aggregate_seed_rows(
        rows=list(source["seed_rows"]),
        experiment="safety_clean",
        source_artifact=SAFETY_CLEAN_SOURCE,
        group_keys=(
            "domain",
            "method",
        ),
        metric_names=metrics,
    )


def aggregate_robustness(
    root: Path,
) -> list[dict[str, Any]]:
    """Aggregate Gaussian, structured-state, and action robustness."""

    records: list[dict[str, Any]] = []

    candidate_metrics = (
        "mean_reward",
        "reward",
        "success_rate",
        "violation_step_rate",
        "executed_violation_step_rate",
        "perturbed_violation_step_rate",
        "intervention_rate",
        "mean_safety_correction_l2",
        "mean_correction_l2",
        "recovery_rate",
        "within_filter_violation_reduction",
        "reward_delta_from_clean",
        "success_delta_from_clean",
        "executed_violation_delta_from_clean",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
    )

    for family, source_path in ROBUSTNESS_SOURCES:
        path = root / source_path

        if not path.exists():
            continue

        source = load_json(path)

        seeds = source.get("principal_seeds")

        if seeds is not None and tuple(seeds) != LOCKED_SEEDS:
            raise ValueError(f"{family} robustness seed mismatch.")

        rows = source.get("seed_rows")

        if not isinstance(
            rows,
            list,
        ):
            continue

        available_keys: set[str] = set()

        for row in rows:
            if isinstance(
                row,
                dict,
            ):
                available_keys.update(row.keys())

        metrics = tuple(
            metric for metric in candidate_metrics if metric in available_keys
        )

        possible_group_keys = [
            "domain",
            "method",
        ]

        for key in (
            "perturbation_family",
            "perturbation_name",
            "perturbation",
            "noise_std",
            "noise_level",
            "state_dimension",
            "state_component",
        ):
            if key in available_keys:
                possible_group_keys.append(key)

        valid_rows = [
            row
            for row in rows
            if isinstance(
                row,
                dict,
            )
        ]

        records.extend(
            aggregate_seed_rows(
                rows=valid_rows,
                experiment=f"robustness_{family}",
                source_artifact=source_path,
                group_keys=tuple(possible_group_keys),
                metric_names=metrics,
            )
        )

    return records


def build_final_statistics(
    root: Path,
) -> dict[str, Any]:
    """Build complete Sprint 7.2 statistics package."""

    records: list[dict[str, Any]] = []

    fixed: list[dict[str, Any]] = []

    baseline_records, baseline_fixed = aggregate_baseline(root)

    records.extend(baseline_records)

    fixed.extend(baseline_fixed)

    compression_records, compression_fixed = aggregate_compression(root)

    records.extend(compression_records)

    fixed.extend(compression_fixed)

    training_records, training_fixed = aggregate_training(root)

    records.extend(training_records)

    fixed.extend(training_fixed)

    rl_records, rl_fixed = aggregate_rl_qml(root)

    records.extend(rl_records)

    fixed.extend(rl_fixed)

    records.extend(aggregate_safety_clean(root))

    records.extend(aggregate_robustness(root))

    recomputed_count = sum(
        record["aggregation_mode"] == "recomputed_from_seed_values"
        for record in records
    )

    canonical_count = sum(
        record["aggregation_mode"] == "validated_canonical_summary"
        for record in records
    )

    return {
        "sprint": "7.2",
        "protocol": "final_mean_sample_standard_deviation",
        "status": "PASS",
        "statistics_definition": {
            "center": "arithmetic_mean",
            "spread": "sample_standard_deviation",
            "ddof": 1,
            "n": 3,
            "locked_seeds": list(LOCKED_SEEDS),
            "reporting": "mean +/- sample_std",
        },
        "records": records,
        "fixed_values": fixed,
        "record_count": len(records),
        "fixed_value_count": len(fixed),
        "aggregation_modes": {
            "recomputed_from_seed_values": recomputed_count,
            "validated_canonical_summary": canonical_count,
        },
        "new_training": False,
        "new_experiments": False,
    }


def write_statistics_csv(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write final statistical records to CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "domain",
        "experiment",
        "method",
        "metric",
        "condition",
        "n",
        "seeds",
        "values",
        "mean",
        "sample_std",
        "minimum",
        "maximum",
        "formatted",
        "unit",
        "aggregation_mode",
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

        for record in payload["records"]:
            row = dict(record)

            row["seeds"] = ",".join(str(seed) for seed in row["seeds"])

            values = row.get("values")

            if isinstance(
                values,
                list,
            ):
                row["values"] = ",".join(str(value) for value in values)
            elif values is None:
                row["values"] = ""

            row["source_artifacts"] = ";".join(row["source_artifacts"])

            writer.writerow(row)


def build_statistics_markdown(
    payload: dict[str, Any],
) -> str:
    """Build compact proposal-facing statistics report."""

    lines = [
        "# Q-VLA Forge - Sprint 7.2 Final Statistics",
        "",
        (
            "All statistical results use arithmetic mean "
            "and sample standard deviation across locked "
            "seeds 42, 123, and 456."
        ),
        "",
        (
            "Deterministic architectural quantities are "
            "reported separately as fixed values and are "
            "not represented as artificial +/- 0 statistics."
        ),
        "",
        "## Baseline Verification",
        "",
        "| Domain | Metric | Mean +/- Sample SD |",
        "|---|---|---:|",
    ]

    baseline_metrics = {
        "test_mse",
        "test_mae",
        "mean_latency_ms",
        "p95_latency_ms",
    }

    for record in payload["records"]:
        if record["experiment"] == "baseline" and record["metric"] in baseline_metrics:
            unit = f" {record['unit']}" if record["unit"] else ""

            lines.append(
                f"| {record['domain']} | "
                f"{record['metric']} | "
                f"{record['formatted']}{unit} |"
            )

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            (f"- Statistical records: " f"{payload['record_count']}"),
            (f"- Fixed-value records: " f"{payload['fixed_value_count']}"),
            (
                "- Recomputed directly from seed values: "
                f"{payload['aggregation_modes']['recomputed_from_seed_values']}"
            ),
            (
                "- Validated canonical n=3 summaries: "
                f"{payload['aggregation_modes']['validated_canonical_summary']}"
            ),
            "",
            "## Scientific Boundary",
            "",
            (
                "Sprint 7.2 performs statistical aggregation "
                "only. It introduces no new training, "
                "experiments, retuning, or scientific claims."
            ),
            "",
        ]
    )

    return "\n".join(lines)
