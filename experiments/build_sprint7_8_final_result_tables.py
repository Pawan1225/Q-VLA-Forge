"""Build Sprint 7.8 canonical final result tables from frozen evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_result_tables import (
    CLAIM_CONTROLS,
    FINAL_TABLES_JSON,
    FINAL_TABLES_MARKDOWN,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    NOT_APPLICABLE,
    NOT_MEASURED,
    NOT_REACHED,
    OUTPUT_DIR,
    SEEDS,
    STATISTICAL_PROTOCOL,
    SYNTHETIC_METRIC_COMPOSITION_ALLOWED,
    TABLE_COLUMNS,
    TABLE_FILENAMES,
    TABLE_IDS,
    TABLE_LIMITATIONS,
    TABLE_MANIFEST_PATH,
    TABLE_METRICS,
    TABLE_SOURCES,
    TABLE_TITLES,
)

ROOT = Path(__file__).resolve().parents[1]

COMPRESSION_CSV = (
    ROOT
    / "results"
    / "final-validation"
    / "compression-ablation"
    / "compression-pareto.csv"
)

TRAINING_JSON = ROOT / "results" / "training" / "training-validation-summary.json"

QML_CSV = (
    ROOT
    / "results"
    / "final-validation"
    / "qml-ablation"
    / "qml-ablation-plot-data.csv"
)

SAFETY_CSV = (
    ROOT
    / "results"
    / "final-validation"
    / "safety-ablation"
    / "safety-ablation-plot-data.csv"
)

CROSS_DOMAIN_JSON = (
    ROOT
    / "results"
    / "final-validation"
    / "cross-domain"
    / "final-cross-domain-summary.json"
)

FULL_SYSTEM_JSON = (
    ROOT
    / "results"
    / "final-validation"
    / "full-system-ablation"
    / "full-system-ablation.json"
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing frozen source artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"Missing frozen source artifact: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _value(
    row: dict[str, str],
    *names: str,
) -> str:
    for name in names:
        if name in row:
            return row[name]

    raise KeyError(f"Missing expected field {names}; " f"available={tuple(row)}")


def _float(
    row: dict[str, str],
    *names: str,
) -> float:
    return float(
        _value(
            row,
            *names,
        )
    )


def _bool(
    value: str,
) -> bool:
    return value.strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _domain_label(
    domain: str,
) -> str:
    mapping = {
        "autonomous_driving": "Driving",
        "driving": "Driving",
        "robotics": "Robotics",
    }

    return mapping.get(
        domain,
        domain,
    )


def _compression_method_label(
    method: str,
) -> str:
    normalized = method.lower().replace("-", "_").replace("/", "_")

    mapping = {
        "fp32": "FP32",
        "int8": "INT8",
        "svd": "SVD",
        "tt_mps": "TT/MPS",
        "tt": "TT/MPS",
        "mps": "TT/MPS",
    }

    return mapping.get(
        normalized,
        method,
    )


def _training_method_label(
    method: str,
) -> str:
    mapping = {
        "trainable_svd": "Trainable SVD",
        "trainable_tt_mps": "Trainable TT/MPS",
    }

    return mapping.get(
        method,
        method,
    )


def _policy_label(
    policy: str,
) -> str:
    mapping = {
        "ppo_mlp": "Classical PPO / MLP",
        "matched_classical": "Matched classical control",
        "ppo_pqc": "QML / PQC",
    }

    return mapping.get(
        policy,
        policy,
    )


def _safety_method_label(
    method: str,
) -> str:
    mapping = {
        "none": "NONE",
        "clipping": "CLIPPING",
        "lyapunov": "LYAPUNOV",
    }

    return mapping.get(
        method.lower(),
        method.upper(),
    )


def _mean_std(
    stats: dict[str, Any] | None,
    *,
    decimals: int = 3,
) -> str:
    if stats is None:
        return NOT_REACHED

    mean = float(stats["mean"])
    std = float(stats["std"])

    return f"{mean:.{decimals}f} ± " f"{std:.{decimals}f}"


def _mean_std_values(
    values: list[float],
    *,
    decimals: int = 3,
) -> str:
    if not values:
        return NOT_REACHED

    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"

    mean = sum(values) / len(values)

    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)

    std = variance**0.5

    return f"{mean:.{decimals}f} ± " f"{std:.{decimals}f}"


def _format_ratio(
    value: float,
) -> str:
    return f"{value:.3f}×"


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in columns
                }
            )


def _markdown_table(
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in rows:
        values = []

        for column in columns:
            value = str(
                row.get(
                    column,
                    "",
                )
            )

            value = value.replace(
                "|",
                "\\|",
            )

            values.append(value)

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _build_compression_rows() -> list[dict[str, Any]]:
    source = _load_csv(COMPRESSION_CSV)

    rows: list[dict[str, Any]] = []

    for item in source:
        domain = _domain_label(
            _value(
                item,
                "domain",
            )
        )

        method = _compression_method_label(
            _value(
                item,
                "method",
            )
        )

        compression_ratio = _float(
            item,
            "compression_ratio",
        )

        mse_mean = _float(
            item,
            "mse_mean",
        )

        mse_std = _float(
            item,
            "mse_sample_std",
        )

        relative_delta_mean_raw = _value(
            item,
            "relative_mse_change_mean_percent",
        ).strip()

        relative_delta_std_raw = _value(
            item,
            "relative_mse_change_sample_std_percent",
        ).strip()

        criterion = _value(
            item,
            "criterion_status",
        ).upper()

        if not relative_delta_mean_raw:
            if method != "FP32":
                raise AssertionError(
                    "Missing relative ΔMSE for non-reference "
                    f"compression row: domain={domain}, "
                    f"method={method}"
                )

            relative_delta = NOT_APPLICABLE

        else:
            relative_delta_mean = float(relative_delta_mean_raw)

            if not relative_delta_std_raw:
                raise AssertionError(
                    "Missing relative ΔMSE sample SD for "
                    f"domain={domain}, method={method}"
                )

            relative_delta_std = float(relative_delta_std_raw)

            relative_delta = (
                f"{relative_delta_mean:.3f}% " f"± {relative_delta_std:.3f}%"
            )

        rows.append(
            {
                "Domain": domain,
                "Method": method,
                "Parameters / Model Size": NOT_MEASURED,
                "Compression Ratio": _format_ratio(compression_ratio),
                "MSE": (f"{mse_mean:.6f} " f"± {mse_std:.6f}"),
                "Relative ΔMSE": relative_delta,
                "Latency": NOT_MEASURED,
                "Criterion": criterion,
            }
        )

    if len(rows) != 8:
        raise AssertionError("Compression table must contain exactly 8 rows.")

    fp32_rows = [row for row in rows if row["Method"] == "FP32"]

    if len(fp32_rows) != 2:
        raise AssertionError(
            "Compression table must contain one FP32 " "reference row per domain."
        )

    if any(row["Relative ΔMSE"] != NOT_APPLICABLE for row in fp32_rows):
        raise AssertionError("FP32 relative ΔMSE must remain Not applicable.")

    non_reference_rows = [row for row in rows if row["Method"] != "FP32"]

    if any(row["Relative ΔMSE"] == NOT_APPLICABLE for row in non_reference_rows):
        raise AssertionError(
            "Only FP32 reference rows may use " "Not applicable for relative ΔMSE."
        )

    return rows


def _build_training_rows() -> list[dict[str, Any]]:
    payload = _load_json(TRAINING_JSON)

    if payload["statistics"]["spread"] != "sample_standard_deviation":
        raise AssertionError("Training statistics must use sample SD.")

    rows: list[dict[str, Any]] = []

    for item in payload["methods"]:
        reach = item["target_reach"]

        reached = int(reach["reached"])

        total = int(reach["total"])

        if reached == 0:
            steps = NOT_REACHED
            training_time = NOT_REACHED
        else:
            steps = _mean_std(
                item["steps_to_target"],
                decimals=3,
            )

            training_time = (
                _mean_std(
                    item["seconds_to_target"],
                    decimals=3,
                )
                + " s"
            )

        final_validation = _mean_std(
            item["final_validation_loss"],
            decimals=6,
        )

        rows.append(
            {
                "Domain": _domain_label(item["domain"]),
                "Method": _training_method_label(item["method"]),
                "Target Reached": (f"{reached}/{total}"),
                "Steps to Target": steps,
                "Training Time": training_time,
                "Memory": NOT_MEASURED,
                "Final Validation Loss / MSE": final_validation,
                "Efficiency Result": (
                    "PASS" if item["robust_ten_percent_step_efficiency"] else "FAIL"
                ),
            }
        )

    if len(rows) != 4:
        raise AssertionError("Training table must contain exactly 4 rows.")

    if any(row["Memory"] != NOT_MEASURED for row in rows):
        raise AssertionError("Unmeasured memory must remain explicit.")

    return rows


def _build_rl_rows() -> list[dict[str, Any]]:
    source = _load_csv(QML_CSV)

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = {}

    for item in source:
        key = (
            _value(
                item,
                "domain",
            ),
            _value(
                item,
                "policy",
            ),
        )

        grouped.setdefault(
            key,
            [],
        ).append(item)

    rows: list[dict[str, Any]] = []

    for (
        domain,
        policy,
    ), records in grouped.items():
        reached_records = [
            record
            for record in records
            if _bool(
                _value(
                    record,
                    "target_reached",
                )
            )
        ]

        steps_values = [
            float(
                _value(
                    record,
                    "environment_steps_to_target",
                )
            )
            for record in reached_records
            if _value(
                record,
                "environment_steps_to_target",
            ).strip()
        ]

        final_rewards = [
            float(
                _value(
                    record,
                    "final_evaluation_reward",
                )
            )
            for record in records
        ]

        final_success = [
            float(
                _value(
                    record,
                    "final_success_rate",
                )
            )
            for record in records
        ]

        actor_parameters = int(
            float(
                _value(
                    records[0],
                    "actor_parameters",
                )
            )
        )

        if policy in {
            "ppo_pqc",
            "matched_classical",
        }:
            reference_records = grouped.get(
                (
                    domain,
                    "ppo_mlp",
                )
            )

            if not reference_records:
                raise AssertionError("Missing PPO reference for actor compactness.")

            ppo_parameters = int(
                float(
                    _value(
                        reference_records[0],
                        "actor_parameters",
                    )
                )
            )

            parameter_reduction = (
                (ppo_parameters - actor_parameters) / ppo_parameters * 100.0
            )

            reduction_text = f"{parameter_reduction:.2f}%"

        else:
            reduction_text = NOT_APPLICABLE

        rows.append(
            {
                "Domain": _domain_label(domain),
                "Policy": _policy_label(policy),
                "Target Reaches": (f"{len(reached_records)}/{len(records)}"),
                "Steps to Target": (
                    _mean_std_values(
                        steps_values,
                        decimals=3,
                    )
                    if steps_values
                    else NOT_REACHED
                ),
                "Final Reward": (
                    _mean_std_values(
                        final_rewards,
                        decimals=3,
                    )
                ),
                "Final Success Rate": (
                    _mean_std_values(
                        final_success,
                        decimals=3,
                    )
                ),
                "Actor Parameters": actor_parameters,
                "Parameter Reduction": reduction_text,
            }
        )

    if len(rows) != 6:
        raise AssertionError("RL/QML table must contain exactly 6 rows.")

    total_ppo = sum(
        int(row["Target Reaches"].split("/")[0])
        for row in rows
        if row["Policy"] == "Classical PPO / MLP"
    )

    total_matched = sum(
        int(row["Target Reaches"].split("/")[0])
        for row in rows
        if row["Policy"] == "Matched classical control"
    )

    total_qml = sum(
        int(row["Target Reaches"].split("/")[0])
        for row in rows
        if row["Policy"] == "QML / PQC"
    )

    if (
        total_ppo,
        total_matched,
        total_qml,
    ) != (
        6,
        1,
        0,
    ):
        raise AssertionError("Frozen RL target-reach counts changed.")

    robotics_matched = [
        row
        for row in rows
        if (
            row["Domain"] == "Robotics" and row["Policy"] == "Matched classical control"
        )
    ]

    if len(robotics_matched) != 1:
        raise AssertionError("Missing robotics matched-classical row.")

    if robotics_matched[0]["Target Reaches"] != "1/3":
        raise AssertionError("Robotics matched-classical target reach changed.")

    if robotics_matched[0]["Steps to Target"] != "20000.000":
        raise AssertionError("Matched-classical final-budget reach must remain 20000.")

    return rows


def _build_safety_rows() -> list[dict[str, Any]]:
    source = _load_csv(SAFETY_CSV)

    rows: list[dict[str, Any]] = []

    for item in source:
        method = _value(
            item,
            "method",
        ).lower()

        violation_mean = _float(
            item,
            "violation_rate_mean",
        )

        violation_std = _float(
            item,
            "violation_rate_sample_std",
        )

        reward_mean = _float(
            item,
            "reward_mean",
        )

        reward_std = _float(
            item,
            "reward_sample_std",
        )

        success_mean = _float(
            item,
            "success_rate_mean",
        )

        success_std = _float(
            item,
            "success_rate_sample_std",
        )

        if (
            "intervention_rate_mean" in item
            and item["intervention_rate_mean"].strip()
            and "intervention_rate_sample_std" in item
            and item["intervention_rate_sample_std"].strip()
        ):
            intervention = (
                f"{_float(item, 'intervention_rate_mean'):.3f} "
                f"± "
                f"{_float(item, 'intervention_rate_sample_std'):.3f}"
            )

        elif method == "none":
            intervention = NOT_APPLICABLE

        else:
            intervention = NOT_MEASURED

        rows.append(
            {
                "Domain": _domain_label(
                    _value(
                        item,
                        "domain",
                    )
                ),
                "Method": _safety_method_label(method),
                "Violation Rate": (f"{violation_mean:.3f} " f"± {violation_std:.3f}"),
                "Reward": (f"{reward_mean:.3f} " f"± {reward_std:.3f}"),
                "Success Rate": (f"{success_mean:.3f} " f"± {success_std:.3f}"),
                "Activation / Intervention": intervention,
                "Evidence Role": (
                    "Baseline"
                    if method == "none"
                    else "Classical empirical safety filter"
                ),
            }
        )

    if len(rows) != 6:
        raise AssertionError("Safety table must contain exactly 6 rows.")

    return rows


def _build_cross_domain_rows(
    compression_rows: list[dict[str, Any]],
    rl_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cross_domain = _load_json(CROSS_DOMAIN_JSON)

    full_system = _load_json(FULL_SYSTEM_JSON)

    phase1 = full_system["phase1"]

    if phase1["direct_count"] != 0 or phase1["component_only_count"] != 16:
        raise AssertionError("Frozen full-system evidence boundary changed.")

    compression_by_method: dict[
        str,
        dict[str, str],
    ] = {}

    for row in compression_rows:
        compression_by_method.setdefault(
            row["Method"],
            {},
        )[
            row["Domain"]
        ] = str(row["Criterion"])

    rl_by_policy: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rl_rows:
        rl_by_policy.setdefault(
            row["Policy"],
            {},
        )[
            row["Domain"]
        ] = str(row["Target Reaches"])

    safety_by_method: dict[
        str,
        dict[str, str],
    ] = {}

    for row in safety_rows:
        safety_by_method.setdefault(
            row["Method"],
            {},
        )[
            row["Domain"]
        ] = str(row["Violation Rate"])

    rows = [
        {
            "Component / Method": "Shared AI/DL architecture",
            "Driving Result": ("Shared architecture components available"),
            "Robotics Result": ("Shared architecture components available"),
            "Shared Across Domains?": (
                "Framework/components: YES; " "same trained policy weights: NO"
            ),
            "Evidence Boundary": (cross_domain["claim"]),
        },
        {
            "Component / Method": "INT8",
            "Driving Result": (compression_by_method["INT8"]["Driving"]),
            "Robotics Result": (compression_by_method["INT8"]["Robotics"]),
            "Shared Across Domains?": ("Same method/protocol family: YES"),
            "Evidence Boundary": (
                "Storage-compression evidence only; " "no production runtime claim."
            ),
        },
        {
            "Component / Method": "SVD",
            "Driving Result": (compression_by_method["SVD"]["Driving"]),
            "Robotics Result": (compression_by_method["SVD"]["Robotics"]),
            "Shared Across Domains?": ("Same method/protocol family: YES"),
            "Evidence Boundary": (
                "Evaluated configuration failed " "the joint criterion."
            ),
        },
        {
            "Component / Method": "TT/MPS",
            "Driving Result": (compression_by_method["TT/MPS"]["Driving"]),
            "Robotics Result": (compression_by_method["TT/MPS"]["Robotics"]),
            "Shared Across Domains?": ("Same method/protocol family: YES"),
            "Evidence Boundary": (
                "Quantum-inspired tensor-network method; " "no superiority claim."
            ),
        },
        {
            "Component / Method": "Classical PPO",
            "Driving Result": (
                "Target reaches " + rl_by_policy["Classical PPO / MLP"]["Driving"]
            ),
            "Robotics Result": (
                "Target reaches " + rl_by_policy["Classical PPO / MLP"]["Robotics"]
            ),
            "Shared Across Domains?": (
                "Evaluation framework: YES; " "same learned weights: NO"
            ),
            "Evidence Boundary": (
                "Separate domain policies and " "reward/state contracts."
            ),
        },
        {
            "Component / Method": "QML/PQC",
            "Driving Result": (
                "Target reaches " + rl_by_policy["QML / PQC"]["Driving"]
            ),
            "Robotics Result": (
                "Target reaches " + rl_by_policy["QML / PQC"]["Robotics"]
            ),
            "Shared Across Domains?": (
                "Evaluation framework/PQC family: YES; " "same trained weights: NO"
            ),
            "Evidence Boundary": (
                "No QML sample-efficiency advantage "
                "or quantum advantage demonstrated."
            ),
        },
        {
            "Component / Method": "Clipping",
            "Driving Result": (
                "Violation rate " + safety_by_method["CLIPPING"]["Driving"]
            ),
            "Robotics Result": (
                "Violation rate " + safety_by_method["CLIPPING"]["Robotics"]
            ),
            "Shared Across Domains?": (
                "Safety abstraction: YES; " "constraints/semantics: domain-specific"
            ),
            "Evidence Boundary": ("Zero observed violations are empirical only."),
        },
        {
            "Component / Method": "Lyapunov",
            "Driving Result": (
                "Violation rate " + safety_by_method["LYAPUNOV"]["Driving"]
            ),
            "Robotics Result": (
                "Violation rate " + safety_by_method["LYAPUNOV"]["Robotics"]
            ),
            "Shared Across Domains?": (
                "Safety abstraction: YES; " "domain-specific predictor/constraints"
            ),
            "Evidence Boundary": (
                "Classical empirical safety potential; "
                "no formal stability guarantee."
            ),
        },
        {
            "Component / Method": "Full-system factorial",
            "Driving Result": "COMPONENT_ONLY",
            "Robotics Result": "COMPONENT_ONLY",
            "Shared Across Domains?": ("Direct integrated evidence: NO"),
            "Evidence Boundary": (
                "DIRECT = 0; COMPONENT_ONLY = 16; "
                "interaction effects remain Phase 2 work."
            ),
        },
    ]

    return rows


def _table_payload(
    table_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "title": TABLE_TITLES[table_id],
        "columns": list(TABLE_COLUMNS[table_id]),
        "rows": rows,
        "source_artifacts": list(TABLE_SOURCES[table_id]),
        "metrics": list(TABLE_METRICS[table_id]),
        "statistical_protocol": STATISTICAL_PROTOCOL,
        "limitations": list(TABLE_LIMITATIONS[table_id]),
        "generated_from_frozen_evidence": True,
    }


def _build_markdown(
    tables: dict[
        str,
        dict[str, Any],
    ],
) -> str:
    sections = [
        "# Q-VLA Forge — Sprint 7.8 Final Result Tables",
        "",
        (
            "Frozen Phase 1 evidence only. "
            "No new training, experiments, retuning, "
            "or scientific results were introduced."
        ),
        "",
        (f"Statistical protocol: " f"{STATISTICAL_PROTOCOL}."),
        "",
        (
            "Missing-value semantics: "
            "`Not reached`, `Not measured`, and "
            "`Not applicable` are distinct."
        ),
        "",
    ]

    for table_id in TABLE_IDS:
        table = tables[table_id]

        sections.extend(
            [
                f"## {table['title']}",
                "",
                _markdown_table(
                    tuple(table["columns"]),
                    table["rows"],
                ),
                "",
                "### Limitations",
                "",
            ]
        )

        sections.extend(f"- {limitation}" for limitation in table["limitations"])

        sections.extend(
            [
                "",
                "### Provenance",
                "",
            ]
        )

        sections.extend(f"- `{source}`" for source in table["source_artifacts"])

        sections.append("")

    sections.extend(
        [
            "## Scientific Controls",
            "",
            "- TT/MPS superiority: blocked.",
            ("- Robust >=10% training-efficiency " "improvement: not demonstrated."),
            "- QML sample-efficiency advantage: blocked.",
            "- Quantum advantage / speedup: blocked.",
            "- Formal Lyapunov stability: blocked.",
            "- Guaranteed zero violations: blocked.",
            "- Zero-shot cross-domain transfer: blocked.",
            "- Full-system superiority: blocked.",
            "- Production readiness / certification: blocked.",
            "",
        ]
    )

    return "\n".join(sections)


def main() -> None:
    if NEW_TRAINING:
        raise AssertionError("Sprint 7.8 cannot introduce training.")

    if NEW_EXPERIMENTS:
        raise AssertionError("Sprint 7.8 cannot introduce experiments.")

    if NEW_SCIENTIFIC_RESULTS:
        raise AssertionError("Sprint 7.8 cannot introduce " "new scientific results.")

    if SYNTHETIC_METRIC_COMPOSITION_ALLOWED:
        raise AssertionError("Synthetic metric composition " "must remain blocked.")

    if tuple(SEEDS) != (
        42,
        123,
        456,
    ):
        raise AssertionError("Locked seeds changed.")

    output_dir = ROOT / OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    compression_rows = _build_compression_rows()

    training_rows = _build_training_rows()

    rl_rows = _build_rl_rows()

    safety_rows = _build_safety_rows()

    cross_domain_rows = _build_cross_domain_rows(
        compression_rows,
        rl_rows,
        safety_rows,
    )

    rows_by_table = {
        "compression": compression_rows,
        "training": training_rows,
        "rl": rl_rows,
        "safety": safety_rows,
        "cross_domain": cross_domain_rows,
    }

    tables = {
        table_id: _table_payload(
            table_id,
            rows_by_table[table_id],
        )
        for table_id in TABLE_IDS
    }

    for table_id in TABLE_IDS:
        path = output_dir / TABLE_FILENAMES[table_id]

        _write_csv(
            path,
            TABLE_COLUMNS[table_id],
            rows_by_table[table_id],
        )

    final_payload = {
        "sprint": "7.8",
        "status": "FROZEN",
        "generated_from_frozen_evidence": True,
        "new_training": False,
        "new_experiments": False,
        "new_scientific_results": False,
        "synthetic_metric_composition": False,
        "seeds": list(SEEDS),
        "statistical_protocol": STATISTICAL_PROTOCOL,
        "missing_value_semantics": {
            "not_reached": NOT_REACHED,
            "not_measured": NOT_MEASURED,
            "not_applicable": NOT_APPLICABLE,
        },
        "tables": tables,
        "claim_controls": CLAIM_CONTROLS,
    }

    (ROOT / FINAL_TABLES_JSON).write_text(
        json.dumps(
            final_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (ROOT / FINAL_TABLES_MARKDOWN).write_text(
        _build_markdown(tables),
        encoding="utf-8",
    )

    manifest = {
        "sprint": "7.8",
        "status": "FROZEN",
        "table_count": len(TABLE_IDS),
        "required_table_ids": list(TABLE_IDS),
        "generated_from_frozen_evidence": True,
        "new_training": False,
        "new_experiments": False,
        "new_scientific_results": False,
        "synthetic_metric_composition": False,
        "statistical_protocol": STATISTICAL_PROTOCOL,
        "claim_controls": CLAIM_CONTROLS,
        "tables": [
            {
                "table_id": table_id,
                "title": TABLE_TITLES[table_id],
                "output_file": str(OUTPUT_DIR / TABLE_FILENAMES[table_id]).replace(
                    "\\",
                    "/",
                ),
                "source_artifacts": list(TABLE_SOURCES[table_id]),
                "row_count": len(rows_by_table[table_id]),
                "domains": (
                    [
                        "autonomous_driving",
                        "robotics",
                    ]
                    if table_id != "cross_domain"
                    else [
                        "autonomous_driving",
                        "robotics",
                        "cross_domain",
                    ]
                ),
                "metrics": list(TABLE_METRICS[table_id]),
                "statistical_protocol": STATISTICAL_PROTOCOL,
                "limitations": list(TABLE_LIMITATIONS[table_id]),
                "claim_controls": CLAIM_CONTROLS,
            }
            for table_id in TABLE_IDS
        ],
    }

    (ROOT / TABLE_MANIFEST_PATH).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 68)
    print(" SPRINT 7.8 FINAL RESULT TABLES BUILD")
    print("=" * 68)
    print()
    print("Compression table                  PASS")
    print("Training table                     PASS")
    print("RL/QML table                       PASS")
    print("Safety table                       PASS")
    print("Cross-domain table                 PASS")
    print()
    print("Final JSON package                 PASS")
    print("Reviewer Markdown                  PASS")
    print("Table manifest                     PASS")
    print("Frozen evidence only               PASS")
    print("Synthetic metric composition       BLOCKED")
    print("New training                       FALSE")
    print("New experiments                    FALSE")
    print("New scientific results             FALSE")
    print()
    print("SPRINT 7.8 FINAL RESULT TABLES BUILD: PASS")


if __name__ == "__main__":
    main()
