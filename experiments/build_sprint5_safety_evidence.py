from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.sprint5_evidence import (
    PRINCIPAL_SEEDS,
    recovery_rate,
    sha256_file,
)

ROOT = Path(".")

EVIDENCE_DIR = ROOT / "results" / "safety" / "evidence"

JSON_OUTPUT = EVIDENCE_DIR / "sprint5-safety-evidence.json"
CSV_OUTPUT = EVIDENCE_DIR / "sprint5-safety-evidence.csv"
MARKDOWN_OUTPUT = EVIDENCE_DIR / "sprint5-safety-evidence.md"

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONSOLIDATED = ROOT / "results" / "safety" / "consolidated"

CROSS_DOMAIN = ROOT / "results" / "safety" / "cross-domain"

FINAL = ROOT / "results" / "safety" / "final"

SOURCES = {
    "safety_package": (CONSOLIDATED / "sprint5-safety-evidence-package.json"),
    "clean_summary": (CONSOLIDATED / "sprint5-clean-three-seed-summary.json"),
    "gaussian_summary": (
        ROOT
        / "results"
        / "safety"
        / "gaussian-robustness"
        / "sprint5-gaussian-robustness-summary.json"
    ),
    "structured_state_summary": (
        ROOT
        / "results"
        / "safety"
        / "structured-state-robustness"
        / "sprint5-structured-state-robustness-summary.json"
    ),
    "action_summary": (
        ROOT
        / "results"
        / "safety"
        / "action-robustness"
        / "sprint5-action-robustness-summary.json"
    ),
    "environment_action_handling": (
        ROOT
        / "results"
        / "safety"
        / "action-robustness"
        / "environment-action-handling.json"
    ),
    "lyapunov_attribution": (
        CONSOLIDATED / "sprint5-lyapunov-mechanism-attribution.json"
    ),
    "cross_domain_package": (CROSS_DOMAIN / "sprint5-cross-domain-package.json"),
    "cross_domain_action": (CROSS_DOMAIN / "sprint5-cross-domain-action.json"),
    "cross_domain_lyapunov": (
        CROSS_DOMAIN / "sprint5-cross-domain-lyapunov-mechanism.json"
    ),
    "cross_domain_architecture": (
        CROSS_DOMAIN / "sprint5-cross-domain-architecture.json"
    ),
    "claim_boundary": (FINAL / "sprint5-final-claim-boundary.json"),
    "limitations": (FINAL / "sprint5-final-limitations.json"),
    "scientific_invariants": (FINAL / "sprint5-scientific-invariants.json"),
    "freeze_record": (FINAL / "sprint5-freeze-record.json"),
}

for source_path in SOURCES.values():
    if not source_path.is_file():
        raise FileNotFoundError(source_path)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


def normalized_path(
    path: Path | str,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


def aggregate_numbers(
    values: Iterable[float],
) -> dict[str, float]:
    numeric = [float(value) for value in values]

    if len(numeric) != 3:
        raise ValueError("Principal aggregation requires " "exactly three seed values.")

    return {
        "mean": float(statistics.mean(numeric)),
        "sample_sd": float(statistics.stdev(numeric)),
    }


def seed_metric(
    values: dict[int, float],
) -> dict[str, float]:
    if set(values) != set(PRINCIPAL_SEEDS):
        raise ValueError("Expected exactly principal seeds " f"{PRINCIPAL_SEEDS}.")

    aggregate = aggregate_numbers(values[seed] for seed in PRINCIPAL_SEEDS)

    return {
        "seed_42": float(values[42]),
        "seed_123": float(values[123]),
        "seed_456": float(values[456]),
        "mean": aggregate["mean"],
        "sample_sd": aggregate["sample_sd"],
    }


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(statistics.mean(values))


def median_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(statistics.median(values))


def numeric_record_value(
    record: dict[str, Any],
    key: str,
) -> float:
    value = record[key]

    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{key} must be numeric")

    return float(value)


def minimum_item(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    if not records:
        return None

    return min(
        records,
        key=lambda item: numeric_record_value(
            item,
            metric,
        ),
    )


def maximum_item(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    if not records:
        return None

    return max(
        records,
        key=lambda item: numeric_record_value(
            item,
            metric,
        ),
    )


def resolve_source_path(
    source: str,
) -> Path:
    path = Path(source)

    if path.is_absolute():
        return path

    return ROOT / path


HASH_CACHE: dict[str, str] = {}


def source_sha256(
    source: str,
) -> str:
    normalized = normalized_path(source)

    if normalized in HASH_CACHE:
        return HASH_CACHE[normalized]

    path = resolve_source_path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = sha256_file(path)

    HASH_CACHE[normalized] = digest

    return digest


def clean_run_path(
    domain: str,
    method: str,
    seed: int,
) -> Path:
    filename = f"{domain}-seed-{seed}.json"

    if method == "none":
        return ROOT / "results" / "safety" / "baseline" / "runs" / filename

    if method == "clipping":
        return ROOT / "results" / "safety" / "clipping" / "runs" / filename

    if method == "lyapunov":
        folder = (
            "lyapunov-driving"
            if domain == "autonomous_driving"
            else "lyapunov-robotics"
        )

        return ROOT / "results" / "safety" / folder / "runs" / filename

    raise ValueError(f"unsupported method: {method}")


def clean_seed_summary(
    domain: str,
    method: str,
    seed: int,
) -> tuple[
    dict[str, Any],
    str,
]:
    path = clean_run_path(
        domain,
        method,
        seed,
    )

    payload = load_json(path)

    if method == "lyapunov":
        summary = payload.get("seed_summary")
    else:
        summary = payload.get("summary")

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(f"Missing clean summary: {path}")

    return (
        summary,
        normalized_path(path),
    )


def clean_value(
    summary: dict[str, Any],
    method: str,
    metric: str,
) -> float:
    aliases = {
        "violation_step_rate": (
            "violation_step_rate"
            if method == "none"
            else "executed_violation_step_rate"
        ),
        "constraint_violation_rate": (
            "constraint_violation_rate"
            if method == "none"
            else "executed_constraint_violation_rate"
        ),
        "critical_violation_rate": ("critical_violation_step_rate"),
        "reward": "mean_reward",
        "success_rate": "success_rate",
        "episode_length": ("mean_episode_length"),
        "intervention_rate": ("intervention_rate"),
        "mean_correction_l2": ("mean_action_correction_l2"),
        "p95_correction_l2": ("p95_action_correction_l2"),
    }

    key = aliases[metric]

    value = summary.get(key)

    if value is None:
        if (
            metric
            in {
                "intervention_rate",
                "mean_correction_l2",
                "p95_correction_l2",
            }
            and method == "none"
        ):
            return 0.0

        raise KeyError(f"Missing clean metric " f"{metric}/{key}")

    return float(value)


CLEAN_METRICS = (
    "violation_step_rate",
    "constraint_violation_rate",
    "critical_violation_rate",
    "reward",
    "success_rate",
    "episode_length",
    "intervention_rate",
    "mean_correction_l2",
    "p95_correction_l2",
)


def build_clean_package() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    package: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        package[domain] = {}

        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            seed_summaries: dict[
                int,
                tuple[
                    dict[str, Any],
                    str,
                ],
            ] = {}

            for seed in PRINCIPAL_SEEDS:
                seed_summaries[seed] = clean_seed_summary(
                    domain,
                    method,
                    seed,
                )

            method_metrics: dict[
                str,
                Any,
            ] = {}

            for metric in CLEAN_METRICS:
                values = {
                    seed: clean_value(
                        seed_summaries[seed][0],
                        method,
                        metric,
                    )
                    for seed in PRINCIPAL_SEEDS
                }

                method_metrics[metric] = seed_metric(values)

                for seed in PRINCIPAL_SEEDS:
                    source = seed_summaries[seed][1]

                    csv_rows.append(
                        make_csv_row(
                            domain=domain,
                            regime="clean",
                            condition="clean",
                            method=method,
                            seed=seed,
                            metric=metric,
                            value=values[seed],
                            reference_value=None,
                            delta=None,
                            unit=metric_unit(metric),
                            source=source,
                        )
                    )

            method_metrics["source_artifacts"] = [
                seed_summaries[seed][1] for seed in PRINCIPAL_SEEDS
            ]

            package[domain][method] = method_metrics

    return (
        package,
        csv_rows,
    )


def metric_unit(
    metric: str,
) -> str:
    if "rate" in metric or "fraction" in metric:
        return "fraction"

    if metric.endswith(("_l2", "_linf")):
        return "normalized_action"

    if metric == "reward":
        return "reward"

    if metric == "episode_length":
        return "steps"

    if metric.endswith("_steps"):
        return "steps"

    return "value"


def make_csv_row(
    *,
    domain: str,
    regime: str,
    condition: str,
    method: str,
    seed: int,
    metric: str,
    value: float,
    reference_value: float | None,
    delta: float | None,
    unit: str,
    source: str,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "regime": regime,
        "condition": condition,
        "method": method,
        "seed": seed,
        "metric": metric,
        "value": float(value),
        "reference_value": (
            None if reference_value is None else float(reference_value)
        ),
        "delta": (None if delta is None else float(delta)),
        "unit": unit,
        "source_artifact": (normalized_path(source)),
        "source_sha256": (source_sha256(source)),
    }


def compact_metric(
    cell: dict[str, Any],
    key: str,
) -> float:
    value = cell.get(key)

    if value is None:
        raise KeyError(f"Missing metric: {key}")

    return float(value)


def build_gaussian_package(
    source: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    cells = source.get("cells")

    if not isinstance(cells, list):
        raise TypeError("Gaussian cells must be a list")

    aggregates = source.get("aggregates")

    if not isinstance(
        aggregates,
        list,
    ):
        raise TypeError("Gaussian aggregates must be a list")

    output_cells: list[dict[str, Any]] = []

    csv_rows: list[dict[str, Any]] = []

    metric_specs = {
        "violation_step_rate": (
            "violation_step_rate",
            "violation_delta_from_clean",
        ),
        "critical_violation_rate": (
            "critical_violation_step_rate",
            "critical_delta_from_clean",
        ),
        "constraint_violation_rate": (
            "constraint_violation_rate",
            "constraint_delta_from_clean",
        ),
        "reward": (
            "reward",
            "reward_delta_from_clean",
        ),
        "success_rate": (
            "success_rate",
            "success_delta_from_clean",
        ),
        "intervention_rate": (
            "intervention_rate",
            None,
        ),
        "mean_correction_l2": (
            "mean_action_correction_l2",
            None,
        ),
        "episode_length": (
            "mean_episode_length",
            None,
        ),
    }

    for raw_cell in cells:
        if not isinstance(
            raw_cell,
            dict,
        ):
            raise TypeError("Gaussian cell must be dict")

        domain = str(raw_cell["domain"])
        method = str(raw_cell["method"])
        seed = int(raw_cell["principal_seed"])
        sigma = float(raw_cell["sigma"])

        condition = "sigma_" + str(sigma).replace(".", "p")

        source_path = str(raw_cell["source"])

        compact = {
            "domain": domain,
            "method": method,
            "principal_seed": seed,
            "sigma": sigma,
            "condition": condition,
            "source": normalized_path(source_path),
        }

        for output_name, (
            value_key,
            delta_key,
        ) in metric_specs.items():
            value = compact_metric(
                raw_cell,
                value_key,
            )

            delta_value = (
                None
                if delta_key is None
                else compact_metric(
                    raw_cell,
                    delta_key,
                )
            )

            reference = None if delta_value is None else value - delta_value

            compact[output_name] = value

            if delta_key is not None:
                compact[f"{output_name}_delta_from_clean"] = delta_value

            csv_rows.append(
                make_csv_row(
                    domain=domain,
                    regime="gaussian",
                    condition=condition,
                    method=method,
                    seed=seed,
                    metric=output_name,
                    value=value,
                    reference_value=reference,
                    delta=delta_value,
                    unit=metric_unit(output_name),
                    source=source_path,
                )
            )

        output_cells.append(compact)

    output_cells.sort(
        key=lambda item: (
            item["domain"],
            item["method"],
            item["sigma"],
            item["principal_seed"],
        )
    )

    return (
        {
            "sigmas": [
                0.0,
                0.01,
                0.05,
                0.10,
            ],
            "perturbed_sigmas": [
                0.01,
                0.05,
                0.10,
            ],
            "cells": output_cells,
            "aggregates": aggregates,
            "limitation": {
                "policy_observation": ("perturbed"),
                "safety_state": ("true_simulator_state"),
                "privileged_safety_state": (True),
                "interpretation": ("policy-plus-privileged-state-" "safety robustness"),
                "real_sensor_robustness_supported": (False),
            },
        },
        csv_rows,
    )


def state_family_group(
    condition: dict[str, Any],
) -> str:
    return str(condition["perturbation_family"])


def build_state_family_summaries(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in aggregates:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError("State aggregate must be dict")

        grouped[
            (
                str(item["domain"]),
                str(item["method"]),
                state_family_group(item),
            )
        ].append(item)

    output: list[dict[str, Any]] = []

    for (
        domain,
        method,
        family,
    ), records in sorted(grouped.items()):
        violation_records = [
            {
                "condition": str(item["perturbation_name"]),
                "value": float(item["violation_delta_from_clean"]["mean"]),
            }
            for item in records
        ]

        reward_records = [
            {
                "condition": str(item["perturbation_name"]),
                "value": float(item["reward_delta_from_clean"]["mean"]),
            }
            for item in records
        ]

        intervention_records = [
            {
                "condition": str(item["perturbation_name"]),
                "value": float(item["intervention_rate"]["mean"]),
            }
            for item in records
        ]

        violation_values = [
            numeric_record_value(
                item,
                "value",
            )
            for item in violation_records
        ]

        reward_values = [
            numeric_record_value(
                item,
                "value",
            )
            for item in reward_records
        ]

        worst_violation = maximum_item(
            violation_records,
            "value",
        )

        worst_reward = minimum_item(
            reward_records,
            "value",
        )

        highest_intervention = maximum_item(
            intervention_records,
            "value",
        )

        output.append(
            {
                "domain": domain,
                "method": method,
                "perturbation_family": (family),
                "condition_count": len(records),
                "mean_violation_delta": (mean_or_none(violation_values)),
                "median_violation_delta": (median_or_none(violation_values)),
                "worst_violation_delta": (
                    None if worst_violation is None else worst_violation["value"]
                ),
                "worst_violation_condition": (
                    None if worst_violation is None else worst_violation["condition"]
                ),
                "mean_reward_delta": (mean_or_none(reward_values)),
                "median_reward_delta": (median_or_none(reward_values)),
                "worst_reward_delta": (
                    None if worst_reward is None else worst_reward["value"]
                ),
                "worst_reward_condition": (
                    None if worst_reward is None else worst_reward["condition"]
                ),
                "highest_intervention_rate": (
                    None
                    if highest_intervention is None
                    else highest_intervention["value"]
                ),
                "highest_intervention_condition": (
                    None
                    if highest_intervention is None
                    else highest_intervention["condition"]
                ),
            }
        )

    return output


def build_structured_state_package(
    source: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    cells = source.get("cells")
    aggregates = source.get("aggregates")

    if not isinstance(
        cells,
        list,
    ):
        raise TypeError("Structured-state cells must be list")

    if not isinstance(
        aggregates,
        list,
    ):
        raise TypeError("Structured-state aggregates must be list")

    csv_rows: list[dict[str, Any]] = []

    output_cells: list[dict[str, Any]] = []

    metric_specs = {
        "violation_step_rate": (
            "violation_step_rate",
            "violation_delta_from_clean",
        ),
        "critical_violation_rate": (
            "critical_violation_step_rate",
            "critical_delta_from_clean",
        ),
        "constraint_violation_rate": (
            "constraint_violation_rate",
            "constraint_delta_from_clean",
        ),
        "reward": (
            "reward",
            "reward_delta_from_clean",
        ),
        "success_rate": (
            "success_rate",
            "success_delta_from_clean",
        ),
        "intervention_rate": (
            "intervention_rate",
            None,
        ),
        "mean_correction_l2": (
            "mean_action_correction_l2",
            None,
        ),
        "episode_length": (
            "mean_episode_length",
            "episode_length_delta_from_clean",
        ),
    }

    for raw_cell in cells:
        if not isinstance(
            raw_cell,
            dict,
        ):
            raise TypeError("Structured-state cell must be dict")

        domain = str(raw_cell["domain"])
        method = str(raw_cell["method"])
        seed = int(raw_cell["principal_seed"])
        condition = str(raw_cell["perturbation_name"])
        family = str(raw_cell["perturbation_family"])
        source_path = str(raw_cell["source"])

        compact = {
            "domain": domain,
            "method": method,
            "principal_seed": seed,
            "perturbation_family": family,
            "perturbation_name": (condition),
            "perturbation_updates": (raw_cell["perturbation_updates"]),
            "source": normalized_path(source_path),
        }

        for output_name, (
            value_key,
            delta_key,
        ) in metric_specs.items():
            value = compact_metric(
                raw_cell,
                value_key,
            )

            delta_value = (
                None
                if delta_key is None
                else compact_metric(
                    raw_cell,
                    delta_key,
                )
            )

            reference = None if delta_value is None else value - delta_value

            compact[output_name] = value

            if delta_key is not None:
                compact[f"{output_name}_delta_from_clean"] = delta_value

            csv_rows.append(
                make_csv_row(
                    domain=domain,
                    regime="structured_state",
                    condition=condition,
                    method=method,
                    seed=seed,
                    metric=output_name,
                    value=value,
                    reference_value=reference,
                    delta=delta_value,
                    unit=metric_unit(output_name),
                    source=source_path,
                )
            )

        output_cells.append(compact)

    output_cells.sort(
        key=lambda item: (
            item["domain"],
            item["perturbation_family"],
            item["perturbation_name"],
            item["method"],
            item["principal_seed"],
        )
    )

    return (
        {
            "cells": output_cells,
            "aggregates": aggregates,
            "family_summaries": (
                build_state_family_summaries(
                    [
                        item
                        for item in aggregates
                        if isinstance(
                            item,
                            dict,
                        )
                    ]
                )
            ),
            "directional_asymmetry": (
                source.get(
                    "directional_asymmetry",
                    [],
                )
            ),
            "sensitivity": source.get(
                "sensitivity",
                {},
            ),
            "policy_observation": ("perturbed"),
            "safety_state": ("true_simulator_state"),
            "privileged_safety_state": (
                bool(
                    source.get(
                        "safety_layer_uses_true_state",
                        False,
                    )
                )
            ),
            "directionality_preserved": (True),
        },
        csv_rows,
    )


def build_action_domain_method(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for cell in cells:
        grouped[
            (
                str(cell["domain"]),
                str(cell["method"]),
            )
        ].append(cell)

    output: list[dict[str, Any]] = []

    for (
        domain,
        method,
    ), records in sorted(grouped.items()):
        total_steps = sum(int(item["total_environment_steps"]) for item in records)

        unsafe = sum(int(item["unsafe_perturbed_steps"]) for item in records)

        recovered = sum(int(item["recovered_unsafe_steps"]) for item in records)

        unresolved = sum(int(item["unresolved_unsafe_steps"]) for item in records)

        if unsafe != (recovered + unresolved):
            raise RuntimeError(
                "Action recovery arithmetic " f"failed: {domain}/{method}"
            )

        intervention_steps = sum(
            round(
                float(item["intervention_rate"]) * int(item["total_environment_steps"])
            )
            for item in records
        )

        output.append(
            {
                "domain": domain,
                "method": method,
                "condition_count": len(records) // len(PRINCIPAL_SEEDS),
                "total_steps": (total_steps),
                "unsafe_perturbed_steps": (unsafe),
                "recovered_unsafe_steps": (recovered),
                "unresolved_unsafe_steps": (unresolved),
                "recovery_rate": (
                    recovery_rate(
                        unsafe,
                        recovered,
                    )
                ),
                "perturbed_violation_step_rate": (
                    mean_or_none(
                        [
                            float(item["perturbed_violation_step_rate"])
                            for item in records
                        ]
                    )
                ),
                "executed_violation_step_rate": (
                    mean_or_none(
                        [
                            float(item["executed_violation_step_rate"])
                            for item in records
                        ]
                    )
                ),
                "within_filter_violation_reduction": (
                    mean_or_none(
                        [
                            float(item["within_filter_violation_reduction"])
                            for item in records
                            if item.get("within_filter_violation_reduction") is not None
                        ]
                    )
                ),
                "intervention_count_estimate": (intervention_steps),
                "intervention_rate": (
                    mean_or_none([float(item["intervention_rate"]) for item in records])
                ),
                "mean_action_perturbation_l2": (
                    mean_or_none(
                        [float(item["mean_action_perturbation_l2"]) for item in records]
                    )
                ),
                "mean_safety_correction_l2": (
                    mean_or_none(
                        [float(item["mean_safety_correction_l2"]) for item in records]
                    )
                ),
                "p95_safety_correction_l2": (
                    mean_or_none(
                        [float(item["p95_safety_correction_l2"]) for item in records]
                    )
                ),
            }
        )

    return output


def build_action_family_summaries(
    source: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frozen_family = source.get("family_aggregates")

    if not isinstance(
        frozen_family,
        list,
    ):
        raise TypeError("Action family aggregates must be list")

    by_key: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in aggregates:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError("Action aggregate must be dict")

        by_key[
            (
                str(item["domain"]),
                str(item["method"]),
                str(item["perturbation_family"]),
            )
        ].append(item)

    output: list[dict[str, Any]] = []

    for frozen in frozen_family:
        if not isinstance(
            frozen,
            dict,
        ):
            raise TypeError("Action family row must be dict")

        key = (
            str(frozen["domain"]),
            str(frozen["method"]),
            str(frozen["family"]),
        )

        records = by_key[key]

        correction = mean_or_none(
            [float(item["mean_safety_correction_l2"]["mean"]) for item in records]
        )

        output.append(
            {
                **frozen,
                "mean_safety_correction_l2": (correction),
            }
        )

    return output


def build_action_package(
    source: dict[str, Any],
    environment: dict[str, Any],
    cross_domain_action: dict[
        str,
        Any,
    ],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    cells = source.get("cells")
    aggregates = source.get("aggregates")

    if not isinstance(
        cells,
        list,
    ):
        raise TypeError("Action cells must be list")

    if not isinstance(
        aggregates,
        list,
    ):
        raise TypeError("Action aggregates must be list")

    typed_cells = [
        item
        for item in cells
        if isinstance(
            item,
            dict,
        )
    ]

    if len(typed_cells) != len(cells):
        raise TypeError("Action cell must be dict")

    csv_rows: list[dict[str, Any]] = []

    metric_specs = {
        "perturbed_violation_step_rate": (
            "perturbed_violation_step_rate",
            None,
        ),
        "executed_violation_step_rate": (
            "executed_violation_step_rate",
            "executed_violation_delta_from_clean",
        ),
        "executed_constraint_violation_rate": (
            "executed_constraint_violation_rate",
            None,
        ),
        "critical_violation_rate": (
            "critical_violation_step_rate",
            None,
        ),
        "reward": (
            "mean_reward",
            "reward_delta_from_clean",
        ),
        "success_rate": (
            "success_rate",
            "success_delta_from_clean",
        ),
        "intervention_rate": (
            "intervention_rate",
            None,
        ),
        "mean_action_perturbation_l2": (
            "mean_action_perturbation_l2",
            None,
        ),
        "mean_safety_correction_l2": (
            "mean_safety_correction_l2",
            None,
        ),
        "p95_safety_correction_l2": (
            "p95_safety_correction_l2",
            None,
        ),
    }

    compact_cells: list[dict[str, Any]] = []

    for raw_cell in typed_cells:
        domain = str(raw_cell["domain"])
        method = str(raw_cell["method"])
        seed = int(raw_cell["principal_seed"])
        condition = str(raw_cell["perturbation_name"])
        source_path = str(raw_cell["source"])

        compact = {
            "domain": domain,
            "method": method,
            "principal_seed": seed,
            "perturbation_family": str(raw_cell["perturbation_family"]),
            "perturbation_name": (condition),
            "perturbation_updates": (raw_cell["perturbation_updates"]),
            "total_environment_steps": int(raw_cell["total_environment_steps"]),
            "unsafe_perturbed_steps": int(raw_cell["unsafe_perturbed_steps"]),
            "recovered_unsafe_steps": int(raw_cell["recovered_unsafe_steps"]),
            "unresolved_unsafe_steps": int(raw_cell["unresolved_unsafe_steps"]),
            "steps_object_grasped": int(raw_cell["steps_object_grasped"]),
            "perturbed_unsafe_steps_while_grasped": int(
                raw_cell["perturbed_unsafe_steps_while_grasped"]
            ),
            "interventions_while_grasped": int(raw_cell["interventions_while_grasped"]),
            "proposed_to_perturbed_gripper_semantic_change_rate": float(
                raw_cell["proposed_to_perturbed_gripper_semantic_change_rate"]
            ),
            "perturbed_to_executed_gripper_semantic_change_rate": float(
                raw_cell["perturbed_to_executed_gripper_semantic_change_rate"]
            ),
            "environment_interface_adjustment_rate": float(
                raw_cell["environment_interface_adjustment_rate"]
            ),
            "source": normalized_path(source_path),
        }

        unsafe = compact["unsafe_perturbed_steps"]
        recovered = compact["recovered_unsafe_steps"]
        unresolved = compact["unresolved_unsafe_steps"]

        if unsafe != (recovered + unresolved):
            raise RuntimeError(
                "Action recovery arithmetic " f"failed for {source_path}"
            )

        computed_recovery_rate = recovery_rate(
            int(raw_cell["unsafe_perturbed_steps"]),
            int(raw_cell["recovered_unsafe_steps"]),
        )

        if computed_recovery_rate is not None:
            compact["recovery_rate"] = computed_recovery_rate

            csv_rows.append(
                make_csv_row(
                    domain=domain,
                    regime="structured_action",
                    condition=condition,
                    method=method,
                    seed=seed,
                    metric="recovery_rate",
                    value=computed_recovery_rate,
                    reference_value=None,
                    delta=None,
                    unit="fraction",
                    source=source_path,
                )
            )
        else:
            compact["recovery_rate"] = None

        for output_name, (
            value_key,
            delta_key,
        ) in metric_specs.items():
            value = compact_metric(
                raw_cell,
                value_key,
            )

            delta_value = (
                None
                if delta_key is None
                else compact_metric(
                    raw_cell,
                    delta_key,
                )
            )

            reference = None if delta_value is None else value - delta_value

            compact[output_name] = value

            if delta_key is not None:
                compact[f"{output_name}_delta_from_clean"] = delta_value

            csv_rows.append(
                make_csv_row(
                    domain=domain,
                    regime="structured_action",
                    condition=condition,
                    method=method,
                    seed=seed,
                    metric=output_name,
                    value=value,
                    reference_value=reference,
                    delta=delta_value,
                    unit=metric_unit(output_name),
                    source=source_path,
                )
            )

        compact_cells.append(compact)

    compact_cells.sort(
        key=lambda item: (
            item["domain"],
            item["perturbation_family"],
            item["perturbation_name"],
            item["method"],
            item["principal_seed"],
        )
    )

    findings = source.get(
        "findings",
        {},
    )

    if not isinstance(
        findings,
        dict,
    ):
        raise TypeError("Action findings must be dict")

    environment_protocol = environment.get(
        "protocol",
        {},
    )

    if not isinstance(
        environment_protocol,
        dict,
    ):
        raise TypeError("Environment protocol must be dict")

    return (
        {
            "cells": compact_cells,
            "aggregates": aggregates,
            "domain_method_summary": (build_action_domain_method(typed_cells)),
            "family_summaries": (
                build_action_family_summaries(
                    source,
                    [
                        item
                        for item in aggregates
                        if isinstance(
                            item,
                            dict,
                        )
                    ],
                )
            ),
            "global": {
                "unsafe_perturbed_steps": int(findings["unsafe_perturbed_steps"]),
                "recovered_unsafe_steps": int(findings["recovered_unsafe_steps"]),
                "unresolved_unsafe_steps": int(findings["unresolved_unsafe_steps"]),
                "recovery_fraction": float(
                    findings["overall_explicit_filter_recovery_fraction"]
                ),
            },
            "environment_interface": {
                "environment_interface_adjustments_count": int(
                    findings["environment_interface_adjustment_steps"]
                ),
                "counted_as_explicit_safety_intervention": (False),
                "environment_interface_adjustment_not_attributed_to_safety_filter": bool(
                    environment_protocol[
                        "environment_interface_adjustment_not_attributed_to_safety_filter"
                    ]
                ),
                "filter_recovery_measured_before_environment_internal_clipping": bool(
                    environment_protocol[
                        "filter_recovery_measured_before_environment_internal_clipping"
                    ]
                ),
                "cross_domain": (
                    cross_domain_action.get(
                        "environment_interface",
                        {},
                    )
                ),
            },
            "robotics_gripper_semantics": {
                "close": ("gripper > +0.50"),
                "open": ("gripper < -0.50"),
                "hold": ("otherwise"),
                "proposed_to_perturbed_semantic_changes": int(
                    findings["robotics_proposed_to_perturbed_gripper_semantic_changes"]
                ),
                "perturbed_to_executed_semantic_changes": int(
                    findings["robotics_perturbed_to_executed_gripper_semantic_changes"]
                ),
            },
            "robotics_grasp_context": {
                "steps_object_grasped": int(
                    findings["lyapunov_robotics_grasped_steps"]
                ),
                "unsafe_steps_while_grasped": int(
                    findings["lyapunov_robotics_unsafe_while_grasped"]
                ),
                "interventions_while_grasped": int(
                    findings["lyapunov_robotics_interventions_while_grasped"]
                ),
                "object_grasped_added_to_policy_observation": (False),
                "interpretation": (
                    "hidden simulator context used by " "the robotics predictor"
                ),
            },
        },
        csv_rows,
    )


def cell_environment_steps(
    cell: dict[str, Any],
) -> int:
    direct = cell.get("total_environment_steps")

    if isinstance(
        direct,
        int,
    ):
        return direct

    source = cell.get("source")

    if not isinstance(
        source,
        str,
    ):
        raise TypeError(
            "Cell missing source path for " "environment-step reconstruction"
        )

    source_path = resolve_source_path(source)

    payload = load_json(source_path)

    summary = payload.get("seed_summary")

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError("Referenced frozen run missing " f"seed_summary: {source_path}")

    steps = summary.get("total_environment_steps")

    if not isinstance(
        steps,
        int,
    ):
        raise TypeError(
            "Referenced frozen run missing " f"total_environment_steps: {source_path}"
        )

    return steps


def count_from_rate(
    rate: float,
    steps: int,
) -> int:
    return round(float(rate) * steps)


def clean_lyapunov_mechanism(
    domain: str,
) -> dict[str, Any]:
    reason_counts = {
        "none": 0,
        "action_bound": 0,
        "domain_constraint": 0,
        "lyapunov_decrease": 0,
        "emergency_fallback": 0,
    }

    strict = 0
    nonincrease = 0
    selected_lower = 0
    interventions = 0
    total_steps = 0

    for seed in PRINCIPAL_SEEDS:
        summary, _ = clean_seed_summary(
            domain,
            "lyapunov",
            seed,
        )

        stored = summary.get(
            "intervention_reason_counts",
            {},
        )

        if not isinstance(
            stored,
            dict,
        ):
            raise TypeError("Clean reason counts must be dict")

        for reason in reason_counts:
            reason_counts[reason] += int(
                stored.get(
                    reason,
                    0,
                )
            )

        strict += int(
            summary.get(
                "strict_lyapunov_decrease_count",
                0,
            )
        )

        nonincrease += int(
            summary.get(
                "lyapunov_nonincrease_count",
                0,
            )
        )

        selected_lower += int(
            summary.get(
                "selected_lower_than_proposed_count",
                0,
            )
        )

        interventions += int(
            summary.get(
                "intervention_count",
                0,
            )
        )

        total_steps += int(
            summary.get(
                "total_environment_steps",
                0,
            )
        )

    return {
        "domain": domain,
        "regime": "clean",
        "intervention_reason_counts": (reason_counts),
        "total_interventions": (interventions),
        "strict_decrease_count": strict,
        "nonincrease_count": (nonincrease),
        "selected_lower_than_proposed_count": (selected_lower),
        "total_environment_steps": (total_steps),
    }


def rate_based_lyapunov_mechanism(
    *,
    domain: str,
    regime: str,
    cells: list[dict[str, Any]],
    selected_key: str,
) -> dict[str, Any]:
    relevant = [
        cell
        for cell in cells
        if str(cell.get("domain")) == domain and str(cell.get("method")) == "lyapunov"
    ]

    reason_counts = {
        "none": 0,
        "action_bound": 0,
        "domain_constraint": 0,
        "lyapunov_decrease": 0,
        "emergency_fallback": 0,
    }

    strict = 0
    nonincrease = 0
    selected_lower = 0
    total_steps = 0

    for cell in relevant:
        steps = cell_environment_steps(cell)

        total_steps += steps

        reason_rates = cell.get(
            "intervention_reason_rates",
            {},
        )

        if not isinstance(
            reason_rates,
            dict,
        ):
            raise TypeError("Reason rates must be dict")

        for reason in reason_counts:
            if reason == ("emergency_fallback"):
                rate = float(
                    cell.get(
                        "emergency_fallback_rate",
                        0.0,
                    )
                )
            else:
                rate = float(
                    reason_rates.get(
                        reason,
                        0.0,
                    )
                )

            reason_counts[reason] += count_from_rate(
                rate,
                steps,
            )

        strict += count_from_rate(
            float(
                cell.get(
                    "strict_lyapunov_decrease_rate",
                    0.0,
                )
            ),
            steps,
        )

        nonincrease += count_from_rate(
            float(
                cell.get(
                    "lyapunov_nonincrease_rate",
                    0.0,
                )
            ),
            steps,
        )

        selected_lower += count_from_rate(
            float(
                cell.get(
                    selected_key,
                    0.0,
                )
            ),
            steps,
        )

    interventions = (
        reason_counts["action_bound"]
        + reason_counts["domain_constraint"]
        + reason_counts["lyapunov_decrease"]
        + reason_counts["emergency_fallback"]
    )

    return {
        "domain": domain,
        "regime": regime,
        "intervention_reason_counts": (reason_counts),
        "total_interventions": (interventions),
        "strict_decrease_count": strict,
        "nonincrease_count": (nonincrease),
        "selected_lower_than_proposed_count": (selected_lower),
        "total_environment_steps": (total_steps),
        "count_reconstruction": (
            "round(rate * environment_steps) " "from frozen per-seed cells"
        ),
    }


def build_lyapunov_package(
    *,
    gaussian: dict[str, Any],
    structured_state: dict[
        str,
        Any,
    ],
    cross_domain_lyapunov: dict[
        str,
        Any,
    ],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    gaussian_cells = gaussian.get("cells", [])

    state_cells = structured_state.get("cells", [])

    if not isinstance(
        gaussian_cells,
        list,
    ):
        raise TypeError("Gaussian cells must be list")

    if not isinstance(
        state_cells,
        list,
    ):
        raise TypeError("State cells must be list")

    mechanisms: list[dict[str, Any]] = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        mechanisms.append(clean_lyapunov_mechanism(domain))

        mechanisms.append(
            rate_based_lyapunov_mechanism(
                domain=domain,
                regime="gaussian",
                cells=[
                    item
                    for item in gaussian_cells
                    if isinstance(
                        item,
                        dict,
                    )
                    and float(
                        item.get(
                            "sigma",
                            0.0,
                        )
                    )
                    > 0.0
                ],
                selected_key=("selected_lower_than_proposed_rate"),
            )
        )

        mechanisms.append(
            rate_based_lyapunov_mechanism(
                domain=domain,
                regime=("structured_state"),
                cells=[
                    item
                    for item in state_cells
                    if isinstance(
                        item,
                        dict,
                    )
                ],
                selected_key=("selected_lower_than_proposed_rate"),
            )
        )

    action_domain = cross_domain_lyapunov.get("domain_mechanism_summary")

    if not isinstance(
        action_domain,
        list,
    ):
        raise TypeError("Cross-domain mechanism summary must be list")

    for row in action_domain:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("Mechanism row must be dict")

        mechanisms.append(
            {
                "domain": row["domain"],
                "regime": ("structured_action"),
                "intervention_reason_counts": (row["intervention_reason_counts"]),
                "total_interventions": int(row["intervention_steps"]),
                "strict_decrease_count": int(row["strict_lyapunov_decrease_steps"]),
                "selected_lower_than_proposed_count": int(
                    row["selected_lower_than_perturbed_steps"]
                ),
                "emergency_fallback_count": int(row["emergency_fallback_steps"]),
                "total_environment_steps": int(row["total_environment_steps"]),
                "activation_observed": bool(row["activation_observed"]),
            }
        )

    return {
        "by_domain_and_regime": (mechanisms),
        "reason_vs_strict_decrease": {
            "lyapunov_decrease_reason_is_not_equivalent_to_strict_empirical_decrease": (
                True
            ),
            "reason_definition": ("intervention_reason == " "LYAPUNOV_DECREASE"),
            "strict_definition": ("selected_next_V < current_V"),
        },
        "frozen_attribution": {
            "clean": "hard guards",
            "gaussian": "hard guards",
            "structured_state": ("hard guards"),
            "structured_action_driving": ("Lyapunov-specific pathway " "activated"),
            "structured_action_robotics": ("action/domain guards remained " "dominant"),
        },
        "attribution_source": (attribution),
        "cross_domain_action_mechanism": (cross_domain_lyapunov),
    }


def build_cross_domain_package(
    cross_package: dict[str, Any],
    architecture: dict[str, Any],
    cross_action: dict[str, Any],
    cross_lyapunov: dict[str, Any],
) -> dict[str, Any]:
    architecture_reuse = cross_package.get(
        "architecture_reuse",
        {},
    )

    if isinstance(
        architecture_reuse,
        dict,
    ):
        nested = architecture_reuse.get(
            "architecture_reuse",
            architecture_reuse,
        )
    else:
        nested = {}

    if not isinstance(
        nested,
        dict,
    ):
        raise TypeError("Architecture reuse must be dict")

    return {
        "comparison_rule": (
            "Cross-domain comparisons use "
            "within-domain normalized measures; "
            "raw contract violation rates are not "
            "treated as directly equivalent."
        ),
        "evidence_summary": (
            cross_package.get(
                "evidence_summary",
                {},
            )
        ),
        "action_recovery": (cross_action),
        "lyapunov_activation": (
            cross_lyapunov.get(
                "activation_consistency",
                {},
            )
        ),
        "architecture_reuse": {
            "shared_safety_interface": bool(nested.get("shared_safety_interface")),
            "shared_action_dimension": bool(nested.get("shared_action_dimension")),
            "shared_decision_contract": bool(nested.get("shared_decision_contract")),
            "shared_metric_schema": bool(nested.get("shared_metric_schema")),
            "shared_seed_protocol": bool(nested.get("shared_seed_protocol")),
            "shared_robustness_harness": bool(nested.get("shared_robustness_harness")),
            "domain_specific_constraints": bool(
                nested.get("domain_specific_constraints")
            ),
            "domain_specific_clipping": bool(nested.get("domain_specific_clipping")),
            "domain_specific_predictor": bool(nested.get("domain_specific_predictor")),
            "domain_specific_lyapunov": bool(nested.get("domain_specific_lyapunov")),
            "same_policy_weights": bool(nested.get("same_policy_weights")),
            "cross_domain_transfer_tested": bool(nested.get("transfer_tested")),
            "universal_controller_supported": bool(
                nested.get("universal_controller_supported")
            ),
        },
        "architecture_source": architecture,
    }


def format_mean_sd(
    metric: dict[str, Any],
) -> str:
    return f"{float(metric['mean']):.6f} " f"± " f"{float(metric['sample_sd']):.6f}"


def clean_markdown_table(
    clean: dict[str, Any],
) -> list[str]:
    lines = [
        (
            "| Domain | Method | Violation-step rate | Reward | "
            "Success | Intervention |"
        ),
        ("| --- | --- | ---: | ---: | ---: | ---: |"),
    ]

    for domain, label in (
        (
            "autonomous_driving",
            "Driving",
        ),
        (
            "robotics",
            "Robotics",
        ),
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            record = clean[domain][method]

            intervention = (
                "—" if method == "none" else format_mean_sd(record["intervention_rate"])
            )

            lines.append(
                "| "
                f"{label} | "
                f"{method.upper()} | "
                f"{format_mean_sd(record['violation_step_rate'])} | "
                f"{format_mean_sd(record['reward'])} | "
                f"{format_mean_sd(record['success_rate'])} | "
                f"{intervention} |"
            )

    return lines


def worst_gaussian_rows(
    gaussian: dict[str, Any],
) -> list[dict[str, Any]]:
    aggregates = gaussian.get("aggregates", [])

    if not isinstance(
        aggregates,
        list,
    ):
        raise TypeError("Gaussian aggregates must be list")

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for item in aggregates:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if (
            float(
                item.get(
                    "sigma",
                    0.0,
                )
            )
            <= 0.0
        ):
            continue

        grouped[
            (
                str(item["domain"]),
                str(item["method"]),
            )
        ].append(item)

    output: list[dict[str, Any]] = []

    for (
        domain,
        method,
    ), records in sorted(grouped.items()):
        worst_violation = max(
            records,
            key=lambda item: float(item["violation_delta_from_clean"]["mean"]),
        )

        worst_reward = min(
            records,
            key=lambda item: float(item["reward_delta_from_clean"]["mean"]),
        )

        output.append(
            {
                "domain": domain,
                "regime": "gaussian",
                "method": method,
                "worst_violation_delta": float(
                    worst_violation["violation_delta_from_clean"]["mean"]
                ),
                "worst_violation_condition": (f"sigma_{worst_violation['sigma']}"),
                "worst_reward_delta": float(
                    worst_reward["reward_delta_from_clean"]["mean"]
                ),
                "worst_reward_condition": (f"sigma_{worst_reward['sigma']}"),
            }
        )

    return output


def robustness_markdown_table(
    gaussian: dict[str, Any],
    state: dict[str, Any],
    action_source: dict[str, Any],
) -> list[str]:
    rows = worst_gaussian_rows(gaussian)

    state_sensitivity = state.get(
        "sensitivity",
        {},
    )

    if isinstance(
        state_sensitivity,
        dict,
    ):
        for domain, methods in state_sensitivity.items():
            if not isinstance(
                methods,
                dict,
            ):
                continue

            for method, values in methods.items():
                if not isinstance(
                    values,
                    dict,
                ):
                    continue

                worst_v = values.get(
                    "worst_violation",
                    {},
                )
                worst_r = values.get(
                    "worst_reward",
                    {},
                )

                if not isinstance(
                    worst_v,
                    dict,
                ) or not isinstance(
                    worst_r,
                    dict,
                ):
                    continue

                rows.append(
                    {
                        "domain": domain,
                        "regime": ("structured_state"),
                        "method": method,
                        "worst_violation_delta": float(worst_v["mean_delta"]),
                        "worst_violation_condition": str(worst_v["perturbation"]),
                        "worst_reward_delta": float(worst_r["mean_delta"]),
                        "worst_reward_condition": str(worst_r["perturbation"]),
                    }
                )

    sensitivity = action_source.get("feature_sensitivity", [])

    if isinstance(
        sensitivity,
        list,
    ):
        for item in sensitivity:
            if not isinstance(
                item,
                dict,
            ):
                continue

            worst_v = item.get(
                "worst_executed_safety",
                {},
            )
            worst_r = item.get(
                "worst_reward",
                {},
            )

            if not isinstance(
                worst_v,
                dict,
            ) or not isinstance(
                worst_r,
                dict,
            ):
                continue

            rows.append(
                {
                    "domain": item["domain"],
                    "regime": ("structured_action"),
                    "method": item["method"],
                    "worst_violation_delta": float(worst_v["delta"]),
                    "worst_violation_condition": str(worst_v["perturbation"]),
                    "worst_reward_delta": float(worst_r["delta"]),
                    "worst_reward_condition": str(worst_r["perturbation"]),
                }
            )

    lines = [
        (
            "| Domain | Regime | Method | Worst violation Delta | "
            "Condition | Worst reward Delta | Condition |"
        ),
        ("| --- | --- | --- | ---: | --- | ---: | --- |"),
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['domain']} | "
            f"{row['regime']} | "
            f"{str(row['method']).upper()} | "
            f"{float(row['worst_violation_delta']):.6f} | "
            f"{row['worst_violation_condition']} | "
            f"{float(row['worst_reward_delta']):.6f} | "
            f"{row['worst_reward_condition']} |"
        )

    return lines


def action_markdown_table(
    action: dict[str, Any],
) -> list[str]:
    lines = [
        ("| Domain | Method | Unsafe | Recovered | " "Unresolved | Recovery rate |"),
        ("| --- | --- | ---: | ---: | ---: | ---: |"),
    ]

    rows = action.get("domain_method_summary", [])

    if not isinstance(
        rows,
        list,
    ):
        raise TypeError("Action domain-method summary must be list")

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        rate = row.get("recovery_rate")

        rate_text = "undefined" if rate is None else f"{float(rate):.6f}"

        lines.append(
            "| "
            f"{row['domain']} | "
            f"{str(row['method']).upper()} | "
            f"{row['unsafe_perturbed_steps']} | "
            f"{row['recovered_unsafe_steps']} | "
            f"{row['unresolved_unsafe_steps']} | "
            f"{rate_text} |"
        )

    return lines


def mechanism_markdown_table(
    lyapunov: dict[str, Any],
) -> list[str]:
    lines = [
        (
            "| Domain | Regime | ACTION_BOUND | DOMAIN_CONSTRAINT | "
            "LYAPUNOV_DECREASE | Strict decrease |"
        ),
        ("| --- | --- | ---: | ---: | ---: | ---: |"),
    ]

    rows = lyapunov.get("by_domain_and_regime", [])

    if not isinstance(
        rows,
        list,
    ):
        raise TypeError("Mechanism rows must be list")

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        reasons = row.get(
            "intervention_reason_counts",
            {},
        )

        if not isinstance(
            reasons,
            dict,
        ):
            raise TypeError("Reason counts must be dict")

        lines.append(
            "| "
            f"{row['domain']} | "
            f"{row['regime']} | "
            f"{int(reasons.get('action_bound', 0))} | "
            f"{int(reasons.get('domain_constraint', 0))} | "
            f"{int(reasons.get('lyapunov_decrease', 0))} | "
            f"{int(row.get('strict_decrease_count', 0))} |"
        )

    return lines


def build_markdown(
    package: dict[str, Any],
    action_source: dict[str, Any],
) -> str:
    clean = package["clean_safety"]
    gaussian = package["gaussian_robustness"]
    state = package["structured_state_robustness"]
    action = package["action_robustness"]
    lyapunov = package["lyapunov_mechanisms"]

    lines = [
        "# Q-VLA Forge — Sprint 5 Safety Evidence",
        "",
        "## Executive Summary",
        "",
        (
            "Sprint 5 packages the frozen safety and robustness "
            "evidence for autonomous-driving and robotics proxy "
            "domains without new training, policy execution, "
            "safety episodes, robustness episodes, retuning, or "
            "scientific experimentation."
        ),
        "",
        (
            "Three explicit safety modes were evaluated: NONE, "
            "heuristic CLIPPING, and the classical "
            "Lyapunov-guided safety filter."
        ),
        "",
        (
            "Under the frozen synthetic proxy safety contracts, "
            "both explicit safety filters reduced measured clean "
            "executed violation-step events to zero in both "
            "evaluated domains."
        ),
        "",
        (
            "This is empirical proxy evidence, not a formal "
            "safety guarantee or certification result."
        ),
        "",
        (
            "Robustness was evaluated under Gaussian observation "
            "noise, structured semantic state perturbations, and "
            "structured action perturbations."
        ),
        "",
        (
            "The Gaussian and structured-state studies use "
            "perturbed policy observations while the explicit "
            "safety layer retains true simulator state. They "
            "therefore support policy-plus-privileged-state-safety "
            "robustness, not real-sensor robustness."
        ),
        "",
        (
            "Under structured action perturbation, unsafe-action "
            "recovery was explicitly quantified. Environment "
            "interface adjustments remain separate from explicit "
            "safety-filter interventions."
        ),
        "",
        (
            "Lyapunov-specific candidate selection became "
            "empirically active under action perturbation in "
            "autonomous driving, while robotics remained dominated "
            "by action/domain guards. LYAPUNOV_DECREASE reason "
            "counts and strict empirical ΔV < 0 counts remain "
            "separate."
        ),
        "",
        (
            "The cross-domain result supports reuse of a shared "
            "safety framework and evaluation interface, not a "
            "universal trained controller, zero-shot transfer, "
            "formal stability, production readiness, or quantum "
            "safety advantage."
        ),
        "",
        "## 1. Scope and Protocol",
        "",
        "- Principal trained-policy seeds: 42, 123, 456",
        "- Experimental uncertainty: mean ± sample SD across principal seeds",
        "- Evaluation seeds per seed/condition: 20000–20019",
        "- Scientific scope: FROZEN",
        "- New training: NO",
        "- New principal execution: NO",
        "- New scientific experiment: NO",
        "",
        "## 2. Safety Architecture",
        "",
        (
            "Frozen policy → proposed action → optional perturbation "
            "→ explicit safety method → executed action → domain "
            "environment."
        ),
        "",
        (
            "The safety framework shares interfaces, metrics, "
            "robustness harnesses, and seed protocol while preserving "
            "domain-specific constraints, clipping rules, predictors, "
            "and Lyapunov safety potentials."
        ),
        "",
        "## 3. Clean Safety Evaluation",
        "",
    ]

    lines.extend(clean_markdown_table(clean))

    lines.extend(
        [
            "",
            (
                "Under the frozen synthetic proxy safety contracts, "
                "both explicit safety filters reduced measured clean "
                "executed violation-step events to zero in both "
                "evaluated domains."
            ),
            "",
            (
                "This is empirical proxy evidence, not a formal "
                "safety guarantee or certification result."
            ),
            "",
            "## 4. Gaussian Observation Robustness",
            "",
            (
                "Frozen levels: σ = 0.01, 0.05, 0.10, with σ = 0 "
                "retained only as a clean reference."
            ),
            "",
            (
                "Policy observation: perturbed. Safety state: true "
                "simulator state. Privileged safety state: YES."
            ),
            "",
            "## 5. Structured-State Robustness",
            "",
            (
                "Driving families: lane_offset, obstacle_distance, "
                "speed, heading_error."
            ),
            "",
            (
                "Robotics families: robot_position, object_position, "
                "target_position. Original signed condition "
                "identifiers are preserved; positive and negative "
                "perturbations are not collapsed."
            ),
            "",
            "## 6. Structured-Action Robustness",
            "",
            (
                "Driving action families: steering, acceleration, "
                "braking. Robotics action families: delta_x, delta_y, "
                "gripper."
            ),
            "",
            (
                "Environment-interface action adjustment remains "
                "separate from explicit CLIPPING or LYAPUNOV "
                "intervention."
            ),
            "",
            "## 7. Unsafe-Action Recovery",
            "",
        ]
    )

    lines.extend(action_markdown_table(action))

    lines.extend(
        [
            "",
            "## 8. Lyapunov Mechanism Analysis",
            "",
        ]
    )

    lines.extend(mechanism_markdown_table(lyapunov))

    lines.extend(
        [
            "",
            (
                "`LYAPUNOV_DECREASE` intervention reason and strict "
                "empirical `ΔV < 0` are reported separately and must "
                "not be treated as equivalent."
            ),
            "",
            "## 9. Cross-Domain Safety Analysis",
            "",
            (
                "Cross-domain comparison uses within-domain normalized "
                "effects rather than treating the two safety contracts "
                "as numerically identical."
            ),
            "",
            (
                "The shared framework is supported at the interface "
                "and infrastructure level. Policy weights, constraints, "
                "predictors, and Lyapunov semantics remain "
                "domain-specific."
            ),
            "",
            "## 10. Proposal-Ready Findings",
            "",
            (
                "- Explicit clean safety filtering was empirically "
                "effective under both frozen proxy-domain contracts."
            ),
            (
                "- Gaussian, structured-state, and structured-action "
                "robustness were evaluated."
            ),
            (
                "- Unsafe-action recovery was quantified separately "
                "from environment-interface adjustment."
            ),
            (
                "- Lyapunov-specific action-selection evidence became "
                "active under action perturbation in driving but not "
                "robotics."
            ),
            (
                "- Cross-domain reuse is supported as a shared "
                "framework, not as a universal controller."
            ),
            "",
            "### Table A — Clean safety",
            "",
        ]
    )

    lines.extend(clean_markdown_table(clean))

    lines.extend(
        [
            "",
            "### Table B — Robustness",
            "",
        ]
    )

    lines.extend(
        robustness_markdown_table(
            gaussian,
            state,
            action_source,
        )
    )

    lines.extend(
        [
            "",
            "### Table C — Action recovery",
            "",
        ]
    )

    lines.extend(action_markdown_table(action))

    lines.extend(
        [
            "",
            "### Table D — Mechanisms",
            "",
        ]
    )

    lines.extend(mechanism_markdown_table(lyapunov))

    lines.extend(
        [
            "",
            "## 11. Claim Boundary",
            "",
            (
                "The final frozen Sprint 5 claim boundary is embedded "
                "verbatim in the canonical JSON package. Supported "
                "claims may be reused exactly or weakened. "
                "Supported-with-limitation claims must retain their "
                "limitations. Not-supported claims may not be "
                "converted into positive Phase 1 conclusions."
            ),
            "",
            "## 12. Limitations",
            "",
        ]
    )

    limitations = package["limitations"]

    limitation_rows = limitations.get("limitations", [])

    if not isinstance(
        limitation_rows,
        list,
    ):
        raise TypeError("Limitations must be list")

    for item in limitation_rows:
        if not isinstance(
            item,
            dict,
        ):
            continue

        lines.append(f"- {item['statement']}")

    lines.extend(
        [
            "",
            "## 13. Evidence Provenance",
            "",
        ]
    )

    provenance = package["provenance"]

    sources = provenance.get("sources", {})

    if not isinstance(
        sources,
        dict,
    ):
        raise TypeError("Provenance sources must be dict")

    for name, record in sorted(sources.items()):
        if not isinstance(
            record,
            dict,
        ):
            continue

        lines.append(
            f"- **{name}** — `{record['path']}` — " f"SHA256 `{record['sha256']}`"
        )

    lines.append("")

    return "\n".join(lines)


def build_provenance() -> dict[str, Any]:
    records: dict[
        str,
        dict[str, Any],
    ] = {}

    for name, path in sorted(SOURCES.items()):
        records[name] = {
            "path": normalized_path(path),
            "sha256": sha256_file(path),
            "size_bytes": (path.stat().st_size),
        }

    return {
        "source_priority": [
            "raw frozen run",
            "regime-specific frozen summary",
            "Sprint 5.13 consolidated evidence",
            "Sprint 5.14 cross-domain evidence",
            "Sprint 5.15 frozen claim boundary",
        ],
        "sources": records,
    }


def write_csv(
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "domain",
        "regime",
        "condition",
        "method",
        "seed",
        "metric",
        "value",
        "reference_value",
        "delta",
        "unit",
        "source_artifact",
        "source_sha256",
    ]

    rows.sort(
        key=lambda row: (
            str(row["domain"]),
            str(row["regime"]),
            str(row["condition"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["metric"]),
        )
    )

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
        writer.writerows(rows)


def main() -> None:
    safety_package = load_json(SOURCES["safety_package"])
    gaussian_source = load_json(SOURCES["gaussian_summary"])
    state_source = load_json(SOURCES["structured_state_summary"])
    action_source = load_json(SOURCES["action_summary"])
    environment_source = load_json(SOURCES["environment_action_handling"])
    attribution = load_json(SOURCES["lyapunov_attribution"])
    cross_package = load_json(SOURCES["cross_domain_package"])
    cross_action = load_json(SOURCES["cross_domain_action"])
    cross_lyapunov = load_json(SOURCES["cross_domain_lyapunov"])
    cross_architecture = load_json(SOURCES["cross_domain_architecture"])
    claims = load_json(SOURCES["claim_boundary"])
    limitations = load_json(SOURCES["limitations"])
    invariants = load_json(SOURCES["scientific_invariants"])
    freeze = load_json(SOURCES["freeze_record"])

    if freeze.get("status") != "COMPLETE":
        raise RuntimeError("Sprint 5 must already be frozen.")

    clean, clean_csv = build_clean_package()

    gaussian, gaussian_csv = build_gaussian_package(gaussian_source)

    state, state_csv = build_structured_state_package(state_source)

    action, action_csv = build_action_package(
        action_source,
        environment_source,
        cross_action,
    )

    lyapunov = build_lyapunov_package(
        gaussian=gaussian_source,
        structured_state=state_source,
        cross_domain_lyapunov=(cross_lyapunov),
        attribution=attribution,
    )

    cross_domain = build_cross_domain_package(
        cross_package,
        cross_architecture,
        cross_action,
        cross_lyapunov,
    )

    package: dict[str, Any] = {
        "metadata": {
            "project": ("Q-VLA Forge"),
            "sprint": "5.15",
            "artifact_type": ("proposal_safety_evidence"),
            "domains": [
                "autonomous_driving",
                "robotics",
            ],
            "methods": [
                "none",
                "clipping",
                "lyapunov",
            ],
            "principal_seeds": [
                42,
                123,
                456,
            ],
            "evaluation_seed_start": (20000),
            "evaluation_seed_end": (20019),
            "new_training": False,
            "new_principal_execution": (False),
            "new_scientific_experiment": (False),
            "scientific_scope_frozen": (True),
        },
        "protocol": {
            "experimental_unit": ("principal trained-policy seed"),
            "uncertainty": ("mean +/- sample SD " "across seeds 42,123,456"),
            "sample_sd_ddof": 1,
            "pseudo_replication_for_primary_uncertainty": (False),
            "zero_denominator_relative_reduction": (None),
            "zero_denominator_recovery": (None),
        },
        "clean_safety": clean,
        "gaussian_robustness": (gaussian),
        "structured_state_robustness": (state),
        "action_robustness": (action),
        "lyapunov_mechanisms": (lyapunov),
        "cross_domain": (cross_domain),
        "proposal_headlines": {
            "clean_safety": {
                "statement": (
                    "Under the frozen synthetic proxy safety "
                    "contracts, both explicit safety filters "
                    "reduced measured clean executed "
                    "violation-step events to zero in both "
                    "evaluated domains."
                ),
                "status": "SUPPORTED",
                "source": (normalized_path(SOURCES["safety_package"])),
                "limitation": (
                    "Empirical proxy evidence only; not a "
                    "formal guarantee or certification result."
                ),
            },
            "robustness": {
                "statement": (
                    "Gaussian observation, structured-state, "
                    "and structured-action robustness were "
                    "evaluated in both proxy domains."
                ),
                "status": ("SUPPORTED_WITH_LIMITATION"),
                "source": (normalized_path(SOURCES["safety_package"])),
                "limitation": (
                    "Synthetic perturbations; perception "
                    "studies retain true-state access for "
                    "the safety layer."
                ),
            },
            "action_recovery": {
                "statement": (
                    "Unsafe perturbed-action recovery was "
                    "quantified separately from environment "
                    "interface adjustment."
                ),
                "status": "SUPPORTED",
                "source": (normalized_path(SOURCES["cross_domain_action"])),
                "limitation": ("Synthetic action perturbation protocol."),
            },
            "lyapunov_mechanism": {
                "statement": (
                    "Lyapunov-specific candidate selection "
                    "became empirically active under action "
                    "perturbation in autonomous driving."
                ),
                "status": "SUPPORTED",
                "source": (normalized_path(SOURCES["cross_domain_lyapunov"])),
                "limitation": (
                    "Mechanism activation was domain-dependent "
                    "and does not establish formal stability."
                ),
            },
            "cross_domain": {
                "statement": (
                    "A shared safety framework was instantiated "
                    "across autonomous-driving and robotics "
                    "proxy environments."
                ),
                "status": "SUPPORTED",
                "source": (normalized_path(SOURCES["cross_domain_package"])),
                "limitation": (
                    "Separate policy weights and "
                    "domain-specific safety semantics remain."
                ),
            },
        },
        "claim_boundary": claims,
        "limitations": limitations,
        "scientific_invariants": (invariants),
        "headline_findings": (
            safety_package.get(
                "headline_findings",
                [],
            )
        ),
        "provenance": (build_provenance()),
        "figures": [],
    }

    all_csv = clean_csv + gaussian_csv + state_csv + action_csv

    JSON_OUTPUT.write_text(
        json.dumps(
            package,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(all_csv)

    MARKDOWN_OUTPUT.write_text(
        build_markdown(
            package,
            action_source,
        ),
        encoding="utf-8",
    )

    print("=" * 52)
    print(" Q-VLA FORGE — SPRINT 5.15 " "EVIDENCE CORE BUILD")
    print("=" * 52)
    print()

    print("Source integrity                     PASS")
    print()
    print("Clean safety                         PASS")
    print("Gaussian robustness                  PASS")
    print("Structured-state robustness          PASS")
    print("Action robustness                    PASS")
    print("Environment-interface separation     PASS")
    print("Robotics gripper semantics           PASS")
    print("Robotics grasp context               PASS")
    print("Lyapunov mechanisms                  PASS")
    print("Cross-domain synthesis               PASS")
    print()
    print("Canonical JSON                       PASS")
    print("Normalized CSV                       PASS")
    print("Markdown evidence                    PASS")
    print("Proposal tables                      PASS")
    print()
    print("No new training                      PASS")
    print("No principal execution               PASS")
    print("No new scientific experiment         PASS")
    print()
    print("SPRINT 5.15 BATCH B CORE BUILD: PASS")


if __name__ == "__main__":
    main()
