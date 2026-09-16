"""Sprint 7.4 — final frozen compression comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMPRESSION_SUMMARY = "results/compression/compression-pareto-summary.json"

EXPECTED_DOMAINS = (
    "driving",
    "robotics",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_compression(
    root: Path,
) -> dict[str, Any]:
    """Build the Sprint 7.4 frozen compression comparison."""

    source_path = root / COMPRESSION_SUMMARY

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source = load_json(source_path)

    pilot = source.get("pilot_feasibility")

    if not isinstance(
        pilot,
        dict,
    ):
        raise KeyError("Missing pilot_feasibility.")

    minimum_ratio = float(pilot["minimum_compression_ratio"])

    maximum_mse = float(pilot["maximum_mse_increase_percent"])

    domains: dict[str, Any] = {}
    pareto_sets: list[set[str]] = []

    for key in EXPECTED_DOMAINS:
        domain_payload = source.get(key)

        if not isinstance(
            domain_payload,
            dict,
        ):
            raise KeyError(f"Missing compression domain: {key}")

        analysis = domain_payload.get("analysis")

        if not isinstance(
            analysis,
            dict,
        ):
            raise KeyError(f"Missing analysis for {key}")

        points = analysis.get("points")

        if not isinstance(
            points,
            list,
        ):
            raise TypeError(f"Invalid compression points for {key}")

        pareto_frontier = analysis.get(
            "pareto_frontier",
            [],
        )

        if not isinstance(
            pareto_frontier,
            list,
        ):
            raise TypeError(f"Invalid Pareto frontier for {key}")

        pareto_sets.append(set(pareto_frontier))

        methods: list[dict[str, Any]] = []

        for point in points:
            if not isinstance(
                point,
                dict,
            ):
                continue

            methods.append(
                {
                    "method": point["method"],
                    "configuration": point["configuration"],
                    "family": point["family"],
                    "compression_ratio_mean": point["compression_ratio_mean"],
                    "compression_ratio_std": point["compression_ratio_std"],
                    "mse_change_mean_percent": point["mse_change_mean"],
                    "mse_change_std_percent": point["mse_change_std"],
                    "mae_change_mean_percent": point["mae_change_mean"],
                    "mae_change_std_percent": point["mae_change_std"],
                    "pilot_feasible_runs": point["pilot_feasible_runs"],
                    "all_runs_pilot_feasible": point["all_runs_pilot_feasible"],
                    "is_reference": point["is_reference"],
                }
            )

        domains[key] = {
            "domain": analysis["domain"],
            "methods": methods,
            "pareto_frontier": (pareto_frontier),
            "dominated": analysis.get(
                "dominated",
                [],
            ),
        }

    shared_pareto = sorted(set.intersection(*pareto_sets))

    int8_supported = shared_pareto == ["INT8"]

    return {
        "sprint": "7.4",
        "protocol": ("final_compression_comparison"),
        "status": "FROZEN",
        "source": COMPRESSION_SUMMARY,
        "criterion": {
            "minimum_compression_ratio": (minimum_ratio),
            "maximum_relative_mse_increase_percent": (maximum_mse),
        },
        "domains": domains,
        "cross_domain_pareto_frontier": (shared_pareto),
        "int8_cross_domain_supported": (int8_supported),
        "claim": (
            "INT8 is the only evaluated method on the "
            "Pareto frontier in both proxy domains under "
            "the frozen Phase 1 compression criterion."
            if int8_supported
            else "No single evaluated compression method "
            "is supported across both domains."
        ),
        "claim_boundaries": [
            (
                "The result applies to the evaluated "
                "Phase 1 proxy tasks and frozen model."
            ),
            (
                "SVD and TT/MPS were evaluated but did "
                "not satisfy the joint frozen criterion."
            ),
            (
                "No superiority claim is made for "
                "unseen architectures or real-world data."
            ),
        ],
        "new_training": False,
    }
