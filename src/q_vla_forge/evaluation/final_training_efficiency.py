"""Sprint 7.5 — final frozen training-efficiency comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAINING_SUMMARY = "results/training/training-validation-summary.json"

EXPECTED_SEEDS = (
    42,
    123,
    456,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_final_training_efficiency(
    root: Path,
) -> dict[str, Any]:
    """Build the Sprint 7.5 training-efficiency summary."""

    source_path = root / TRAINING_SUMMARY

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source = load_json(source_path)

    seeds = tuple(
        source.get(
            "seeds",
            [],
        )
    )

    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            "Training-efficiency summary does not " "use locked seeds 42, 123, 456."
        )

    strong_rule = source.get("strong_efficiency_rule")

    if not isinstance(
        strong_rule,
        dict,
    ):
        raise KeyError("Missing strong_efficiency_rule.")

    methods = source.get("methods")

    if not isinstance(
        methods,
        list,
    ):
        raise TypeError("Training methods must be a list.")

    validated_methods: list[dict[str, Any]] = []

    robust_methods: list[str] = []

    for method in methods:
        if not isinstance(
            method,
            dict,
        ):
            continue

        method_seeds = tuple(
            method.get(
                "seeds",
                [],
            )
        )

        if method_seeds != EXPECTED_SEEDS:
            raise ValueError(
                "Training method seed mismatch: " f"{method.get('method')}"
            )

        robust = bool(
            method.get(
                "robust_ten_percent_step_efficiency",
                False,
            )
        )

        identifier = f"{method['domain']}:" f"{method['method']}"

        if robust:
            robust_methods.append(identifier)

        validated_methods.append(
            {
                "domain": method["domain"],
                "method": method["method"],
                "target_reach": method["target_reach"],
                "trainable_parameters": method["trainable_parameters"],
                "effective_parameters": method["effective_parameters"],
                "best_validation_loss": method["best_validation_loss"],
                "test_mse": method["test_mse"],
                "steps_to_target": method["steps_to_target"],
                "samples_to_target": method["samples_to_target"],
                "seconds_to_target": method["seconds_to_target"],
                "step_reduction_percent": method["step_reduction_percent"],
                "robust_ten_percent_step_efficiency": (robust),
            }
        )

    demonstrated = bool(robust_methods)

    return {
        "sprint": "7.5",
        "protocol": ("final_training_efficiency_comparison"),
        "status": "FROZEN",
        "source": TRAINING_SUMMARY,
        "required_seeds": list(EXPECTED_SEEDS),
        "primary_efficiency_metric": source["primary_efficiency_metric"],
        "strong_efficiency_rule": (strong_rule),
        "wall_clock_interpretation": source["wall_clock_interpretation"],
        "methods": validated_methods,
        "robust_methods": robust_methods,
        "robust_ten_percent_efficiency_demonstrated": (demonstrated),
        "claim": (
            "At least one evaluated method demonstrated "
            "the locked strong training-efficiency target."
            if demonstrated
            else "Phase 1 did not demonstrate a robust "
            "10% or greater optimizer-step efficiency "
            "improvement across all three locked seeds."
        ),
        "limitations": source.get(
            "limitations",
            [],
        ),
        "new_training": False,
    }
