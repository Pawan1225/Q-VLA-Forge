"""Independent verification for Sprint 7.3 compression ablation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.compression_ablation import (
    DOMAINS,
    MAX_RELATIVE_MSE_DEGRADATION_PERCENT,
    METHODS,
    MIN_COMPRESSION_RATIO,
    REQUIRED_SEEDS,
    compression_quality_pass,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "compression-ablation"


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    _require(
        isinstance(payload, dict),
        "Expected JSON object.",
    )

    return payload


def _load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _verify_protocol(
    payload: dict[str, Any],
) -> None:
    protocol = payload["protocol"]

    _require(
        tuple(protocol["methods"]) == METHODS,
        "Method protocol mismatch.",
    )

    _require(
        tuple(protocol["domains"]) == DOMAINS,
        "Domain protocol mismatch.",
    )

    _require(
        tuple(payload["required_seeds"]) == REQUIRED_SEEDS,
        "Seed protocol mismatch.",
    )

    _require(
        math.isclose(
            float(protocol["minimum_compression_ratio"]),
            MIN_COMPRESSION_RATIO,
        ),
        "Compression threshold mismatch.",
    )

    _require(
        math.isclose(
            float(protocol["maximum_relative_mse_degradation_percent"]),
            MAX_RELATIVE_MSE_DEGRADATION_PERCENT,
        ),
        "Quality threshold mismatch.",
    )

    statistics = protocol["statistics"]

    _require(
        statistics["center"] == "arithmetic_mean",
        "Mean convention mismatch.",
    )

    _require(
        statistics["spread"] == "sample_standard_deviation",
        "Sample SD convention mismatch.",
    )

    _require(
        statistics["ddof"] == 1,
        "Expected ddof=1.",
    )

    _require(
        statistics["n"] == 3,
        "Expected n=3.",
    )


def _verify_coverage(
    records: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    _require(
        len(records) == 8,
        "Expected exactly eight records.",
    )

    observed = {
        (
            record["domain"],
            record["method"],
        )
        for record in records
    }

    expected = {
        (
            domain,
            method,
        )
        for domain in DOMAINS
        for method in METHODS
    }

    _require(
        observed == expected,
        "Domain/method coverage mismatch.",
    )

    return observed


def _verify_reference_and_criterion(
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        if record["method"] == "fp32":
            _require(
                record["classification"] == "reference",
                "FP32 must be REFERENCE.",
            )

            _require(
                record["criterion_pass"] is None,
                "FP32 must not be PASS/FAIL.",
            )

            _require(
                record["seed_pass_count"] is None,
                "FP32 must not have compressed seed pass count.",
            )

            _require(
                math.isclose(
                    float(record["compression_ratio"]),
                    1.0,
                ),
                "FP32 compression ratio must be 1.0x.",
            )

            continue

        ratio = float(record["compression_ratio"])

        delta_mse = float(record["relative_mse_change_mean_percent"])

        derived = compression_quality_pass(
            ratio,
            delta_mse,
        )

        _require(
            derived == record["criterion_pass"],
            (
                f"Derived criterion mismatch: "
                f"{record['domain']}/"
                f"{record['method']}"
            ),
        )

        seed_pass_count = int(record["seed_pass_count"])

        _require(
            0 <= seed_pass_count <= len(REQUIRED_SEEDS),
            "Invalid seed pass count.",
        )

    _require(
        compression_quality_pass(
            3.846,
            -0.225,
        ),
        "Negative Delta MSE must remain valid.",
    )

    _require(
        not compression_quality_pass(
            1.99,
            -100.0,
        ),
        ("Compression threshold must still " "apply for negative Delta MSE."),
    )


def _verify_statistics(
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        _require(
            record["mse_mean"] is not None,
            "Missing MSE mean.",
        )

        _require(
            record["mse_sample_std"] is not None,
            "Missing MSE sample SD.",
        )

        _require(
            float(record["mse_sample_std"]) >= 0.0,
            "Negative sample SD.",
        )

        if record["method"] != "fp32":
            _require(
                record["relative_mse_change_sample_std_percent"] is not None,
                "Missing Delta MSE sample SD.",
            )

            _require(
                float(record["relative_mse_change_sample_std_percent"]) >= 0.0,
                "Negative Delta MSE sample SD.",
            )

    deterministic_ratios = {
        (
            record["domain"],
            record["method"],
            record["compression_ratio"],
        )
        for record in records
    }

    _require(
        len(deterministic_ratios) == 8,
        "Deterministic compression values malformed.",
    )


def _verify_provenance(
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        sources = record["source_artifacts"]

        _require(
            bool(sources),
            "Missing source provenance.",
        )

        for source in sources:
            _require(
                source.startswith("results/compression/"),
                ("Source is not frozen " "Sprint 2 compression evidence."),
            )

            _require(
                "final-validation" not in source,
                ("Generated final-validation " "artifact used as source evidence."),
            )


def _verify_pareto(
    pareto_rows: list[dict[str, str]],
    combinations: set[tuple[str, str]],
) -> None:
    _require(
        len(pareto_rows) == 8,
        "Pareto dataset must contain eight rows.",
    )

    observed = {
        (
            row["domain"],
            row["method"],
        )
        for row in pareto_rows
    }

    _require(
        observed == combinations,
        "Pareto domain/method coverage mismatch.",
    )

    allowed_columns = {
        "domain",
        "method",
        "compression_ratio",
        "mse_mean",
        "mse_sample_std",
        "relative_mse_change_mean_percent",
        "relative_mse_change_sample_std_percent",
        "criterion_status",
    }

    _require(
        set(pareto_rows[0].keys()) == allowed_columns,
        "Unexpected Pareto plotting columns.",
    )


def _verify_claim_controls(
    payload: dict[str, Any],
) -> None:
    controls = payload["claim_controls"]

    blocked = (
        "tt_mps_superiority_claim_allowed",
        "quantum_inspired_compression_advantage_claim_allowed",
        "quantum_advantage_claim_allowed",
        "production_scale_extrapolation_allowed",
        "seven_b_vla_generalization_allowed",
    )

    for key in blocked:
        _require(
            controls[key] is False,
            f"Claim control must be blocked: {key}",
        )


def main() -> None:
    json_path = OUTPUT_DIR / "compression-ablation.json"

    csv_path = OUTPUT_DIR / "compression-ablation.csv"

    markdown_path = OUTPUT_DIR / "compression-ablation.md"

    pareto_path = OUTPUT_DIR / "compression-pareto.csv"

    for path in (
        json_path,
        csv_path,
        markdown_path,
        pareto_path,
    ):
        _require(
            path.exists(),
            f"Missing artifact: {path}",
        )

    payload = _load_json(json_path)

    ablation_rows = _load_csv(csv_path)

    pareto_rows = _load_csv(pareto_path)

    records = payload["records"]

    _verify_protocol(payload)

    combinations = _verify_coverage(records)

    _verify_reference_and_criterion(records)

    _verify_statistics(records)

    _verify_provenance(records)

    _verify_pareto(
        pareto_rows,
        combinations,
    )

    _verify_claim_controls(payload)

    _require(
        len(ablation_rows) == 8,
        "Ablation CSV coverage mismatch.",
    )

    markdown_text = markdown_path.read_text(encoding="utf-8")

    _require(
        "## Autonomous Driving" in markdown_text,
        "Driving Markdown section missing.",
    )

    _require(
        "## Robotics" in markdown_text,
        "Robotics Markdown section missing.",
    )

    print("=" * 64)
    print(" SPRINT 7.3 - COMPRESSION ABLATION VERIFICATION")
    print("=" * 64)
    print()

    print("Protocol:                  PASS")
    print("Coverage:                  PASS")
    print("Reference handling:        PASS")
    print("Statistics:                PASS")
    print("Joint criterion:           PASS")
    print("Provenance:                PASS")
    print("Pareto data:               PASS")
    print("Claim controls:            PASS")
    print()
    print("INDEPENDENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
