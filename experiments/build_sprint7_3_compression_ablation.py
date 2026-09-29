"""Build Sprint 7.3 final compression ablation artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.compression_ablation import (
    DOMAINS,
    MAX_RELATIVE_MSE_DEGRADATION_PERCENT,
    METHODS,
    MIN_COMPRESSION_RATIO,
    REQUIRED_SEEDS,
    load_canonical_compression_ablation,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "compression-ablation"


def _method_label(method: str) -> str:
    if method == "fp32":
        return "FP32"

    if method == "tt_mps":
        return "TT/MPS"

    return method.upper()


def _status(record: dict[str, Any]) -> str:
    if record["method"] == "fp32":
        return "REFERENCE"

    return "PASS" if record["criterion_pass"] else "FAIL"


def _build_payload() -> dict[str, Any]:
    records = [asdict(record) for record in load_canonical_compression_ablation(ROOT)]

    domain_conclusions: dict[str, Any] = {}

    for domain in DOMAINS:
        domain_records = [record for record in records if record["domain"] == domain]

        domain_conclusions[domain] = {
            "passing_compressed_methods": [
                record["method"]
                for record in domain_records
                if record["criterion_pass"] is True
            ],
            "failing_compressed_methods": [
                record["method"]
                for record in domain_records
                if record["criterion_pass"] is False
            ],
        }

    cross_domain_pass = [
        method
        for method in METHODS
        if method != "fp32"
        and all(
            any(
                record["domain"] == domain
                and record["method"] == method
                and record["criterion_pass"] is True
                for record in records
            )
            for domain in DOMAINS
        )
    ]

    return {
        "sprint": "7.3",
        "title": "Final Compression Ablation",
        "new_experiments": False,
        "new_training": False,
        "required_seeds": list(REQUIRED_SEEDS),
        "protocol": {
            "methods": list(METHODS),
            "domains": list(DOMAINS),
            "minimum_compression_ratio": (MIN_COMPRESSION_RATIO),
            "maximum_relative_mse_degradation_percent": (
                MAX_RELATIVE_MSE_DEGRADATION_PERCENT
            ),
            "statistics": {
                "center": "arithmetic_mean",
                "spread": "sample_standard_deviation",
                "ddof": 1,
                "n": 3,
            },
        },
        "records": records,
        "domain_conclusions": domain_conclusions,
        "cross_domain_conclusion": {
            "methods_meeting_joint_criterion_in_both_domains": (cross_domain_pass)
        },
        "claim_controls": {
            "tt_mps_superiority_claim_allowed": False,
            "quantum_inspired_compression_advantage_claim_allowed": False,
            "quantum_advantage_claim_allowed": False,
            "production_scale_extrapolation_allowed": False,
            "seven_b_vla_generalization_allowed": False,
        },
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


def _write_ablation_csv(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "domain",
        "method",
        "compression_ratio",
        "mse_mean",
        "mse_sample_std",
        "relative_mse_change_mean_percent",
        "relative_mse_change_sample_std_percent",
        "seed_pass_count",
        "criterion_pass",
        "classification",
        "latency_mean_ms",
        "latency_sample_std_ms",
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

            row["source_artifacts"] = ";".join(row["source_artifacts"])

            writer.writerow(row)


def _write_pareto_csv(
    path: Path,
    payload: dict[str, Any],
) -> None:
    fieldnames = [
        "domain",
        "method",
        "compression_ratio",
        "mse_mean",
        "mse_sample_std",
        "relative_mse_change_mean_percent",
        "relative_mse_change_sample_std_percent",
        "criterion_status",
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
            writer.writerow(
                {
                    "domain": record["domain"],
                    "method": record["method"],
                    "compression_ratio": record["compression_ratio"],
                    "mse_mean": record["mse_mean"],
                    "mse_sample_std": record["mse_sample_std"],
                    "relative_mse_change_mean_percent": (
                        record["relative_mse_change_mean_percent"]
                    ),
                    "relative_mse_change_sample_std_percent": (
                        record["relative_mse_change_sample_std_percent"]
                    ),
                    "criterion_status": _status(record),
                }
            )


def _build_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge - Sprint 7.3 Final Compression Ablation",
        "",
        (
            "Frozen Sprint 2 evidence only. "
            "No new compression experiments, retraining, "
            "retuning, or rank search."
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
                    "| Method | Compression | MSE mean +/- sample SD | "
                    "Delta MSE mean +/- sample SD (%) | "
                    "Seeds Passing | Status |"
                ),
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for record in payload["records"]:
            if record["domain"] != domain:
                continue

            method = _method_label(record["method"])

            compression = f"{record['compression_ratio']:.3f}x"

            mse = f"{record['mse_mean']:.6f} +/- " f"{record['mse_sample_std']:.6f}"

            if record["method"] == "fp32":
                delta_mse = "Reference"
                seeds_passing = "-"
            else:
                delta_mse = (
                    f"{record['relative_mse_change_mean_percent']:.3f} "
                    "+/- "
                    f"{record['relative_mse_change_sample_std_percent']:.3f}"
                )

                seeds_passing = f"{record['seed_pass_count']}/" f"{len(REQUIRED_SEEDS)}"

            lines.append(
                f"| {method} | {compression} | "
                f"{mse} | {delta_mse} | "
                f"{seeds_passing} | {_status(record)} |"
            )

        lines.append("")

    cross_domain = payload["cross_domain_conclusion"][
        "methods_meeting_joint_criterion_in_both_domains"
    ]

    if cross_domain:
        method_text = ", ".join(_method_label(method) for method in cross_domain)
    else:
        method_text = "None"

    lines.extend(
        [
            "## Scientific Conclusion",
            "",
            (
                "Methods meeting the frozen >=2.0x compression "
                "and <=5% relative MSE-degradation criterion in "
                f"both evaluated domains: {method_text}."
            ),
            "",
            (
                "Methods that do not satisfy the joint criterion "
                "remain retained as negative Phase 1 evidence."
            ),
            "",
            "## Claim Controls",
            "",
            "- TT/MPS superiority: blocked",
            "- Quantum-inspired compression advantage: blocked",
            "- Quantum advantage: blocked",
            "- Production-scale extrapolation: blocked",
            "- 7B VLA generalization: blocked",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = _build_payload()

    json_path = OUTPUT_DIR / "compression-ablation.json"

    csv_path = OUTPUT_DIR / "compression-ablation.csv"

    markdown_path = OUTPUT_DIR / "compression-ablation.md"

    pareto_path = OUTPUT_DIR / "compression-pareto.csv"

    _write_json(
        json_path,
        payload,
    )

    _write_ablation_csv(
        csv_path,
        payload,
    )

    _write_pareto_csv(
        pareto_path,
        payload,
    )

    markdown_path.write_text(
        _build_markdown(payload),
        encoding="utf-8",
    )

    print("=" * 64)
    print(" SPRINT 7.3 COMPRESSION ABLATION BUILD")
    print("=" * 64)
    print()

    print("Artifacts:")
    print("  compression-ablation.json     PASS")
    print("  compression-ablation.csv      PASS")
    print("  compression-ablation.md       PASS")
    print("  compression-pareto.csv        PASS")
    print()

    print("Records:")

    for record in payload["records"]:
        print(
            f"  {record['domain']:<20} / "
            f"{record['method']:<8} "
            f"{_status(record)}"
        )

    print()
    print("SPRINT 7.3 COMPRESSION ABLATION BUILD: PASS")


if __name__ == "__main__":
    main()
