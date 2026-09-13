from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.training_evidence import (
    build_cross_domain_claim,
    build_method_claim,
    prohibited_claims,
)

ROOT = Path("results") / "training"

ANALYSIS_PATH = ROOT / "analysis" / "training-efficiency-analysis.json"

ABLATION_PATH = ROOT / "ablation" / "classical-vs-qi-ablation-summary.json"

CROSS_DOMAIN_PATH = ROOT / "cross-domain" / "cross-domain-training-comparison.json"

EVIDENCE_DIR = ROOT / "evidence"

JSON_PATH = EVIDENCE_DIR / "sprint3-training-evidence.json"

MARKDOWN_PATH = EVIDENCE_DIR / "sprint3-training-evidence.md"

CSV_PATH = EVIDENCE_DIR / "sprint3-training-evidence.csv"


def _load(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _method_claims(
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    claims = []

    for point in analysis["points"]:
        claim = build_method_claim(
            domain=point["domain"],
            method=point["method"],
            reached=int(point["target_reach_count"]),
            total=int(point["target_reach_total"]),
            step_reduction_mean=(point["step_reduction_percent_mean"]),
            step_reduction_std=(point["step_reduction_percent_std"]),
            robust_ten_percent=bool(point["robust_ten_percent_efficiency"]),
        )

        claims.append(asdict(claim))

    return claims


def _cross_domain_claims(
    cross_domain: dict[str, Any],
) -> list[dict[str, Any]]:
    claims = []

    for method in cross_domain["methods"]:
        claim = build_cross_domain_claim(
            method=method["method"],
            robust_cross_domain=bool(method["robust_cross_domain_efficiency"]),
            target_consistency=(method["target_consistency"]),
            efficiency_consistency=(method["efficiency_consistency"]),
        )

        claims.append(asdict(claim))

    return claims


def _ablation_summary(
    ablation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []

    for domain in ablation["domains"]:
        svd_reach = bool(domain["svd"]["target_reach"]["reached_target"])

        tt_reach = bool(domain["tt_mps"]["target_reach"]["reached_target"])

        if svd_reach and tt_reach:
            interpretation = (
                "Both matched representations " "reached the paired FP32 target."
            )

        elif svd_reach and not tt_reach:
            interpretation = (
                "Classical SVD reached the paired " "FP32 target while TT/MPS did not."
            )

        elif tt_reach and not svd_reach:
            interpretation = (
                "Quantum-inspired TT/MPS reached the "
                "paired FP32 target while SVD did not."
            )

        else:
            interpretation = (
                "Neither matched representation " "reached the paired FP32 target."
            )

        rows.append(
            {
                "domain": domain["domain"],
                "seed": int(domain["seed"]),
                "svd_configuration": (domain["matched_pair"]["svd"]["configuration"]),
                "tt_mps_configuration": (
                    domain["matched_pair"]["tt_mps"]["configuration"]
                ),
                "parameter_difference_percent": (
                    domain["matched_pair"]["relative_difference_percent"]
                ),
                "svd_reached_target": (svd_reach),
                "tt_mps_reached_target": (tt_reach),
                "svd_test_mse": float(domain["svd"]["test_mse"]),
                "tt_mps_test_mse": float(domain["tt_mps"]["test_mse"]),
                "interpretation": (interpretation),
            }
        )

    return rows


def _evidence_rows(
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []

    for point in analysis["points"]:
        rows.append(
            {
                "domain": (point["domain"]),
                "method": (point["method"]),
                "target_reach": (
                    f"{point['target_reach_count']}/" f"{point['target_reach_total']}"
                ),
                "parameter_reduction_percent": (point["parameter_reduction_percent"]),
                "step_reduction_mean_percent": (point["step_reduction_percent_mean"]),
                "step_reduction_std_percent": (point["step_reduction_percent_std"]),
                "test_mse_mean": (point["test_mse_mean"]),
                "test_mse_std": (point["test_mse_std"]),
                "test_mae_mean": (point["test_mae_mean"]),
                "test_mae_std": (point["test_mae_std"]),
                "robust_ten_percent_efficiency": (
                    point["robust_ten_percent_efficiency"]
                ),
            }
        )

    return rows


def _write_csv(
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("no evidence rows available")

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()

        writer.writerows(rows)


def _format_optional_percent(
    value: float | None,
    std: float | None,
) -> str:
    if value is None:
        return "N/A"

    if std is None:
        return f"{value:.2f}%"

    return f"{value:.2f} ± " f"{std:.2f}%"


def _write_markdown(
    payload: dict[str, Any],
) -> None:
    lines = [
        "# Sprint 3 — Training Efficiency Evidence",
        "",
        (
            "Q-VLA Forge evaluates classical trainable SVD "
            "and quantum-inspired TT/MPS structured "
            "parameterizations on compact autonomous-driving "
            "and robotics VLA proxy tasks."
        ),
        "",
        "## Experimental Protocol",
        "",
        "- Seeds: 42, 123, 456",
        "- Training samples: 512",
        "- Validation samples: 128",
        "- Test samples: 128",
        "- Training budget: 20 epochs",
        "- Batch size: 32",
        "- Optimizer: AdamW",
        "- Scheduler: cosine annealing",
        "- Primary metric: optimizer steps to paired FP32 target",
        (
            "- Robust efficiency criterion: all three seeds "
            "reach target and mean optimizer-step reduction "
            "is at least 10%"
        ),
        "",
        "## Three-Seed Results",
        "",
        (
            "| Domain | Method | Target Reach | "
            "Parameter Reduction | Step Reduction | "
            "Test MSE | Robust ≥10% |"
        ),
        ("|---|---|---:|---:|---:|---:|---:|"),
    ]

    for row in payload["table"]:
        step_text = _format_optional_percent(
            row["step_reduction_mean_percent"],
            row["step_reduction_std_percent"],
        )

        test_mse = f"{row['test_mse_mean']:.6f} ± " f"{row['test_mse_std']:.6f}"

        lines.append(
            f"| {row['domain']} "
            f"| {row['method']} "
            f"| {row['target_reach']} "
            f"| {row['parameter_reduction_percent']:.2f}% "
            f"| {step_text} "
            f"| {test_mse} "
            f"| {'Yes' if row['robust_ten_percent_efficiency'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Supported Findings",
            "",
        ]
    )

    for claim in payload["supported_claims"]:
        lines.append(f"- {claim['statement']}")

    lines.extend(
        [
            "",
            "## Classical vs Quantum-Inspired Matched Ablation",
            "",
        ]
    )

    for row in payload["matched_ablation"]:
        lines.append(
            f"- **{row['domain']}**: "
            f"{row['svd_configuration']} vs "
            f"{row['tt_mps_configuration']}; "
            f"parameter-budget difference "
            f"{row['parameter_difference_percent']:.2f}%. "
            f"{row['interpretation']}"
        )

    lines.extend(
        [
            "",
            "## Explicitly Unsupported Claims",
            "",
        ]
    )

    for claim in payload["unsupported_claims"]:
        lines.append(f"- {claim['statement']}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in payload["limitations"]:
        lines.append(f"- {limitation}")

    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis = _load(ANALYSIS_PATH)

    ablation = _load(ABLATION_PATH)

    cross_domain = _load(CROSS_DOMAIN_PATH)

    table = _evidence_rows(analysis)

    supported_claims = _method_claims(analysis) + _cross_domain_claims(cross_domain)

    unsupported_claims = [asdict(claim) for claim in prohibited_claims()]

    matched_ablation = _ablation_summary(ablation)

    tt_cross_domain = next(
        item for item in cross_domain["methods"] if item["method"] == "trainable_tt_mps"
    )

    payload = {
        "sprint": "3",
        "title": "Training Efficiency Evidence",
        "seeds": [
            42,
            123,
            456,
        ],
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "methods": [
            "shared_vla_fp32",
            "trainable_svd",
            "trainable_tt_mps",
        ],
        "primary_metric": ("optimizer_steps_to_paired_fp32_target"),
        "challenge_efficiency_threshold_percent": 10.0,
        "table": table,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "matched_ablation": matched_ablation,
        "cross_domain": {
            "tt_mps_robust_cross_domain_efficiency": (
                tt_cross_domain["robust_cross_domain_efficiency"]
            ),
            "tt_mps_target_consistency": (tt_cross_domain["target_consistency"]),
            "tt_mps_efficiency_consistency": (
                tt_cross_domain["efficiency_consistency"]
            ),
            "shared_architecture_claim": (
                cross_domain["architecture_interpretation"]["shared_architecture_claim"]
            ),
            "universal_trained_model_claim": (
                cross_domain["architecture_interpretation"][
                    "universal_trained_model_claim"
                ]
            ),
        },
        "quantum_inspired_method": (
            "Trainable Tensor Train / open-boundary "
            "Matrix Product State parameterization "
            "(TT/MPS), executed classically."
        ),
        "quantum_hardware_used": False,
        "figures": {
            "driving_convergence": (
                "figures/training/" "driving-training-convergence.png"
            ),
            "robotics_convergence": (
                "figures/training/" "robotics-training-convergence.png"
            ),
            "parameter_vs_efficiency": (
                "figures/training/" "parameter-vs-training-efficiency.png"
            ),
            "cross_domain_efficiency": (
                "figures/training/cross-domain/" "cross-domain-step-efficiency.png"
            ),
            "cross_domain_parameter_reduction": (
                "figures/training/cross-domain/" "cross-domain-parameter-reduction.png"
            ),
        },
        "limitations": [
            (
                "Experiments use compact synthetic "
                "autonomous-driving and robotics proxy tasks."
            ),
            (
                "The pilot does not train or evaluate a "
                "full 7B-class production VLA model."
            ),
            (
                "The lightweight language component is a "
                "trainable hashed-token embedding encoder, "
                "not a large Transformer language model."
            ),
            (
                "CPU wall-clock timing is descriptive; "
                "optimizer steps and samples to target are "
                "the primary efficiency measures."
            ),
            (
                "TT/MPS executes classically and does not "
                "constitute a quantum-hardware speedup."
            ),
            (
                "The matched SVD-vs-TT/MPS ablation uses "
                "seed 42 as a controlled isolation experiment."
            ),
            (
                "Architectural reuse across domains does "
                "not imply one universally trained model."
            ),
        ],
    }

    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_csv(table)

    _write_markdown(payload)

    print()
    print("============================================")
    print(" SPRINT 3 TRAINING EVIDENCE")
    print("============================================")

    print()
    print("Supported claims:")

    for claim in supported_claims:
        print(
            " -",
            claim["statement"],
        )

    print()
    print("Explicitly unsupported claims:")

    for claim in unsupported_claims:
        print(
            " -",
            claim["statement"],
        )

    print()
    print(
        "Evidence JSON:",
        JSON_PATH,
    )

    print(
        "Evidence Markdown:",
        MARKDOWN_PATH,
    )

    print(
        "Evidence CSV:",
        CSV_PATH,
    )


if __name__ == "__main__":
    main()
