from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.sprint3_manifest import (
    sha256_file,
    validate_ablation_run_count,
    validate_primary_run_count,
    validate_seed_set,
    validate_unique_run_keys,
)

ROOT = Path(".")

TRAINING_ROOT = Path("results") / "training"

FP32_DIR = TRAINING_ROOT / "fp32"

VALIDATION_DIR = TRAINING_ROOT / "validation"

ABLATION_PATH = TRAINING_ROOT / "ablation" / "classical-vs-qi-ablation-summary.json"

VALIDATION_SUMMARY = TRAINING_ROOT / "training-validation-summary.json"

ANALYSIS_PATH = TRAINING_ROOT / "analysis" / "training-efficiency-analysis.json"

CROSS_DOMAIN_PATH = (
    TRAINING_ROOT / "cross-domain" / "cross-domain-training-comparison.json"
)

EVIDENCE_JSON = TRAINING_ROOT / "evidence" / "sprint3-training-evidence.json"

EVIDENCE_MD = TRAINING_ROOT / "evidence" / "sprint3-training-evidence.md"

EVIDENCE_CSV = TRAINING_ROOT / "evidence" / "sprint3-training-evidence.csv"

OUTPUT = TRAINING_ROOT / "sprint3-training-manifest.json"


EXPECTED_SEEDS = {
    42,
    123,
    456,
}

EXPECTED_DOMAINS = {
    "autonomous_driving",
    "robotics",
}

EXPECTED_STRUCTURED_METHODS = {
    "trainable_svd",
    "trainable_tt_mps",
}


def _load(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _fp32_files() -> list[Path]:
    return sorted(FP32_DIR.glob("*-fp32-seed-*.json"))


def _structured_files() -> list[Path]:
    return sorted(VALIDATION_DIR.glob("*.json"))


def _structured_run_matrix(
    files: list[Path],
) -> list[
    tuple[
        str,
        str,
        int,
    ]
]:
    keys = []

    for path in files:
        payload = _load(path)

        key = (
            str(payload["domain"]),
            str(payload["method"]),
            int(payload["seed"]),
        )

        keys.append(key)

    return keys


def _fp32_matrix(
    files: list[Path],
) -> list[
    tuple[
        str,
        str,
        int,
    ]
]:
    keys = []

    for path in files:
        payload = _load(path)

        keys.append(
            (
                str(payload["domain"]),
                "shared_vla_fp32",
                int(payload["seed"]),
            )
        )

    return keys


def _validate_fp32_matrix(
    keys: list[
        tuple[
            str,
            str,
            int,
        ]
    ],
) -> None:
    validate_unique_run_keys(keys)

    if {domain for domain, _, _ in keys} != EXPECTED_DOMAINS:
        raise RuntimeError("FP32 domain matrix incomplete")

    for domain in EXPECTED_DOMAINS:
        seeds = {seed for run_domain, _, seed in keys if run_domain == domain}

        validate_seed_set(seeds)


def _validate_structured_matrix(
    keys: list[
        tuple[
            str,
            str,
            int,
        ]
    ],
) -> None:
    validate_unique_run_keys(keys)

    if {domain for domain, _, _ in keys} != EXPECTED_DOMAINS:
        raise RuntimeError("structured domain matrix incomplete")

    if {method for _, method, _ in keys} != EXPECTED_STRUCTURED_METHODS:
        raise RuntimeError("structured method matrix incomplete")

    for domain in EXPECTED_DOMAINS:
        for method in EXPECTED_STRUCTURED_METHODS:
            seeds = {
                seed
                for run_domain, run_method, seed in keys
                if (run_domain == domain and run_method == method)
            }

            validate_seed_set(seeds)


def _validate_ablation() -> tuple[
    dict[str, Any],
    int,
]:
    payload = _load(ABLATION_PATH)

    domains = payload["domains"]

    if len(domains) != 2:
        raise RuntimeError("expected two ablation domains")

    if {item["domain"] for item in domains} != EXPECTED_DOMAINS:
        raise RuntimeError("ablation domains incomplete")

    run_count = 0

    for item in domains:
        if int(item["seed"]) != 42:
            raise RuntimeError("matched ablation must use seed 42")

        if item["selection_uses_performance_data"]:
            raise RuntimeError("ablation pair selection used " "performance data")

        control = item["scientific_control"]

        if not control["same_selected_layers"]:
            raise RuntimeError("ablation layer control invalid")

        if not control["approximately_matched_parameters"]:
            raise RuntimeError("ablation parameter control invalid")

        run_count += 2

    validate_ablation_run_count(run_count)

    return (
        payload,
        run_count,
    )


def _validate_evidence() -> dict[str, Any]:
    evidence = _load(EVIDENCE_JSON)

    validate_seed_set(set(evidence["seeds"]))

    if set(evidence["domains"]) != EXPECTED_DOMAINS:
        raise RuntimeError("evidence domain set invalid")

    if float(evidence["challenge_efficiency_threshold_percent"]) != 10.0:
        raise RuntimeError("challenge efficiency threshold changed")

    for row in evidence["table"]:
        if row["robust_ten_percent_efficiency"] and row["target_reach"] != "3/3":
            raise RuntimeError("robust claim without 3/3 target reach")

        if row["robust_ten_percent_efficiency"] and (
            row["step_reduction_mean_percent"] is None
            or float(row["step_reduction_mean_percent"]) < 10.0
        ):
            raise RuntimeError("robust claim below 10% threshold")

    unsupported = {item["claim_id"]: item for item in evidence["unsupported_claims"]}

    required_unsupported = {
        "quantum-advantage",
        "quantum-speedup",
        "native-tt-runtime-speedup",
        "production-vla-validation",
    }

    if not required_unsupported.issubset(unsupported):
        raise RuntimeError("required unsupported claims missing")

    for claim_id in required_unsupported:
        if unsupported[claim_id]["supported"]:
            raise RuntimeError(f"prohibited claim marked supported: " f"{claim_id}")

    if evidence["quantum_hardware_used"]:
        raise RuntimeError("Sprint 3 must not claim quantum hardware")

    return evidence


def _core_artifacts(
    fp32_files: list[Path],
    structured_files: list[Path],
) -> list[Path]:
    paths = (
        fp32_files
        + structured_files
        + [
            VALIDATION_SUMMARY,
            ANALYSIS_PATH,
            ABLATION_PATH,
            CROSS_DOMAIN_PATH,
            EVIDENCE_JSON,
            EVIDENCE_MD,
            EVIDENCE_CSV,
        ]
    )

    unique = []

    seen = set()

    for path in paths:
        key = path.as_posix()

        if key not in seen:
            seen.add(key)

            unique.append(path)

    return unique


def main() -> None:
    fp32_files = _fp32_files()

    structured_files = _structured_files()

    validate_primary_run_count(
        fp32_count=len(fp32_files),
        structured_count=len(structured_files),
    )

    fp32_matrix = _fp32_matrix(fp32_files)

    structured_matrix = _structured_run_matrix(structured_files)

    _validate_fp32_matrix(fp32_matrix)

    _validate_structured_matrix(structured_matrix)

    (
        ablation,
        ablation_run_count,
    ) = _validate_ablation()

    evidence = _validate_evidence()

    analysis = _load(ANALYSIS_PATH)

    cross_domain = _load(CROSS_DOMAIN_PATH)

    validation_summary = _load(VALIDATION_SUMMARY)

    artifacts = [
        asdict(sha256_file(path))
        for path in _core_artifacts(
            fp32_files,
            structured_files,
        )
    ]

    manifest = {
        "sprint": "3",
        "title": ("Training Efficiency"),
        "status": ("ready_for_final_quality_gate"),
        "protocol": {
            "domains": [
                "autonomous_driving",
                "robotics",
            ],
            "seeds": [
                42,
                123,
                456,
            ],
            "train_size": 512,
            "validation_size": 128,
            "test_size": 128,
            "epochs": 20,
            "batch_size": 32,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": ("CosineAnnealingLR"),
            "minimum_learning_rate": (1e-5),
            "loss": "MSE",
            "primary_efficiency_metric": ("optimizer_steps_to_paired_fp32_target"),
            "robust_efficiency_threshold_percent": (10.0),
        },
        "methods": {
            "reference": ("shared_vla_fp32"),
            "classical_structured": ("trainable_svd"),
            "quantum_inspired_structured": ("trainable_tt_mps"),
        },
        "primary_run_matrix": {
            "fp32_runs": len(fp32_files),
            "structured_runs": len(structured_files),
            "total_runs": (len(fp32_files) + len(structured_files)),
            "expected_total": 18,
        },
        "ablation": {
            "seed": 42,
            "runs": (ablation_run_count),
            "type": ("matched_layer_matched_parameter_budget"),
            "performance_blind_pair_selection": True,
        },
        "total_scientific_training_runs": (
            len(fp32_files) + len(structured_files) + ablation_run_count
        ),
        "evidence_chain": {
            "three_seed_validation": (VALIDATION_SUMMARY.as_posix()),
            "training_analysis": (ANALYSIS_PATH.as_posix()),
            "matched_ablation": (ABLATION_PATH.as_posix()),
            "cross_domain_comparison": (CROSS_DOMAIN_PATH.as_posix()),
            "proposal_evidence_json": (EVIDENCE_JSON.as_posix()),
            "proposal_evidence_markdown": (EVIDENCE_MD.as_posix()),
            "proposal_evidence_csv": (EVIDENCE_CSV.as_posix()),
        },
        "headline_controls": {
            "three_seed_requirement": True,
            "sample_standard_deviation": True,
            "paired_fp32_targets": True,
            "robust_claim_requires_all_seeds": True,
            "robust_claim_requires_ten_percent": True,
            "negative_results_retained": True,
        },
        "quantum_claim_control": {
            "tt_mps_is_quantum_inspired": True,
            "quantum_hardware_used": False,
            "quantum_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "native_tt_runtime_speedup_claimed": False,
        },
        "architecture_claim_control": {
            "shared_architecture": True,
            "universal_trained_model": False,
            "language_component": ("lightweight trainable text encoder"),
        },
        "derived_evidence": {
            "validation_methods": len(validation_summary["methods"]),
            "analysis_points": len(analysis["points"]),
            "cross_domain_methods": len(cross_domain["methods"]),
            "proposal_table_rows": len(evidence["table"]),
            "supported_claims": len(evidence["supported_claims"]),
            "unsupported_claims": len(evidence["unsupported_claims"]),
            "ablation_domains": len(ablation["domains"]),
        },
        "artifacts": artifacts,
    }

    OUTPUT.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print(" SPRINT 3 MANIFEST BUILT")
    print("============================================")

    print(
        "FP32 runs:",
        len(fp32_files),
    )

    print(
        "Structured runs:",
        len(structured_files),
    )

    print(
        "Principal runs:",
        (len(fp32_files) + len(structured_files)),
    )

    print(
        "Ablation runs:",
        ablation_run_count,
    )

    print(
        "Total scientific training runs:",
        manifest["total_scientific_training_runs"],
    )

    print(
        "Hashed artifacts:",
        len(artifacts),
    )

    print(
        "Manifest:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
