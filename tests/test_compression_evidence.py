from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.compression_evidence import (
    build_compression_evidence,
    evidence_to_markdown,
)


def _validation_method(
    configuration: str,
) -> dict:
    return {
        "method": "test",
        "configuration": configuration,
        "seeds": [
            42,
            123,
            456,
        ],
        "compression_ratio": {
            "mean": 2.5,
            "std": 0.0,
            "values": [
                2.5,
                2.5,
                2.5,
            ],
        },
        "storage_reduction_percent": {
            "mean": 60.0,
            "std": 0.0,
            "values": [
                60.0,
                60.0,
                60.0,
            ],
        },
        "parameter_reduction_percent": {
            "mean": 60.0,
            "std": 0.0,
            "values": [
                60.0,
                60.0,
                60.0,
            ],
        },
        "baseline_test_mse": {
            "mean": 1.0,
            "std": 0.1,
            "values": [
                1.0,
                1.1,
                0.9,
            ],
        },
        "compressed_test_mse": {
            "mean": 1.02,
            "std": 0.1,
            "values": [
                1.02,
                1.12,
                0.92,
            ],
        },
        "mse_change_percent": {
            "mean": 2.0,
            "std": 0.5,
            "values": [
                1.5,
                2.0,
                2.5,
            ],
        },
        "baseline_test_mae": {
            "mean": 1.0,
            "std": 0.1,
            "values": [
                1.0,
                1.1,
                0.9,
            ],
        },
        "compressed_test_mae": {
            "mean": 1.01,
            "std": 0.1,
            "values": [
                1.01,
                1.11,
                0.91,
            ],
        },
        "mae_change_percent": {
            "mean": 1.0,
            "std": 0.3,
            "values": [
                0.7,
                1.0,
                1.3,
            ],
        },
        "baseline_mean_latency_ms": {
            "mean": 1.0,
            "std": 0.1,
            "values": [
                1.0,
                1.1,
                0.9,
            ],
        },
        "compressed_mean_latency_ms": {
            "mean": 1.0,
            "std": 0.1,
            "values": [
                1.0,
                1.1,
                0.9,
            ],
        },
        "latency_change_percent": {
            "mean": 0.0,
            "std": 0.1,
            "values": [
                0.0,
                0.1,
                -0.1,
            ],
        },
        "pilot_feasible_runs": 3,
        "all_runs_pilot_feasible": True,
    }


def _write_inputs(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    validation_path = tmp_path / "validation.json"

    driving_pareto_path = tmp_path / "driving-pareto.json"

    robotics_pareto_path = tmp_path / "robotics-pareto.json"

    ablation_path = tmp_path / "ablation.json"

    mps_path = tmp_path / "mps.json"

    validation = {
        "seeds": [
            42,
            123,
            456,
        ],
        "driving": {
            "int8": _validation_method("INT8"),
            "svd": _validation_method("SVD-50%"),
            "tensor_network": (_validation_method("TT-rank-4")),
        },
        "robotics": {
            "int8": _validation_method("INT8"),
            "svd": _validation_method("SVD-50%"),
            "tensor_network": (_validation_method("TT-rank-4")),
        },
    }

    validation_path.write_text(
        json.dumps(validation),
        encoding="utf-8",
    )

    driving_pareto_path.write_text(
        json.dumps(
            {
                "pareto_frontier": [
                    "INT8",
                    "TT-rank-4",
                ]
            }
        ),
        encoding="utf-8",
    )

    robotics_pareto_path.write_text(
        json.dumps(
            {
                "pareto_frontier": [
                    "TT-rank-4",
                ]
            }
        ),
        encoding="utf-8",
    )

    ablation_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "points": [{"id": index} for index in range(20)],
            }
        ),
        encoding="utf-8",
    )

    mps_path.write_text(
        json.dumps(
            {
                "independent_compression_method": False,
                "quantum_inspired": True,
                "quantum_hardware_used": False,
            }
        ),
        encoding="utf-8",
    )

    return (
        validation_path,
        driving_pareto_path,
        robotics_pareto_path,
        ablation_path,
        mps_path,
    )


def test_build_evidence_has_six_methods(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    assert len(evidence.methods) == 6


def test_evidence_uses_three_seeds(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    assert evidence.seeds == (
        42,
        123,
        456,
    )


def test_tt_pareto_flag_detected(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    driving_tt = next(
        method
        for method in evidence.methods
        if (
            method.domain == "autonomous_driving"
            and method.method == "tensor_train_mps"
        )
    )

    assert driving_tt.pareto_efficient


def test_svd_not_pareto_when_absent(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    driving_svd = next(
        method
        for method in evidence.methods
        if (method.domain == "autonomous_driving" and method.method == "svd")
    )

    assert not (driving_svd.pareto_efficient)


def test_ablation_count_is_twenty(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    assert evidence.ablation_points == 20


def test_no_quantum_hardware_claim(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    assert not (evidence.quantum_hardware_used)


def test_markdown_contains_results_table(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    evidence = build_compression_evidence(
        validation_path=paths[0],
        driving_pareto_path=paths[1],
        robotics_pareto_path=paths[2],
        ablation_path=paths[3],
        mps_mapping_path=paths[4],
    )

    markdown = evidence_to_markdown(evidence)

    assert "Validated Results" in markdown

    assert "TT-rank-4" in markdown


def test_missing_source_rejected(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        build_compression_evidence(
            validation_path=missing,
            driving_pareto_path=paths[1],
            robotics_pareto_path=paths[2],
            ablation_path=paths[3],
            mps_mapping_path=paths[4],
        )
