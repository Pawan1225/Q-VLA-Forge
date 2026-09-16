import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.final_compression import (
    build_final_compression,
)
from q_vla_forge.evaluation.final_rl_qml import (
    build_final_rl_qml,
    parameter_reduction_percent,
)
from q_vla_forge.evaluation.final_training_efficiency import (
    build_final_training_efficiency,
)


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_parameter_reduction() -> None:
    reduction = parameter_reduction_percent(
        100,
        10,
    )

    assert reduction == pytest.approx(90.0)


def test_final_compression(
    tmp_path: Path,
) -> None:
    point_reference = {
        "method": "fp32",
        "configuration": "FP32",
        "family": "reference",
        "compression_ratio_mean": 1.0,
        "compression_ratio_std": 0.0,
        "mse_change_mean": 0.0,
        "mse_change_std": 0.0,
        "mae_change_mean": 0.0,
        "mae_change_std": 0.0,
        "pilot_feasible_runs": 0,
        "all_runs_pilot_feasible": False,
        "is_reference": True,
    }

    point_int8 = {
        **point_reference,
        "method": "int8",
        "configuration": "INT8",
        "family": "classical_quantization",
        "compression_ratio_mean": 3.85,
        "pilot_feasible_runs": 3,
        "all_runs_pilot_feasible": True,
        "is_reference": False,
    }

    source = {
        "driving": {
            "analysis": {
                "domain": "autonomous_driving",
                "points": [
                    point_reference,
                    point_int8,
                ],
                "pareto_frontier": ["INT8"],
                "dominated": [],
            }
        },
        "robotics": {
            "analysis": {
                "domain": "robotics",
                "points": [
                    point_reference,
                    point_int8,
                ],
                "pareto_frontier": ["INT8"],
                "dominated": [],
            }
        },
        "pilot_feasibility": {
            "minimum_compression_ratio": 2.0,
            "maximum_mse_increase_percent": 5.0,
        },
    }

    write_json(
        tmp_path / "results" / "compression" / "compression-pareto-summary.json",
        source,
    )

    result = build_final_compression(tmp_path)

    assert result["cross_domain_pareto_frontier"] == ["INT8"]

    assert result["int8_cross_domain_supported"]


def test_final_training_efficiency(
    tmp_path: Path,
) -> None:
    method = {
        "domain": "autonomous_driving",
        "method": "trainable_svd",
        "seeds": [42, 123, 456],
        "target_reach": {
            "reached": 3,
            "total": 3,
        },
        "trainable_parameters": {
            "mean": 10.0,
        },
        "effective_parameters": {
            "mean": 10.0,
        },
        "best_validation_loss": {
            "mean": 0.1,
        },
        "test_mse": {
            "mean": 0.1,
        },
        "steps_to_target": {
            "mean": 10.0,
        },
        "samples_to_target": {
            "mean": 100.0,
        },
        "seconds_to_target": {
            "mean": 1.0,
        },
        "step_reduction_percent": {
            "mean": -20.0,
        },
        "robust_ten_percent_step_efficiency": False,
    }

    source = {
        "seeds": [42, 123, 456],
        "primary_efficiency_metric": ("optimizer_steps_to_paired_fp32_target"),
        "wall_clock_interpretation": ("descriptive"),
        "strong_efficiency_rule": {
            "all_three_seeds_must_reach_target": True,
            "mean_step_reduction_percent_at_least": 10.0,
        },
        "methods": [method],
        "limitations": [],
    }

    write_json(
        tmp_path / "results" / "training" / "training-validation-summary.json",
        source,
    )

    result = build_final_training_efficiency(tmp_path)

    assert not result["robust_ten_percent_efficiency_demonstrated"]


def test_final_rl_qml(
    tmp_path: Path,
) -> None:
    def domain_payload(
        ppo_actor: int,
        compact_actor: int,
    ) -> dict[str, object]:
        return {
            "methods": {
                "full_ppo": {
                    "target_reach_count": 3,
                    "target_reach_rate": 1.0,
                    "actor_parameters": ppo_actor,
                },
                "matched_classical": {
                    "target_reach_count": 1,
                    "target_reach_rate": (1.0 / 3.0),
                    "actor_parameters": compact_actor,
                },
                "qml": {
                    "target_reach_count": 0,
                    "target_reach_rate": 0.0,
                    "actor_parameters": compact_actor,
                },
            },
            "paired_delta_summary": {},
        }

    source = {
        "principal_seeds": [
            42,
            123,
            456,
        ],
        "parameter_matched_ablation": True,
        "domains": {
            "autonomous_driving": (
                domain_payload(
                    1318,
                    54,
                )
            ),
            "robotics": (
                domain_payload(
                    1382,
                    62,
                )
            ),
        },
        "scientific_boundary": {
            "quantum_speedup_claimed": False,
        },
    }

    write_json(
        tmp_path
        / "results"
        / "rl"
        / "ablation"
        / "sprint4-classical-vs-qml-ablation.json",
        source,
    )

    result = build_final_rl_qml(tmp_path)

    assert result["aggregate_target_reach"]["full_ppo"] == 6

    assert result["aggregate_target_reach"]["qml"] == 0

    assert result["qml_actor_parameter_reduction_vs_ppo_percent"][
        "autonomous_driving"
    ] == pytest.approx(
        95.90,
        abs=0.01,
    )

    assert not result["qml_sample_efficiency_advantage_demonstrated"]
