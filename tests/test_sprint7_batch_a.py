import json
from pathlib import Path

from q_vla_forge.evaluation.final_baseline import (
    build_final_baseline,
)
from q_vla_forge.evaluation.phase1_evidence import (
    build_phase1_evidence,
)
from q_vla_forge.evaluation.three_seed_validation import (
    REQUIRED_GROUPS,
    REQUIRED_SEEDS,
    extract_observed_seeds,
    path_seed,
    validate_three_seed_evidence,
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


def test_locked_seeds() -> None:
    assert REQUIRED_SEEDS == (
        42,
        123,
        456,
    )


def test_required_groups() -> None:
    assert REQUIRED_GROUPS == (
        "baseline",
        "compression",
        "training_efficiency",
        "ppo",
        "qml",
        "safety",
        "robustness",
    )


def test_required_seeds_are_not_observed() -> None:
    payload = {
        "required_seeds": [
            42,
            123,
            456,
        ]
    }

    assert (
        extract_observed_seeds(
            payload,
            allow_aggregate_seeds=False,
        )
        == set()
    )


def test_explicit_seed_is_observed() -> None:
    payload = {
        "seed": 123,
    }

    assert extract_observed_seeds(
        payload,
        allow_aggregate_seeds=False,
    ) == {123}


def test_seed_from_filename() -> None:
    path = Path("results/example-seed-456.json")

    assert path_seed(path) == {456}


def test_generated_validation_is_excluded(
    tmp_path: Path,
) -> None:
    canonical_paths = {
        "baseline": ("results/" "driving-baseline-seed-{seed}.json"),
        "compression": ("results/compression/" "compression-seed-{seed}.json"),
        "training_efficiency": (
            "results/training/" "training-efficiency-seed-{seed}.json"
        ),
        "ppo": ("results/rl/ppo/" "autonomous_driving-seed-{seed}.json"),
        "qml": ("results/rl/qml/" "autonomous_driving-seed-{seed}.json"),
        "safety": (
            "results/safety/baseline/runs/" "autonomous_driving-seed-{seed}.json"
        ),
        "robustness": (
            "results/safety/"
            "gaussian-robustness/runs/"
            "autonomous-driving-none-"
            "seed-{seed}-sigma-0p01.json"
        ),
    }

    for group, template in canonical_paths.items():
        for seed in REQUIRED_SEEDS:
            write_json(
                tmp_path / template.format(seed=seed),
                {
                    "seed": seed,
                    "method": group,
                    "test_mse": 0.1,
                },
            )

    write_json(
        tmp_path / "results" / "final-validation" / "fake-baseline-seed-999.json",
        {
            "seed": 999,
            "method": "baseline",
            "test_mse": 0.0,
        },
    )

    validation = validate_three_seed_evidence(tmp_path)

    assert validation.passed

    for group in validation.groups:
        assert group.observed_seeds == (
            42,
            123,
            456,
        )


def test_phase1_evidence_excludes_generated(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "results" / "baseline-seed-42.json",
        {
            "experiment_id": ("baseline-42"),
            "method": ("shared_vla_fp32"),
            "domain": "robotics",
            "seed": 42,
            "test_mse": 0.01,
        },
    )

    write_json(
        tmp_path / "results" / "final-validation" / "generated.json",
        {
            "seed": 999,
        },
    )

    payload = build_phase1_evidence(tmp_path)

    assert payload["artifact_count"] == 1

    assert payload["artifacts"][0]["artifact_path"] == "results/baseline-seed-42.json"


def test_final_baseline_summary(
    tmp_path: Path,
) -> None:
    manifest = {
        "baseline_name": ("shared_vla_fp32"),
        "baseline_version": "1.0",
        "seeds": [
            42,
            123,
            456,
        ],
        "vision_dim": 64,
        "language_dim": 32,
        "state_dim": 16,
        "fusion_dim": 64,
        "latent_dim": 32,
        "action_dim": 3,
        "optimizer": "AdamW",
        "scheduler": ("CosineAnnealingLR"),
        "loss": "MSELoss",
        "epochs": 20,
        "batch_size": 32,
        "train_size": 512,
        "validation_size": 128,
        "test_size": 128,
        "parameters": 76179,
        "fp32_model_size_bytes": (304716),
        "evidence_files": [],
    }

    metric = {
        "mean": 0.1,
        "std": 0.01,
        "values": [
            0.09,
            0.10,
            0.11,
        ],
    }

    domain = {
        "domain": "robotics",
        "seeds": [
            42,
            123,
            456,
        ],
        "test_mse": metric,
        "test_mae": metric,
        "mean_latency_ms": metric,
        "p95_latency_ms": metric,
        "parameters": 76179,
        "fp32_model_size_bytes": (304716),
    }

    summary = {
        "method": ("shared_vla_fp32"),
        "seeds": [
            42,
            123,
            456,
        ],
        "driving": {
            **domain,
            "domain": ("autonomous_driving"),
        },
        "robotics": domain,
    }

    write_json(
        tmp_path / "results" / "sprint1-baseline-manifest.json",
        manifest,
    )

    write_json(
        tmp_path / "results" / "baseline-validation-summary.json",
        summary,
    )

    result = build_final_baseline(tmp_path)

    assert result["architecture"]["parameters"] == 76179

    assert result["required_seeds"] == [
        42,
        123,
        456,
    ]
