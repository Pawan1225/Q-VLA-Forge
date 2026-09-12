from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.baseline_manifest import (
    build_baseline_manifest,
    load_json,
    save_baseline_manifest,
)


def _write_seed_file(
    results_dir: Path,
    domain: str,
    seed: int,
) -> None:
    path = results_dir / f"{domain}-baseline-seed-{seed}.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )


def _write_summary(
    results_dir: Path,
    *,
    driving_parameters: int = 1000,
    robotics_parameters: int = 1000,
    driving_size: int = 4000,
    robotics_size: int = 4000,
) -> None:
    payload = {
        "method": "shared_vla_fp32",
        "seeds": [
            42,
            123,
            456,
        ],
        "driving": {
            "parameters": driving_parameters,
            "fp32_model_size_bytes": driving_size,
        },
        "robotics": {
            "parameters": robotics_parameters,
            "fp32_model_size_bytes": robotics_size,
        },
    }

    (results_dir / "baseline-validation-summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _prepare_results(
    results_dir: Path,
) -> None:
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in (
        42,
        123,
        456,
    ):
        _write_seed_file(
            results_dir,
            "driving",
            seed,
        )

        _write_seed_file(
            results_dir,
            "robotics",
            seed,
        )

    _write_summary(results_dir)


def test_build_manifest(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    manifest = build_baseline_manifest(tmp_path)

    assert manifest.project == "Q-VLA Forge"

    assert manifest.baseline_name == "shared_vla_fp32"

    assert manifest.seeds == (
        42,
        123,
        456,
    )

    assert manifest.latent_dim == 32
    assert manifest.action_dim == 3
    assert manifest.parameters == 1000
    assert manifest.fp32_model_size_bytes == 4000


def test_manifest_records_training_setup(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    manifest = build_baseline_manifest(tmp_path)

    assert manifest.optimizer == "AdamW"

    assert manifest.scheduler == "CosineAnnealingLR"

    assert manifest.loss == "MSELoss"

    assert manifest.epochs == 20
    assert manifest.batch_size == 32


def test_manifest_records_architecture(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    manifest = build_baseline_manifest(tmp_path)

    assert manifest.vision_dim == 64
    assert manifest.language_dim == 32
    assert manifest.state_dim == 16
    assert manifest.fusion_dim == 64
    assert manifest.latent_dim == 32
    assert manifest.action_dim == 3


def test_manifest_rejects_parameter_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    _write_summary(
        tmp_path,
        driving_parameters=1000,
        robotics_parameters=999,
    )

    with pytest.raises(ValueError):
        build_baseline_manifest(tmp_path)


def test_manifest_rejects_size_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    _write_summary(
        tmp_path,
        driving_size=4000,
        robotics_size=3996,
    )

    with pytest.raises(ValueError):
        build_baseline_manifest(tmp_path)


def test_manifest_rejects_missing_evidence(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    (tmp_path / "driving-baseline-seed-456.json").unlink()

    with pytest.raises(FileNotFoundError):
        build_baseline_manifest(tmp_path)


def test_manifest_saves_json(
    tmp_path: Path,
) -> None:
    _prepare_results(tmp_path)

    manifest = build_baseline_manifest(tmp_path)

    output = tmp_path / "baseline-manifest.json"

    save_baseline_manifest(
        manifest,
        output,
    )

    payload = load_json(output)

    assert payload["baseline_name"] == "shared_vla_fp32"

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]


def test_load_json_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")
