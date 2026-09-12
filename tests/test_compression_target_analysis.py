"""Tests for compression target analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch import nn

from q_vla_forge.compression import (
    analyze_compression_targets,
    save_compression_target_report,
)
from q_vla_forge.models import SharedVLAModel


def test_analysis_matches_frozen_parameter_count() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.total_parameters == 76179
    assert report.trainable_parameters == 76179


def test_candidate_parameter_partition() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.candidate_weight_parameters == 75184
    assert report.uncategorized_parameters == 995

    assert (
        report.candidate_weight_parameters + report.uncategorized_parameters
        == report.total_parameters
    )


def test_analysis_finds_exact_layer_count() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert len(report.layers) == 15


def test_int8_targets_all_candidate_weights() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.int8_target_parameters == report.candidate_weight_parameters

    assert all(layer.int8_eligible for layer in report.layers)


def test_int8_coverage_matches_model() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.int8_coverage_percent == pytest.approx(98.69386602291997)


def test_svd_target_parameter_count() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.svd_target_parameters == 35328


def test_svd_coverage_matches_model() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.svd_coverage_percent == pytest.approx(46.37498523215059)


def test_tt_target_parameter_count() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.tensor_train_target_parameters == 34816


def test_tt_coverage_matches_model() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.tensor_train_coverage_percent == pytest.approx(45.70288399690203)


def test_mps_matches_tt_coverage() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    assert report.mps_target_parameters == report.tensor_train_target_parameters

    assert report.mps_coverage_percent == report.tensor_train_coverage_percent


def test_exact_tt_target_set() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    actual = {layer.name for layer in report.layers if layer.tensor_train_eligible}

    expected = {
        "fusion.network.0",
        "fusion.network.2",
        "vision_encoder.projection.1",
        "latent.network.0",
        "latent.network.2",
        "language_encoder.projection.0",
        "action_head.network.0",
    }

    assert actual == expected


def test_tt_targets_are_linear() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    targets = [layer for layer in report.layers if layer.tensor_train_eligible]

    assert all(layer.module_type == "Linear" for layer in targets)


def test_tt_targets_meet_minimum_size() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    for layer in report.layers:
        if not layer.tensor_train_eligible:
            continue

        assert layer.weight_parameters >= 1024
        assert len(layer.weight_shape) == 2
        assert min(layer.weight_shape) >= 16


def test_small_dense_layers_are_excluded_from_tt() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    excluded = {
        layer.name for layer in report.layers if not layer.tensor_train_eligible
    }

    assert "state_encoder.shared_encoder.0" in excluded
    assert "state_encoder.driving_adapter.0" in excluded
    assert "state_encoder.robotics_adapter.0" in excluded
    assert "action_head.network.2" in excluded


def test_expected_high_priority_targets() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    actual = {layer.name for layer in report.layers if layer.priority == "high"}

    expected = {
        "vision_encoder.features.4",
        "language_encoder.embedding",
        "fusion.network.0",
        "fusion.network.2",
        "vision_encoder.features.2",
        "vision_encoder.projection.1",
        "latent.network.0",
    }

    assert actual == expected


def test_largest_layer_is_final_vision_convolution() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    largest = report.layers[0]

    assert largest.name == "vision_encoder.features.4"
    assert largest.weight_parameters == 18432
    assert largest.module_type == "Conv2d"


def test_fusion_zero_is_largest_tt_target() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    tt_targets = [layer for layer in report.layers if layer.tensor_train_eligible]

    assert tt_targets[0].name == "fusion.network.0"
    assert tt_targets[0].weight_shape == (128, 112)
    assert tt_targets[0].weight_parameters == 14336


def test_layers_sorted_largest_first() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    counts = [layer.weight_parameters for layer in report.layers]

    assert counts == sorted(
        counts,
        reverse=True,
    )


def test_detects_expected_module_types() -> None:
    report = analyze_compression_targets(SharedVLAModel())

    types = {layer.module_type for layer in report.layers}

    assert "Conv2d" in types
    assert "Linear" in types
    assert "Embedding" in types


def test_save_target_report(
    tmp_path: Path,
) -> None:
    report = analyze_compression_targets(SharedVLAModel())

    output = tmp_path / "compression-target-analysis.json"

    save_compression_target_report(
        report,
        output,
    )

    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["total_parameters"] == 76179
    assert payload["candidate_weight_parameters"] == 75184
    assert len(payload["layers"]) == 15


def test_parameterless_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model must contain parameters",
    ):
        analyze_compression_targets(nn.Identity())
