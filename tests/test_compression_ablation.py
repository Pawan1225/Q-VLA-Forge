import math
from pathlib import Path

import pytest

from q_vla_forge.evaluation.compression_ablation import (
    DOMAINS,
    MAX_RELATIVE_MSE_DEGRADATION_PERCENT,
    METHODS,
    MIN_COMPRESSION_RATIO,
    REQUIRED_SEEDS,
    build_compressed_record,
    build_reference_record,
    classify_method,
    compression_quality_pass,
    load_canonical_compression_ablation,
    validate_required_seeds,
)

ROOT = Path(__file__).resolve().parents[1]


def test_methods_are_locked() -> None:
    assert METHODS == (
        "fp32",
        "int8",
        "svd",
        "tt_mps",
    )


def test_domains_are_locked() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_thresholds_are_locked() -> None:
    assert MIN_COMPRESSION_RATIO == 2.0

    assert MAX_RELATIVE_MSE_DEGRADATION_PERCENT == 5.0


def test_required_seeds_are_locked() -> None:
    assert REQUIRED_SEEDS == (
        42,
        123,
        456,
    )


def test_int8_negative_mse_change_passes() -> None:
    assert compression_quality_pass(
        3.846,
        -0.225,
    )


def test_quality_change_4p9_passes() -> None:
    assert compression_quality_pass(
        3.846,
        4.9,
    )


def test_quality_change_5p1_fails() -> None:
    assert not compression_quality_pass(
        3.846,
        5.1,
    )


def test_ratio_below_two_fails() -> None:
    assert not compression_quality_pass(
        1.99,
        0.0,
    )


def test_negative_quality_change_is_not_absolute() -> None:
    assert compression_quality_pass(
        2.0,
        -100.0,
    )


def test_fp32_classified_as_reference() -> None:
    assert classify_method("fp32") == "reference"


def test_compressed_method_requires_ratio() -> None:
    with pytest.raises(ValueError):
        classify_method(
            "int8",
            relative_mse_change_percent=0.0,
        )


def test_compressed_method_requires_quality_evidence() -> None:
    with pytest.raises(ValueError):
        classify_method(
            "int8",
            compression_ratio=3.846,
        )


def test_missing_evidence_cannot_become_zero() -> None:
    with pytest.raises(ValueError):
        classify_method(
            "svd",
            compression_ratio=None,
            relative_mse_change_percent=None,
        )


def test_non_finite_evidence_rejected() -> None:
    with pytest.raises(ValueError):
        compression_quality_pass(
            math.nan,
            0.0,
        )


def test_wrong_seed_contract_rejected() -> None:
    with pytest.raises(ValueError):
        validate_required_seeds(
            (
                42,
                123,
            )
        )


def test_seed_order_may_vary() -> None:
    validate_required_seeds(
        (
            456,
            42,
            123,
        )
    )


def test_reference_record_retains_provenance() -> None:
    record = build_reference_record(
        domain="autonomous_driving",
        mse_mean=0.012,
        mse_sample_std=0.001,
        source_artifacts=("results/baseline-validation-summary.json",),
    )

    assert record.classification == "reference"

    assert record.criterion_pass is None

    assert record.seed_pass_count is None

    assert record.source_artifacts == ("results/baseline-validation-summary.json",)


def test_compressed_record_derives_pass() -> None:
    record = build_compressed_record(
        domain="autonomous_driving",
        method="int8",
        compression_ratio=3.846,
        mse_mean=0.012,
        mse_sample_std=0.001,
        relative_mse_change_mean_percent=-0.225,
        relative_mse_change_sample_std_percent=0.133,
        seed_pass_count=3,
        source_artifacts=("results/compression/compression-validation-summary.json",),
    )

    assert record.criterion_pass

    assert record.classification == "pass"

    assert record.seed_pass_count == 3


def test_compressed_record_derives_fail() -> None:
    record = build_compressed_record(
        domain="autonomous_driving",
        method="svd",
        compression_ratio=1.136,
        mse_mean=0.058,
        mse_sample_std=0.009,
        relative_mse_change_mean_percent=382.624,
        relative_mse_change_sample_std_percent=15.187,
        seed_pass_count=0,
        source_artifacts=("results/compression/compression-validation-summary.json",),
    )

    assert not record.criterion_pass

    assert record.classification == "fail"


def test_source_provenance_required() -> None:
    with pytest.raises(ValueError):
        build_compressed_record(
            domain="robotics",
            method="tt_mps",
            compression_ratio=1.8,
            mse_mean=0.1,
            mse_sample_std=0.01,
            relative_mse_change_mean_percent=100.0,
            relative_mse_change_sample_std_percent=10.0,
            seed_pass_count=0,
            source_artifacts=(),
        )


def test_reference_to_dict_is_serializable_shape() -> None:
    record = build_reference_record(
        domain="robotics",
        mse_mean=0.008,
        mse_sample_std=0.002,
        source_artifacts=("results/baseline-validation-summary.json",),
    )

    payload = record.to_dict()

    assert payload["domain"] == "robotics"

    assert payload["method"] == "fp32"

    assert payload["classification"] == "reference"


def test_canonical_adapter_returns_eight_records() -> None:
    records = load_canonical_compression_ablation(ROOT)

    assert len(records) == 8


def test_canonical_adapter_has_all_combinations() -> None:
    records = load_canonical_compression_ablation(ROOT)

    observed = {
        (
            record.domain,
            record.method,
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

    assert observed == expected


def test_canonical_int8_passes_both_domains() -> None:
    records = load_canonical_compression_ablation(ROOT)

    int8_records = [record for record in records if record.method == "int8"]

    assert len(int8_records) == 2

    assert all(record.criterion_pass for record in int8_records)

    assert all(record.seed_pass_count == 3 for record in int8_records)


def test_canonical_svd_fails_both_domains() -> None:
    records = load_canonical_compression_ablation(ROOT)

    svd_records = [record for record in records if record.method == "svd"]

    assert len(svd_records) == 2

    assert all(record.criterion_pass is False for record in svd_records)


def test_canonical_tt_mps_fails_both_domains() -> None:
    records = load_canonical_compression_ablation(ROOT)

    tt_records = [record for record in records if record.method == "tt_mps"]

    assert len(tt_records) == 2

    assert all(record.criterion_pass is False for record in tt_records)


def test_canonical_fp32_is_reference_only() -> None:
    records = load_canonical_compression_ablation(ROOT)

    fp32_records = [record for record in records if record.method == "fp32"]

    assert len(fp32_records) == 2

    assert all(record.classification == "reference" for record in fp32_records)

    assert all(record.criterion_pass is None for record in fp32_records)


def test_canonical_provenance_is_frozen_sprint2() -> None:
    records = load_canonical_compression_ablation(ROOT)

    for record in records:
        assert record.source_artifacts

        assert all(
            source.startswith("results/compression/")
            for source in record.source_artifacts
        )

        assert not any(
            "final-validation" in source for source in record.source_artifacts
        )


def test_canonical_driving_int8_matches_frozen_evidence() -> None:
    records = load_canonical_compression_ablation(ROOT)

    record = next(
        record
        for record in records
        if record.domain == "autonomous_driving" and record.method == "int8"
    )

    assert record.compression_ratio == pytest.approx(3.846258709481975)

    assert record.relative_mse_change_mean_percent == pytest.approx(
        -0.22516096466215424
    )

    assert record.relative_mse_change_sample_std_percent == pytest.approx(
        0.13334689015909043
    )

    assert record.seed_pass_count == 3


def test_canonical_robotics_int8_matches_frozen_evidence() -> None:
    records = load_canonical_compression_ablation(ROOT)

    record = next(
        record
        for record in records
        if record.domain == "robotics" and record.method == "int8"
    )

    assert record.compression_ratio == pytest.approx(3.846258709481975)

    assert record.relative_mse_change_mean_percent == pytest.approx(
        -0.15251875574570348
    )

    assert record.relative_mse_change_sample_std_percent == pytest.approx(
        1.1282471790678248
    )

    assert record.seed_pass_count == 3
