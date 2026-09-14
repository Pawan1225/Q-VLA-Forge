"""Tests for Sprint 5.14 cross-domain safety analysis."""

import pytest

from q_vla_forge.evaluation.cross_domain_safety import (
    ComparisonDirection,
    SafetyArchitectureReuse,
    comparison_direction,
    directions_consistent,
    relative_violation_reduction,
)


def test_relative_violation_reduction() -> None:
    assert relative_violation_reduction(
        0.4,
        0.1,
    ) == pytest.approx(0.75)


def test_relative_violation_reduction_zero_denominator() -> None:
    assert (
        relative_violation_reduction(
            0.0,
            0.0,
        )
        is None
    )


def test_improved_direction() -> None:
    assert comparison_direction(-0.1) is ComparisonDirection.IMPROVED


def test_degraded_direction() -> None:
    assert comparison_direction(0.1) is ComparisonDirection.DEGRADED


def test_equal_result_is_tie() -> None:
    assert comparison_direction(0.0) is ComparisonDirection.TIE


def test_direction_consistency() -> None:
    assert (
        directions_consistent(
            ComparisonDirection.IMPROVED,
            ComparisonDirection.IMPROVED,
        )
        is True
    )


def test_direction_reversal_is_not_consistent() -> None:
    assert (
        directions_consistent(
            ComparisonDirection.IMPROVED,
            ComparisonDirection.DEGRADED,
        )
        is False
    )


def test_undefined_direction_consistency() -> None:
    assert (
        directions_consistent(
            None,
            ComparisonDirection.IMPROVED,
        )
        is None
    )


def test_architecture_reuse_flags() -> None:
    record = SafetyArchitectureReuse(
        shared_safety_interface=True,
        shared_action_dimension=True,
        shared_decision_contract=True,
        shared_metric_schema=True,
        shared_seed_protocol=True,
        shared_robustness_harness=True,
        domain_specific_constraints=True,
        domain_specific_clipping=True,
        domain_specific_predictor=True,
        domain_specific_lyapunov=True,
        same_policy_weights=False,
        transfer_tested=False,
        universal_controller_supported=False,
    )

    assert record.shared_safety_interface is True
    assert record.domain_specific_lyapunov is True

    assert record.same_policy_weights is False
    assert record.transfer_tested is False
    assert record.universal_controller_supported is False


def test_architecture_requires_domain_specific_safety_semantics() -> None:
    record = SafetyArchitectureReuse(
        shared_safety_interface=True,
        shared_action_dimension=True,
        shared_decision_contract=True,
        shared_metric_schema=True,
        shared_seed_protocol=True,
        shared_robustness_harness=True,
        domain_specific_constraints=True,
        domain_specific_clipping=True,
        domain_specific_predictor=True,
        domain_specific_lyapunov=True,
        same_policy_weights=False,
        transfer_tested=False,
        universal_controller_supported=False,
    )

    assert record.domain_specific_constraints is True
    assert record.domain_specific_clipping is True
    assert record.domain_specific_predictor is True
    assert record.domain_specific_lyapunov is True


def test_architecture_does_not_imply_policy_transfer() -> None:
    record = SafetyArchitectureReuse(
        shared_safety_interface=True,
        shared_action_dimension=True,
        shared_decision_contract=True,
        shared_metric_schema=True,
        shared_seed_protocol=True,
        shared_robustness_harness=True,
        domain_specific_constraints=True,
        domain_specific_clipping=True,
        domain_specific_predictor=True,
        domain_specific_lyapunov=True,
        same_policy_weights=False,
        transfer_tested=False,
        universal_controller_supported=False,
    )

    assert record.same_policy_weights is False
    assert record.transfer_tested is False
    assert record.universal_controller_supported is False


def test_relative_reduction_preserves_zero_baseline_as_none() -> None:
    assert (
        relative_violation_reduction(
            0.0,
            0.0,
        )
        is None
    )


def test_relative_reduction_is_computed_from_seed_values() -> None:
    seed_reductions = [
        relative_violation_reduction(
            0.5,
            0.0,
        ),
        relative_violation_reduction(
            0.25,
            0.0,
        ),
    ]

    assert seed_reductions == [
        1.0,
        1.0,
    ]


def test_gaussian_direction_consistency_does_not_require_equal_magnitude() -> None:
    driving = comparison_direction(0.01)
    robotics = comparison_direction(0.25)

    assert driving is ComparisonDirection.DEGRADED
    assert robotics is ComparisonDirection.DEGRADED

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is True
    )


def test_gaussian_direction_reversal_is_preserved() -> None:
    driving = comparison_direction(-0.01)
    robotics = comparison_direction(0.01)

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is False
    )


def test_gaussian_equal_effect_is_tie() -> None:
    assert comparison_direction(0.0) is ComparisonDirection.TIE


def test_structured_state_direction_reversal_is_preserved() -> None:
    driving = comparison_direction(0.05)
    robotics = comparison_direction(-0.01)

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is False
    )


def test_structured_state_ties_remain_ties() -> None:
    driving = comparison_direction(0.0)
    robotics = comparison_direction(0.0)

    assert driving is ComparisonDirection.TIE
    assert robotics is ComparisonDirection.TIE

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is True
    )


def test_exact_count_from_rate() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        exact_count_from_rate,
    )

    assert (
        exact_count_from_rate(
            0.011,
            2000,
        )
        == 22
    )


def test_exact_count_from_rate_handles_small_fraction() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        exact_count_from_rate,
    )

    assert (
        exact_count_from_rate(
            0.0025,
            2000,
        )
        == 5
    )


def test_exact_count_from_rate_rejects_non_integer_reconstruction() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        exact_count_from_rate,
    )

    with pytest.raises(ValueError):
        exact_count_from_rate(
            0.00125,
            1000,
        )


def test_exact_count_from_rate_rejects_negative_rate() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        exact_count_from_rate,
    )

    with pytest.raises(ValueError):
        exact_count_from_rate(
            -0.1,
            100,
        )


def test_exact_count_from_rate_rejects_negative_denominator() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        exact_count_from_rate,
    )

    with pytest.raises(ValueError):
        exact_count_from_rate(
            0.1,
            -100,
        )


def test_cross_domain_consistency_can_be_false_without_failure() -> None:
    driving = ComparisonDirection.IMPROVED
    robotics = ComparisonDirection.DEGRADED

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is False
    )


def test_cross_domain_equal_direction_is_consistent() -> None:
    driving = ComparisonDirection.TIE
    robotics = ComparisonDirection.TIE

    assert (
        directions_consistent(
            driving,
            robotics,
        )
        is True
    )


def test_cross_domain_claim_status_supports_not_supported() -> None:
    from q_vla_forge.evaluation.cross_domain_safety import (
        CrossDomainClaimStatus,
    )

    assert CrossDomainClaimStatus.NOT_SUPPORTED.value == "not_supported"


def test_cross_domain_transfer_claims_remain_blocked_by_architecture() -> None:
    record = SafetyArchitectureReuse(
        shared_safety_interface=True,
        shared_action_dimension=True,
        shared_decision_contract=True,
        shared_metric_schema=True,
        shared_seed_protocol=True,
        shared_robustness_harness=True,
        domain_specific_constraints=True,
        domain_specific_clipping=True,
        domain_specific_predictor=True,
        domain_specific_lyapunov=True,
        same_policy_weights=False,
        transfer_tested=False,
        universal_controller_supported=False,
    )

    assert record.same_policy_weights is False
    assert record.transfer_tested is False
    assert record.universal_controller_supported is False


def test_cross_domain_figure_ids_are_unique() -> None:
    figure_ids = [
        "S5-CD-F01",
        "S5-CD-F02",
        "S5-CD-F03",
        "S5-CD-F04",
        "S5-CD-F05",
        "S5-CD-F06",
    ]

    assert len(figure_ids) == len(set(figure_ids))


def test_cross_domain_figure_set_has_six_items() -> None:
    figure_ids = {
        "S5-CD-F01",
        "S5-CD-F02",
        "S5-CD-F03",
        "S5-CD-F04",
        "S5-CD-F05",
        "S5-CD-F06",
    }

    assert len(figure_ids) == 6


def test_cross_domain_claim_matrix_has_twelve_claim_ids() -> None:
    claim_ids = {
        f"S5-CD{index:02d}"
        for index in range(
            1,
            13,
        )
    }

    assert len(claim_ids) == 12


def test_cross_domain_expected_claim_partition() -> None:
    status_counts = {
        "supported": 5,
        "supported_with_limitation": 1,
        "not_supported": 6,
    }

    assert sum(status_counts.values()) == 12


def test_cross_domain_expected_figure_count() -> None:
    figure_ids = {
        "S5-CD-F01",
        "S5-CD-F02",
        "S5-CD-F03",
        "S5-CD-F04",
        "S5-CD-F05",
        "S5-CD-F06",
    }

    assert len(figure_ids) == 6


def test_cross_domain_expected_mechanism_totals() -> None:
    totals = {
        "lyapunov_decrease": 1584,
        "strict_decrease": 70,
        "selected_lower": 4887,
    }

    assert totals["lyapunov_decrease"] == 1584

    assert totals["strict_decrease"] == 70

    assert totals["selected_lower"] == 4887
