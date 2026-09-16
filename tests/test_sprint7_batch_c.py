import json
from pathlib import Path

from q_vla_forge.evaluation.challenge_scorecard import (
    build_challenge_scorecard,
)
from q_vla_forge.evaluation.final_cross_domain import (
    build_final_cross_domain,
)
from q_vla_forge.evaluation.final_safety_robustness import (
    build_final_safety_robustness,
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


def test_safety_summary(
    tmp_path: Path,
) -> None:
    payload = {
        "principal_seeds": [42, 123, 456],
        "clean_evidence": {
            "autonomous_driving": {
                "none": {"violation_step_rate": 0.4},
                "clipping": {"violation_step_rate": 0.0},
                "lyapunov": {"violation_step_rate": 0.0},
            },
            "robotics": {
                "none": {"violation_step_rate": 0.02},
                "clipping": {"violation_step_rate": 0.0},
                "lyapunov": {"violation_step_rate": 0.0},
            },
        },
        "robustness_corpus": {
            "gaussian": {},
            "structured_state": {},
            "action": {
                "unsafe_perturbed_steps": 100,
                "recovered_unsafe_steps": 60,
                "unresolved_unsafe_steps": 40,
                "recovery_fraction": 0.6,
            },
        },
        "lyapunov_mechanism": {},
        "global_limitations": [],
    }

    write_json(
        tmp_path / "results/safety/consolidated/"
        "sprint5-safety-evidence-package.json",
        payload,
    )

    result = build_final_safety_robustness(tmp_path)

    assert result["clean_explicit_filtering_zero_violation_rate_both_domains"]

    assert not result["formal_safety_guarantee"]


def test_cross_domain_summary(
    tmp_path: Path,
) -> None:
    payload = {
        "domains": [
            "autonomous_driving",
            "robotics",
        ],
        "architecture_reuse": {
            "supported_statement": ("Shared framework supported."),
        },
        "evidence_summary": {
            "clean": {"all_consistent": True},
            "structured_state": {"all_consistent": True},
            "action_recovery": {"all_consistent": True},
            "gaussian": {
                "all_consistent": False,
                "consistent": 7,
                "comparisons": 9,
            },
            "lyapunov_activation": {
                "cross_domain_consistent": False,
            },
        },
        "domain_specific_findings": [],
        "proposal_safe_conclusions": [],
        "limitations": [],
    }

    write_json(
        tmp_path / "results/safety/cross-domain/" "sprint5-cross-domain-package.json",
        payload,
    )

    result = build_final_cross_domain(tmp_path)

    assert (
        result["classifications"]["framework_reuse"]["classification"]
        == "Cross-domain supported"
    )

    assert (
        result["classifications"]["gaussian_robustness_direction"]["classification"]
        == "Domain-dependent"
    )


def test_scorecard_uses_four_bottlenecks(
    tmp_path: Path,
) -> None:
    base = tmp_path / "results" / "final-validation"

    write_json(
        base / "compression" / "final-compression-summary.json",
        {
            "claim": "compression",
        },
    )

    write_json(
        base / "training-efficiency" / "final-training-efficiency-summary.json",
        {
            "claim": "training",
        },
    )

    write_json(
        base / "rl-qml" / "final-rl-qml-summary.json",
        {
            "claim": "rl-qml",
        },
    )

    write_json(
        base / "safety-robustness" / "final-safety-robustness-summary.json",
        {
            "claim": "safety",
        },
    )

    result = build_challenge_scorecard(tmp_path)

    assert len(result["bottlenecks"]) == 4

    assert result["all_four_bottlenecks_experimentally_addressed"]

    assert not result["all_four_bottlenecks_solved"]
