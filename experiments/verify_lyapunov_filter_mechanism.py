"""Independent verifier for Sprint 5.7 Lyapunov-filter mechanism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-filter"
    / "sprint5-lyapunov-filter-mechanism.json"
)


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main() -> None:
    _require(
        ARTIFACT_PATH.exists(),
        "artifact exists",
    )

    artifact: dict[str, Any] = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    _require(
        artifact["sprint"] == "5.7",
        "artifact sprint is 5.7",
    )

    _require(
        artifact["artifact"] == "lyapunov_safety_filter_mechanism",
        "artifact identity",
    )

    scope = artifact["scope"]

    for key in (
        "policy_loading",
        "checkpoint_loading",
        "ppo_training",
        "policy_fine_tuning",
        "principal_policy_evaluation",
        "robustness_evaluation",
        "empirical_parameter_tuning",
        "formal_stability_proof",
        "formal_safety_guarantee",
    ):
        _require(
            scope[key] is False,
            f"scope excludes {key}",
        )

    provenance = artifact["provenance"]

    for name, entry in provenance.items():
        path = ROOT / entry["path"]

        _require(
            path.exists(),
            f"provenance source exists: {name}",
        )

        _require(
            _sha256(path) == entry["sha256"],
            f"provenance SHA matches: {name}",
        )

    constants = artifact["constants"]

    _require(
        constants["action_deduplication_tolerance"] == 1.0e-12,
        "candidate deduplication tolerance",
    )

    _require(
        constants["intervention_tolerance_l2"] == 1.0e-8,
        "intervention tolerance",
    )

    _require(
        constants["lyapunov_improvement_tolerance"] == 1.0e-12,
        "Lyapunov improvement tolerance",
    )

    _require(
        constants["max_driving_candidates"] == 10,
        "driving candidate ceiling",
    )

    _require(
        constants["max_robotics_candidates"] == 9,
        "robotics candidate ceiling",
    )

    expected_order = [
        "hard_guard_pass",
        "lower_predicted_next_v",
        "lower_delta_v",
        "lower_action_correction_l2",
        "lower_candidate_index",
    ]

    _require(
        artifact["selection_order"] == expected_order,
        "selection order",
    )

    checks = artifact["property_checks"]

    for name, passed in checks.items():
        _require(
            passed is True,
            f"property check: {name}",
        )

    probes = artifact["synthetic_probes"]

    _require(
        probes["driving_safe_preservation"]["intervened"] is False,
        "driving safe action preserved",
    )

    _require(
        probes["robotics_safe_preservation"]["intervened"] is False,
        "robotics safe action preserved",
    )

    _require(
        probes["driving_hard_guard"]["intervention_reason"] == "domain_constraint",
        "driving hard guard intervention",
    )

    _require(
        probes["robotics_hard_guard"]["intervention_reason"] == "domain_constraint",
        "robotics hard guard intervention",
    )

    _require(
        probes["driving_lyapunov_decrease"]["selected_delta_v"] < 0.0,
        "driving strict Lyapunov decrease",
    )

    _require(
        probes["robotics_lyapunov_decrease"]["selected_delta_v"] < 0.0,
        "robotics strict Lyapunov decrease",
    )

    claims = artifact["claim_controls"]

    _require(
        claims["filter_implementation_verified"] is True,
        "filter implementation verified",
    )

    for key in (
        "principal_violation_reduction_tested",
        "principal_reward_effect_tested",
        "principal_success_effect_tested",
        "lyapunov_vs_clipping_tested",
        "robustness_tested",
        "formal_stability_established",
        "formal_safety_guarantee_established",
    ):
        _require(
            claims[key] is False,
            f"claim remains controlled: {key}",
        )

    filter_source = (
        ROOT / "src" / "q_vla_forge" / "safety" / "lyapunov_filter.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "torch.load",
        "load_state_dict",
        "optimizer",
        ".backward(",
        "PPO",
    ):
        _require(
            forbidden not in filter_source,
            f"filter excludes {forbidden}",
        )

    print()
    print("Source provenance: PASS")
    print("Predictor integration: PASS")
    print("Candidate selection: PASS")
    print("Hard guards: PASS")
    print("Lyapunov decrease behavior: PASS")
    print("Safe-action preservation: PASS")
    print("Policy independence: PASS")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.7 LYAPUNOV FILTER MECHANISM: PASS")


if __name__ == "__main__":
    main()
