"""Independent verification for Sprint 5.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

HANDOFF_PATH = ROOT / "results" / "rl" / "final" / "sprint4-handoff.json"
CONFIG_PATH = ROOT / "configs" / "safety_evaluation.yaml"
PROTOCOL_PATH = (
    ROOT / "results" / "safety" / "protocol" / "sprint5-safety-protocol.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main() -> None:
    require(HANDOFF_PATH.exists(), "Sprint 4 handoff exists")
    require(CONFIG_PATH.exists(), "Safety evaluation config exists")
    require(PROTOCOL_PATH.exists(), "Safety protocol artifact exists")

    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    require(
        handoff["from_sprint"] == "4",
        "Handoff source is Sprint 4",
    )

    require(
        handoff["to_sprint"] == "5",
        "Handoff target is Sprint 5",
    )

    require(
        handoff["handoff_status"] == "ready",
        "Sprint 4 handoff is READY",
    )

    require(
        protocol["sprint4_provenance"]["handoff_sha256"] == sha256_file(HANDOFF_PATH),
        "Sprint 4 handoff SHA256 matches",
    )

    frozen = protocol["protocol"]

    require(
        frozen["primary_policy"] == "classical_ppo",
        "Primary safety policy is classical PPO",
    )

    require(
        frozen["methods"]
        == [
            "none",
            "clipping",
            "lyapunov",
        ],
        "Exactly three safety methods are frozen",
    )

    require(
        frozen["seeds"] == [42, 123, 456],
        "Principal seeds are frozen",
    )

    require(
        frozen["evaluation_seeds"] == list(range(20_000, 20_020)),
        "Twenty held-out evaluation seeds are frozen",
    )

    require(
        frozen["primary_metric"] == "violation_step_rate",
        "Primary metric is violation_step_rate",
    )

    require(
        frozen["required_violation_reduction_fraction"] == 0.20,
        "20% violation-reduction threshold is frozen",
    )

    require(
        frozen["maximum_reward_degradation_fraction"] == 0.10,
        "10% reward degradation threshold is frozen",
    )

    require(
        frozen["maximum_success_rate_drop"] == 0.10,
        "10-point success-rate threshold is frozen",
    )

    require(
        frozen["gaussian_noise_stds"] == [0.0, 0.01, 0.05, 0.10],
        "Gaussian perturbation levels are frozen",
    )

    require(
        frozen["safety_filter_uses_true_state"] is True,
        "Safety filter true-state access is explicit",
    )

    require(
        frozen["significance_testing_enabled"] is False,
        "Inferential significance testing is disabled",
    )

    require(
        frozen["formal_certification_claimed"] is False,
        "Formal certification claim is disabled",
    )

    require(
        frozen["production_safety_guarantee_claimed"] is False,
        "Production safety guarantee is disabled",
    )

    require(
        config["metrics"]["primary"] == frozen["primary_metric"],
        "YAML and Python agree on primary metric",
    )

    require(
        config["seeds"] == frozen["seeds"],
        "YAML and Python agree on principal seeds",
    )

    require(
        config["safety_methods"] == frozen["methods"],
        "YAML and Python agree on safety methods",
    )

    results = protocol["results"]

    require(
        results["safety_results_present"] is False,
        "No premature safety results exist",
    )

    require(
        results["filter_comparison_present"] is False,
        "No premature filter comparison exists",
    )

    require(
        results["robustness_results_present"] is False,
        "No premature robustness results exist",
    )

    serialized = json.dumps(protocol).lower()

    forbidden_result_fields = (
        "clipping_violation_reduction_actual",
        "lyapunov_violation_reduction_actual",
        "best_method",
        "winning_method",
        "qml_safety_advantage",
    )

    for field in forbidden_result_fields:
        require(
            field not in serialized,
            f"Premature result field absent: {field}",
        )

    claims = config["claims"]

    for name in (
        "clipping_safety_improvement",
        "lyapunov_safety_improvement",
        "lyapunov_over_clipping",
        "perturbation_robustness",
        "cross_domain_safety_reuse",
    ):
        require(
            claims[name] == "NOT_YET_TESTED",
            f"{name} remains NOT_YET_TESTED",
        )

    print()
    print("=" * 56)
    print(" SPRINT 5.1 — SAFETY PROTOCOL & CONTRACTS")
    print("=" * 56)
    print()
    print("Protocol: PASS")
    print("Sprint 4 provenance: PASS")
    print("Safety contracts: PASS")
    print("Safety methods: PASS")
    print("Seed controls: PASS")
    print("Metrics: PASS")
    print("Pilot thresholds: PASS")
    print("Robustness taxonomy: PASS")
    print("Claim boundaries: PASS")
    print("Premature-result protection: PASS")
    print()
    print("SPRINT 5.1 FINAL GATE: PASS")


if __name__ == "__main__":
    main()
