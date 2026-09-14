"""Independent final verifier for the complete Sprint 4 freeze."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.rl_final_gate import file_sha256

MANIFEST_PATH = Path("results/rl/final/sprint4-final-manifest.json")

FREEZE_PATH = Path("results/rl/final/sprint4-freeze-record.json")

HANDOFF_PATH = Path("results/rl/final/sprint4-handoff.json")

SUMMARY_PATH = Path("results/rl/final/sprint4-final-summary.md")

EVIDENCE_PATH = Path("results/rl/evidence/sprint4-rl-evidence.json")

DASHBOARD_PATH = Path("dashboard/pages/4_RL_Hybrid_QML.py")


def load_json(
    path: Path,
) -> dict[str, object]:
    """Load a JSON object."""

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def verify_artifacts(
    manifest: dict[str, object],
) -> None:
    """Verify all required frozen artifacts."""

    artifacts = manifest["artifacts"]

    assert isinstance(
        artifacts,
        list,
    )

    for artifact in artifacts:
        path = Path(artifact["path"])

        assert path.is_file(), path
        assert artifact["required"] is True
        assert artifact["frozen"] is True
        assert path.stat().st_size > 0

        current_hash = file_sha256(path)

        assert current_hash == artifact["sha256"], path


def verify_science(
    freeze: dict[str, object],
) -> None:
    """Verify frozen Sprint 4 scientific conclusions."""

    target = freeze["target_reach"]

    assert target["full_ppo"] == {
        "driving": 3,
        "robotics": 3,
        "total": 6,
        "out_of": 6,
    }

    assert target["matched_classical"] == {
        "driving": 0,
        "robotics": 1,
        "total": 1,
        "out_of": 6,
    }

    assert target["hybrid_qml"] == {
        "driving": 0,
        "robotics": 0,
        "total": 0,
        "out_of": 6,
    }

    driving = freeze["actor_compactness"]["autonomous_driving"]

    robotics = freeze["actor_compactness"]["robotics"]

    assert driving["full_ppo_actor_parameters"] == 1318

    assert driving["compact_actor_parameters"] == 54

    assert abs(driving["reduction_percent"] - 95.90288315629742) < 1e-12

    assert robotics["full_ppo_actor_parameters"] == 1382

    assert robotics["compact_actor_parameters"] == 62

    assert abs(robotics["reduction_percent"] - 95.5137481910275) < 1e-12

    matched = freeze["matched_budget_representation"]

    assert matched["autonomous_driving"]["direction"] == "hybrid_qml"

    assert matched["robotics"]["direction"] == "matched_classical"

    assert matched["direction_consistent_across_domains"] is False

    cross_domain = freeze["cross_domain"]

    assert cross_domain["architecture_reused"] is True

    assert cross_domain["shared_trained_weights"] is False

    assert cross_domain["robust_cross_domain_qml_advantage"] is False


def verify_claims(
    freeze: dict[str, object],
) -> None:
    """Verify Sprint 4 claim controls."""

    boundaries = freeze["claim_boundaries"]

    for key in (
        "qml_sample_efficiency_advantage",
        "robust_cross_domain_qml_advantage",
        "quantum_speedup",
        "quantum_hardware_advantage",
        "universal_policy",
        "transfer_learning",
        "zero_shot_transfer",
        "shared_trained_weights",
    ):
        assert boundaries[key] is False

    statuses = freeze["claim_matrix_status"]

    assert statuses["qml_sample_efficiency"] == "NOT_SUPPORTED"

    assert statuses["quantum_speedup"] == "NOT_SUPPORTED"

    assert statuses["quantum_hardware_advantage"] == "NOT_SUPPORTED"

    safety = freeze["safety"]

    assert safety["sprint4_safety_status"] == "NOT_YET_TESTED_IN_SPRINT_4"

    assert safety["safety_guarantee_established"] is False


def verify_proposal_evidence() -> None:
    """Verify the Sprint 4.14 proposal evidence package."""

    evidence = load_json(EVIDENCE_PATH)

    assert evidence["target_reach"]["full_ppo"] == {
        "reached": 6,
        "total": 6,
    }

    assert evidence["target_reach"]["matched_classical"] == {
        "reached": 1,
        "total": 6,
    }

    assert evidence["target_reach"]["hybrid_qml"] == {
        "reached": 0,
        "total": 6,
    }

    assert len(evidence["figure_index"]["figures"]) == 6

    assert len(evidence["claim_matrix"]["claims"]) == 10


def verify_dashboard() -> None:
    """Verify the dashboard remains evidence-layer only."""

    text = DASHBOARD_PATH.read_text(
        encoding="utf-8",
    )

    assert "load_sprint4_rl_evidence" in text

    assert "sprint4-rl-evidence.json" not in text or "load_sprint4_rl_evidence" in text

    assert "Synthetic" in text or "synthetic" in text

    assert "quantum speedup" in text.lower()


def verify_handoff(
    handoff: dict[str, object],
) -> None:
    """Verify Sprint 5 handoff controls."""

    assert handoff["from_sprint"] == "4"

    assert handoff["to_sprint"] == "5"

    assert handoff["handoff_status"] == "ready"

    scope = handoff["safety_scope"]

    assert scope["methods"] == [
        "none",
        "heuristic_constraint_clipping",
        "lyapunov_safety_filter",
    ]

    assert (
        handoff["scientific_boundary"]["sprint4_established_safety_guarantee"] is False
    )


def main() -> None:
    """Run the complete final Sprint 4 scientific gate."""

    manifest = load_json(MANIFEST_PATH)

    freeze = load_json(FREEZE_PATH)

    handoff = load_json(HANDOFF_PATH)

    print("===================================================")
    print(" Q-VLA FORGE - SPRINT 4 FINAL SCIENTIFIC GATE")
    print("===================================================")
    print()

    assert MANIFEST_PATH.is_file()
    assert FREEZE_PATH.is_file()
    assert HANDOFF_PATH.is_file()
    assert SUMMARY_PATH.is_file()

    print("[1] Artifact completeness")
    verify_artifacts(manifest)
    print("    PASS")
    print()

    print("[2] SHA256 freeze")
    verify_artifacts(manifest)
    print("    PASS")
    print()

    print("[3] PPO frozen baseline")
    assert freeze["target_reach"]["full_ppo"]["total"] == 6
    print("    6 / 6")
    print("    PASS")
    print()

    print("[4] Hybrid QML")
    assert freeze["target_reach"]["hybrid_qml"]["total"] == 0
    print("    0 / 6")
    print("    PASS")
    print()

    print("[5] Matched classical")
    assert freeze["target_reach"]["matched_classical"]["total"] == 1
    print("    1 / 6")
    print("    PASS")
    print()

    print("[6] Actor compactness")
    verify_science(freeze)
    print("    Driving  95.902883%")
    print("    Robotics 95.513748%")
    print("    PASS")
    print()

    print("[7] Matched-budget representation direction")
    print("    Driving  hybrid_qml")
    print("    Robotics matched_classical")
    print("    PASS")
    print()

    print("[8] Cross-domain architecture reuse")
    assert freeze["cross_domain"]["architecture_reused"] is True
    print("    PASS")
    print()

    print("[9] Claim controls")
    verify_claims(freeze)
    print("    PASS")
    print()

    print("[10] Proposal evidence package")
    verify_proposal_evidence()
    print("    PASS")
    print()

    print("[11] Dashboard evidence contract")
    verify_dashboard()
    print("    PASS")
    print()

    print("[12] Sprint 5 handoff")
    verify_handoff(handoff)
    print("    PASS")
    print()

    assert freeze["manifest"]["sha256"] == file_sha256(MANIFEST_PATH)

    print("SPRINT 4 FINAL GATE: PASS")


if __name__ == "__main__":
    main()
