from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.data_loader import (
    load_sprint4_rl_evidence,
)

EVIDENCE_PATH = Path("results/rl/evidence/" "sprint4-rl-evidence.json")

PAGE_PATH = Path("dashboard/pages/" "4_RL_Hybrid_QML.py")


def test_sprint4_dashboard_evidence_contract() -> None:
    data = load_sprint4_rl_evidence(EVIDENCE_PATH)

    assert data["sprint"] == "4.14"

    assert data["target_reach"]["full_ppo"] == {
        "reached": 6,
        "total": 6,
    }

    assert data["target_reach"]["matched_classical"] == {
        "reached": 1,
        "total": 6,
    }

    assert data["target_reach"]["hybrid_qml"] == {
        "reached": 0,
        "total": 6,
    }

    assert set(data["domains"]) == {
        "autonomous_driving",
        "robotics",
    }

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        assert set(data["domains"][domain]["methods"]) == {
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        }


def test_sprint4_dashboard_compactness() -> None:
    data = load_sprint4_rl_evidence(EVIDENCE_PATH)

    driving = data["domains"]["autonomous_driving"]

    robotics = data["domains"]["robotics"]

    assert driving["compactness"]["full_actor_parameters"] == 1318

    assert driving["compactness"]["compact_actor_parameters"] == 54

    assert driving["compactness"]["parameter_reduction_percent"] == pytest.approx(
        95.90288315629742
    )

    assert robotics["compactness"]["full_actor_parameters"] == 1382

    assert robotics["compactness"]["compact_actor_parameters"] == 62

    assert robotics["compactness"]["parameter_reduction_percent"] == pytest.approx(
        95.5137481910275
    )


def test_sprint4_dashboard_representation_direction() -> None:
    data = load_sprint4_rl_evidence(EVIDENCE_PATH)

    driving = data["domains"]["autonomous_driving"]

    robotics = data["domains"]["robotics"]

    assert driving["matched_budget_representation"]["direction"] == "hybrid_qml"

    assert robotics["matched_budget_representation"]["direction"] == "matched_classical"

    assert (
        data["cross_domain"]["matched_auc_direction_consistent_across_domains"] is False
    )


def test_sprint4_dashboard_claim_matrix() -> None:
    data = load_sprint4_rl_evidence(EVIDENCE_PATH)

    matrix = data["claim_matrix"]

    assert matrix["claim_count"] == 10

    claims = {claim["claim_id"]: claim["status"] for claim in matrix["claims"]}

    assert claims["classical-ppo-baseline"] == "SUPPORTED"

    assert claims["hybrid-pqc-implementation"] == "SUPPORTED"

    assert claims["hybrid-actor-compactness"] == "SUPPORTED"

    assert claims["qml-sample-efficiency"] == "NOT_SUPPORTED"

    assert claims["quantum-speedup"] == "NOT_SUPPORTED"

    assert claims["quantum-hardware-advantage"] == "NOT_SUPPORTED"


def test_sprint4_dashboard_figure_index() -> None:
    data = load_sprint4_rl_evidence(EVIDENCE_PATH)

    figure_index = data["figure_index"]

    assert figure_index["figure_count"] == 6

    for record in figure_index["figures"]:
        path = Path(record["path"])

        assert path.exists()
        assert path.is_file()
        assert path.stat().st_size > 0


def test_sprint4_dashboard_selected_figures() -> None:
    selected = [
        Path("figures/rl/" "driving-matched-ablation.png"),
        Path("figures/rl/" "robotics-matched-ablation.png"),
        Path("figures/rl/" "cross-domain-matched-delta.png"),
        Path("figures/rl/" "cross-domain-parameter-compactness.png"),
    ]

    assert len(selected) == 4

    for path in selected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_sprint4_dashboard_uses_packaged_evidence_only() -> None:
    source = PAGE_PATH.read_text(encoding="utf-8")

    assert "load_sprint4_rl_evidence" in source

    assert "sprint4-rl-evidence.json" in source

    historical_sources = [
        "sprint4-three-seed-validation.json",
        "sprint4-sample-efficiency.json",
        "sprint4-classical-vs-qml-ablation.json",
        "sprint4-cross-domain.json",
        "sprint4-ppo-summary.json",
        "sprint4-ppo-targets.json",
    ]

    for historical_source in historical_sources:
        assert historical_source not in source


def test_sprint4_dashboard_boundary_content() -> None:
    source = PAGE_PATH.read_text(encoding="utf-8")

    required = [
        "Synthetic proxy environments",
        "3 principal seeds",
        "20,000 interaction steps",
        "4-qubit simulated PQC",
        "default.qubit",
        "No quantum hardware",
        "No quantum speedup claim",
        "No quantum-hardware advantage claim",
        "No formal statistical significance claim",
        "No transfer learning or shared trained weights",
        "No Sprint 4 safety claim",
    ]

    for text in required:
        assert text in source
