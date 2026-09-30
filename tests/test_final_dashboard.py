from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP = ROOT / "dashboard" / "app.py"


def source() -> str:
    return APP.read_text(encoding="utf-8")


def parsed_tree() -> ast.Module:
    return ast.parse(source())


def string_content() -> str:
    """Return normalized Python string-literal content from the dashboard."""
    values = [
        node.value
        for node in ast.walk(parsed_tree())
        if isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            str,
        )
    ]

    return " ".join(" ".join(value.split()) for value in values)


def test_final_dashboard_exists() -> None:
    assert APP.is_file()


def test_final_dashboard_parses() -> None:
    parsed_tree()


def test_dashboard_uses_final_evidence_adapter() -> None:
    text = source()

    assert "load_dashboard_evidence" in text

    assert "dashboard_evidence" in text


def test_dashboard_does_not_use_raw_result_scanning() -> None:
    text = source()

    forbidden = (
        "load_result_files",
        "experiments_dataframe",
        'Path("results")',
        "glob(",
        "rglob(",
    )

    for token in forbidden:
        assert token not in text


def test_dashboard_has_six_final_sections() -> None:
    text = string_content()

    required = (
        "Overview",
        "Final Figures",
        "Final Tables",
        "Ablation",
        "Proposal Claims",
        "Provenance",
    )

    for item in required:
        assert item in text


def test_dashboard_has_four_bottlenecks() -> None:
    text = string_content()

    required = (
        "Compression",
        "Training Efficiency",
        "RL / QML",
        "Safety",
    )

    for item in required:
        assert item in text


def test_dashboard_has_domain_selector() -> None:
    text = string_content()

    assert "All Domains" in text
    assert "Autonomous Driving" in text
    assert "Robotics" in text


def test_dashboard_displays_locked_seed_protocol() -> None:
    text = string_content()
    raw = source()

    assert "Locked Seeds" in text
    assert "mean ± sample SD" in text
    assert "REQUIRED_SEEDS" in raw


def test_dashboard_displays_full_system_boundary() -> None:
    raw = source()
    text = string_content()

    assert '"DIRECT"' in raw
    assert '"COMPONENT_ONLY"' in raw
    assert '"NOT_EVALUATED"' in raw

    assert ("not combined into synthetic full-system " "performance estimates") in text


def test_dashboard_separates_qml_compactness() -> None:
    text = string_content()

    assert "PQC Actor Compactness" in text

    assert ("Actor compactness is separate from " "sample efficiency") in text


def test_dashboard_preserves_empirical_safety_boundary() -> None:
    text = string_content()

    assert "Empirical proxy result only" in text

    assert ("Zero observed violations do not establish " "formal stability") in text


def test_dashboard_has_claim_controls() -> None:
    text = string_content()

    assert "Blocked Claim Controls" in text

    assert "full-system superiority" in text

    assert "production readiness" in text


def test_dashboard_fails_loudly_for_missing_evidence() -> None:
    raw = source()
    text = string_content()

    assert "MISSING_CANONICAL_EVIDENCE" in raw

    assert "will not substitute historical artifacts" in text

    assert "generate zeros" in text


def test_dashboard_has_no_training_or_scientific_execution() -> None:
    text = source().lower()

    forbidden = (
        "optimizer.step(",
        ".backward(",
        "env.step(",
        "model.train(",
        "ppo.train(",
        "evaluate_policy(",
    )

    for token in forbidden:
        assert token not in text


def test_dashboard_has_no_future_sprint7_language() -> None:
    text = string_content().lower()

    assert "future system-level experiment records" not in text

    assert "pending sprint 7" not in text


def test_dashboard_has_no_generic_best_method_claim() -> None:
    text = string_content().lower()

    assert "best method" not in text
    assert "best algorithm" not in text


def test_dashboard_is_final_phase1_branded() -> None:
    text = string_content()

    assert "Q-VLA Forge — Phase 1 Evidence Dashboard" in text

    assert "Evidence Status: PHASE 1 FROZEN" in text
