from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = ROOT / "README.md"


def readme() -> str:
    return README.read_text(encoding="utf-8")


def normalized() -> str:
    return " ".join(readme().split())


def test_readme_exists() -> None:
    assert README.is_file()


def test_required_sections_present() -> None:
    text = readme()

    required = (
        "## Challenge",
        "## Pilot Objective",
        "## Research Questions",
        "## Architecture",
        "## Algorithms",
        "## Installation",
        "## Quick Start",
        "## Experiments",
        "## Dashboard",
        "## Results",
        "## Ablation",
        "## Limitations",
        "## Reproduction",
        "## Phase 2 Roadmap",
    )

    for heading in required:
        assert heading in text


def test_phase1_identity_present() -> None:
    text = normalized()

    assert "Q-VLA Forge" in text
    assert "Phase 1" in text
    assert "Autonomous Driving" in text
    assert "Robotics" in text


def test_locked_seeds_present() -> None:
    text = readme()

    assert "42" in text
    assert "123" in text
    assert "456" in text


def test_four_bottlenecks_present() -> None:
    text = normalized().lower()

    assert "compression" in text
    assert "training efficiency" in text
    assert "rl alignment" in text
    assert "safety" in text


def test_install_command_present() -> None:
    text = readme()

    assert "pip install -r requirements.txt" in text
    assert '$env:PYTHONPATH="src;."' in text


def test_dashboard_command_present() -> None:
    text = readme()

    assert "streamlit run dashboard\\app.py" in text


def test_final_evidence_paths_present() -> None:
    text = readme()

    required = (
        "results/final-validation/statistics/",
        "results/final-validation/compression-ablation/",
        "results/final-validation/qml-ablation/",
        "results/final-validation/safety-ablation/",
        "results/final-validation/full-system-ablation/",
        "results/final-validation/figures/",
        "results/final-validation/tables/",
    )

    for path in required:
        assert path in text


def test_architecture_figure_path_present() -> None:
    text = readme()

    assert ("results/final-validation/figures/" "figure-01-architecture.png") in text


def test_int8_result_present() -> None:
    text = normalized()

    assert "INT8" in text
    assert "3.85×" in text
    assert "PASS" in text


def test_training_negative_result_present() -> None:
    text = normalized()

    assert (
        "No robust >=10% optimizer-step efficiency " "improvement was demonstrated"
    ) in text


def test_rl_target_counts_present() -> None:
    text = normalized()

    assert "Classical PPO / MLP" in text
    assert "6/6" in text
    assert "Matched classical control" in text
    assert "1/6" in text
    assert "QML / PQC" in text
    assert "0/6" in text


def test_qml_compactness_present() -> None:
    text = normalized()

    assert "95.90%" in text
    assert "95.51%" in text

    assert (
        "Actor compactness is reported separately " "from sample efficiency"
    ) in text


def test_qml_negative_result_present() -> None:
    text = normalized()

    assert "QML sample-efficiency superiority" in text


def test_safety_empirical_boundary_present() -> None:
    text = normalized()

    assert "zero observed violation-step rate" in text
    assert "This is empirical proxy evidence only" in text
    assert "formal closed-loop stability" in text


def test_full_system_counts_present() -> None:
    text = normalized()

    assert "DIRECT = 0" in text
    assert "COMPONENT_ONLY = 16" in text
    assert "NOT_EVALUATED = 0" in text


def test_no_integrated_factorial_claim() -> None:
    text = normalized()

    assert (
        "No matched integrated Compression × QML × "
        "Safety factorial configuration was directly executed"
    ) in text

    assert "No synthetic end-to-end metrics are reported" in text


def test_limitations_are_explicit() -> None:
    text = normalized()

    required = (
        "pilot-scale proxy tasks",
        "synthetic environments",
        "no full-scale 7B VLA validation",
        "no physical QPU execution",
        "no TT/MPS superiority",
        "no ISO 26262 certification",
        "no production deployment validation",
        "no zero-shot cross-domain policy transfer",
    )

    for statement in required:
        assert statement in text


def test_phase2_integrated_factorial_objective_present() -> None:
    text = normalized()

    assert (
        "16 matched Compression × QML × Safety " "domain/configuration experiments"
    ) in text


def test_no_mojibake() -> None:
    text = readme()

    forbidden = (
        "Ã—",
        "â‰¥",
        "â€”",
        "Â±",
        "ï¸",
    )

    for token in forbidden:
        assert token not in text


def test_unsupported_success_language_absent() -> None:
    text = normalized().lower()

    forbidden = (
        "quantum advantage achieved.",
        "quantum speedup achieved.",
        "qml superiority demonstrated.",
        "tt/mps superiority demonstrated.",
        "formally safe.",
        "certified safe.",
        "production ready.",
        "zero-shot transfer demonstrated.",
        "full-system superiority demonstrated.",
    )

    for phrase in forbidden:
        assert phrase not in text


def test_reproduction_quality_commands_present() -> None:
    text = readme()

    assert "pytest tests -q" in text
    assert "ruff check src tests experiments dashboard" in text
    assert "black --check src tests experiments dashboard" in text
    assert "mypy src" in text


def test_frozen_vs_rerun_distinction_present() -> None:
    text = normalized()

    assert "Frozen Evidence Verification vs Full Experiment Rerun" in text

    assert "A full experimental rerun additionally executes" in text


def test_requirements_manifest_is_complete() -> None:
    requirements = ROOT / "requirements.txt"

    assert requirements.is_file()

    content = requirements.read_text(encoding="utf-8")

    required = (
        "torch==",
        "numpy==",
        "pandas==",
        "matplotlib==",
        "PyYAML==",
        "gymnasium==",
        "PennyLane==",
        "qiskit==",
        "tensorly==",
        "streamlit==",
        "pytest==",
        "ruff==",
        "black==",
        "mypy==",
        "types-PyYAML==",
    )

    for dependency in required:
        assert dependency in content
