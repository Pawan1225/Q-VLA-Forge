from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = ROOT / "requirements.txt"

REQUIRED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "matplotlib",
    "yaml",
    "gymnasium",
    "pennylane",
    "qiskit",
    "tensorly",
    "streamlit",
    "pytest",
)

REQUIRED_REQUIREMENTS = (
    "torch==2.14.0",
    "numpy==2.5.3",
    "pandas==3.0.5",
    "matplotlib==3.11.1",
    "PyYAML==6.0.3",
    "gymnasium==1.3.0",
    "PennyLane==0.45.1",
    "qiskit==2.5.2",
    "tensorly==0.9.0",
    "streamlit==1.63.0",
    "pytest==9.1.1",
    "ruff==0.16.7",
    "black==26.5.1",
    "mypy==2.3.1",
    "types-PyYAML==6.0.12.20260906",
)


def test_requirements_file_exists() -> None:
    assert REQUIREMENTS.is_file()


def test_required_versions_are_pinned() -> None:
    content = REQUIREMENTS.read_text(encoding="utf-8")

    for requirement in REQUIRED_REQUIREMENTS:
        assert requirement in content


def test_runtime_imports_available() -> None:
    for package in REQUIRED_PACKAGES:
        module = importlib.import_module(package)

        assert module is not None


def test_core_project_imports_available() -> None:
    modules = (
        "q_vla_forge",
        "q_vla_forge.models",
        "q_vla_forge.compression",
        "q_vla_forge.rl",
        "q_vla_forge.quantum",
        "q_vla_forge.safety",
        "q_vla_forge.evaluation",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)

        assert module is not None


def test_final_validation_artifacts_exist() -> None:
    required_paths = (
        ROOT / "results" / "final-validation" / "figures",
        ROOT / "results" / "final-validation" / "tables",
        ROOT / "results" / "final-validation" / "full-system-ablation",
    )

    for path in required_paths:
        assert path.exists()
