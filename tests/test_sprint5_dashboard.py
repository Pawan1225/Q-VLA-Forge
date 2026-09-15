from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = ROOT / "dashboard" / "pages" / "5_Safety_Evidence.py"

EVIDENCE = ROOT / "results" / "safety" / "evidence" / "sprint5-safety-evidence.json"

CLAIMS = ROOT / "results" / "safety" / "evidence" / "sprint5-safety-claim-matrix.json"

FIGURE_INDEX = (
    ROOT / "results" / "safety" / "evidence" / "sprint5-safety-figure-index.json"
)


def load_json(
    path: Path,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    assert isinstance(
        payload,
        dict,
    )

    return payload


def test_dashboard_page_exists() -> None:
    assert PAGE.is_file()


def test_dashboard_page_parses() -> None:
    source = PAGE.read_text(encoding="utf-8")

    ast.parse(source)


def test_dashboard_uses_only_canonical_evidence() -> None:
    source = PAGE.read_text(encoding="utf-8")

    assert "sprint5-safety-evidence.json" in source

    assert "sprint5-safety-claim-matrix.json" in source

    assert "sprint5-safety-figure-index.json" in source

    forbidden = (
        "baseline/runs",
        "clipping/runs",
        "gaussian-robustness/runs",
        "structured-state-robustness/runs",
        "action-robustness/runs",
        "lyapunov-driving/runs",
        "lyapunov-robotics/runs",
    )

    for token in forbidden:
        assert token not in source


def test_dashboard_has_no_training_or_execution() -> None:
    source = PAGE.read_text(encoding="utf-8").lower()

    forbidden = (
        "optimizer.step(",
        ".backward(",
        "env.step(",
        "model.train(",
        "ppo.train",
        "evaluate_policy(",
    )

    for token in forbidden:
        assert token not in source


def test_dashboard_evidence_scope_is_frozen() -> None:
    payload = load_json(EVIDENCE)

    metadata = payload["metadata"]

    assert metadata["new_training"] is False

    assert metadata["new_principal_execution"] is False

    assert metadata["new_scientific_experiment"] is False

    assert metadata["scientific_scope_frozen"] is True


def test_dashboard_has_five_figures() -> None:
    payload = load_json(FIGURE_INDEX)

    figures = payload["figures"]

    assert len(figures) == 5

    for figure in figures:
        path = ROOT / figure["path"]

        assert path.is_file()

        assert path.stat().st_size > 0


def test_dashboard_claim_matrix_complete() -> None:
    payload = load_json(CLAIMS)

    proposal = payload["proposal_claims"]

    ids = {row["claim_id"] for row in proposal}

    assert ids == {
        "S5-E01",
        "S5-E02",
        "S5-E03",
        "S5-E04",
        "S5-E05",
        "S5-E06",
    }


def test_dashboard_unsupported_controls_present() -> None:
    payload = load_json(CLAIMS)

    unsupported = payload["explicitly_unsupported"]

    text = " ".join(row["statement"] for row in unsupported).lower()

    assert "lyapunov stability" in text

    assert "iso 26262" in text

    assert "production safety" in text

    assert "quantum safety advantage" in text


def test_clean_filter_dashboard_values_zero() -> None:
    payload = load_json(EVIDENCE)

    clean = payload["clean_safety"]

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "clipping",
            "lyapunov",
        ):
            value = float(clean[domain][method]["violation_step_rate"]["mean"])

            assert abs(value) <= 1e-12


def test_action_recovery_arithmetic() -> None:
    payload = load_json(EVIDENCE)

    result = payload["action_robustness"]["global"]

    assert int(result["unsafe_perturbed_steps"]) == int(
        result["recovered_unsafe_steps"]
    ) + int(result["unresolved_unsafe_steps"])
