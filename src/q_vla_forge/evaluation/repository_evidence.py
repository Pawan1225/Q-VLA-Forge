"""Sprint 7.13 — submission-facing repository evidence layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def build_readme(
    root: Path,
) -> str:
    """Build final Q-VLA Forge README from frozen evidence."""

    final_root = root / "results" / "final-validation"

    baseline = load_json(final_root / "baseline" / "final-baseline-summary.json")

    rl_qml = load_json(final_root / "rl-qml" / "final-rl-qml-summary.json")

    safety = load_json(
        final_root / "safety-robustness" / "final-safety-robustness-summary.json"
    )

    scorecard = load_json(
        final_root / "scorecard" / "challenge-bottleneck-scorecard.json"
    )

    driving_reduction = rl_qml["qml_actor_parameter_reduction_vs_ppo_percent"][
        "autonomous_driving"
    ]

    robotics_reduction = rl_qml["qml_actor_parameter_reduction_vs_ppo_percent"][
        "robotics"
    ]

    recovery = safety["action_recovery"]["recovery_fraction"]

    lines = [
        "# Q-VLA Forge",
        "",
        (
            "A reproducible Phase 1 pilot for the "
            "Volkswagen Group Global Quantum + AI Challenge "
            "2026, evaluating a shared lightweight "
            "vision-language-action-style framework across "
            "autonomous-driving and robotics proxy tasks."
        ),
        "",
        "## Phase 1 Scope",
        "",
        ("Q-VLA Forge experimentally addresses four " "challenge bottlenecks:"),
        "",
        "1. Model compression and footprint",
        "2. Training efficiency",
        "3. RL alignment and sample efficiency",
        "4. Safety and robustness",
        "",
        "The pilot uses the locked seeds **42, 123, 456**.",
        "",
        "## Shared Baseline",
        "",
        (f"- Model: `{baseline['baseline_name']}`"),
        ("- Parameters: " f"**{baseline['architecture']['parameters']:,}**"),
        "",
        "## Final Phase 1 Findings",
        "",
        (
            "- **Compression:** INT8 is the only evaluated "
            "method on the Pareto frontier in both proxy "
            "domains under the frozen criterion, at "
            "approximately **3.85× effective compression**."
        ),
        (
            "- **Training efficiency:** No robust "
            "≥10% optimizer-step efficiency improvement "
            "was demonstrated across all three locked seeds."
        ),
        ("- **RL:** Full PPO reached " "**6/6 frozen targets**."),
        (
            "- **QML compactness:** PQC/QML actor parameter "
            f"reduction was approximately **{driving_reduction:.2f}%** "
            "for autonomous driving and "
            f"**{robotics_reduction:.2f}%** for robotics."
        ),
        (
            "- **QML performance boundary:** No QML "
            "sample-efficiency or computational advantage "
            "was demonstrated."
        ),
        (
            "- **Safety:** Explicit clipping and Lyapunov "
            "filtering reduced observed clean violation-step "
            "rate to **0** in both evaluated proxy domains."
        ),
        (
            "- **Action robustness:** Explicit filtering "
            f"recovered approximately **{recovery:.1%}** of "
            "unsafe directly perturbed action steps."
        ),
        (
            "- **Cross-domain reuse:** Supported at the "
            "framework/interface/protocol level, not as one "
            "universal trained policy."
        ),
        "",
        "## Challenge Bottleneck Status",
        "",
        "| Bottleneck | Phase 1 status |",
        "|---|---|",
    ]

    for item in scorecard["bottlenecks"]:
        lines.append(f"| {item['bottleneck']} | " f"{item['phase1_status']} |")

    lines.extend(
        [
            "",
            "## Scientific Boundaries",
            "",
            (
                "Phase 1 does **not** claim quantum advantage, "
                "quantum speedup, QML superiority, TT/MPS "
                "superiority, formal Lyapunov stability, "
                "formal functional-safety certification, "
                "production readiness, physical vehicle or "
                "robot validation, or zero-shot cross-domain "
                "policy transfer."
            ),
            "",
            "## Final Evidence",
            "",
            ("Canonical Sprint 7 evidence is under " "`results/final-validation/`."),
            "",
            ("Final proposal-ready figures are under " "`figures/final/`."),
            "",
            (
                "Final proposal-ready CSV tables are under "
                "`results/final-validation/tables/`."
            ),
            "",
            "Key evidence:",
            "",
            ("- `results/final-validation/baseline/" "final-baseline-summary.json`"),
            (
                "- `results/final-validation/compression/"
                "final-compression-summary.json`"
            ),
            (
                "- `results/final-validation/training-efficiency/"
                "final-training-efficiency-summary.json`"
            ),
            ("- `results/final-validation/rl-qml/" "final-rl-qml-summary.json`"),
            (
                "- `results/final-validation/safety-robustness/"
                "final-safety-robustness-summary.json`"
            ),
            (
                "- `results/final-validation/cross-domain/"
                "final-cross-domain-summary.json`"
            ),
            ("- `results/final-validation/claims/" "final-claim-registry.json`"),
            "",
            "## Reproducibility",
            "",
            "```powershell",
            '$env:PYTHONPATH="src;."',
            "pytest tests -q",
            "ruff check src tests dashboard",
            "black --check src tests dashboard",
            "```",
            "",
            "## Phase 1 Status",
            "",
            (
                "**Frozen.** Sprint 7 performs final validation, "
                "evidence synthesis, claim control, and "
                "submission packaging only. No new principal "
                "training is introduced."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def build_evidence_index(
    root: Path,
) -> str:
    """Build repository evidence navigation document."""

    del root

    return """# Q-VLA Forge — Final Evidence Index

## Sprint 7

- 7.1 Final Three-Seed Validation
- 7.2 Cross-Sprint Evidence Aggregation
- 7.3 Final Baseline Summary
- 7.4 Final Compression Comparison
- 7.5 Final Training-Efficiency Comparison
- 7.6 Final RL/QML Ablation
- 7.7 Final Safety & Robustness Summary
- 7.8 Cross-Domain Comparison
- 7.9 Final Ablation Matrix
- 7.10 Challenge-Bottleneck Scorecard
- 7.11 Final Claim Registry Freeze
- 7.12 Final Figures & Tables
- 7.13 README / Repository Evidence Layer

## Canonical Root

`results/final-validation/`

## Figures

`figures/final/`

## Scientific Rule

Submission claims must be traceable to frozen Phase 1 artifacts and must respect the final claim registry.
"""


def build_repository_evidence_layer(
    root: Path,
) -> dict[str, Any]:
    """Write README and final evidence index."""

    readme = build_readme(root)

    evidence_index = build_evidence_index(root)

    readme_path = root / "README.md"

    index_path = root / "docs" / "final_evidence_index.md"

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    index_path.write_text(
        evidence_index,
        encoding="utf-8",
    )

    return {
        "sprint": "7.13",
        "protocol": "repository_evidence_layer",
        "status": "FROZEN",
        "readme": "README.md",
        "evidence_index": ("docs/final_evidence_index.md"),
        "new_training": False,
        "new_experiments": False,
    }
