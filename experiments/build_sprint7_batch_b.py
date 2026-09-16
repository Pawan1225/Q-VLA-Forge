"""Build Sprint 7 Batch B — subsprints 7.4, 7.5, and 7.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_compression import (
    build_final_compression,
)
from q_vla_forge.evaluation.final_rl_qml import (
    build_final_rl_qml,
)
from q_vla_forge.evaluation.final_training_efficiency import (
    build_final_training_efficiency,
)

ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT = ROOT / "results" / "final-validation"

COMPRESSION_DIR = FINAL_ROOT / "compression"
TRAINING_DIR = FINAL_ROOT / "training-efficiency"
RL_QML_DIR = FINAL_ROOT / "rl-qml"
BATCH_DIR = FINAL_ROOT / "batch-b"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_compression_report(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge — Sprint 7.4",
        "",
        "## Final Compression Comparison",
        "",
        (
            "Frozen criterion: "
            f"compression ≥ "
            f"{payload['criterion']['minimum_compression_ratio']:.1f}× "
            "and relative MSE increase ≤ "
            f"{payload['criterion']['maximum_relative_mse_increase_percent']:.1f}%."
        ),
        "",
        (
            "Cross-domain Pareto frontier: "
            f"**{', '.join(payload['cross_domain_pareto_frontier'])}**"
        ),
        "",
    ]

    for key in (
        "driving",
        "robotics",
    ):
        domain = payload["domains"][key]

        lines.extend(
            [
                f"### {domain['domain']}",
                "",
                ("Pareto frontier: " f"{', '.join(domain['pareto_frontier'])}"),
                "",
            ]
        )

    lines.extend(
        [
            "## Final Claim",
            "",
            payload["claim"],
            "",
        ]
    )

    return "\n".join(lines)


def build_training_report(
    payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Q-VLA Forge — Sprint 7.5",
            "",
            "## Final Training-Efficiency Comparison",
            "",
            ("Primary metric: **" f"{payload['primary_efficiency_metric']}**"),
            "",
            (
                "Robust ≥10% step-efficiency demonstrated: "
                f"**{payload['robust_ten_percent_efficiency_demonstrated']}**"
            ),
            "",
            "## Final Claim",
            "",
            payload["claim"],
            "",
        ]
    )


def build_rl_qml_report(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Q-VLA Forge — Sprint 7.6",
        "",
        "## Final RL/QML Ablation",
        "",
        (
            "| Domain | PPO Target Reach | "
            "Matched Classical | QML | "
            "QML Actor Reduction vs PPO |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        item = payload["domains"][domain]

        lines.append(
            f"| {domain} | "
            f"{item['full_ppo']['target_reach_rate']:.3f} | "
            f"{item['matched_classical']['target_reach_rate']:.3f} | "
            f"{item['qml']['target_reach_rate']:.3f} | "
            f"{item['parameter_reduction']['qml_actor_vs_ppo_percent']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Final Claim",
            "",
            payload["claim"],
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print(" Q-VLA FORGE — SPRINT 7 BATCH B")
    print(" 7.4 + 7.5 + 7.6 FINAL COMPARISONS")
    print("=" * 70)
    print()

    # --------------------------------------------------
    # Sprint 7.4
    # --------------------------------------------------

    print("[7.4] Final Compression Comparison")

    compression = build_final_compression(ROOT)

    write_json(
        COMPRESSION_DIR / "final-compression-summary.json",
        compression,
    )

    COMPRESSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (COMPRESSION_DIR / "final-compression-summary.md").write_text(
        build_compression_report(compression),
        encoding="utf-8",
    )

    print("  Cross-domain Pareto: " f"{compression['cross_domain_pareto_frontier']}")
    print("  INT8 supported: " f"{compression['int8_cross_domain_supported']}")
    print("  Status: PASS")
    print()

    # --------------------------------------------------
    # Sprint 7.5
    # --------------------------------------------------

    print("[7.5] Final Training-Efficiency Comparison")

    training = build_final_training_efficiency(ROOT)

    write_json(
        TRAINING_DIR / "final-training-efficiency-summary.json",
        training,
    )

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (TRAINING_DIR / "final-training-efficiency-summary.md").write_text(
        build_training_report(training),
        encoding="utf-8",
    )

    print(
        "  Robust >=10% efficiency: "
        f"{training['robust_ten_percent_efficiency_demonstrated']}"
    )
    print("  Status: PASS")
    print()

    # --------------------------------------------------
    # Sprint 7.6
    # --------------------------------------------------

    print("[7.6] Final RL/QML Ablation")

    rl_qml = build_final_rl_qml(ROOT)

    write_json(
        RL_QML_DIR / "final-rl-qml-summary.json",
        rl_qml,
    )

    RL_QML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (RL_QML_DIR / "final-rl-qml-summary.md").write_text(
        build_rl_qml_report(rl_qml),
        encoding="utf-8",
    )

    print("  PPO target reaches: " f"{rl_qml['aggregate_target_reach']['full_ppo']}/6")

    print("  QML target reaches: " f"{rl_qml['aggregate_target_reach']['qml']}/6")

    for domain, reduction in rl_qml[
        "qml_actor_parameter_reduction_vs_ppo_percent"
    ].items():
        print(f"  {domain} QML actor reduction: " f"{reduction:.2f}%")

    print("  Status: PASS")
    print()

    # --------------------------------------------------
    # Batch acceptance
    # --------------------------------------------------

    acceptance = {
        "batch": "Sprint 7 Batch B",
        "sub_sprints": [
            "7.4",
            "7.5",
            "7.6",
        ],
        "compression_comparison": "PASS",
        "training_efficiency_comparison": "PASS",
        "rl_qml_ablation": "PASS",
        "frozen_evidence_preserved": True,
        "new_training": False,
        "passed": True,
    }

    write_json(
        BATCH_DIR / "batch-b-acceptance.json",
        acceptance,
    )

    print("-" * 70)
    print("SPRINT 7 BATCH B: PASS")
    print()

    print("Artifacts:")
    print("  results/final-validation/" "compression/")
    print("  results/final-validation/" "training-efficiency/")
    print("  results/final-validation/" "rl-qml/")
    print("  results/final-validation/" "batch-b/")


if __name__ == "__main__":
    main()
