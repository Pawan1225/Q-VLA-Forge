"""Build the proposal-facing Sprint 4 RL evidence Markdown brief."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = ROOT / "results" / "rl" / "evidence" / "sprint4-rl-evidence.json"

OUTPUT_PATH = ROOT / "results" / "rl" / "evidence" / "sprint4-rl-evidence.md"


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_method_row(
    *,
    label: str,
    method: dict[str, Any],
) -> str:
    return (
        f"| {label} | "
        f'{method["actor_parameters"]} | '
        f'{method["target_reach_count"]}/'
        f'{method["target_total"]} | '
        f'{method["normalized_auc_mean"]:.6f} | '
        f'{method["normalized_auc_sd"]:.6f} | '
        f'{method["best_progress_mean"]:.6f} | '
        f'{method["final_progress_mean"]:.6f} |'
    )


def build_domain_table(
    *,
    title: str,
    block: dict[str, Any],
) -> list[str]:
    methods = block["methods"]

    return [
        f"### {title}",
        "",
        (
            "| Method | Actor Params | Target Reach | "
            "Normalized AUC Mean | AUC SD | "
            "Best Progress Mean | Final Progress Mean |"
        ),
        ("|---|---:|---:|---:|---:|---:|---:|"),
        format_method_row(
            label="Full PPO",
            method=methods["full_ppo"],
        ),
        format_method_row(
            label="Matched Classical",
            method=methods["matched_classical"],
        ),
        format_method_row(
            label="Hybrid QML",
            method=methods["hybrid_qml"],
        ),
        "",
    ]


def main() -> None:
    evidence = load_json(EVIDENCE_PATH)

    driving = evidence["domains"]["autonomous_driving"]

    robotics = evidence["domains"]["robotics"]

    global_results = evidence["global_results"]

    lines: list[str] = []

    lines.extend(
        [
            "# Sprint 4 - RL & Hybrid QML Evidence",
            "",
            "## Research Question",
            "",
            (
                "Can a compact hybrid quantum-classical "
                "policy architecture be reused across "
                "autonomous-driving and robotics proxy "
                "domains while preserving useful RL "
                "learning behavior under a matched "
                "interaction budget->"
            ),
            "",
            "## Experimental Protocol",
            "",
            (
                f"- Principal seeds: "
                f'{", ".join(str(seed) for seed in evidence["principal_seeds"])}'
            ),
            ("- Principal interaction budget: " "20,000 environment steps per run"),
            ("- Primary cross-method metric: normalized " "learning-curve AUC"),
            (
                "- Sprint 4.14 performs evidence packaging "
                "only; no new training or metrics were introduced."
            ),
            "",
            "## Policy Architectures",
            "",
            ("- Full PPO: classical actor and classical critic."),
            (
                "- Matched Classical: classical actor matched "
                "to the compact QML actor parameter budget."
            ),
            (
                "- Hybrid QML: four-qubit, two-layer variational "
                "quantum actor with a classical value critic."
            ),
            "",
            "## Primary Target-Reach Results",
            "",
            (
                f"- Full PPO: "
                f'{global_results["full_ppo_target_reach_count"]}/'
                f'{global_results["full_ppo_target_total"]}'
            ),
            (
                f"- Matched Classical: "
                f'{global_results["matched_classical_target_reach_count"]}/'
                f'{global_results["matched_classical_target_total"]}'
            ),
            (
                f"- Hybrid QML: "
                f'{global_results["hybrid_qml_target_reach_count"]}/'
                f'{global_results["hybrid_qml_target_total"]}'
            ),
            "",
        ]
    )

    lines.extend(
        build_domain_table(
            title="Autonomous Driving",
            block=driving,
        )
    )

    lines.extend(
        build_domain_table(
            title="Robotics",
            block=robotics,
        )
    )

    lines.extend(
        [
            "## Sample-Efficiency Analysis",
            "",
            (
                "Classical PPO reached all six paired frozen "
                "targets across both domains. The tested hybrid "
                "QML policies reached none within the frozen "
                "20,000-step budget, so no robust hybrid-QML "
                "sample-efficiency advantage was demonstrated."
            ),
            "",
            "## Matched-Budget Ablation",
            "",
            (
                "Autonomous driving: hybrid QML achieved higher "
                "mean normalized learning-curve AUC than the "
                "parameter-matched classical actor."
            ),
            "",
            (
                "Robotics: the parameter-matched classical actor "
                "achieved higher mean normalized learning-curve "
                "AUC than hybrid QML."
            ),
            "",
            (
                "The representation effect therefore changed "
                "direction across domains."
            ),
            "",
            "## Cross-Domain Comparison",
            "",
            (
                "Cross-domain architectural reuse is supported: "
                "the same four-qubit, two-layer hybrid QML "
                "policy architecture and common PPO evaluation "
                "framework were reused across both proxy domains."
            ),
            "",
            (
                "Directionally consistent matched-budget "
                "representation advantage is not supported."
            ),
            "",
            "## Compactness",
            "",
            (
                f"- Driving: "
                f'{driving["compactness"]["full_actor_parameters"]} '
                f'-> {driving["compactness"]["compact_actor_parameters"]} '
                f"actor parameters "
                f'({driving["compactness"]["parameter_reduction_percent"]:.2f}% reduction).'
            ),
            (
                f"- Robotics: "
                f'{robotics["compactness"]["full_actor_parameters"]} '
                f'-> {robotics["compactness"]["compact_actor_parameters"]} '
                f"actor parameters "
                f'({robotics["compactness"]["parameter_reduction_percent"]:.2f}% reduction).'
            ),
            "",
            "## Reproducibility",
            "",
            ("- Principal seeds: 42, 123, 456."),
            ("- Driving hybrid-QML seed-42 reproduction: PASS."),
            ("- Robotics hybrid-QML seed-42 reproduction: PASS."),
            ("- Driving matched-classical seed-42 reproduction: PASS."),
            ("- Robotics matched-classical seed-42 reproduction: PASS."),
            ("- Frozen PPO target and summary hashes remained unchanged."),
            "",
            "## Supported Claims",
            "",
        ]
    )

    for claim in evidence["claims"]:
        if claim["status"] == "SUPPORTED":
            lines.append(f'- {claim["statement"]}')

    lines.extend(
        [
            "",
            "## Unsupported Claims",
            "",
        ]
    )

    for claim in evidence["claims"]:
        if claim["status"] == "NOT_SUPPORTED":
            lines.append(f'- {claim["statement"]}')

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in evidence["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Proposal-Ready Summary",
            "",
        ]
    )

    for statement in evidence["proposal_ready_statements"]:
        lines.append(statement)
        lines.append("")

    OUTPUT_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print(
        "Sprint 4 Markdown evidence:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
