from __future__ import annotations

from pathlib import Path

from q_vla_forge.evaluation import (
    build_compression_evidence,
    evidence_to_markdown,
    save_compression_evidence,
)

COMPRESSION_DIR = Path("results") / "compression"

EVIDENCE_DIR = COMPRESSION_DIR / "evidence"

JSON_OUTPUT = EVIDENCE_DIR / "sprint2-compression-evidence.json"

MARKDOWN_OUTPUT = EVIDENCE_DIR / "sprint2-compression-evidence.md"


def main() -> None:
    evidence = build_compression_evidence(
        validation_path=(COMPRESSION_DIR / "compression-validation-summary.json"),
        driving_pareto_path=(COMPRESSION_DIR / "driving-compression-pareto.json"),
        robotics_pareto_path=(COMPRESSION_DIR / "robotics-compression-pareto.json"),
        ablation_path=(
            COMPRESSION_DIR / "ablation" / "compression-ablation-seed-42.json"
        ),
        mps_mapping_path=(COMPRESSION_DIR / "mps-mapping-verification.json"),
    )

    save_compression_evidence(
        evidence,
        JSON_OUTPUT,
    )

    MARKDOWN_OUTPUT.write_text(
        evidence_to_markdown(evidence),
        encoding="utf-8",
    )

    print()
    print("===== Q-VLA Forge " "Sprint 2 Proposal Evidence =====")

    print()
    print(
        f"{'Domain':22}"
        f"{'Method':20}"
        f"{'Compression':>20}"
        f"{'MSE Δ':>20}"
        f"{'Feasible':>12}"
        f"{'Pareto':>10}"
    )

    print("-" * 106)

    for method in evidence.methods:
        compression = (
            f"{method.compression_ratio_mean:.3f}"
            f" ± "
            f"{method.compression_ratio_std:.3f}x"
        )

        mse = f"{method.mse_change_mean:.3f}" f" ± " f"{method.mse_change_std:.3f}%"

        feasible = f"{method.pilot_feasible_runs}/3"

        print(
            f"{method.domain:22}"
            f"{method.method:20}"
            f"{compression:>20}"
            f"{mse:>20}"
            f"{feasible:>12}"
            f"{method.pareto_efficient!s:>10}"
        )

    print()
    print(
        "Driving Pareto:",
        evidence.driving_pareto_frontier,
    )

    print(
        "Robotics Pareto:",
        evidence.robotics_pareto_frontier,
    )

    print()
    print(
        "Ablation points:",
        evidence.ablation_points,
    )

    print(
        "Quantum hardware used:",
        evidence.quantum_hardware_used,
    )

    print()
    print(
        "JSON:",
        JSON_OUTPUT,
    )

    print(
        "Markdown:",
        MARKDOWN_OUTPUT,
    )


if __name__ == "__main__":
    main()
