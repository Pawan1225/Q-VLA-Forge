from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

PHASE1_STATUS: Final[str] = "evaluated_component_evidence"

PHASE2_STATUS: Final[str] = "candidate_integrated_validation"

DIRECT_FULL_SYSTEM_RUNS: Final[int] = 0

COMPONENT_ONLY_CELLS: Final[int] = 16

PRINCIPAL_SEEDS: Final[tuple[int, int, int]] = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class PhaseBoundaryRow:
    area: str
    phase1_executed: str
    phase1_evidence: str
    phase1_boundary: str
    phase2_objective: str
    phase2_status: str
    evidence_artifact: str

    def to_dict(
        self,
    ) -> dict[str, str]:
        return asdict(self)


def build_phase_separation_matrix() -> tuple[
    PhaseBoundaryRow,
    ...,
]:
    return (
        PhaseBoundaryRow(
            area="shared_architecture",
            phase1_executed=(
                "Shared lightweight VLA-style framework "
                "evaluated across autonomous-driving and "
                "robotics proxy domains."
            ),
            phase1_evidence=(
                "Framework, interface, evaluation, "
                "robustness-harness, metric-schema, and "
                "protocol reuse."
            ),
            phase1_boundary=(
                "No universal trained policy, shared trained "
                "policy weights, zero-shot transfer, or universal "
                "safety controller was demonstrated."
            ),
            phase2_objective=(
                "Evaluate stronger shared representations, "
                "explicit weight-sharing controls, and matched "
                "cross-domain transfer experiments."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=(
                "results/final-validation/tables/" "cross-domain-table.csv"
            ),
        ),
        PhaseBoundaryRow(
            area="compression",
            phase1_executed=(
                "FP32, INT8, SVD, and TT/MPS compression "
                "were evaluated under the frozen criterion."
            ),
            phase1_evidence=(
                "INT8 passed the joint compression-quality "
                "criterion in both proxy domains at approximately "
                "3.85x effective storage compression."
            ),
            phase1_boundary=(
                "Evaluated SVD and TT/MPS configurations did "
                "not satisfy the same joint criterion. No TT/MPS "
                "superiority claim is supported."
            ),
            phase2_objective=(
                "Evaluate stronger low-rank and tensor-network "
                "configurations on larger VLA components."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=(
                "results/final-validation/tables/" "compression-table.csv"
            ),
        ),
        PhaseBoundaryRow(
            area="training_efficiency",
            phase1_executed=(
                "Training-efficiency experiments were evaluated "
                "under the locked three-seed target-reaching protocol."
            ),
            phase1_evidence=(
                "The evaluated methods produced measurable "
                "target-reaching and validation statistics."
            ),
            phase1_boundary=(
                "Phase 1 did not demonstrate a robust 10% or "
                "greater optimizer-step efficiency improvement "
                "across all three locked seeds."
            ),
            phase2_objective=(
                "Evaluate stronger optimization and representation "
                "strategies under matched target-reaching protocols."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=("results/final-validation/tables/" "training-table.csv"),
        ),
        PhaseBoundaryRow(
            area="rl",
            phase1_executed=(
                "Classical PPO was evaluated across both domains "
                "and all three principal seeds."
            ),
            phase1_evidence=(
                "Classical PPO reached all 6 of 6 frozen " "domain-seed targets."
            ),
            phase1_boundary=(
                "This result is limited to the frozen pilot-scale "
                "proxy environments and does not establish "
                "production policy performance."
            ),
            phase2_objective=(
                "Scale policy evaluation to stronger environments "
                "and more realistic VLA policy workloads."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=("results/final-validation/tables/" "rl-table.csv"),
        ),
        PhaseBoundaryRow(
            area="qml",
            phase1_executed=(
                "A compact 4-qubit PQC/QML actor and matched "
                "classical control were evaluated under the "
                "frozen RL protocol."
            ),
            phase1_evidence=(
                "The QML actor was substantially more "
                "parameter-compact than the full PPO actor."
            ),
            phase1_boundary=(
                "QML reached 0 of 6 frozen targets. No QML "
                "sample-efficiency, computational, quantum-speedup, "
                "or quantum-hardware advantage was demonstrated."
            ),
            phase2_objective=(
                "Screen alternative encodings, circuit structures, "
                "matched compact classical controls, and optional "
                "QPU execution only where scientifically justified."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=("results/final-validation/tables/" "rl-table.csv"),
        ),
        PhaseBoundaryRow(
            area="safety",
            phase1_executed=(
                "No-filter, clipping, and classical "
                "Lyapunov-guided filtering were evaluated under "
                "the frozen proxy safety contracts."
            ),
            phase1_evidence=(
                "Clipping and Lyapunov filtering produced zero "
                "observed clean violation-step rate in both "
                "proxy domains."
            ),
            phase1_boundary=(
                "This is empirical pilot evidence only. It does "
                "not establish formal safety, Lyapunov stability, "
                "invariance, certification, production readiness, "
                "or real-world vehicle or robot safety."
            ),
            phase2_objective=(
                "Evaluate stronger formal constraints, intervention "
                "latency, realistic failure modes, and closed-loop "
                "safety validation."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=("results/final-validation/tables/" "safety-table.csv"),
        ),
        PhaseBoundaryRow(
            area="robustness",
            phase1_executed=(
                "Gaussian observation noise, structured-state "
                "perturbation, and direct action perturbation "
                "were evaluated."
            ),
            phase1_evidence=(
                "Robustness behavior and direct-action recovery "
                "were quantified under the frozen synthetic "
                "perturbation protocol."
            ),
            phase1_boundary=(
                "Synthetic perturbation results do not establish "
                "real-world robustness or production fault coverage."
            ),
            phase2_objective=(
                "Expand to realistic sensor, actuator, environment, "
                "and distribution-shift failure models."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=(
                "results/safety/consolidated/" "sprint5-safety-evidence-package.json"
            ),
        ),
        PhaseBoundaryRow(
            area="full_system",
            phase1_executed=(
                "Compression, QML policy, and safety-filter "
                "components were independently evaluated across "
                "both proxy domains."
            ),
            phase1_evidence=(
                "The intended factorial contains 16 "
                "domain/configuration cells classified as "
                "component-only evidence."
            ),
            phase1_boundary=(
                "DIRECT = 0 and COMPONENT_ONLY = 16. No matched "
                "end-to-end Compression x QML x Safety factorial "
                "configuration was directly executed. No synthetic "
                "full-system metric, interaction effect, or "
                "full-system superiority claim is supported."
            ),
            phase2_objective=(
                "Execute all 16 matched Compression x QML x Safety "
                "domain/configuration cells and estimate main and "
                "interaction effects from direct integrated evidence."
            ),
            phase2_status=PHASE2_STATUS,
            evidence_artifact=(
                "results/final-validation/full-system-ablation/"
                "full-system-ablation.json"
            ),
        ),
    )


def phase_separation_as_dicts() -> list[dict[str, str]]:
    return [row.to_dict() for row in build_phase_separation_matrix()]


def validate_phase_separation_matrix(
    rows: tuple[
        PhaseBoundaryRow,
        ...,
    ],
) -> None:
    expected_areas = (
        "shared_architecture",
        "compression",
        "training_efficiency",
        "rl",
        "qml",
        "safety",
        "robustness",
        "full_system",
    )

    if tuple(row.area for row in rows) != expected_areas:
        raise ValueError("Phase separation areas changed.")

    if len(rows) != 8:
        raise ValueError("Phase separation matrix must contain " "exactly 8 rows.")

    for row in rows:
        if not row.phase1_executed:
            raise ValueError(f"{row.area} is missing Phase 1 execution.")

        if not row.phase1_evidence:
            raise ValueError(f"{row.area} is missing Phase 1 evidence.")

        if not row.phase1_boundary:
            raise ValueError(f"{row.area} is missing a Phase 1 boundary.")

        if not row.phase2_objective:
            raise ValueError(f"{row.area} is missing a Phase 2 objective.")

        if row.phase2_status != PHASE2_STATUS:
            raise ValueError(f"{row.area} has an invalid Phase 2 status.")

        if not row.evidence_artifact:
            raise ValueError(f"{row.area} is missing evidence provenance.")
