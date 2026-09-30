from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

PHASE1_SEEDS: Final[tuple[int, int, int]] = (
    42,
    123,
    456,
)

DIRECT_FULL_SYSTEM_RUNS: Final[int] = 0

COMPONENT_ONLY_CELLS: Final[int] = 16


@dataclass(frozen=True)
class ProposalEvidenceRow:
    evidence_id: str
    challenge_bottleneck: str
    research_question: str
    method: str
    metric_or_criterion: str
    observed_result: str
    phase1_status: str
    proposal_safe_claim: str
    limitation: str
    evidence_artifact: str
    phase2_follow_up: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_proposal_evidence_matrix() -> tuple[
    ProposalEvidenceRow,
    ...,
]:
    return (
        ProposalEvidenceRow(
            evidence_id="S7-C01",
            challenge_bottleneck="shared_architecture",
            research_question=(
                "Can one computational framework support "
                "both autonomous-driving and robotics proxy domains?"
            ),
            method=(
                "Shared VLA-style framework with domain-specific "
                "observation interfaces"
            ),
            metric_or_criterion=(
                "Cross-domain framework, interface, evaluation, " "and protocol reuse"
            ),
            observed_result=(
                "Reusable computational framework instantiated "
                "across both proxy domains."
            ),
            phase1_status="supported",
            proposal_safe_claim=(
                "A shared computational framework was instantiated "
                "across autonomous-driving and robotics proxy domains."
            ),
            limitation=(
                "Does not demonstrate one universal trained policy, "
                "shared policy weights, or zero-shot policy transfer."
            ),
            evidence_artifact=(
                "results/safety/cross-domain/" "sprint5-cross-domain-package.json"
            ),
            phase2_follow_up=(
                "Evaluate matched integrated policies and stronger "
                "cross-domain transfer settings."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C02",
            challenge_bottleneck="model_footprint",
            research_question=(
                "Can model storage be reduced while remaining within "
                "the frozen predictive-quality criterion?"
            ),
            method="INT8, SVD, TT/MPS",
            metric_or_criterion=(
                "Compression ratio >= 2.0x and relative MSE " "degradation <= 5%."
            ),
            observed_result=(
                "INT8 achieved approximately 3.85x effective "
                "compression and passed the joint criterion in both "
                "domains; evaluated SVD and TT/MPS settings did not."
            ),
            phase1_status="demonstrated",
            proposal_safe_claim=(
                "INT8 was the only evaluated compression method on "
                "the Pareto frontier in both Phase 1 proxy domains "
                "under the frozen compression criterion."
            ),
            limitation=("No TT/MPS superiority was demonstrated."),
            evidence_artifact=(
                "results/compression/" "compression-pareto-summary.json"
            ),
            phase2_follow_up=(
                "Evaluate stronger low-rank and tensor-network "
                "settings on larger VLA components."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C03",
            challenge_bottleneck="training_efficiency",
            research_question=(
                "Can the evaluated training methods reduce "
                "optimizer steps to the frozen target by at least "
                "10% robustly across all three seeds?"
            ),
            method="Trainable SVD and TT/MPS",
            metric_or_criterion=(
                "Robust >=10% optimizer-step reduction across "
                "seeds 42, 123, and 456."
            ),
            observed_result=("The frozen efficiency criterion was not satisfied."),
            phase1_status="not_demonstrated",
            proposal_safe_claim=(
                "Phase 1 did not demonstrate a robust 10% or greater "
                "optimizer-step efficiency improvement across all "
                "three locked seeds."
            ),
            limitation=(
                "Training-efficiency improvement is a negative " "Phase 1 result."
            ),
            evidence_artifact=("results/training/" "training-validation-summary.json"),
            phase2_follow_up=(
                "Test stronger optimization and representation "
                "strategies under matched target-reaching protocols."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C04",
            challenge_bottleneck=("rl_alignment_sample_efficiency"),
            research_question=(
                "Can the classical PPO reference reliably reach the "
                "frozen domain-specific reward targets?"
            ),
            method="Classical PPO / MLP",
            metric_or_criterion=(
                "Environment steps to frozen target across " "2 domains x 3 seeds."
            ),
            observed_result=(
                "Classical PPO reached 6 of 6 frozen " "domain-seed targets."
            ),
            phase1_status="supported",
            proposal_safe_claim=(
                "Classical PPO reached all six frozen domain-seed "
                "targets in the final matched comparison."
            ),
            limitation=("Pilot-scale proxy environments only."),
            evidence_artifact=(
                "results/rl/ablation/" "sprint4-classical-vs-qml-ablation.json"
            ),
            phase2_follow_up=(
                "Scale PPO evaluation to stronger environments "
                "and more realistic VLA policy workloads."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C05",
            challenge_bottleneck=("rl_alignment_sample_efficiency"),
            research_question=(
                "Can a compact PQC/QML policy reduce actor size "
                "while preserving or improving sample efficiency?"
            ),
            method=("4-qubit RY/RZ + CNOT PQC actor with " "matched classical control"),
            metric_or_criterion=(
                "Target-reaching efficiency and actor parameter count."
            ),
            observed_result=(
                "QML reached 0/6 frozen targets; matched classical "
                "reached 1/6; the QML actor reduced parameters by "
                "approximately 95.90% in driving and 95.51% in robotics."
            ),
            phase1_status="supported_with_limitation",
            proposal_safe_claim=(
                "The evaluated PQC/QML actor was substantially more "
                "parameter-compact than the full PPO actor."
            ),
            limitation=(
                "No QML sample-efficiency, computational, "
                "quantum-speedup, or quantum-hardware advantage "
                "was demonstrated."
            ),
            evidence_artifact=(
                "results/rl/ablation/" "sprint4-classical-vs-qml-ablation.json"
            ),
            phase2_follow_up=(
                "Screen alternative encodings, circuit structures, "
                "and matched compact classical controls."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C06",
            challenge_bottleneck="safety",
            research_question=(
                "Can explicit safety filtering reduce observed "
                "constraint violations in the proxy domains?"
            ),
            method=("No filter, clipping, and classical " "Lyapunov-guided filtering"),
            metric_or_criterion=("Observed clean violation-step rate."),
            observed_result=(
                "Clipping and Lyapunov filtering produced zero "
                "observed clean violation-step rate in both domains."
            ),
            phase1_status="empirically_supported",
            proposal_safe_claim=(
                "Explicit safety filtering reduced the observed "
                "clean violation-step rate to zero in both evaluated "
                "proxy domains."
            ),
            limitation=(
                "This is empirical pilot evidence and does not establish "
                "a formal safety guarantee. Empirical pilot evidence only; "
                "no formal safety, stability, invariance, certification, "
                "or production guarantee is claimed."
            ),
            evidence_artifact=(
                "results/safety/consolidated/" "sprint5-safety-evidence-package.json"
            ),
            phase2_follow_up=(
                "Evaluate stronger formal constraints, latency, "
                "failure modes, and realistic closed-loop scenarios."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C07",
            challenge_bottleneck="robustness",
            research_question=(
                "How do the evaluated policies and filters behave "
                "under controlled observation, state, and action "
                "perturbations?"
            ),
            method=(
                "Gaussian observation noise, structured-state "
                "perturbation, and direct action perturbation"
            ),
            metric_or_criterion=("Frozen perturbation robustness protocol."),
            observed_result=(
                "All three perturbation families were evaluated "
                "under the frozen Phase 1 protocol."
            ),
            phase1_status="supported",
            proposal_safe_claim=(
                "Gaussian observation, structured-state, and direct "
                "action perturbation robustness were evaluated under "
                "the frozen Phase 1 protocol."
            ),
            limitation=(
                "Synthetic perturbations do not establish real-world " "robustness."
            ),
            evidence_artifact=(
                "results/safety/consolidated/" "sprint5-safety-evidence-package.json"
            ),
            phase2_follow_up=(
                "Expand perturbations to realistic sensor, actuator, "
                "and environment failure distributions."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C08",
            challenge_bottleneck="robustness",
            research_question=(
                "Can explicit safety filtering recover unsafe actions "
                "introduced by direct action perturbations?"
            ),
            method="Explicit safety filtering",
            metric_or_criterion=("Recovered unsafe directly perturbed steps."),
            observed_result=(
                "69,367 of 111,341 unsafe directly perturbed " "steps were recovered."
            ),
            phase1_status="supported",
            proposal_safe_claim=(
                "Explicit filtering recovered a measurable fraction "
                "of unsafe directly perturbed actions under the "
                "evaluated synthetic action-perturbation protocol."
            ),
            limitation=(
                "Recovery is specific to the frozen synthetic " "perturbation protocol."
            ),
            evidence_artifact=(
                "results/safety/consolidated/" "sprint5-safety-evidence-package.json"
            ),
            phase2_follow_up=(
                "Measure recovery, intervention cost, and failure "
                "coverage under more realistic actuator faults."
            ),
        ),
        ProposalEvidenceRow(
            evidence_id="S7-C09",
            challenge_bottleneck="cross_domain",
            research_question=(
                "Which components can be reused across driving and "
                "robotics without claiming universal policy transfer?"
            ),
            method=(
                "Shared framework, interfaces, evaluation schema, "
                "robustness harness, and protocol"
            ),
            metric_or_criterion=("Cross-domain reuse without shared trained weights."),
            observed_result=(
                "Reuse was demonstrated at framework, interface, "
                "evaluation, robustness-harness, metric-schema, and "
                "protocol levels."
            ),
            phase1_status="supported_with_limitation",
            proposal_safe_claim=(
                "Cross-domain reuse was demonstrated at the framework, "
                "interface, evaluation, and protocol levels."
            ),
            limitation=(
                "The Phase 1 evidence does not demonstrate one universal "
                "trained policy, zero-shot policy transfer, or one universal "
                "safety controller."
            ),
            evidence_artifact=(
                "results/safety/cross-domain/" "sprint5-cross-domain-package.json"
            ),
            phase2_follow_up=(
                "Evaluate matched shared-representation and transfer "
                "experiments with explicit weight-sharing controls."
            ),
        ),
    )


def proposal_evidence_as_dicts() -> list[dict[str, str]]:
    return [row.to_dict() for row in build_proposal_evidence_matrix()]


def validate_proposal_evidence_matrix(
    rows: tuple[
        ProposalEvidenceRow,
        ...,
    ],
) -> None:
    if len(rows) != 9:
        raise ValueError(
            "Proposal evidence matrix must contain " "exactly 9 frozen Phase 1 rows."
        )

    ids = [row.evidence_id for row in rows]

    expected_ids = [
        f"S7-C{index:02d}"
        for index in range(
            1,
            10,
        )
    ]

    if ids != expected_ids:
        raise ValueError("Proposal evidence IDs must match " "S7-C01 through S7-C09.")

    for row in rows:
        if not row.proposal_safe_claim:
            raise ValueError(f"{row.evidence_id} is missing " "a proposal-safe claim.")

        if not row.evidence_artifact:
            raise ValueError(f"{row.evidence_id} is missing " "evidence provenance.")

        if not row.phase2_follow_up:
            raise ValueError(f"{row.evidence_id} is missing " "a Phase 2 follow-up.")
