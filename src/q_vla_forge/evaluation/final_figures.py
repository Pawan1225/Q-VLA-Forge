"""Sprint 7.7 final-figure evidence contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FIGURE_IDS = (
    "architecture",
    "compression_pareto",
    "training_convergence",
    "rl_sample_efficiency",
    "safety_violations",
    "cross_domain",
    "ablation_summary",
)

FIGURE_FILENAMES = {
    "architecture": "figure-01-architecture.png",
    "compression_pareto": "figure-02-compression-pareto.png",
    "training_convergence": "figure-03-training-convergence.png",
    "rl_sample_efficiency": "figure-04-rl-sample-efficiency.png",
    "safety_violations": "figure-05-safety-violations.png",
    "cross_domain": "figure-06-cross-domain.png",
    "ablation_summary": "figure-07-ablation-evidence.png",
}

OUTPUT_DIR = Path("results/final-validation/figures")

FIGURE_MANIFEST_PATH = OUTPUT_DIR / "figure-manifest.json"

FIGURE_CAPTIONS_PATH = OUTPUT_DIR / "figure-captions.md"

COMPRESSION_RATIO_THRESHOLD = 2.0
COMPRESSION_MSE_DEGRADATION_THRESHOLD_PERCENT = 5.0

REQUIRED_SEEDS = (
    42,
    123,
    456,
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

GENERATED_FROM_FROZEN_EVIDENCE = True

NEW_TRAINING = False
NEW_EXPERIMENTS = False
NEW_SCIENTIFIC_RESULTS = False

ALLOW_SYNTHETIC_METRICS = False
ALLOW_SYNTHETIC_CROSS_DOMAIN_SCORE = False
ALLOW_FABRICATED_CONVERGENCE = False
ALLOW_QML_NONATTAINMENT_CENSORING = False
ALLOW_FULL_SYSTEM_PERFORMANCE_FABRICATION = False

ABLATION_DIRECT_COUNT = 0
ABLATION_COMPONENT_ONLY_COUNT = 16
ABLATION_NOT_EVALUATED_COUNT = 0

QML_TARGET_REACH_COUNTS = {
    "ppo_mlp": {
        "reached": 6,
        "total": 6,
    },
    "ppo_pqc": {
        "reached": 0,
        "total": 6,
    },
    "matched_classical": {
        "reached": 1,
        "total": 6,
    },
}

QML_ACTOR_PARAMETER_REDUCTION_PERCENT = {
    "autonomous_driving": 95.90,
    "robotics": 95.51,
}

CLAIM_CONTROLS = {
    "quantum_advantage_claim_allowed": False,
    "quantum_speedup_claim_allowed": False,
    "qml_sample_efficiency_superiority_claim_allowed": False,
    "tt_mps_superiority_claim_allowed": False,
    "training_efficiency_superiority_claim_allowed": False,
    "formal_safety_guarantee_claim_allowed": False,
    "production_safety_claim_allowed": False,
    "cross_domain_zero_shot_claim_allowed": False,
    "full_system_superiority_claim_allowed": False,
    "integrated_phase1_pipeline_claim_allowed": False,
}

FIGURE_TITLES = {
    "architecture": ("Q-VLA Forge Phase 1 Architecture and Integration Boundary"),
    "compression_pareto": ("Compression Pareto Under Frozen Phase 1 Criteria"),
    "training_convergence": ("Training Efficiency and Target Attainment"),
    "rl_sample_efficiency": ("RL/QML Target Attainment and Actor Compactness"),
    "safety_violations": ("Clean Safety Violation Rate"),
    "cross_domain": ("Cross-Domain Phase 1 Evidence Comparison"),
    "ablation_summary": ("Full-System Factorial Evidence Status"),
}

FIGURE_SOURCES = {
    "architecture": (
        "results/integration/sprint5-unified-bridge-package.json",
        ("results/safety/cross-domain/" "sprint5-cross-domain-architecture.json"),
        ("results/final-validation/claims/" "final-claim-registry.json"),
        ("results/final-validation/full-system-ablation/" "full-system-ablation.json"),
    ),
    "compression_pareto": (
        ("results/final-validation/compression-ablation/" "compression-pareto.csv"),
    ),
    "training_convergence": (
        (
            "results/final-validation/training-efficiency/"
            "final-training-efficiency-summary.json"
        ),
        "results/training/training-validation-summary.json",
    ),
    "rl_sample_efficiency": (
        ("results/final-validation/qml-ablation/" "qml-ablation.json"),
        ("results/final-validation/qml-ablation/" "qml-ablation-plot-data.csv"),
    ),
    "safety_violations": (
        ("results/final-validation/safety-ablation/" "safety-ablation.json"),
        ("results/final-validation/safety-ablation/" "safety-ablation-plot-data.csv"),
    ),
    "cross_domain": (
        ("results/final-validation/cross-domain/" "final-cross-domain-summary.json"),
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
    "ablation_summary": (
        ("results/final-validation/full-system-ablation/" "full-system-ablation.json"),
        ("results/final-validation/full-system-ablation/" "full-system-ablation.csv"),
    ),
}

FIGURE_METRICS = {
    "architecture": (
        "architectural_component_reuse",
        "integration_boundary",
    ),
    "compression_pareto": (
        "compression_ratio",
        "relative_mse_degradation_percent",
    ),
    "training_convergence": (
        "target_reach",
        "optimizer_steps_to_target",
    ),
    "rl_sample_efficiency": (
        "target_reach",
        "environment_steps_to_target",
        "actor_parameters",
    ),
    "safety_violations": (
        "violation_rate_mean",
        "violation_rate_sample_std",
    ),
    "cross_domain": (
        "criterion_status",
        "target_reach",
        "observed_clean_violation_status",
    ),
    "ablation_summary": ("evidence_status",),
}

FIGURE_LIMITATIONS = {
    "architecture": (
        (
            "The diagram represents the modular architecture and integration "
            "pathway. It does not imply that the complete shared-latent, "
            "compression, QML, and safety pipeline was executed end-to-end "
            "in Phase 1."
        ),
        "Driving and robotics used separate trained policy weights.",
        "Zero-shot cross-domain policy transfer was not evaluated.",
    ),
    "compression_pareto": (
        (
            "Compression results were obtained on the compact Phase 1 proxy "
            "model and do not establish production-scale VLA compression."
        ),
        (
            "INT8 storage compression does not imply native compressed-runtime "
            "speedup."
        ),
    ),
    "training_convergence": (
        (
            "Target-reaching quantities remain absent when a candidate did "
            "not reach its paired FP32 target."
        ),
        (
            "Phase 1 did not demonstrate a robust >=10% optimizer-step "
            "efficiency improvement across all locked seeds."
        ),
        "CPU wall-clock timing is descriptive.",
    ),
    "rl_sample_efficiency": (
        (
            "QML target non-attainment remains non-attainment and is not "
            "replaced with the maximum environment-step budget."
        ),
        "Actor compactness is reported separately from sample efficiency.",
        (
            "No quantum advantage, quantum speedup, or QML sample-efficiency "
            "advantage was demonstrated."
        ),
    ),
    "safety_violations": (
        (
            "Zero violation rate means zero observed violations under the "
            "frozen clean proxy evaluation only."
        ),
        (
            "The implemented Lyapunov quantity is an empirical classical "
            "safety potential, not a formal closed-loop stability proof."
        ),
        (
            "During perception perturbations the safety layer retained true "
            "simulator state."
        ),
    ),
    "cross_domain": (
        (
            "Metrics with different physical meanings and units are not "
            "combined into one synthetic aggregate score."
        ),
        (
            "Cross-domain reuse is supported at framework, interface, "
            "evaluation, and protocol levels rather than as one universal "
            "trained policy."
        ),
    ),
    "ablation_summary": (
        (
            "The matrix visualizes evidence status rather than fabricated "
            "full-system performance."
        ),
        (
            "No matched integrated Compression x QML x Safety configuration "
            "was directly executed in Phase 1."
        ),
        "Interaction effects remain a Phase 2 validation objective.",
    ),
}

FIGURE_SCOPES = {
    "architecture": (
        "Conceptual representation of frozen Phase 1 components, shared "
        "interfaces, and the experimentally unclosed integration boundary."
    ),
    "compression_pareto": (
        "Frozen Phase 1 compression tradeoff under the >=2x compression "
        "and <=5% relative MSE-degradation joint criterion."
    ),
    "training_convergence": (
        "Frozen target-attainment and optimizer-step evidence without "
        "extrapolating missing convergence points."
    ),
    "rl_sample_efficiency": (
        "Frozen classical PPO, PQC/QML, and matched-classical evidence; "
        "parameter compactness is separated from target attainment."
    ),
    "safety_violations": (
        "Observed clean proxy violation rates for NONE, CLIPPING, and "
        "classical LYAPUNOV filtering."
    ),
    "cross_domain": (
        "Categorical cross-domain evidence comparison without synthetic "
        "metric normalization."
    ),
    "ablation_summary": (
        "Evidence-status visualization of the 2x2x2 full-system factorial "
        "design across both proxy domains."
    ),
}


@dataclass(frozen=True)
class FigureEvidence:
    """Frozen evidence/provenance contract for one final figure."""

    figure_id: str
    title: str
    source_artifacts: tuple[str, ...]
    output_path: str
    domains: tuple[str, ...]
    metrics: tuple[str, ...]
    statistical_protocol: str
    scientific_scope: str
    limitations: tuple[str, ...]
    generated_from_frozen_evidence: bool = True

    def __post_init__(
        self,
    ) -> None:
        validate_figure_id(self.figure_id)

        if self.title != FIGURE_TITLES[self.figure_id]:
            raise ValueError("Figure title does not match " "the frozen contract.")

        if not self.source_artifacts:
            raise ValueError(
                "Every final evidence figure " "requires source provenance."
            )

        for source in self.source_artifacts:
            if not source.strip():
                raise ValueError("Figure source path cannot " "be blank.")

        if not self.output_path.strip():
            raise ValueError("Figure output path cannot " "be blank.")

        if not self.metrics:
            raise ValueError("Every figure must identify " "its evidence dimensions.")

        if not self.scientific_scope.strip():
            raise ValueError("Figure scientific scope " "cannot be blank.")

        if not self.limitations:
            raise ValueError("Every final figure must " "retain limitations.")

        if self.generated_from_frozen_evidence is not True:
            raise ValueError("Final figures must be generated " "from frozen evidence.")

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Serialize figure provenance metadata."""

        return asdict(self)


def validate_figure_id(
    figure_id: str,
) -> None:
    """Validate a required Sprint 7.7 figure ID."""

    if figure_id not in FIGURE_IDS:
        raise ValueError("Unknown Sprint 7.7 figure ID: " f"{figure_id!r}.")


def figure_output_path(
    figure_id: str,
) -> str:
    """Return canonical output path for a final figure."""

    validate_figure_id(figure_id)

    return str(OUTPUT_DIR / FIGURE_FILENAMES[figure_id])


def build_figure_contract(
    figure_id: str,
) -> FigureEvidence:
    """Build frozen metadata contract for one required figure."""

    validate_figure_id(figure_id)

    if figure_id == "architecture":
        domains = DOMAINS
        protocol = "frozen_architecture_and_" "integration_boundary"
    elif figure_id == "cross_domain":
        domains = DOMAINS
        protocol = "frozen_cross_domain_" "categorical_evidence"
    elif figure_id == "ablation_summary":
        domains = DOMAINS
        protocol = "frozen_full_system_" "evidence_classification"
    else:
        domains = DOMAINS
        protocol = "three_seed_mean_sample_sd"

    return FigureEvidence(
        figure_id=figure_id,
        title=(FIGURE_TITLES[figure_id]),
        source_artifacts=(FIGURE_SOURCES[figure_id]),
        output_path=(figure_output_path(figure_id)),
        domains=domains,
        metrics=(FIGURE_METRICS[figure_id]),
        statistical_protocol=protocol,
        scientific_scope=(FIGURE_SCOPES[figure_id]),
        limitations=(FIGURE_LIMITATIONS[figure_id]),
        generated_from_frozen_evidence=True,
    )


def build_all_figure_contracts() -> tuple[
    FigureEvidence,
    ...,
]:
    """Build the seven required Sprint 7.7 figure contracts."""

    contracts = tuple(build_figure_contract(figure_id) for figure_id in FIGURE_IDS)

    if len(contracts) != 7:
        raise ValueError("Sprint 7.7 requires exactly " "seven final figures.")

    return contracts


def claim_controls() -> dict[
    str,
    bool,
]:
    """Return a copy of final-figure claim controls."""

    return dict(CLAIM_CONTROLS)


def training_target_nonattainment_may_be_fabricated() -> bool:
    """Prevent missing target attainment from becoming synthetic data."""

    return ALLOW_FABRICATED_CONVERGENCE


def qml_nonattainment_may_be_censored_to_budget() -> bool:
    """Prevent QML non-attainment being plotted as measured budget reach."""

    return ALLOW_QML_NONATTAINMENT_CENSORING


def cross_domain_synthetic_score_allowed() -> bool:
    """Prevent scientifically meaningless aggregate scores."""

    return ALLOW_SYNTHETIC_CROSS_DOMAIN_SCORE


def full_system_performance_fabrication_allowed() -> bool:
    """Prevent the Sprint 7.6 evidence matrix becoming fake performance."""

    return ALLOW_FULL_SYSTEM_PERFORMANCE_FABRICATION
