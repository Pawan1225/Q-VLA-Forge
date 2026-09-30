"""Sprint 7.8 canonical final-result-table contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TABLE_IDS = (
    "compression",
    "training",
    "rl",
    "safety",
    "cross_domain",
)

TABLE_FILENAMES = {
    "compression": "compression-table.csv",
    "training": "training-table.csv",
    "rl": "rl-table.csv",
    "safety": "safety-table.csv",
    "cross_domain": "cross-domain-table.csv",
}

OUTPUT_DIR = Path("results/final-validation/tables")

FINAL_TABLES_JSON = OUTPUT_DIR / "final-result-tables.json"

FINAL_TABLES_MARKDOWN = OUTPUT_DIR / "final-result-tables.md"

TABLE_MANIFEST_PATH = OUTPUT_DIR / "table-manifest.json"

SEEDS = (
    42,
    123,
    456,
)

STATISTICAL_PROTOCOL = (
    "mean ± sample SD across n=3 locked seeds "
    "(42, 123, 456) for seed-dependent quantities; "
    "deterministic values remain plain values"
)

NEW_TRAINING = False
NEW_EXPERIMENTS = False
NEW_SCIENTIFIC_RESULTS = False
SYNTHETIC_METRIC_COMPOSITION_ALLOWED = False

NOT_REACHED = "Not reached"
NOT_MEASURED = "Not measured"
NOT_APPLICABLE = "Not applicable"

MISSING_VALUE_SEMANTICS = {
    "not_reached": NOT_REACHED,
    "not_measured": NOT_MEASURED,
    "not_applicable": NOT_APPLICABLE,
}

COMPRESSION_METHODS = (
    "fp32",
    "int8",
    "svd",
    "tt_mps",
)

SAFETY_METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

RL_POLICIES = (
    "ppo_mlp",
    "matched_classical",
    "ppo_pqc",
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

COMPRESSION_CRITERION = {
    "fp32": "REFERENCE",
    "int8": "PASS",
    "svd": "FAIL",
    "tt_mps": "FAIL",
}

QML_TARGET_REACH_COUNTS = {
    "ppo_mlp": {
        "reached": 6,
        "total": 6,
    },
    "matched_classical": {
        "reached": 1,
        "total": 6,
    },
    "ppo_pqc": {
        "reached": 0,
        "total": 6,
    },
}

QML_PARAMETER_REDUCTION_PERCENT = {
    "autonomous_driving": 95.90,
    "robotics": 95.51,
}

FULL_SYSTEM_COUNTS = {
    "direct": 0,
    "component_only": 16,
    "not_evaluated": 0,
}

CLAIM_CONTROLS = {
    "tt_mps_superiority_claim_allowed": False,
    "robust_training_efficiency_improvement_claim_allowed": False,
    "qml_sample_efficiency_advantage_claim_allowed": False,
    "qml_performance_superiority_claim_allowed": False,
    "quantum_advantage_claim_allowed": False,
    "quantum_speedup_claim_allowed": False,
    "formal_lyapunov_stability_claim_allowed": False,
    "guaranteed_zero_violations_claim_allowed": False,
    "iso_26262_certification_claim_allowed": False,
    "production_readiness_claim_allowed": False,
    "zero_shot_cross_domain_transfer_claim_allowed": False,
    "full_system_superiority_claim_allowed": False,
}

TABLE_COLUMNS = {
    "compression": (
        "Domain",
        "Method",
        "Parameters / Model Size",
        "Compression Ratio",
        "MSE",
        "Relative ΔMSE",
        "Latency",
        "Criterion",
    ),
    "training": (
        "Domain",
        "Method",
        "Target Reached",
        "Steps to Target",
        "Training Time",
        "Memory",
        "Final Validation Loss / MSE",
        "Efficiency Result",
    ),
    "rl": (
        "Domain",
        "Policy",
        "Target Reaches",
        "Steps to Target",
        "Final Reward",
        "Final Success Rate",
        "Actor Parameters",
        "Parameter Reduction",
    ),
    "safety": (
        "Domain",
        "Method",
        "Violation Rate",
        "Reward",
        "Success Rate",
        "Activation / Intervention",
        "Evidence Role",
    ),
    "cross_domain": (
        "Component / Method",
        "Driving Result",
        "Robotics Result",
        "Shared Across Domains?",
        "Evidence Boundary",
    ),
}

TABLE_TITLES = {
    "compression": "Compression Results",
    "training": "Training Efficiency Results",
    "rl": "RL / QML Results",
    "safety": "Safety Results",
    "cross_domain": "Cross-Domain Evidence Summary",
}

TABLE_SOURCES = {
    "compression": (
        ("results/final-validation/compression-ablation/" "compression-ablation.json"),
        ("results/final-validation/compression-ablation/" "compression-pareto.csv"),
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
    "training": (
        (
            "results/final-validation/training-efficiency/"
            "final-training-efficiency-summary.json"
        ),
        "results/training/training-validation-summary.json",
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
    "rl": (
        ("results/final-validation/qml-ablation/" "qml-ablation.json"),
        ("results/final-validation/qml-ablation/" "qml-ablation-plot-data.csv"),
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
    "safety": (
        ("results/final-validation/safety-ablation/" "safety-ablation.json"),
        ("results/final-validation/safety-ablation/" "safety-ablation-plot-data.csv"),
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
    "cross_domain": (
        ("results/final-validation/cross-domain/" "final-cross-domain-summary.json"),
        ("results/final-validation/full-system-ablation/" "full-system-ablation.json"),
        ("results/final-validation/statistics/" "final-statistics.json"),
    ),
}

TABLE_METRICS = {
    "compression": (
        "compression_ratio",
        "mse",
        "relative_mse_change_percent",
        "latency",
        "criterion_status",
    ),
    "training": (
        "target_reach",
        "steps_to_target",
        "training_time",
        "memory_if_measured",
        "final_validation_loss",
        "test_mse",
        "efficiency_result",
    ),
    "rl": (
        "target_reach",
        "steps_to_target",
        "final_reward",
        "final_success_rate",
        "actor_parameters",
        "parameter_reduction_percent",
    ),
    "safety": (
        "violation_rate",
        "reward",
        "success_rate",
        "intervention_rate",
        "evidence_role",
    ),
    "cross_domain": (
        "component_result",
        "shared_framework_status",
        "shared_weights_status",
        "evidence_boundary",
    ),
}

TABLE_LIMITATIONS = {
    "compression": (
        "Results are from the compact Phase 1 proxy VLA model.",
        (
            "MSE and MAE are supervised regression metrics; "
            "they must not be relabeled as generic accuracy."
        ),
        (
            "INT8 storage compression does not establish "
            "native compressed-runtime speedup."
        ),
        (
            "Evaluated SVD and TT/MPS configurations did not "
            "satisfy the joint frozen Phase 1 criterion."
        ),
    ),
    "training": (
        (
            "Phase 1 did not demonstrate a robust >=10% "
            "optimizer-step efficiency improvement across "
            "both domains and all locked seeds."
        ),
        (
            "Target-reaching quantities remain missing when "
            "the paired target was not reached."
        ),
        (
            "Memory must be reported as Not measured when "
            "the Sprint 3 evidence does not contain it."
        ),
        "CPU wall-clock timing is descriptive.",
    ),
    "rl": (
        (
            "QML target non-attainment remains Not reached "
            "and is not replaced with the 20,000-step budget."
        ),
        (
            "Actor compactness is reported separately from "
            "sample efficiency and performance."
        ),
        (
            "No quantum advantage, quantum speedup, "
            "or QML sample-efficiency advantage was demonstrated."
        ),
    ),
    "safety": (
        (
            "Zero violation rate means zero observed violations "
            "under the frozen proxy evaluation only."
        ),
        (
            "The Lyapunov quantity is an empirical classical "
            "safety potential, not a formal stability proof."
        ),
        (
            "Action-perturbation recovery evidence must not be "
            "mixed into the primary clean safety-ablation rows."
        ),
        (
            "During perception perturbations the safety layer "
            "retained true simulator state."
        ),
    ),
    "cross_domain": (
        (
            "Cross-domain reuse is supported at framework, "
            "interface, protocol, and evaluation levels."
        ),
        ("Driving and robotics used separate trained " "policy weights."),
        (
            "Domain-specific constraints, predictors, and "
            "safety semantics remain distinct."
        ),
        "No synthetic aggregate score is permitted.",
        (
            "No matched integrated Compression x QML x Safety "
            "factorial pipeline was directly executed in Phase 1."
        ),
    ),
}


@dataclass(frozen=True)
class FinalResultTable:
    """Scientific contract for one canonical Sprint 7.8 table."""

    table_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    source_artifacts: tuple[str, ...]
    metrics: tuple[str, ...]
    statistical_protocol: str
    limitations: tuple[str, ...]
    generated_from_frozen_evidence: bool = True

    def __post_init__(
        self,
    ) -> None:
        validate_table_id(self.table_id)

        if not self.title:
            raise ValueError("Table title cannot be empty.")

        if not self.columns:
            raise ValueError("Table columns cannot be empty.")

        if not self.source_artifacts:
            raise ValueError("Table provenance cannot be empty.")

        if not self.metrics:
            raise ValueError("Table metrics cannot be empty.")

        if not self.statistical_protocol:
            raise ValueError("Statistical protocol cannot be empty.")

        if not self.limitations:
            raise ValueError("Table limitations cannot be empty.")

        if self.generated_from_frozen_evidence is not True:
            raise ValueError("Sprint 7.8 tables must use frozen evidence.")

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


def validate_table_id(
    table_id: str,
) -> None:
    if table_id not in TABLE_IDS:
        raise ValueError(f"Unknown final-result table: {table_id}")


def table_output_path(
    table_id: str,
) -> str:
    validate_table_id(table_id)

    return str(OUTPUT_DIR / TABLE_FILENAMES[table_id]).replace(
        "\\",
        "/",
    )


def build_table_contract(
    table_id: str,
) -> FinalResultTable:
    validate_table_id(table_id)

    return FinalResultTable(
        table_id=table_id,
        title=TABLE_TITLES[table_id],
        columns=TABLE_COLUMNS[table_id],
        rows=(),
        source_artifacts=TABLE_SOURCES[table_id],
        metrics=TABLE_METRICS[table_id],
        statistical_protocol=STATISTICAL_PROTOCOL,
        limitations=TABLE_LIMITATIONS[table_id],
    )


def build_all_table_contracts() -> tuple[FinalResultTable, ...]:
    return tuple(build_table_contract(table_id) for table_id in TABLE_IDS)


def claim_controls() -> dict[str, bool]:
    return dict(CLAIM_CONTROLS)


def missing_value_semantics() -> dict[str, str]:
    return dict(MISSING_VALUE_SEMANTICS)


def synthetic_metric_composition_allowed() -> bool:
    return SYNTHETIC_METRIC_COMPOSITION_ALLOWED


def qml_nonattainment_value() -> str:
    return NOT_REACHED


def unmeasured_value() -> str:
    return NOT_MEASURED


def not_applicable_value() -> str:
    return NOT_APPLICABLE
