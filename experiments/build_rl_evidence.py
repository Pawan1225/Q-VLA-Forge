"""Build the consolidated Sprint 4 RL proposal-evidence package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_evidence import (
    ProposalCompactnessSummary,
    ProposalMethodSummary,
    ProposalRepresentationSummary,
    compactness_summary_to_dict,
    method_summary_to_dict,
    representation_summary_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_EFFICIENCY_PATH = (
    ROOT / "results" / "rl" / "analysis" / "sprint4-sample-efficiency.json"
)

ABLATION_PATH = (
    ROOT / "results" / "rl" / "ablation" / "sprint4-classical-vs-qml-ablation.json"
)

CROSS_DOMAIN_PATH = (
    ROOT / "results" / "rl" / "cross-domain" / "sprint4-cross-domain.json"
)

VALIDATION_PATH = (
    ROOT / "results" / "rl" / "validation" / "sprint4-three-seed-validation.json"
)

CLAIM_REGISTRY_PATH = ROOT / "results" / "pilot-readiness" / "claim-registry.json"

OUTPUT_DIR = ROOT / "results" / "rl" / "evidence"

OUTPUT_JSON = OUTPUT_DIR / "sprint4-rl-evidence.json"

OUTPUT_CSV = OUTPUT_DIR / "sprint4-rl-evidence.csv"


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_cross_domain_record(
    records: list[dict[str, Any]],
    *,
    domain: str,
    method: str,
) -> dict[str, Any]:
    for record in records:
        if record["domain"] == domain and record["method"] == method:
            return record

    raise KeyError(f"missing cross-domain record: " f"{domain}/{method}")


def build_method_summary(
    record: dict[str, Any],
) -> ProposalMethodSummary:
    return ProposalMethodSummary(
        domain=record["domain"],
        method=record["method"],
        actor_parameters=int(record["actor_parameters"]),
        target_reach_count=int(record["target_reach_count"]),
        target_total=int(record["target_total"]),
        normalized_auc_mean=float(record["normalized_auc_mean"]),
        normalized_auc_sd=float(record["normalized_auc_sd"]),
        best_progress_mean=float(record["best_progress_mean"]),
        best_progress_sd=float(record["best_progress_sd"]),
        final_progress_mean=float(record["final_progress_mean"]),
        final_progress_sd=float(record["final_progress_sd"]),
    )


def build_domain_block(
    *,
    domain: str,
    records: list[dict[str, Any]],
    compactness: dict[str, Any],
) -> dict[str, Any]:
    full_ppo_record = get_cross_domain_record(
        records,
        domain=domain,
        method="full_ppo",
    )

    matched_record = get_cross_domain_record(
        records,
        domain=domain,
        method="matched_classical",
    )

    qml_record = get_cross_domain_record(
        records,
        domain=domain,
        method="hybrid_qml",
    )

    full_ppo = build_method_summary(full_ppo_record)
    matched = build_method_summary(matched_record)
    qml = build_method_summary(qml_record)

    compactness_record = compactness[domain]

    compactness_summary = ProposalCompactnessSummary(
        domain=domain,
        full_actor_parameters=int(full_ppo_record["actor_parameters"]),
        compact_actor_parameters=int(qml_record["actor_parameters"]),
        parameter_reduction_percent=float(
            compactness_record["compact_actor_reduction_percent"]
        ),
    )

    matched_minus_qml_auc = float(matched_record["normalized_auc_mean"]) - float(
        qml_record["normalized_auc_mean"]
    )

    if matched_minus_qml_auc > 0:
        direction = "matched_classical"
    elif matched_minus_qml_auc < 0:
        direction = "hybrid_qml"
    else:
        direction = "tie"

    representation_summary = ProposalRepresentationSummary(
        domain=domain,
        matched_actor_parameters=int(matched_record["actor_parameters"]),
        qml_actor_parameters=int(qml_record["actor_parameters"]),
        matched_auc_mean=float(matched_record["normalized_auc_mean"]),
        qml_auc_mean=float(qml_record["normalized_auc_mean"]),
        matched_minus_qml_auc=(matched_minus_qml_auc),
        direction=direction,
    )

    return {
        "methods": {
            "full_ppo": (method_summary_to_dict(full_ppo)),
            "matched_classical": (method_summary_to_dict(matched)),
            "hybrid_qml": (method_summary_to_dict(qml)),
        },
        "compactness": (compactness_summary_to_dict(compactness_summary)),
        "matched_budget_representation": (
            representation_summary_to_dict(representation_summary)
        ),
    }


def select_sprint4_claims(
    claim_registry: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_ids = {
        "ppo-sample-efficiency",
        "pqc-vqc-sample-efficiency",
        "hybrid-qml-policy-advantage",
        "cross-domain-rl-consistency",
        "sprint4-classical-ppo-baseline",
        "sprint4-hybrid-pqc-ppo-implementation",
        "sprint4-hybrid-actor-compactness",
        "sprint4-driving-qml-sample-efficiency-advantage",
        "sprint4-robotics-qml-sample-efficiency-advantage",
        "sprint4-robust-qml-sample-efficiency-advantage",
        "sprint4-quantum-computational-speedup",
        "sprint4-quantum-hardware-advantage",
        "sprint4-cross-domain-architecture-reuse",
        "sprint4-cross-domain-representation-consistency",
    }

    claims = []

    for claim in claim_registry["claims"]:
        claim_id = claim.get(
            "claim_id",
            claim.get("id"),
        )

        if claim_id in selected_ids:
            claims.append(claim)

    return claims


def build_limitations() -> list[str]:
    return [
        (
            "Synthetic proxy environments rather than "
            "production autonomous-driving or robotics "
            "benchmarks."
        ),
        (
            "Compact state vectors were used instead of "
            "a full production-scale VLA latent input."
        ),
        ("No CARLA, RLBench, or LIBERO deployment was " "performed in Sprint 4."),
        "No quantum hardware was used.",
        ("PennyLane default.qubit analytic simulation " "was used for the PQC."),
        "The PQC used four qubits.",
        "The PQC used two variational layers.",
        "Three principal seeds were evaluated.",
        ("The principal interaction budget was " "20,000 environment steps per run."),
        (
            "No formal statistical significance test "
            "was used for the three-seed comparison."
        ),
        "No transfer learning was evaluated.",
        ("The same trained weights were not reused " "across both domains."),
        ("No wall-clock quantum computational " "advantage is claimed."),
        (
            "Safety filtering is outside Sprint 4 and "
            "is evaluated separately in Sprint 5."
        ),
    ]


def build_proposal_statements() -> list[str]:
    return [
        (
            "Classical PPO reached all six paired frozen "
            "reward targets across three seeds in both "
            "proxy domains, whereas the tested hybrid "
            "QML policies reached none within the "
            "20,000-step interaction budget."
        ),
        (
            "The hybrid PQC actors reduced trainable "
            "actor parameters from 1,318 to 54 in "
            "autonomous driving and from 1,382 to 62 in "
            "robotics, corresponding to reductions of "
            "approximately 95.9% and 95.5%."
        ),
        (
            "Parameter-matched ablation showed "
            "domain-dependent representation behavior: "
            "hybrid QML achieved higher mean normalized "
            "learning-curve AUC than the matched "
            "classical control in driving, while the "
            "matched classical control achieved higher "
            "mean AUC in robotics."
        ),
        (
            "The same four-qubit, two-layer hybrid QML "
            "policy architecture was reused across both "
            "proxy domains under a common PPO and "
            "evaluation protocol."
        ),
        (
            "These pilot results support compact "
            "cross-domain architectural reuse, but do "
            "not demonstrate robust QML sample-efficiency "
            "advantage, quantum computational speedup, "
            "or quantum-hardware advantage."
        ),
    ]


def build_blacklisted_wording() -> list[str]:
    return [
        "Quantum advantage demonstrated",
        "Quantum speedup demonstrated",
        "QML outperforms PPO",
        "QML is more sample efficient",
        "QML generalizes across domains",
        "One universal policy",
        "Production autonomous driving",
        "Production robotics",
        "Safety certified",
        "ISO 26262 compliant",
        "ISO 10218 certified",
    ]


def write_csv(
    evidence: dict[str, Any],
) -> None:
    critic_parameters = {
        "autonomous_driving": 1249,
        "robotics": 1313,
    }

    fieldnames = [
        "domain",
        "method",
        "actor_parameters",
        "total_parameters",
        "target_reach_count",
        "target_total",
        "normalized_auc_mean",
        "normalized_auc_sd",
        "best_progress_mean",
        "best_progress_sd",
        "final_progress_mean",
        "final_progress_sd",
        "compactness_vs_full_ppo_percent",
    ]

    rows = []

    for domain, block in evidence["domains"].items():
        reduction = float(block["compactness"]["parameter_reduction_percent"])

        for method_name in (
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        ):
            method = block["methods"][method_name]

            if method_name == "full_ppo":
                compactness_value = 0.0
            else:
                compactness_value = reduction

            rows.append(
                {
                    "domain": domain,
                    "method": method_name,
                    "actor_parameters": (method["actor_parameters"]),
                    "total_parameters": (
                        method["actor_parameters"] + critic_parameters[domain]
                    ),
                    "target_reach_count": (method["target_reach_count"]),
                    "target_total": (method["target_total"]),
                    "normalized_auc_mean": (method["normalized_auc_mean"]),
                    "normalized_auc_sd": (method["normalized_auc_sd"]),
                    "best_progress_mean": (method["best_progress_mean"]),
                    "best_progress_sd": (method["best_progress_sd"]),
                    "final_progress_mean": (method["final_progress_mean"]),
                    "final_progress_sd": (method["final_progress_sd"]),
                    ("compactness_vs_" "full_ppo_percent"): compactness_value,
                }
            )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_figure_index() -> dict[str, Any]:
    """Build the deterministic Sprint 4 proposal figure index."""
    figures = [
        {
            "proposal_priority": 1,
            "category": "primary_result",
            "path": "figures/rl/driving-normalized-progress.png",
            "purpose": (
                "Driving normalized learning progress across "
                "Full PPO, Matched Classical, and Hybrid QML."
            ),
        },
        {
            "proposal_priority": 2,
            "category": "primary_result",
            "path": "figures/rl/robotics-normalized-progress.png",
            "purpose": (
                "Robotics normalized learning progress across "
                "Full PPO, Matched Classical, and Hybrid QML."
            ),
        },
        {
            "proposal_priority": 3,
            "category": "ablation",
            "path": "figures/rl/driving-matched-ablation.png",
            "purpose": (
                "Driving matched-budget classical versus "
                "Hybrid QML representation comparison."
            ),
        },
        {
            "proposal_priority": 4,
            "category": "ablation",
            "path": "figures/rl/robotics-matched-ablation.png",
            "purpose": (
                "Robotics matched-budget classical versus "
                "Hybrid QML representation comparison."
            ),
        },
        {
            "proposal_priority": 5,
            "category": "cross_domain",
            "path": "figures/rl/cross-domain-matched-delta.png",
            "purpose": ("Cross-domain matched-minus-QML normalized " "AUC comparison."),
        },
        {
            "proposal_priority": 6,
            "category": "compactness",
            "path": ("figures/rl/" "cross-domain-parameter-compactness.png"),
            "purpose": ("Cross-domain actor-parameter compactness " "comparison."),
        },
    ]

    recommendation = {
        "recommended_group_count": 3,
        "recommended_figure_groups": [
            {
                "group": "normalized_learning_curves",
                "source_figures": [
                    ("figures/rl/" "driving-normalized-progress.png"),
                    ("figures/rl/" "robotics-normalized-progress.png"),
                ],
            },
            {
                "group": "matched_budget_ablation",
                "source_figures": [
                    ("figures/rl/" "driving-matched-ablation.png"),
                    ("figures/rl/" "robotics-matched-ablation.png"),
                ],
            },
            {
                "group": ("cross_domain_compactness_" "and_representation"),
                "source_figures": [
                    ("figures/rl/" "cross-domain-matched-delta.png"),
                    ("figures/rl/" "cross-domain-parameter-compactness.png"),
                ],
            },
        ],
        "combined_panels_deferred_to_final_report": True,
        "new_scientific_analysis_required": False,
    }

    return {
        "sprint": "4.14",
        "figure_count": len(figures),
        "figures": figures,
        "proposal_recommendation": recommendation,
    }


def build_claim_matrix(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build proposal-safe Sprint 4 claim matrix."""
    claims_by_id = {claim["claim_id"]: claim for claim in evidence["claims"]}

    specifications = [
        {
            "claim_id": "classical-ppo-baseline",
            "registry_claim_id": ("sprint4-classical-ppo-baseline"),
            "proposal_safe_wording": (
                "The classical PPO baseline reached the "
                "defined target in all six principal runs."
            ),
            "limitation": (
                "Result is limited to the tested synthetic "
                "proxy environments and protocol."
            ),
        },
        {
            "claim_id": "hybrid-pqc-implementation",
            "registry_claim_id": ("sprint4-hybrid-pqc-ppo-implementation"),
            "proposal_safe_wording": (
                "A hybrid PPO policy using a simulated "
                "parameterized quantum circuit was "
                "implemented and evaluated."
            ),
            "limitation": ("The PQC used simulation rather than " "quantum hardware."),
        },
        {
            "claim_id": "hybrid-actor-compactness",
            "registry_claim_id": ("sprint4-hybrid-actor-compactness"),
            "proposal_safe_wording": (
                "The tested compact actors reduced actor "
                "parameter count by approximately 95.9% "
                "in driving and 95.5% in robotics."
            ),
            "limitation": (
                "Parameter compactness is descriptive and "
                "does not establish computational speedup."
            ),
        },
        {
            "claim_id": "qml-sample-efficiency",
            "registry_claim_id": ("sprint4-robust-qml-sample-efficiency-advantage"),
            "proposal_safe_wording": (
                "A robust Hybrid QML sample-efficiency "
                "advantage was not demonstrated."
            ),
            "limitation": (
                "Three principal seeds and a 20,000-step "
                "interaction budget were evaluated."
            ),
        },
        {
            "claim_id": "matched-budget-driving",
            "registry_claim_id": ("sprint4-driving-qml-sample-efficiency-advantage"),
            "proposal_safe_wording": (
                "Under the tested matched-budget driving "
                "comparison, Hybrid QML had the higher mean "
                "normalized AUC, but no robust advantage is "
                "claimed."
            ),
            "limitation": (
                "The direction is domain dependent and no "
                "formal statistical significance test was "
                "performed."
            ),
        },
        {
            "claim_id": "matched-budget-robotics",
            "registry_claim_id": ("sprint4-robotics-qml-sample-efficiency-advantage"),
            "proposal_safe_wording": (
                "Under the tested matched-budget robotics "
                "comparison, matched classical had the "
                "higher mean normalized AUC."
            ),
            "limitation": (
                "The direction is domain dependent and no "
                "formal statistical significance test was "
                "performed."
            ),
        },
        {
            "claim_id": "cross-domain-architecture-reuse",
            "registry_claim_id": ("sprint4-cross-domain-architecture-reuse"),
            "proposal_safe_wording": (
                "The same hybrid policy architecture was "
                "reused across driving and robotics."
            ),
            "limitation": (
                "Architecture reuse does not mean the same "
                "trained weights were reused."
            ),
        },
        {
            "claim_id": ("cross-domain-performance-consistency"),
            "registry_claim_id": ("sprint4-cross-domain-representation-consistency"),
            "proposal_safe_wording": (
                "The matched-budget representation result "
                "was not directionally consistent across "
                "the two domains."
            ),
            "limitation": (
                "The tested evidence does not support a "
                "universal cross-domain representation "
                "advantage."
            ),
        },
        {
            "claim_id": "quantum-speedup",
            "registry_claim_id": ("sprint4-quantum-computational-speedup"),
            "proposal_safe_wording": (
                "No quantum computational speedup is " "claimed from Sprint 4."
            ),
            "limitation": (
                "The experiment used simulated PQCs and "
                "did not establish quantum runtime "
                "advantage."
            ),
        },
        {
            "claim_id": "quantum-hardware-advantage",
            "registry_claim_id": ("sprint4-quantum-hardware-advantage"),
            "proposal_safe_wording": ("No quantum-hardware advantage is claimed."),
            "limitation": (
                "No quantum hardware was used in the " "Sprint 4 experiments."
            ),
        },
    ]

    records = []

    for specification in specifications:
        registry_claim_id = specification["registry_claim_id"]

        if registry_claim_id not in claims_by_id:
            raise RuntimeError(
                "required registry claim missing: " f"{registry_claim_id}"
            )

        source = claims_by_id[registry_claim_id]

        records.append(
            {
                "claim_id": (specification["claim_id"]),
                "registry_claim_id": registry_claim_id,
                "claim": source["statement"],
                "status": source["status"],
                "evidence_basis": source["evidence_basis"],
                "evidence_artifacts": source.get(
                    "evidence_artifacts",
                    [],
                ),
                "proposal_safe_wording": (specification["proposal_safe_wording"]),
                "limitation": (specification["limitation"]),
            }
        )

    return {
        "sprint": "4.14",
        "claim_count": len(records),
        "claims": records,
    }


def write_supplemental_evidence(
    evidence: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Write deterministic figure and claim evidence."""
    figure_index = build_figure_index()
    claim_matrix = build_claim_matrix(evidence)

    figure_path = OUTPUT_DIR / "sprint4-figure-index.json"

    claim_path = OUTPUT_DIR / "sprint4-claim-matrix.json"

    figure_path.write_text(
        json.dumps(
            figure_index,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    claim_path.write_text(
        json.dumps(
            claim_matrix,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        figure_index,
        claim_matrix,
    )


def main() -> None:
    sample_efficiency = load_json(SAMPLE_EFFICIENCY_PATH)
    ablation = load_json(ABLATION_PATH)
    cross_domain = load_json(CROSS_DOMAIN_PATH)
    validation = load_json(VALIDATION_PATH)
    claim_registry = load_json(CLAIM_REGISTRY_PATH)

    records = cross_domain["method_records"]

    compactness = cross_domain["compactness"]

    driving = build_domain_block(
        domain="autonomous_driving",
        records=records,
        compactness=compactness,
    )

    robotics = build_domain_block(
        domain="robotics",
        records=records,
        compactness=compactness,
    )

    ppo_target_reach = (
        driving["methods"]["full_ppo"]["target_reach_count"]
        + robotics["methods"]["full_ppo"]["target_reach_count"]
    )

    matched_target_reach = (
        driving["methods"]["matched_classical"]["target_reach_count"]
        + robotics["methods"]["matched_classical"]["target_reach_count"]
    )

    qml_target_reach = (
        driving["methods"]["hybrid_qml"]["target_reach_count"]
        + robotics["methods"]["hybrid_qml"]["target_reach_count"]
    )

    conclusion = cross_domain["cross_domain_conclusion"]

    evidence = {
        "metadata": {
            "sprint": "4.14",
            "title": "Sprint 4 RL and Hybrid QML Evidence",
            "principal_seeds": cross_domain["principal_seeds"],
            "new_training_performed": False,
            "new_metrics_introduced": False,
            "evidence_packaging_only": True,
        },
        "sprint": "4.14",
        "title": ("Sprint 4 RL and Hybrid QML Evidence"),
        "principal_seeds": (cross_domain["principal_seeds"]),
        "new_training_performed": False,
        "new_metrics_introduced": False,
        "protocol": {
            "interaction_budget_environment_steps": 20000,
            "principal_seed_count_per_domain": 3,
            "principal_seeds": (cross_domain["principal_seeds"]),
            "primary_metric": (sample_efficiency["primary_metric"]),
            "evidence_packaging_only": True,
        },
        "domains": {
            "autonomous_driving": driving,
            "robotics": robotics,
        },
        "methods": [
            "full_ppo",
            "matched_classical",
            "hybrid_qml",
        ],
        "target_reach": {
            "full_ppo": {
                "reached": ppo_target_reach,
                "total": 6,
            },
            "matched_classical": {
                "reached": matched_target_reach,
                "total": 6,
            },
            "hybrid_qml": {
                "reached": qml_target_reach,
                "total": 6,
            },
        },
        "global_results": {
            "full_ppo_target_reach_count": (ppo_target_reach),
            "full_ppo_target_total": 6,
            "matched_classical_target_reach_count": (matched_target_reach),
            "matched_classical_target_total": 6,
            "hybrid_qml_target_reach_count": (qml_target_reach),
            "hybrid_qml_target_total": 6,
            "robust_qml_sample_efficiency_advantage": (False),
            "cross_domain_architecture_reuse": (
                bool(conclusion["architecture_reused"])
            ),
            ("cross_domain_representation_" "direction_consistent"): bool(
                conclusion[("matched_auc_direction_" "consistent_across_domains")]
            ),
        },
        "compactness": (cross_domain["compactness"]),
        "sample_efficiency": {
            "source_sprint": (sample_efficiency["sprint"]),
            "source_title": (sample_efficiency["title"]),
            "primary_metric": (sample_efficiency["primary_metric"]),
        },
        "matched_budget_ablation": {
            "source_sprint": ablation["sprint"],
            "source_title": ablation["title"],
            "driving_direction": (
                driving["matched_budget_representation"]["direction"]
            ),
            "robotics_direction": (
                robotics["matched_budget_representation"]["direction"]
            ),
            "direction_consistent_across_domains": (False),
        },
        "cross_domain": {
            "source_sprint": (cross_domain["sprint"]),
            "source_title": (cross_domain["title"]),
            "architecture_reuse": (conclusion["architecture_reused"]),
            ("robust_cross_domain_" "qml_advantage"): conclusion[
                "robust_cross_domain_qml_advantage"
            ],
            ("robust_cross_domain_" "matched_classical_advantage"): conclusion[
                ("robust_cross_domain_" "matched_classical_advantage")
            ],
            ("matched_auc_direction_" "consistent_across_domains"): conclusion[
                ("matched_auc_direction_" "consistent_across_domains")
            ],
        },
        "reproducibility": {
            "principal_seeds": (cross_domain["principal_seeds"]),
            "three_seed_validation_source": (validation["sprint"]),
            "claim_registry_overall_passed": (claim_registry["overall_passed"]),
            "driving_qml_seed42_reproduction": (
                validation["global_validation"]["driving_seed42_reproduction_exists"]
            ),
            "robotics_qml_seed42_reproduction": (
                validation["global_validation"]["robotics_seed42_reproduction_exists"]
            ),
            "driving_matched_classical_seed42_reproduction": True,
            "robotics_matched_classical_seed42_reproduction": True,
            "matched_classical_reproduction_verified_by": (
                "experiments/verify_rl_ablation.py"
            ),
            "ppo_targets_sha256": (
                "F78C5CFBB30CE20AA2DE53E75CB3C033" "A5A1AC846E71810D6C06006424DDA5B1"
            ),
            "ppo_summary_sha256": (
                "CA090235E3ED490F7B03C33D7B4D750B" "FBA34872495F6E5C67BC87365960F559"
            ),
            "ppo_target_hash_unchanged": True,
            "ppo_summary_hash_unchanged": True,
        },
        "claims": (select_sprint4_claims(claim_registry)),
        "figures": [],
        "figure_index": [],
        "limitations": build_limitations(),
        "proposal_ready_statements": (build_proposal_statements()),
        "proposal_ready_paragraph": (
            "Across the Sprint 4 pilot, the classical PPO baseline reached "
            "the predefined performance target in all 6 of 6 principal runs, "
            "while the parameter-matched classical policy reached the target "
            "in 1 of 6 runs and the tested hybrid QML policy in 0 of 6. "
            "The hybrid policy architecture reduced trainable actor parameters "
            "by approximately 95.9% in autonomous driving and 95.5% in robotics, "
            "while reusing the same 4-qubit, 2-layer PQC architecture across "
            "both domains. Under the matched-budget comparison, Hybrid QML "
            "achieved the higher mean normalized AUC in driving, whereas the "
            "matched classical representation achieved the higher mean "
            "normalized AUC in robotics, indicating a domain-dependent "
            "representation effect rather than a consistent cross-domain "
            "advantage. These results support compact hybrid policy construction "
            "and cross-domain architecture reuse, but do not establish robust "
            "QML sample-efficiency advantage, quantum computational speedup, "
            "quantum-hardware advantage, transfer learning, or shared "
            "trained-weight generalization."
        ),
        "proposal_ready_bullets": [
            (
                "Classical PPO reached the predefined target in 6/6 principal "
                "runs across driving and robotics."
            ),
            (
                "The parameter-matched classical policy reached the target in "
                "1/6 runs, while the tested Hybrid QML policy reached it in 0/6."
            ),
            (
                "The compact hybrid actor reduced trainable actor parameters "
                "by approximately 95.9% in driving and 95.5% in robotics."
            ),
            (
                "Under matched-budget comparison, Hybrid QML had the higher "
                "mean normalized AUC in driving, while matched classical had "
                "the higher mean normalized AUC in robotics, indicating a "
                "domain-dependent representation effect."
            ),
            (
                "Sprint 4 supports hybrid architecture reuse and actor "
                "compactness, but does not establish robust QML "
                "sample-efficiency advantage, quantum speedup, "
                "quantum-hardware advantage, transfer learning, or "
                "shared-weight cross-domain generalization."
            ),
        ],
        "blacklisted_wording": (build_blacklisted_wording()),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_index, claim_matrix = write_supplemental_evidence(evidence)

    evidence["figures"] = figure_index["figures"]
    evidence["figure_index"] = figure_index
    evidence["claim_matrix"] = claim_matrix

    OUTPUT_JSON.write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(evidence)

    print(
        "Sprint 4 evidence JSON:",
        OUTPUT_JSON,
    )
    print(
        "Sprint 4 evidence CSV:",
        OUTPUT_CSV,
    )
    print(
        "Full PPO target reach:",
        ppo_target_reach,
        "/ 6",
    )
    print(
        "Matched target reach:",
        matched_target_reach,
        "/ 6",
    )
    print(
        "Hybrid QML target reach:",
        qml_target_reach,
        "/ 6",
    )
    print(
        "Driving representation:",
        driving["matched_budget_representation"]["direction"],
    )
    print(
        "Robotics representation:",
        robotics["matched_budget_representation"]["direction"],
    )


if __name__ == "__main__":
    main()
