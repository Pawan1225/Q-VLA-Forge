"""Build Sprint 4.13 cross-domain RL comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_cross_domain import (
    CrossDomainMethodRecord,
    MatchedRepresentationComparison,
    matched_comparison_to_dict,
    parameter_reduction_percent,
    record_to_dict,
    representation_direction,
)

SAMPLE_EFFICIENCY_PATH = (
    Path("results") / "rl" / "analysis" / "sprint4-sample-efficiency.json"
)

ABLATION_PATH = (
    Path("results") / "rl" / "ablation" / "sprint4-classical-vs-qml-ablation.json"
)

VALIDATION_PATH = (
    Path("results") / "rl" / "validation" / "sprint4-three-seed-validation.json"
)

TARGET_PATH = Path("results") / "rl" / "ppo" / "sprint4-ppo-targets.json"

OUTPUT_DIRECTORY = Path("results") / "rl" / "cross-domain"

JSON_OUTPUT = OUTPUT_DIRECTORY / "sprint4-cross-domain.json"

CSV_OUTPUT = OUTPUT_DIRECTORY / "sprint4-cross-domain.csv"

MARKDOWN_OUTPUT = OUTPUT_DIRECTORY / "sprint4-cross-domain.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "full_ppo",
    "matched_classical",
    "hybrid_qml",
)

SEEDS = (
    42,
    123,
    456,
)

DISPLAY_NAMES = {
    "autonomous_driving": "Autonomous Driving",
    "robotics": "Robotics",
}

EXPECTED_ACTOR_PARAMETERS = {
    "autonomous_driving": {
        "full_ppo": 1318,
        "matched_classical": 54,
        "hybrid_qml": 54,
    },
    "robotics": {
        "full_ppo": 1382,
        "matched_classical": 62,
        "hybrid_qml": 62,
    },
}

EXPECTED_TARGET_REACH = {
    "autonomous_driving": {
        "full_ppo": 3,
        "matched_classical": 0,
        "hybrid_qml": 0,
    },
    "robotics": {
        "full_ppo": 3,
        "matched_classical": 1,
        "hybrid_qml": 0,
    },
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_inputs(
    *,
    sample_efficiency: dict[str, Any],
    ablation: dict[str, Any],
    validation: dict[str, Any],
    targets: dict[str, Any],
) -> None:
    """Validate the frozen evidence entering Sprint 4.13."""

    if sample_efficiency["sprint"] != "4.11":
        raise RuntimeError("unexpected Sprint 4.11 artifact")

    if sample_efficiency["new_training_performed"] is not False:
        raise RuntimeError("Sprint 4.11 must remain analysis-only")

    if ablation["sprint"] != "4.12":
        raise RuntimeError("unexpected Sprint 4.12 artifact")

    if ablation["new_qml_training_performed"] is not False:
        raise RuntimeError("Sprint 4.12 must not contain new QML training")

    if ablation["parameter_matched_ablation"] is not True:
        raise RuntimeError("Sprint 4.12 parameter matching is not frozen")

    if validation["sprint"] != "4.10":
        raise RuntimeError("unexpected Sprint 4.10 artifact")

    if validation["principal_training_performed"] is not False:
        raise RuntimeError("Sprint 4.10 must not contain new principal training")

    if tuple(int(seed) for seed in validation["principal_seeds"]) != SEEDS:
        raise RuntimeError("Sprint 4.10 seed set mismatch")

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("frozen PPO target history is invalid")

    for domain in DOMAINS:
        target_domain = targets["domains"][domain]

        if target_domain["targets_frozen_before_qml"] is not True:
            raise RuntimeError(f"{domain} targets were not frozen before QML")


def metric_pair(
    payload: dict[str, Any],
    key: str,
) -> tuple[
    float,
    float,
]:
    """Return mean and sample SD from one aggregate metric."""

    metric = payload[key]

    return (
        float(metric["mean"]),
        float(metric["sample_standard_deviation"]),
    )


def build_method_record(
    *,
    domain: str,
    method: str,
    sample_efficiency: dict[str, Any],
    ablation: dict[str, Any],
) -> CrossDomainMethodRecord:
    """Build one domain/method cross-domain record."""

    if method == "full_ppo":
        aggregate = sample_efficiency["domains"][domain]["policies"]["ppo"]["aggregate"]

    elif method == "hybrid_qml":
        aggregate = sample_efficiency["domains"][domain]["policies"]["qml"]["aggregate"]

    elif method == "matched_classical":
        aggregate = ablation["domains"][domain]["methods"]["matched_classical"]

    else:
        raise ValueError(f"unsupported method: {method}")

    (
        auc_mean,
        auc_sd,
    ) = metric_pair(
        aggregate,
        "normalized_auc",
    )

    (
        best_mean,
        best_sd,
    ) = metric_pair(
        aggregate,
        "best_normalized_progress",
    )

    (
        final_mean,
        final_sd,
    ) = metric_pair(
        aggregate,
        "final_normalized_progress",
    )

    actor_parameters = int(aggregate["actor_parameters"])

    target_reach_count = int(aggregate["target_reach_count"])

    expected_actor = EXPECTED_ACTOR_PARAMETERS[domain][method]

    expected_reach = EXPECTED_TARGET_REACH[domain][method]

    if actor_parameters != expected_actor:
        raise RuntimeError(f"{domain} {method} actor parameter mismatch")

    if target_reach_count != expected_reach:
        raise RuntimeError(f"{domain} {method} target reach mismatch")

    return CrossDomainMethodRecord(
        domain=domain,
        method=method,
        actor_parameters=(actor_parameters),
        target_reach_count=(target_reach_count),
        target_total=3,
        normalized_auc_mean=(auc_mean),
        normalized_auc_sd=(auc_sd),
        best_progress_mean=(best_mean),
        best_progress_sd=(best_sd),
        final_progress_mean=(final_mean),
        final_progress_sd=(final_sd),
    )


def validate_against_three_seed_validation(
    *,
    records: list[CrossDomainMethodRecord],
    validation: dict[str, Any],
) -> None:
    """Cross-check PPO and QML target reach against Sprint 4.10."""

    for domain in DOMAINS:
        summary = validation["domains"][domain]["summary"]

        ppo_records = [
            record
            for record in records
            if (record.domain == domain and record.method == "full_ppo")
        ]

        qml_records = [
            record
            for record in records
            if (record.domain == domain and record.method == "hybrid_qml")
        ]

        if len(ppo_records) != 1:
            raise RuntimeError(f"{domain} PPO record lookup failed")

        if len(qml_records) != 1:
            raise RuntimeError(f"{domain} QML record lookup failed")

        if ppo_records[0].target_reach_count != int(
            summary["classical_target_reach_count"]
        ):
            raise RuntimeError(f"{domain} PPO validation mismatch")

        if qml_records[0].target_reach_count != int(summary["qml_target_reach_count"]):
            raise RuntimeError(f"{domain} QML validation mismatch")


def record_lookup(
    *,
    records: list[CrossDomainMethodRecord],
    domain: str,
    method: str,
) -> CrossDomainMethodRecord:
    """Return exactly one method record."""

    matches = [
        record
        for record in records
        if (record.domain == domain and record.method == method)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one {domain} {method} record")

    return matches[0]


def build_matched_comparison(
    *,
    records: list[CrossDomainMethodRecord],
    domain: str,
) -> MatchedRepresentationComparison:
    """Build one equal-parameter matched-vs-QML comparison."""

    matched = record_lookup(
        records=records,
        domain=domain,
        method="matched_classical",
    )

    qml = record_lookup(
        records=records,
        domain=domain,
        method="hybrid_qml",
    )

    if matched.actor_parameters != qml.actor_parameters:
        raise RuntimeError(f"{domain} compact actor parameter mismatch")

    difference = float(matched.normalized_auc_mean - qml.normalized_auc_mean)

    return MatchedRepresentationComparison(
        domain=domain,
        matched_actor_parameters=(matched.actor_parameters),
        qml_actor_parameters=(qml.actor_parameters),
        matched_auc_mean=(matched.normalized_auc_mean),
        qml_auc_mean=(qml.normalized_auc_mean),
        matched_minus_qml_auc=(difference),
        direction=(
            representation_direction(
                matched_value=(matched.normalized_auc_mean),
                qml_value=(qml.normalized_auc_mean),
            )
        ),
    )


def matched_metric_direction(
    *,
    records: list[CrossDomainMethodRecord],
    domain: str,
    metric: str,
) -> str:
    """Return direction for another matched-budget aggregate metric."""

    matched = record_lookup(
        records=records,
        domain=domain,
        method="matched_classical",
    )

    qml = record_lookup(
        records=records,
        domain=domain,
        method="hybrid_qml",
    )

    if metric == "best":
        matched_value = matched.best_progress_mean

        qml_value = qml.best_progress_mean

    elif metric == "final":
        matched_value = matched.final_progress_mean

        qml_value = qml.final_progress_mean

    else:
        raise ValueError(f"unsupported metric: {metric}")

    return representation_direction(
        matched_value=matched_value,
        qml_value=qml_value,
    )


def build_compactness(
    *,
    records: list[CrossDomainMethodRecord],
) -> dict[str, Any]:
    """Build cross-domain compactness accounting."""

    payload: dict[
        str,
        Any,
    ] = {}

    reductions: list[float] = []

    for domain in DOMAINS:
        full = record_lookup(
            records=records,
            domain=domain,
            method="full_ppo",
        )

        qml = record_lookup(
            records=records,
            domain=domain,
            method="hybrid_qml",
        )

        matched = record_lookup(
            records=records,
            domain=domain,
            method="matched_classical",
        )

        if qml.actor_parameters != matched.actor_parameters:
            raise RuntimeError(f"{domain} compact actors are not parameter matched")

        reduction = parameter_reduction_percent(
            full_actor_parameters=(full.actor_parameters),
            compact_actor_parameters=(qml.actor_parameters),
        )

        reductions.append(reduction)

        payload[domain] = {
            "full_ppo_actor_parameters": (full.actor_parameters),
            "matched_classical_actor_parameters": (matched.actor_parameters),
            "hybrid_qml_actor_parameters": (qml.actor_parameters),
            "compact_actor_reduction_percent": (reduction),
        }

    difference = abs(reductions[0] - reductions[1])

    payload["cross_domain"] = {
        "absolute_reduction_difference_percentage_points": (difference),
        "compactness_consistent": True,
        "compactness_consistency_basis": ("descriptive_similarity_only"),
        "compactness_described_as_highly_similar": True,
        "formal_consistency_test_performed": False,
    }

    return payload


def build_architecture_reuse() -> dict[str, Any]:
    """Record the supported cross-domain reuse boundary."""

    scorecard = [
        {
            "property": "same_4_qubit_pqc_core",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_2_layer_circuit",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_16_pqc_parameters",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_3d_action_output",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_ppo_protocol",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_20000_step_budget",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_training_seeds",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_evaluation_seed_set",
            "driving": True,
            "robotics": True,
            "cross_domain": "reused",
        },
        {
            "property": "same_trained_weights",
            "driving": False,
            "robotics": False,
            "cross_domain": "not_claimed",
        },
        {
            "property": "same_observation_dimension",
            "driving": False,
            "robotics": False,
            "cross_domain": "domain_specific",
        },
        {
            "property": "robust_qml_target_reach",
            "driving": False,
            "robotics": False,
            "cross_domain": "not_supported",
        },
        {
            "property": "compactness_approximately_95_percent",
            "driving": True,
            "robotics": True,
            "cross_domain": "consistent",
        },
    ]

    return {
        "scorecard": scorecard,
        "architecture_reused": True,
        "shared_qml_core": True,
        "shared_qubit_count": 4,
        "shared_pqc_layers": 2,
        "shared_quantum_parameters": 16,
        "shared_action_dimension": 3,
        "shared_ppo_protocol": True,
        "shared_training_seed_set": True,
        "training_seeds": list(SEEDS),
        "shared_evaluation_seed_set": True,
        "shared_environment_step_budget": 20_000,
        "driving_observation_dimension": 4,
        "robotics_observation_dimension": 6,
        "same_observation_dimension": False,
        "same_trained_weights_across_domains": False,
        "zero_shot_transfer_performed": False,
        "transfer_learning_performed": False,
        "universal_policy_claimed": False,
        "shared_sprint1_latent_used_directly_in_rl": False,
    }


def write_csv(
    records: list[CrossDomainMethodRecord],
) -> None:
    """Write six cross-domain method rows."""

    fieldnames = [
        "domain",
        "method",
        "actor_parameters",
        "target_reach_count",
        "target_total",
        "normalized_auc_mean",
        "normalized_auc_sd",
        "best_progress_mean",
        "best_progress_sd",
        "final_progress_mean",
        "final_progress_sd",
    ]

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            payload = record_to_dict(record)

            writer.writerow({field: payload[field] for field in fieldnames})


def write_markdown(
    *,
    payload: dict[str, Any],
) -> None:
    """Write proposal-readable Sprint 4.13 evidence."""

    lines = [
        "# Sprint 4.13 — Cross-Domain RL Comparison",
        "",
        (
            "Sprint 4.13 performs no new RL training. It compares "
            "the frozen Full PPO, matched-classical, and hybrid-QML "
            "evidence across the autonomous-driving and robotics proxies."
        ),
        "",
        "## Target Reach",
        "",
        "| Domain | Full PPO | Matched Classical | Hybrid QML |",
        "|---|---:|---:|---:|",
    ]

    for domain in DOMAINS:
        domain_records = {
            record["method"]: record
            for record in payload["method_records"]
            if record["domain"] == domain
        }

        lines.append(
            "| "
            f"{DISPLAY_NAMES[domain]} | "
            f"{domain_records['full_ppo']['target_reach_count']}/3 | "
            f"{domain_records['matched_classical']['target_reach_count']}/3 | "
            f"{domain_records['hybrid_qml']['target_reach_count']}/3 |"
        )

    lines.extend(
        [
            "",
            "## Normalized Learning-Curve AUC",
            "",
            (
                "| Domain | Method | Actor Params | "
                "Normalized AUC | Best Progress | Final Progress |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for record in payload["method_records"]:
        lines.append(
            "| "
            f"{DISPLAY_NAMES[record['domain']]} | "
            f"{record['method']} | "
            f"{record['actor_parameters']} | "
            f"{record['normalized_auc_mean']:.6f} ± "
            f"{record['normalized_auc_sd']:.6f} | "
            f"{record['best_progress_mean']:.6f} ± "
            f"{record['best_progress_sd']:.6f} | "
            f"{record['final_progress_mean']:.6f} ± "
            f"{record['final_progress_sd']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Matched-Budget Representation Comparison",
            "",
            (
                "| Domain | Matched Params | QML Params | "
                "Matched AUC | QML AUC | Δ Matched−QML | Direction |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for comparison in payload["matched_representation_comparison"]:
        lines.append(
            "| "
            f"{DISPLAY_NAMES[comparison['domain']]} | "
            f"{comparison['matched_actor_parameters']} | "
            f"{comparison['qml_actor_parameters']} | "
            f"{comparison['matched_auc_mean']:.6f} | "
            f"{comparison['qml_auc_mean']:.6f} | "
            f"{comparison['matched_minus_qml_auc']:+.6f} | "
            f"{comparison['direction']} |"
        )

    compactness = payload["compactness"]

    lines.extend(
        [
            "",
            "## Actor Compactness",
            "",
            "| Domain | Full PPO | Compact Actor | Reduction |",
            "|---|---:|---:|---:|",
        ]
    )

    for domain in DOMAINS:
        row = compactness[domain]

        lines.append(
            "| "
            f"{DISPLAY_NAMES[domain]} | "
            f"{row['full_ppo_actor_parameters']} | "
            f"{row['hybrid_qml_actor_parameters']} | "
            f"{row['compact_actor_reduction_percent']:.6f}% |"
        )

    architecture = payload["architecture_reuse"]

    lines.extend(
        [
            "",
            "## Architecture Reuse Scorecard",
            "",
            "| Property | Driving | Robotics | Cross-Domain |",
            "|---|---|---|---|",
        ]
    )

    scorecard_labels = {
        "same_4_qubit_pqc_core": "Same 4-qubit PQC core",
        "same_2_layer_circuit": "Same 2-layer circuit",
        "same_16_pqc_parameters": "Same 16 PQC parameters",
        "same_3d_action_output": "Same 3-D action output",
        "same_ppo_protocol": "Same PPO protocol",
        "same_20000_step_budget": "Same 20k budget",
        "same_training_seeds": "Same training seeds",
        "same_evaluation_seed_set": "Same evaluation seeds",
        "same_trained_weights": "Same trained weights",
        "same_observation_dimension": "Same observation dimension",
        "robust_qml_target_reach": "Robust QML target reach",
        "compactness_approximately_95_percent": "Compactness ~95%",
    }

    cross_domain_labels = {
        "reused": "Reused",
        "not_claimed": "Not claimed",
        "domain_specific": "Domain-specific",
        "not_supported": "Not supported",
        "consistent": "Consistent",
    }

    for row in architecture["scorecard"]:
        lines.append(
            "| "
            f"{scorecard_labels[row['property']]} | "
            f"{'Yes' if row['driving'] else 'No'} | "
            f"{'Yes' if row['robotics'] else 'No'} | "
            f"{cross_domain_labels[row['cross_domain']]} |"
        )

    conclusion = payload["cross_domain_conclusion"]

    lines.extend(
        [
            "",
            "## Cross-Domain Interpretation",
            "",
            (
                "The common hybrid policy architecture was reused "
                "across both proxy domains, but the matched-budget "
                "representation effect was not directionally consistent."
            ),
            "",
            (
                f"Driving matched-budget AUC direction: "
                f"`{conclusion['matched_auc_direction_driving']}`."
            ),
            "",
            (
                f"Robotics matched-budget AUC direction: "
                f"`{conclusion['matched_auc_direction_robotics']}`."
            ),
            "",
            (
                "Therefore, the pilot does not support a robust "
                "cross-domain QML learning advantage or a robust "
                "cross-domain matched-classical advantage."
            ),
            "",
            "## Domain-Dependent Representation Behavior",
            "",
            conclusion["domain_dependent_behavior_statement"],
            "",
            (
                "Best normalized progress favored "
                f"`{conclusion['best_progress_direction_driving']}` "
                "in driving and "
                f"`{conclusion['best_progress_direction_robotics']}` "
                "in robotics."
            ),
            "",
            (
                "Final normalized progress favored "
                f"`{conclusion['final_progress_direction_driving']}` "
                "in driving and "
                f"`{conclusion['final_progress_direction_robotics']}` "
                "in robotics."
            ),
            "",
            (
                "These results are interpreted as domain-dependent "
                "behavior under the tested proxy environments, not as "
                "a universal property of either representation."
            ),
            "",
            "## Seed Variability",
            "",
            (
                "Hybrid-QML normalized-AUC sample SD was "
                f"{conclusion['qml_auc_sd_driving']:.6f} in driving "
                "and "
                f"{conclusion['qml_auc_sd_robotics']:.6f} in robotics."
            ),
            "",
            (
                "Under the tested protocol, the observed hybrid-QML "
                "learning trajectories showed greater seed-to-seed "
                "dispersion in the driving proxy than in the robotics "
                "proxy."
            ),
            "",
            (
                "This is a descriptive observation from three principal "
                "seeds per domain and is not claimed as a general "
                "property of autonomous driving versus robotics."
            ),
            "",
            "## Scientific Boundary",
            "",
            (
                "Cross-domain architectural reuse means that the same "
                "conceptual hybrid policy framework, PQC design, PPO "
                "protocol, budget, seed set, and validation methodology "
                "were used in both domains."
            ),
            "",
            (
                "It does not mean that the same trained weights were "
                "shared across domains."
            ),
            "",
            (
                "No transfer learning, zero-shot transfer, universal-policy "
                "claim, quantum speedup, or quantum hardware advantage is claimed."
            ),
            "",
            (
                "Cross-domain comparisons use normalized target-relative "
                "metrics rather than raw rewards because the two environment "
                "reward functions differ."
            ),
            "",
            (
                "No formal significance testing is performed because "
                "there are only three principal seeds per domain."
            ),
            "",
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build Sprint 4.13 cross-domain evidence."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_efficiency = load_json(SAMPLE_EFFICIENCY_PATH)

    ablation = load_json(ABLATION_PATH)

    validation = load_json(VALIDATION_PATH)

    targets = load_json(TARGET_PATH)

    validate_frozen_inputs(
        sample_efficiency=(sample_efficiency),
        ablation=ablation,
        validation=validation,
        targets=targets,
    )

    method_records: list[CrossDomainMethodRecord] = []

    for domain in DOMAINS:
        for method in METHODS:
            method_records.append(
                build_method_record(
                    domain=domain,
                    method=method,
                    sample_efficiency=(sample_efficiency),
                    ablation=ablation,
                )
            )

    validate_against_three_seed_validation(
        records=method_records,
        validation=validation,
    )

    matched_comparisons = [
        build_matched_comparison(
            records=method_records,
            domain=domain,
        )
        for domain in DOMAINS
    ]

    comparison_by_domain = {
        comparison.domain: comparison for comparison in matched_comparisons
    }

    driving_direction = comparison_by_domain["autonomous_driving"].direction

    robotics_direction = comparison_by_domain["robotics"].direction

    direction_consistent = driving_direction == robotics_direction

    driving_best_direction = matched_metric_direction(
        records=method_records,
        domain="autonomous_driving",
        metric="best",
    )

    robotics_best_direction = matched_metric_direction(
        records=method_records,
        domain="robotics",
        metric="best",
    )

    driving_final_direction = matched_metric_direction(
        records=method_records,
        domain="autonomous_driving",
        metric="final",
    )

    robotics_final_direction = matched_metric_direction(
        records=method_records,
        domain="robotics",
        metric="final",
    )

    domain_dependent_representation_behavior = driving_direction != robotics_direction

    domain_dependent_behavior_statement = (
        "At the matched compact actor budget, the representation "
        "effect was domain dependent. Hybrid QML achieved higher "
        "mean normalized learning-curve AUC than the matched "
        "classical control in the autonomous-driving proxy, while "
        "the matched classical control achieved higher mean "
        "normalized AUC in robotics."
    )

    if driving_best_direction == robotics_best_direction:
        best_progress_interpretation = f"{driving_best_direction}_in_both_domains"
    else:
        best_progress_interpretation = "mixed"

    if driving_final_direction == robotics_final_direction:
        final_progress_interpretation = f"{driving_final_direction}_in_both_domains"
    else:
        final_progress_interpretation = "mixed"

    compactness = build_compactness(records=method_records)

    architecture_reuse = build_architecture_reuse()

    domain_payload = {
        domain: {
            "methods": {
                method: record_to_dict(
                    record_lookup(
                        records=method_records,
                        domain=domain,
                        method=method,
                    )
                )
                for method in METHODS
            }
        }
        for domain in DOMAINS
    }

    payload = {
        "sprint": "4.13",
        "title": ("Cross-Domain RL Comparison"),
        "new_training_performed": False,
        "new_ppo_training_performed": False,
        "new_qml_training_performed": False,
        "new_matched_classical_training_performed": False,
        "principal_seeds": list(SEEDS),
        "domains": domain_payload,
        "method_records": [record_to_dict(record) for record in method_records],
        "matched_representation_comparison": [
            matched_comparison_to_dict(comparison) for comparison in matched_comparisons
        ],
        "compactness": compactness,
        "architecture_reuse": (architecture_reuse),
        "cross_domain_conclusion": {
            "qml_auc_sd_driving": (
                record_lookup(
                    records=method_records,
                    domain="autonomous_driving",
                    method="hybrid_qml",
                ).normalized_auc_sd
            ),
            "qml_auc_sd_robotics": (
                record_lookup(
                    records=method_records,
                    domain="robotics",
                    method="hybrid_qml",
                ).normalized_auc_sd
            ),
            "qml_auc_dispersion_larger_in_driving": (
                record_lookup(
                    records=method_records,
                    domain="autonomous_driving",
                    method="hybrid_qml",
                ).normalized_auc_sd
                > record_lookup(
                    records=method_records,
                    domain="robotics",
                    method="hybrid_qml",
                ).normalized_auc_sd
            ),
            "seed_variability_interpretation_scope": (
                "descriptive_under_tested_protocol_only"
            ),
            "matched_auc_direction_driving": (driving_direction),
            "matched_auc_direction_robotics": (robotics_direction),
            "matched_auc_direction_consistent_across_domains": (direction_consistent),
            "robust_cross_domain_qml_advantage": (
                direction_consistent and driving_direction == "hybrid_qml"
            ),
            "robust_cross_domain_matched_classical_advantage": (
                direction_consistent and driving_direction == "matched_classical"
            ),
            "domain_dependent_representation_behavior": (
                domain_dependent_representation_behavior
            ),
            "domain_dependent_behavior_statement": (
                domain_dependent_behavior_statement
            ),
            "best_progress_direction_driving": (driving_best_direction),
            "best_progress_direction_robotics": (robotics_best_direction),
            "best_progress_direction_consistent_across_domains": (
                driving_best_direction == robotics_best_direction
            ),
            "best_progress_cross_domain_interpretation": (best_progress_interpretation),
            "final_progress_direction_driving": (driving_final_direction),
            "final_progress_direction_robotics": (robotics_final_direction),
            "final_progress_direction_consistent_across_domains": (
                driving_final_direction == robotics_final_direction
            ),
            "final_progress_cross_domain_interpretation": (
                final_progress_interpretation
            ),
            "architecture_reused": True,
            "shared_trained_weights": False,
        },
        "scientific_boundary": {
            "raw_rewards_compared_across_domains": False,
            "normalized_metrics_used_for_cross_domain_comparison": True,
            "formal_significance_test_performed": False,
            "seed_variability_analysis_is_descriptive_only": True,
            "seed_variability_generalized_beyond_tested_protocol": False,
            "transfer_learning_performed": False,
            "zero_shot_transfer_performed": False,
            "universal_policy_claimed": False,
            "robust_cross_domain_qml_advantage_claimed": False,
            "quantum_speedup_claimed": False,
            "quantum_hardware_advantage_claimed": False,
        },
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(method_records)

    write_markdown(payload=payload)

    print()
    print("=" * 52)

    print(" SPRINT 4.13 CROSS-DOMAIN RL COMPARISON")

    print("=" * 52)

    print()

    for domain in DOMAINS:
        print(DISPLAY_NAMES[domain])

        for method in METHODS:
            record = record_lookup(
                records=method_records,
                domain=domain,
                method=method,
            )

            print(f"  {method}:")

            print(
                "    actor parameters:",
                record.actor_parameters,
            )

            print(
                "    target reach:",
                record.target_reach_count,
                "/",
                record.target_total,
            )

            print(
                "    normalized AUC:",
                record.normalized_auc_mean,
                "±",
                record.normalized_auc_sd,
            )

        comparison = comparison_by_domain[domain]

        print(
            "  Matched vs QML AUC delta:",
            comparison.matched_minus_qml_auc,
        )

        print(
            "  Matched vs QML AUC direction:",
            comparison.direction,
        )

        print()

    print(
        "Cross-domain AUC direction consistent:",
        direction_consistent,
    )

    print(
        "Robust cross-domain QML advantage:",
        payload["cross_domain_conclusion"]["robust_cross_domain_qml_advantage"],
    )

    print(
        "Robust cross-domain matched-classical advantage:",
        payload["cross_domain_conclusion"][
            "robust_cross_domain_matched_classical_advantage"
        ],
    )

    print(
        "Architecture reused:",
        architecture_reuse["architecture_reused"],
    )

    print(
        "Shared trained weights:",
        architecture_reuse["same_trained_weights_across_domains"],
    )

    print(
        "New training:",
        False,
    )

    print()
    print("SPRINT 4.13 CROSS-DOMAIN ANALYSIS: BUILT")

    print(
        "JSON:",
        JSON_OUTPUT,
    )

    print(
        "CSV:",
        CSV_OUTPUT,
    )

    print(
        "Markdown:",
        MARKDOWN_OUTPUT,
    )


if __name__ == "__main__":
    main()
