"""Independent verification for Sprint 4.13 cross-domain RL comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_cross_domain import (
    parameter_reduction_percent,
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

CROSS_DOMAIN_PATH = (
    Path("results") / "rl" / "cross-domain" / "sprint4-cross-domain.json"
)

CSV_PATH = Path("results") / "rl" / "cross-domain" / "sprint4-cross-domain.csv"

MARKDOWN_PATH = Path("results") / "rl" / "cross-domain" / "sprint4-cross-domain.md"

FIGURES = (
    Path("figures") / "rl" / "cross-domain-normalized-auc.png",
    Path("figures") / "rl" / "cross-domain-target-reach.png",
    Path("figures") / "rl" / "cross-domain-parameter-compactness.png",
    Path("figures") / "rl" / "cross-domain-matched-delta.png",
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SEEDS = (
    42,
    123,
    456,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    """Require numerical equality within tight floating tolerance."""

    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{label} mismatch: " f"{actual} != {expected}")


def method_record(
    *,
    payload: dict[str, Any],
    domain: str,
    method: str,
) -> dict[str, Any]:
    """Return exactly one stored Sprint 4.13 method record."""

    matches = [
        record
        for record in payload["method_records"]
        if (record["domain"] == domain and record["method"] == method)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one {domain} {method} record")

    return matches[0]


def matched_comparison(
    *,
    payload: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    """Return one stored matched representation comparison."""

    matches = [
        record
        for record in payload["matched_representation_comparison"]
        if record["domain"] == domain
    ]

    if len(matches) != 1:
        raise RuntimeError(f"expected one matched comparison for {domain}")

    return matches[0]


def main() -> None:
    """Independently verify Sprint 4.13 evidence."""

    sample_efficiency = load_json(SAMPLE_EFFICIENCY_PATH)

    ablation = load_json(ABLATION_PATH)

    validation = load_json(VALIDATION_PATH)

    targets = load_json(TARGET_PATH)

    cross_domain = load_json(CROSS_DOMAIN_PATH)

    if cross_domain["sprint"] != "4.13":
        raise RuntimeError("unexpected Sprint 4.13 artifact")

    if cross_domain["new_training_performed"] is not False:
        raise RuntimeError("Sprint 4.13 must remain analysis-only")

    if cross_domain["new_ppo_training_performed"] is not False:
        raise RuntimeError("Sprint 4.13 cannot retrain PPO")

    if cross_domain["new_qml_training_performed"] is not False:
        raise RuntimeError("Sprint 4.13 cannot retrain QML")

    if cross_domain["new_matched_classical_training_performed"] is not False:
        raise RuntimeError("Sprint 4.13 cannot retrain matched classical")

    if tuple(int(seed) for seed in cross_domain["principal_seeds"]) != SEEDS:
        raise RuntimeError("Sprint 4.13 principal seed mismatch")

    if sample_efficiency["new_training_performed"] is not False:
        raise RuntimeError("Sprint 4.11 freeze invalid")

    if ablation["new_qml_training_performed"] is not False:
        raise RuntimeError("Sprint 4.12 QML freeze invalid")

    if ablation["parameter_matched_ablation"] is not True:
        raise RuntimeError("Sprint 4.12 parameter-match boundary invalid")

    if validation["principal_training_performed"] is not False:
        raise RuntimeError("Sprint 4.10 freeze invalid")

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("historical PPO target freeze invalid")

    reconstructed_directions: dict[
        str,
        str,
    ] = {}

    reconstructed_deltas: dict[
        str,
        float,
    ] = {}

    for domain in DOMAINS:
        sprint411_ppo = sample_efficiency["domains"][domain]["policies"]["ppo"][
            "aggregate"
        ]

        sprint411_qml = sample_efficiency["domains"][domain]["policies"]["qml"][
            "aggregate"
        ]

        sprint412_matched = ablation["domains"][domain]["methods"]["matched_classical"]

        stored_ppo = method_record(
            payload=cross_domain,
            domain=domain,
            method="full_ppo",
        )

        stored_matched = method_record(
            payload=cross_domain,
            domain=domain,
            method="matched_classical",
        )

        stored_qml = method_record(
            payload=cross_domain,
            domain=domain,
            method="hybrid_qml",
        )

        source_ppo_actor_parameters = int(sprint411_ppo["actor_parameters"])

        source_qml_actor_parameters = int(sprint411_qml["actor_parameters"])

        source_matched_actor_parameters = int(sprint412_matched["actor_parameters"])

        if int(stored_ppo["actor_parameters"]) != source_ppo_actor_parameters:
            raise RuntimeError(f"{domain} PPO actor count mismatch")

        if int(stored_matched["actor_parameters"]) != source_matched_actor_parameters:
            raise RuntimeError(f"{domain} matched actor count mismatch")

        if int(stored_qml["actor_parameters"]) != source_qml_actor_parameters:
            raise RuntimeError(f"{domain} QML actor count mismatch")

        if source_matched_actor_parameters != source_qml_actor_parameters:
            raise RuntimeError(f"{domain} frozen matched/QML actor equality failed")

        if int(stored_matched["actor_parameters"]) != int(
            stored_qml["actor_parameters"]
        ):
            raise RuntimeError(f"{domain} stored matched/QML actor equality failed")

        validation_summary = validation["domains"][domain]["summary"]

        if int(stored_ppo["target_reach_count"]) != int(
            validation_summary["classical_target_reach_count"]
        ):
            raise RuntimeError(f"{domain} PPO target reach mismatch")

        if int(stored_qml["target_reach_count"]) != int(
            validation_summary["qml_target_reach_count"]
        ):
            raise RuntimeError(f"{domain} QML target reach mismatch")

        if int(stored_matched["target_reach_count"]) != int(
            sprint412_matched["target_reach_count"]
        ):
            raise RuntimeError(f"{domain} matched target reach mismatch")

        source_records = (
            (
                stored_ppo,
                sprint411_ppo,
                "PPO",
            ),
            (
                stored_qml,
                sprint411_qml,
                "QML",
            ),
            (
                stored_matched,
                sprint412_matched,
                "matched",
            ),
        )

        for (
            stored,
            source,
            label,
        ) in source_records:
            assert_close(
                float(stored["normalized_auc_mean"]),
                float(source["normalized_auc"]["mean"]),
                label=(f"{domain} {label} AUC mean"),
            )

            assert_close(
                float(stored["normalized_auc_sd"]),
                float(source["normalized_auc"]["sample_standard_deviation"]),
                label=(f"{domain} {label} AUC SD"),
            )

            assert_close(
                float(stored["best_progress_mean"]),
                float(source["best_normalized_progress"]["mean"]),
                label=(f"{domain} {label} best mean"),
            )

            assert_close(
                float(stored["best_progress_sd"]),
                float(source["best_normalized_progress"]["sample_standard_deviation"]),
                label=(f"{domain} {label} best SD"),
            )

            assert_close(
                float(stored["final_progress_mean"]),
                float(source["final_normalized_progress"]["mean"]),
                label=(f"{domain} {label} final mean"),
            )

            assert_close(
                float(stored["final_progress_sd"]),
                float(source["final_normalized_progress"]["sample_standard_deviation"]),
                label=(f"{domain} {label} final SD"),
            )

        matched_auc = float(sprint412_matched["normalized_auc"]["mean"])

        qml_auc = float(sprint411_qml["normalized_auc"]["mean"])

        delta = matched_auc - qml_auc

        direction = representation_direction(
            matched_value=matched_auc,
            qml_value=qml_auc,
        )

        reconstructed_deltas[domain] = delta

        reconstructed_directions[domain] = direction

        stored_comparison = matched_comparison(
            payload=cross_domain,
            domain=domain,
        )

        assert_close(
            float(stored_comparison["matched_minus_qml_auc"]),
            delta,
            label=(f"{domain} matched-QML AUC delta"),
        )

        if stored_comparison["direction"] != direction:
            raise RuntimeError(f"{domain} representation direction mismatch")

        compactness = cross_domain["compactness"][domain]

        expected_reduction = parameter_reduction_percent(
            full_actor_parameters=(source_ppo_actor_parameters),
            compact_actor_parameters=(source_qml_actor_parameters),
        )

        assert_close(
            float(compactness["compact_actor_reduction_percent"]),
            expected_reduction,
            label=(f"{domain} compactness"),
        )

    driving_direction = reconstructed_directions["autonomous_driving"]

    robotics_direction = reconstructed_directions["robotics"]

    if driving_direction != "hybrid_qml":
        raise RuntimeError("driving frozen evidence must favor hybrid QML on mean AUC")

    if robotics_direction != "matched_classical":
        raise RuntimeError(
            "robotics frozen evidence must favor matched classical on mean AUC"
        )

    if not (reconstructed_deltas["autonomous_driving"] < 0.0):
        raise RuntimeError("driving matched-QML AUC sign mismatch")

    if not (reconstructed_deltas["robotics"] > 0.0):
        raise RuntimeError("robotics matched-QML AUC sign mismatch")

    conclusion = cross_domain["cross_domain_conclusion"]

    if conclusion["matched_auc_direction_consistent_across_domains"] is not False:
        raise RuntimeError("cross-domain direction consistency must be false")

    if conclusion["robust_cross_domain_qml_advantage"] is not False:
        raise RuntimeError("robust cross-domain QML advantage must be false")

    if conclusion["robust_cross_domain_matched_classical_advantage"] is not False:
        raise RuntimeError(
            "robust cross-domain matched-classical advantage must be false"
        )

    if conclusion["architecture_reused"] is not True:
        raise RuntimeError("architecture reuse must be supported")

    if conclusion["shared_trained_weights"] is not False:
        raise RuntimeError("shared trained weights must remain false")

    architecture = cross_domain["architecture_reuse"]

    expected_architecture = {
        "architecture_reused": True,
        "shared_qml_core": True,
        "shared_qubit_count": 4,
        "shared_pqc_layers": 2,
        "shared_quantum_parameters": 16,
        "shared_action_dimension": 3,
        "shared_ppo_protocol": True,
        "shared_training_seed_set": True,
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

    for (
        key,
        expected,
    ) in expected_architecture.items():
        if architecture[key] != expected:
            raise RuntimeError(f"architecture reuse mismatch: {key}")

    if tuple(int(seed) for seed in architecture["training_seeds"]) != SEEDS:
        raise RuntimeError("architecture training seed metadata mismatch")

    boundary = cross_domain["scientific_boundary"]

    expected_boundary = {
        "raw_rewards_compared_across_domains": False,
        "normalized_metrics_used_for_cross_domain_comparison": True,
        "formal_significance_test_performed": False,
        "transfer_learning_performed": False,
        "zero_shot_transfer_performed": False,
        "universal_policy_claimed": False,
        "robust_cross_domain_qml_advantage_claimed": False,
        "quantum_speedup_claimed": False,
        "quantum_hardware_advantage_claimed": False,
    }

    for (
        key,
        expected,
    ) in expected_boundary.items():
        if boundary[key] != expected:
            raise RuntimeError(f"scientific boundary mismatch: {key}")

    if not CSV_PATH.exists():
        raise RuntimeError("cross-domain CSV missing")

    if not MARKDOWN_PATH.exists():
        raise RuntimeError("cross-domain Markdown missing")

    figures_verified = 0

    for path in FIGURES:
        if not path.exists():
            raise RuntimeError(f"missing figure: {path}")

        if path.stat().st_size <= 0:
            raise RuntimeError(f"empty figure: {path}")

        figures_verified += 1

    print()
    print("=" * 54)

    print(" SPRINT 4.13 CROSS-DOMAIN RL VERIFIER")

    print("=" * 54)

    print()

    print(
        "Driving AUC direction:",
        driving_direction,
    )

    print(
        "Driving matched-QML AUC delta:",
        reconstructed_deltas["autonomous_driving"],
    )

    print()

    print(
        "Robotics AUC direction:",
        robotics_direction,
    )

    print(
        "Robotics matched-QML AUC delta:",
        reconstructed_deltas["robotics"],
    )

    print()

    print(
        "Cross-domain direction consistent:",
        False,
    )

    print(
        "Robust cross-domain QML advantage:",
        False,
    )

    print(
        "Robust cross-domain matched-classical advantage:",
        False,
    )

    print(
        "Architecture reused:",
        True,
    )

    print(
        "Shared trained weights:",
        False,
    )

    print(
        "Figures verified:",
        figures_verified,
    )

    print(
        "New training:",
        False,
    )

    print()

    print("SPRINT 4.13 CROSS-DOMAIN RL COMPARISON: PASS")


if __name__ == "__main__":
    main()
