"""Independent verification for Sprint 4.12 matched-budget ablation."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.rl_sample_efficiency import (
    summarize_learning_curve,
)

RESULTS_DIRECTORY = Path("results") / "rl" / "ablation"

QML_DIRECTORY = Path("results") / "rl" / "qml"

PPO_DIRECTORY = Path("results") / "rl" / "ppo"

ENVIRONMENT_AUDIT_PATH = (
    Path("results") / "rl" / "environment-audit" / "sprint4-environment-audit.json"
)

TARGET_PATH = PPO_DIRECTORY / "sprint4-ppo-targets.json"

ANALYSIS_PATH = RESULTS_DIRECTORY / "sprint4-classical-vs-qml-ablation.json"

CSV_PATH = RESULTS_DIRECTORY / "sprint4-classical-vs-qml-ablation.csv"

MARKDOWN_PATH = RESULTS_DIRECTORY / "sprint4-classical-vs-qml-ablation.md"

FIGURES = (
    Path("figures") / "rl" / "driving-matched-ablation.png",
    Path("figures") / "rl" / "robotics-matched-ablation.png",
    Path("figures") / "rl" / "matched-ablation-auc.png",
    Path("figures") / "rl" / "matched-budget-performance.png",
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

RANDOM_REFERENCE_KEYS = {
    "autonomous_driving": "driving_random",
    "robotics": "robotics_random",
}

EXPECTED_COUNTS = {
    "autonomous_driving": {
        "matched_actor": 54,
        "qml_actor": 54,
        "matched_critic": 1249,
        "qml_critic": 1249,
        "matched_total": 1303,
        "qml_total": 1303,
    },
    "robotics": {
        "matched_actor": 62,
        "qml_actor": 62,
        "matched_critic": 1313,
        "qml_critic": 1313,
        "matched_total": 1375,
        "qml_total": 1375,
    },
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def target_for_seed(
    *,
    targets: dict[str, Any],
    domain: str,
    seed: int,
) -> float:
    """Return the frozen Sprint 4.5 target."""

    records = targets["domains"][domain]["seeds"]

    matches = [record for record in records if int(record["seed"]) == seed]

    if len(matches) != 1:
        raise RuntimeError(f"expected one target for {domain} seed {seed}")

    return float(matches[0]["target_evaluation_reward"])


def random_reference_for_domain(
    *,
    audit: dict[str, Any],
    domain: str,
) -> float:
    """Return the frozen Sprint 4.4 random reference."""

    key = RANDOM_REFERENCE_KEYS[domain]

    summary = audit["summaries"][key]

    if summary["domain"] != domain:
        raise RuntimeError("random-reference domain mismatch")

    if summary["policy"] != "random":
        raise RuntimeError("random-reference policy mismatch")

    return float(summary["mean_reward"])


def matched_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return one matched-classical principal run."""

    return RESULTS_DIRECTORY / f"matched-classical-{domain}-seed-{seed}.json"


def matched_repro_path(
    *,
    domain: str,
) -> Path:
    """Return one seed-42 matched reproduction."""

    return RESULTS_DIRECTORY / f"matched-classical-{domain}-seed-42-repro.json"


def qml_path(
    *,
    domain: str,
    seed: int,
) -> Path:
    """Return one frozen QML principal run."""

    return QML_DIRECTORY / f"{domain}-seed-{seed}.json"


def first_crossing(
    *,
    evaluations: list[dict[str, Any]],
    target: float,
) -> int | None:
    """Return first target crossing."""

    for row in evaluations:
        if float(row["mean_reward"]) >= target:
            return int(row["environment_steps"])

    return None


def sample_mean_sd(
    values: list[float],
) -> tuple[
    float,
    float,
]:
    """Return arithmetic mean and sample SD."""

    if not values:
        raise ValueError("values cannot be empty")

    return (
        float(statistics.mean(values)),
        float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    )


def assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    """Require numerically equivalent stored and reconstructed values."""

    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{label} mismatch: " f"{actual} != {expected}")


def verify_reproduction(
    *,
    domain: str,
) -> None:
    """Verify exact seed-42 matched-classical reproducibility."""

    principal = load_json(
        matched_path(
            domain=domain,
            seed=42,
        )
    )

    reproduction = load_json(
        matched_repro_path(
            domain=domain,
        )
    )

    principal_rows = principal["evaluations"]

    reproduction_rows = reproduction["evaluations"]

    if len(principal_rows) != 21:
        raise RuntimeError("principal reproduction source must contain 21 evaluations")

    if len(reproduction_rows) != 21:
        raise RuntimeError("reproduction must contain 21 evaluations")

    for index, (
        principal_row,
        reproduction_row,
    ) in enumerate(
        zip(
            principal_rows,
            reproduction_rows,
            strict=True,
        )
    ):
        for key in (
            "environment_steps",
            "completed_training_episodes",
            "mean_reward",
            "reward_standard_deviation",
            "success_rate",
        ):
            if principal_row[key] != reproduction_row[key]:
                raise RuntimeError(
                    f"{domain} seed-42 reproduction mismatch "
                    f"at checkpoint {index}: {key}"
                )

    if principal["parameter_counts"] != reproduction["parameter_counts"]:
        raise RuntimeError(f"{domain} seed-42 parameter reproduction mismatch")


def main() -> None:
    """Independently verify Sprint 4.12 evidence."""

    targets = load_json(TARGET_PATH)

    audit = load_json(ENVIRONMENT_AUDIT_PATH)

    analysis = load_json(ANALYSIS_PATH)

    if targets["qml_results_seen"] is not False:
        raise RuntimeError("frozen PPO target history is invalid")

    reconstructed: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    total_matched_reaches = 0
    total_qml_reaches = 0

    principal_matched_runs = 0
    principal_qml_runs = 0

    for domain in DOMAINS:
        random_reference = random_reference_for_domain(
            audit=audit,
            domain=domain,
        )

        matched_aucs: list[float] = []
        qml_aucs: list[float] = []

        matched_best: list[float] = []
        qml_best: list[float] = []

        matched_final: list[float] = []
        qml_final: list[float] = []

        auc_deltas: list[float] = []
        best_deltas: list[float] = []
        final_deltas: list[float] = []

        matched_reaches = 0
        qml_reaches = 0

        for seed in SEEDS:
            target = target_for_seed(
                targets=targets,
                domain=domain,
                seed=seed,
            )

            matched = load_json(
                matched_path(
                    domain=domain,
                    seed=seed,
                )
            )

            qml = load_json(
                qml_path(
                    domain=domain,
                    seed=seed,
                )
            )

            principal_matched_runs += 1
            principal_qml_runs += 1

            if matched["domain"] != domain:
                raise RuntimeError("matched domain mismatch")

            if int(matched["seed"]) != seed:
                raise RuntimeError("matched seed mismatch")

            if qml["domain"] != domain:
                raise RuntimeError("QML domain mismatch")

            if int(qml["seed"]) != seed:
                raise RuntimeError("QML seed mismatch")

            matched_rows = matched["evaluations"]

            qml_rows = qml["evaluations"]

            if len(matched_rows) != 21:
                raise RuntimeError("matched run must contain 21 evaluations")

            if len(qml_rows) != 21:
                raise RuntimeError("QML run must contain 21 evaluations")

            expected_steps = list(
                range(
                    0,
                    20_001,
                    1_000,
                )
            )

            matched_steps = [int(row["environment_steps"]) for row in matched_rows]

            qml_steps = [int(row["environment_steps"]) for row in qml_rows]

            if matched_steps != expected_steps:
                raise RuntimeError("matched evaluation schedule mismatch")

            if qml_steps != expected_steps:
                raise RuntimeError("QML evaluation schedule mismatch")

            matched_counts = matched["parameter_counts"]

            expected_counts = EXPECTED_COUNTS[domain]

            if (
                int(matched_counts["actor_parameters"])
                != expected_counts["matched_actor"]
            ):
                raise RuntimeError(f"{domain} matched actor count mismatch")

            if (
                int(matched_counts["critic_parameters"])
                != expected_counts["matched_critic"]
            ):
                raise RuntimeError(f"{domain} matched critic count mismatch")

            if (
                int(matched_counts["total_parameters"])
                != expected_counts["matched_total"]
            ):
                raise RuntimeError(f"{domain} matched total count mismatch")

            if int(qml["actor_parameters"]) != expected_counts["qml_actor"]:
                raise RuntimeError(f"{domain} QML actor count mismatch")

            if int(qml["critic_parameters"]) != expected_counts["qml_critic"]:
                raise RuntimeError(f"{domain} QML critic count mismatch")

            if int(qml["total_parameters"]) != expected_counts["qml_total"]:
                raise RuntimeError(f"{domain} QML total count mismatch")

            if expected_counts["matched_actor"] != expected_counts["qml_actor"]:
                raise RuntimeError("actor parameter matching failed")

            if expected_counts["matched_critic"] != expected_counts["qml_critic"]:
                raise RuntimeError("critic parameter matching failed")

            if expected_counts["matched_total"] != expected_counts["qml_total"]:
                raise RuntimeError("total parameter matching failed")

            matched_rewards = [float(row["mean_reward"]) for row in matched_rows]

            qml_rewards = [float(row["mean_reward"]) for row in qml_rows]

            matched_summary = summarize_learning_curve(
                steps=matched_steps,
                rewards=matched_rewards,
                random_reference=(random_reference),
                target_reward=(target),
            )

            qml_summary = summarize_learning_curve(
                steps=qml_steps,
                rewards=qml_rewards,
                random_reference=(random_reference),
                target_reward=(target),
            )

            matched_crossing = first_crossing(
                evaluations=(matched_rows),
                target=target,
            )

            qml_crossing = first_crossing(
                evaluations=(qml_rows),
                target=target,
            )

            if matched_crossing is not None:
                matched_reaches += 1

            if qml_crossing is not None:
                qml_reaches += 1

            matched_aucs.append(float(matched_summary.normalized_auc))

            qml_aucs.append(float(qml_summary.normalized_auc))

            matched_best.append(float(matched_summary.best_normalized_progress))

            qml_best.append(float(qml_summary.best_normalized_progress))

            matched_final.append(float(matched_summary.final_normalized_progress))

            qml_final.append(float(qml_summary.final_normalized_progress))

            auc_deltas.append(
                float(matched_summary.normalized_auc - qml_summary.normalized_auc)
            )

            best_deltas.append(
                float(
                    matched_summary.best_normalized_progress
                    - qml_summary.best_normalized_progress
                )
            )

            final_deltas.append(
                float(
                    matched_summary.final_normalized_progress
                    - qml_summary.final_normalized_progress
                )
            )

        total_matched_reaches += matched_reaches

        total_qml_reaches += qml_reaches

        matched_auc_mean, matched_auc_sd = sample_mean_sd(matched_aucs)

        qml_auc_mean, qml_auc_sd = sample_mean_sd(qml_aucs)

        auc_delta_mean, auc_delta_sd = sample_mean_sd(auc_deltas)

        best_delta_mean, best_delta_sd = sample_mean_sd(best_deltas)

        final_delta_mean, final_delta_sd = sample_mean_sd(final_deltas)

        reconstructed[domain] = {
            "matched_target_reach": (matched_reaches),
            "qml_target_reach": (qml_reaches),
            "matched_auc_mean": (matched_auc_mean),
            "matched_auc_sd": (matched_auc_sd),
            "qml_auc_mean": (qml_auc_mean),
            "qml_auc_sd": (qml_auc_sd),
            "auc_delta_mean": (auc_delta_mean),
            "auc_delta_sd": (auc_delta_sd),
            "best_delta_mean": (best_delta_mean),
            "best_delta_sd": (best_delta_sd),
            "final_delta_mean": (final_delta_mean),
            "final_delta_sd": (final_delta_sd),
            "auc_win_counts": {
                "matched_classical_wins": sum(value > 0.0 for value in auc_deltas),
                "qml_wins": sum(value < 0.0 for value in auc_deltas),
                "ties": sum(value == 0.0 for value in auc_deltas),
            },
            "best_win_counts": {
                "matched_classical_wins": sum(value > 0.0 for value in best_deltas),
                "qml_wins": sum(value < 0.0 for value in best_deltas),
                "ties": sum(value == 0.0 for value in best_deltas),
            },
            "final_win_counts": {
                "matched_classical_wins": sum(value > 0.0 for value in final_deltas),
                "qml_wins": sum(value < 0.0 for value in final_deltas),
                "ties": sum(value == 0.0 for value in final_deltas),
            },
        }

    for domain in DOMAINS:
        stored_domain = analysis["domains"][domain]

        stored_matched = stored_domain["methods"]["matched_classical"]

        stored_qml = stored_domain["methods"]["qml"]

        stored_delta = stored_domain["paired_delta_summary"]

        rebuilt = reconstructed[domain]

        if int(stored_matched["target_reach_count"]) != rebuilt["matched_target_reach"]:
            raise RuntimeError(f"{domain} matched target reach mismatch")

        if int(stored_qml["target_reach_count"]) != rebuilt["qml_target_reach"]:
            raise RuntimeError(f"{domain} QML target reach mismatch")

        assert_close(
            float(stored_matched["normalized_auc"]["mean"]),
            float(rebuilt["matched_auc_mean"]),
            label=(f"{domain} matched AUC mean"),
        )

        assert_close(
            float(stored_matched["normalized_auc"]["sample_standard_deviation"]),
            float(rebuilt["matched_auc_sd"]),
            label=(f"{domain} matched AUC SD"),
        )

        assert_close(
            float(stored_qml["normalized_auc"]["mean"]),
            float(rebuilt["qml_auc_mean"]),
            label=(f"{domain} QML AUC mean"),
        )

        assert_close(
            float(stored_qml["normalized_auc"]["sample_standard_deviation"]),
            float(rebuilt["qml_auc_sd"]),
            label=(f"{domain} QML AUC SD"),
        )

        assert_close(
            float(stored_delta["normalized_auc_delta"]["mean"]),
            float(rebuilt["auc_delta_mean"]),
            label=(f"{domain} delta AUC mean"),
        )

        assert_close(
            float(stored_delta["best_progress_delta"]["mean"]),
            float(rebuilt["best_delta_mean"]),
            label=(f"{domain} delta best mean"),
        )

        assert_close(
            float(stored_delta["final_progress_delta"]["mean"]),
            float(rebuilt["final_delta_mean"]),
            label=(f"{domain} delta final mean"),
        )

        if (
            stored_delta["normalized_auc_delta"]["win_counts"]
            != rebuilt["auc_win_counts"]
        ):
            raise RuntimeError(f"{domain} AUC win-count mismatch")

        if (
            stored_delta["best_progress_delta"]["win_counts"]
            != rebuilt["best_win_counts"]
        ):
            raise RuntimeError(f"{domain} best-progress win-count mismatch")

        if (
            stored_delta["final_progress_delta"]["win_counts"]
            != rebuilt["final_win_counts"]
        ):
            raise RuntimeError(f"{domain} final-progress win-count mismatch")

    primary = analysis["primary_target_reach"]

    if int(primary["matched_classical"]["reached"]) != total_matched_reaches:
        raise RuntimeError("total matched target reach mismatch")

    if int(primary["qml"]["reached"]) != total_qml_reaches:
        raise RuntimeError("total QML target reach mismatch")

    verify_reproduction(domain="autonomous_driving")

    verify_reproduction(domain="robotics")

    if not CSV_PATH.exists():
        raise RuntimeError("ablation CSV missing")

    if not MARKDOWN_PATH.exists():
        raise RuntimeError("ablation markdown missing")

    figures_verified = 0

    for path in FIGURES:
        if not path.exists():
            raise RuntimeError(f"missing figure: {path}")

        if path.stat().st_size <= 0:
            raise RuntimeError(f"empty figure: {path}")

        figures_verified += 1

    boundary = analysis["scientific_boundary"]

    if boundary["qml_retrained"] is not False:
        raise RuntimeError("QML must not be retrained in Sprint 4.12")

    if boundary["matched_classical_new_training"] is not True:
        raise RuntimeError("matched-classical training boundary mismatch")

    if boundary["parameter_matched_representation_ablation"] is not True:
        raise RuntimeError("parameter-matched boundary mismatch")

    if boundary["formal_significance_test_performed"] is not False:
        raise RuntimeError("formal significance testing must remain disabled")

    if boundary["quantum_speedup_claimed"] is not False:
        raise RuntimeError("quantum speedup must not be claimed")

    if boundary["quantum_hardware_advantage_claimed"] is not False:
        raise RuntimeError("quantum hardware advantage must not be claimed")

    print()
    print("=" * 54)

    print(" SPRINT 4.12 MATCHED-BUDGET ABLATION VERIFIER")

    print("=" * 54)

    print()

    print(
        "Principal matched-classical runs:",
        principal_matched_runs,
    )

    print(
        "Principal QML runs:",
        principal_qml_runs,
    )

    print(
        "Matched target reaches:",
        total_matched_reaches,
        "/ 6",
    )

    print(
        "QML target reaches:",
        total_qml_reaches,
        "/ 6",
    )

    print("Driving seed-42 reproduction: PASS")

    print("Robotics seed-42 reproduction: PASS")

    print(
        "Figures verified:",
        figures_verified,
    )

    print("Exact parameter matching: PASS")

    print(
        "New QML training:",
        False,
    )

    print(
        "New matched-classical training:",
        True,
    )

    print()

    print("SPRINT 4.12 MATCHED-BUDGET ABLATION: PASS")


if __name__ == "__main__":
    main()
