"""Independent verifier for Sprint 5.8 driving Lyapunov safety evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RUNS_DIR = ROOT / "results" / "safety" / "lyapunov-driving" / "runs"

SUMMARY_PATH = (
    ROOT
    / "results"
    / "safety"
    / "lyapunov-driving"
    / "sprint5-driving-lyapunov-summary.json"
)

NONE_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "baseline" / "sprint5-no-filter-safety-summary.json"
)

CLIPPING_SUMMARY_PATH = (
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json"
)

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)

EXPECTED_CHECKPOINT_HASHES = {
    42: "671ec3b742dbfdc1468f9ff9944a59dfad35aca7765fa3302e611baf9f533db2",
    123: "af8b7ea9efe3eab81c45589f04a706d7c8b473382ccda3c0f110c6b7a93d4563",
    456: "71e35c17743d5f422a4b50e5dd1d6dea641c3e071713896d5acdd2c2f6a24788",
}

FLOAT_ATOL = 1.0e-10


def _load(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(65_536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def _close(
    actual: float,
    expected: float,
    *,
    message: str,
) -> None:
    _require(
        bool(
            np.isclose(
                actual,
                expected,
                rtol=0.0,
                atol=FLOAT_ATOL,
            )
        ),
        message,
    )


def _p95(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            95,
        )
    )


def main() -> None:
    print("=" * 68)
    print(" Q-VLA FORGE - SPRINT 5.8 " "DRIVING LYAPUNOV SAFETY VERIFICATION")
    print("=" * 68)
    print()

    summary = _load(SUMMARY_PATH)

    none_summary = _load(NONE_SUMMARY_PATH)

    clipping_summary = _load(CLIPPING_SUMMARY_PATH)

    _require(
        summary["sprint"] == "5.8",
        "summary sprint is 5.8",
    )

    _require(
        summary["domain"] == "autonomous_driving",
        "summary domain is autonomous_driving",
    )

    _require(
        summary["condition"] == "clean",
        "summary condition is clean",
    )

    _require(
        summary["principal_seeds"] == list(PRINCIPAL_SEEDS),
        "principal seeds exactly 42,123,456",
    )

    _require(
        summary["evaluation_seeds"] == list(EVALUATION_SEEDS),
        "evaluation seeds exactly 20000..20019",
    )

    _require(
        summary["new_training_performed"] is False,
        "summary records no new training",
    )

    seed_violation_rates: list[float] = []

    seed_rewards: list[float] = []

    seed_success_rates: list[float] = []

    seed_intervention_rates: list[float] = []

    for seed in PRINCIPAL_SEEDS:
        path = RUNS_DIR / f"autonomous_driving-seed-{seed}.json"

        _require(
            path.exists(),
            f"principal run exists seed {seed}",
        )

        payload = _load(path)

        _require(
            payload["sprint"] == "5.8",
            f"run sprint valid seed {seed}",
        )

        _require(
            payload["artifact"] == "driving-lyapunov-safety-principal-run",
            f"artifact identity seed {seed}",
        )

        _require(
            payload["domain"] == "autonomous_driving",
            f"domain valid seed {seed}",
        )

        _require(
            payload["principal_seed"] == seed,
            f"principal seed valid seed {seed}",
        )

        _require(
            payload["safety_method"] == "lyapunov",
            f"method valid seed {seed}",
        )

        _require(
            payload["robustness_condition"] == "clean",
            f"condition valid seed {seed}",
        )

        _require(
            payload["new_training_performed"] is False,
            f"no training seed {seed}",
        )

        _require(
            payload["policy_fine_tuning_performed"] is False,
            f"no policy fine-tuning seed {seed}",
        )

        _require(
            payload["filter_tuning_performed"] is False,
            f"no filter tuning seed {seed}",
        )

        _require(
            payload["perturbation_used"] is False,
            f"no perturbation seed {seed}",
        )

        _require(
            payload["evaluation_seeds"] == list(EVALUATION_SEEDS),
            f"evaluation seeds valid seed {seed}",
        )

        _require(
            payload["episode_count"] == 20,
            f"20 episodes seed {seed}",
        )

        _require(
            len(payload["episodes"]) == 20,
            f"episode payload count seed {seed}",
        )

        _require(
            payload["policy_checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASHES[seed],
            f"checkpoint SHA seed {seed}",
        )

        checkpoint_path = ROOT / payload["policy_checkpoint"]

        _require(
            _sha256(checkpoint_path) == payload["policy_checkpoint_sha256"],
            f"current checkpoint provenance seed {seed}",
        )

        recovery_path = ROOT / payload["checkpoint_recovery_record"]

        _require(
            _sha256(recovery_path) == payload["checkpoint_recovery_record_sha256"],
            f"current recovery provenance seed {seed}",
        )

        for (
            provenance_name,
            provenance,
        ) in payload["provenance"].items():
            provenance_path = ROOT / provenance["path"]

            _require(
                provenance_path.exists(),
                (f"provenance exists " f"{provenance_name} " f"seed {seed}"),
            )

            _require(
                _sha256(provenance_path) == provenance["sha256"],
                (f"provenance SHA " f"{provenance_name} " f"seed {seed}"),
            )

        episodes = payload["episodes"]

        total_steps = sum(int(episode["episode_length"]) for episode in episodes)

        _require(
            total_steps == int(payload["seed_summary"]["total_environment_steps"]),
            f"total step arithmetic seed {seed}",
        )

        executed_violation_steps = sum(
            int(episode["executed_violation_step_count"]) for episode in episodes
        )

        reconstructed_violation_rate = executed_violation_steps / total_steps

        _close(
            reconstructed_violation_rate,
            float(payload["seed_summary"]["executed_violation_step_rate"]),
            message=(f"violation rate arithmetic seed {seed}"),
        )

        executed_constraints = sum(
            int(episode["executed_constraint_violation_count"]) for episode in episodes
        )

        _close(
            executed_constraints / total_steps,
            float(payload["seed_summary"]["executed_constraint_violation_rate"]),
            message=(f"constraint rate arithmetic seed {seed}"),
        )

        critical_steps = sum(
            int(episode["critical_violation_step_count"]) for episode in episodes
        )

        _close(
            critical_steps / total_steps,
            float(payload["seed_summary"]["critical_violation_step_rate"]),
            message=(f"critical rate arithmetic seed {seed}"),
        )

        interventions = sum(int(episode["intervention_count"]) for episode in episodes)

        reconstructed_intervention_rate = interventions / total_steps

        _close(
            reconstructed_intervention_rate,
            float(payload["seed_summary"]["intervention_rate"]),
            message=(f"intervention rate arithmetic seed {seed}"),
        )

        corrections = [
            float(value)
            for episode in episodes
            for value in episode["action_corrections_l2"]
        ]

        _close(
            float(statistics.mean(corrections)),
            float(payload["seed_summary"]["mean_action_correction_l2"]),
            message=(f"mean correction arithmetic seed {seed}"),
        )

        _close(
            _p95(corrections),
            float(payload["seed_summary"]["p95_action_correction_l2"]),
            message=(f"P95 correction arithmetic seed {seed}"),
        )

        _close(
            max(
                corrections,
                default=0.0,
            ),
            float(payload["seed_summary"]["max_action_correction_l2"]),
            message=(f"max correction arithmetic seed {seed}"),
        )

        reason_counts: dict[
            str,
            int,
        ] = {}

        candidate_counts: dict[
            str,
            int,
        ] = {}

        strict_decreases = 0
        nonincreases = 0
        fallback_count = 0

        for episode in episodes:
            for (
                reason,
                count,
            ) in episode["intervention_reason_counts"].items():
                reason_counts[reason] = reason_counts.get(
                    reason,
                    0,
                ) + int(count)

            for (
                source,
                count,
            ) in episode["selected_candidate_source_counts"].items():
                candidate_counts[source] = candidate_counts.get(
                    source,
                    0,
                ) + int(count)

            strict_decreases += int(episode["strict_lyapunov_decrease_count"])

            nonincreases += int(episode["lyapunov_nonincrease_count"])

            fallback_count += int(episode["emergency_fallback_count"])

            _require(
                max(episode["candidate_counts"]) <= 10,
                f"candidate ceiling seed {seed}",
            )

        summary_reason_counts = {
            str(key): int(value)
            for key, value in payload["seed_summary"][
                "intervention_reason_counts"
            ].items()
        }

        normalized_reason_counts = {
            reason: int(
                reason_counts.get(
                    reason,
                    0,
                )
            )
            for reason in summary_reason_counts
        }

        _require(
            normalized_reason_counts == summary_reason_counts,
            f"reason accounting seed {seed}",
        )

        summary_candidate_counts = {
            str(key): int(value)
            for key, value in payload["seed_summary"][
                "selected_candidate_source_counts"
            ].items()
        }

        normalized_candidate_counts = {
            source: int(
                candidate_counts.get(
                    source,
                    0,
                )
            )
            for source in summary_candidate_counts
        }

        _require(
            normalized_candidate_counts == summary_candidate_counts,
            f"candidate-source accounting seed {seed}",
        )

        _close(
            strict_decreases / total_steps,
            float(payload["seed_summary"]["strict_lyapunov_decrease_rate"]),
            message=(f"strict decrease rate seed {seed}"),
        )

        _close(
            nonincreases / total_steps,
            float(payload["seed_summary"]["lyapunov_nonincrease_rate"]),
            message=(f"nonincrease rate seed {seed}"),
        )

        _close(
            fallback_count / total_steps,
            float(payload["seed_summary"]["emergency_fallback_rate"]),
            message=(f"fallback rate seed {seed}"),
        )

        _require(
            int(payload["seed_summary"]["selected_hard_guard_failure_count"]) == 0,
            f"selected hard guards valid seed {seed}",
        )

        _require(
            int(payload["seed_summary"]["executed_action_out_of_bounds_count"]) == 0,
            f"executed actions in bounds seed {seed}",
        )

        seed_violation_rates.append(reconstructed_violation_rate)

        rewards = [float(episode["reward"]) for episode in episodes]

        successes = [float(episode["success"]) for episode in episodes]

        seed_rewards.append(float(statistics.mean(rewards)))

        seed_success_rates.append(float(statistics.mean(successes)))

        seed_intervention_rates.append(reconstructed_intervention_rate)

    none_domain = none_summary["domains"]["autonomous_driving"]

    none_mean_violation = float(none_domain["violation_step_rate"]["mean"])

    lyapunov_mean_violation = float(statistics.mean(seed_violation_rates))

    reconstructed_reduction = (
        none_mean_violation - lyapunov_mean_violation
    ) / none_mean_violation

    _close(
        reconstructed_reduction,
        float(summary["effectiveness"]["domain_mean_violation_reduction"]),
        message=("domain mean violation reduction"),
    )

    clipping_rows = {
        int(row["principal_seed"]): row
        for row in clipping_summary["seed_level_comparisons"]
        if (row["domain"] == "autonomous_driving")
    }

    for seed in PRINCIPAL_SEEDS:
        none_rate = float(clipping_rows[seed]["none_violation_step_rate"])

        lyapunov_rate = seed_violation_rates[PRINCIPAL_SEEDS.index(seed)]

        if none_rate == 0.0:
            passes = lyapunov_rate <= none_rate
        else:
            passes = lyapunov_rate < none_rate

        _require(
            passes,
            (f"frozen seed safety " f"criterion seed {seed}"),
        )

    none_reward = float(none_domain["reward"]["mean_of_seed_means"])

    lyapunov_reward = float(statistics.mean(seed_rewards))

    reward_degradation = max(
        0.0,
        none_reward - lyapunov_reward,
    ) / abs(none_reward)

    _close(
        reward_degradation,
        float(summary["effectiveness"]["reward_degradation_fraction"]),
        message=("reward degradation reconstruction"),
    )

    none_success = float(none_domain["success"]["mean_of_seed_rates"])

    lyapunov_success = float(statistics.mean(seed_success_rates))

    success_drop = max(
        0.0,
        none_success - lyapunov_success,
    )

    _close(
        success_drop,
        float(summary["effectiveness"]["success_rate_drop"]),
        message=("success drop reconstruction"),
    )

    _require(
        reconstructed_reduction >= 0.20,
        "mean violation reduction >= 20%",
    )

    _require(
        reward_degradation <= 0.10,
        "reward degradation <= 10%",
    )

    _require(
        success_drop <= 0.10,
        "success-rate drop <= 10 percentage points",
    )

    _require(
        summary["effectiveness"]["driving_lyapunov_empirical_effectiveness_supported"]
        is True,
        "mechanical effectiveness is supported",
    )

    clipping_mean = float(
        summary["three_method_summary"]["clipping"]["violation_step_rate"]["mean"]
    )

    lyapunov_mean = float(
        summary["three_method_summary"]["lyapunov"]["violation_step_rate"]["mean"]
    )

    _require(
        clipping_mean == 0.0 and lyapunov_mean == 0.0,
        ("clipping and Lyapunov both " "have zero clean violation-step rate"),
    )

    _require(
        summary["mechanism_interpretation"][
            "principal_lyapunov_decrease_interventions_observed"
        ]
        is False,
        ("no principal LYAPUNOV_DECREASE " "intervention observed"),
    )

    _require(
        summary["mechanism_interpretation"][
            "all_selected_lyapunov_transitions_nonincreasing"
        ]
        is True,
        ("selected transitions are " "Lyapunov non-increasing"),
    )

    _require(
        summary["mechanism_interpretation"]["emergency_fallback_observed"] is False,
        "no emergency fallback observed",
    )

    claims = summary["claims"]

    for key in (
        "formal_stability_proven",
        "formal_safety_guarantee",
        "collision_free_driving_claim",
        "production_safety_claim",
        "iso_26262_claim",
        "lyapunov_superior_to_clipping_claim",
    ):
        _require(
            claims[key] is False,
            f"claim remains controlled: {key}",
        )

    _require(
        math.isfinite(
            float(
                summary["three_method_summary"]["lyapunov"]["mean_filter_latency_ms"][
                    "mean"
                ]
            )
        ),
        "CPU proxy filter latency is finite",
    )

    print()
    print("Protocol: PASS")
    print("Provenance: PASS")
    print("Policy fairness: PASS")
    print("Episode pairing: PASS")
    print("Safety accounting: PASS")
    print("Correction accounting: PASS")
    print("Candidate accounting: PASS")
    print("Lyapunov accounting: PASS")
    print("Fallback accounting: PASS")
    print("NONE comparison: PASS")
    print("CLIPPING comparison: PASS")
    print("Claim controls: PASS")
    print()
    print("Driving empirical effectiveness: SUPPORTED")
    print()
    print("SPRINT 5.8 DRIVING LYAPUNOV SAFETY: PASS")


if __name__ == "__main__":
    main()
