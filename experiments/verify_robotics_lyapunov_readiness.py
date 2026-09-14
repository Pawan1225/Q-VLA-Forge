"""Closure-readiness gate for Sprint 5.9 robotics Lyapunov evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from q_vla_forge.evaluation.robotics_lyapunov_safety import (
    ROBOTICS_DOMAIN,
)
from q_vla_forge.evaluation.safety_baseline import (
    EXPECTED_CHECKPOINT_SELECTION,
    EXPECTED_TRAINING_BUDGET,
    PRINCIPAL_SEEDS,
    SAFETY_EVALUATION_SEEDS,
)
from q_vla_forge.safety.lyapunov_candidates import (
    MAX_ROBOTICS_CANDIDATES,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CHECKPOINT_HASHES = {
    42: ("ce4703c3080b4485d8788ed7e7225a61" "f821cfdbd8d473a3f99c33511df1cd63"),
    123: ("54694c3691fe8cd255b4c85b1433ceef" "65dfd04aca9b28c3f466a7a9447fdb13"),
    456: ("ae48eebf0b3e0dae09414d2fd895ab64" "4178296800c13ffd293b3b812087ab13"),
}

EXPECTED_RECOVERY_HASHES = {
    42: ("9c706cb46380b8355a995bae397fad607" "ffafcb35a0ca1bbd8f0740b1b52391f"),
    123: ("fea58ea92fd035838c3362d196e68182" "9637886d1587f79b323adffd45420dd4"),
    456: ("ab386492c47809cb92756d8e42281fb1" "1bea6aed485381a421d40f4f77cedeb1"),
}

FROZEN_PATHS = {
    "robotics_constraints": (
        ROOT / "src" / "q_vla_forge" / "safety" / "robotics_constraints.py"
    ),
    "lyapunov_foundation": (
        ROOT
        / "results"
        / "safety"
        / "lyapunov-foundation"
        / "sprint5-lyapunov-foundation.json"
    ),
    "lyapunov_filter_mechanism": (
        ROOT
        / "results"
        / "safety"
        / "lyapunov-filter"
        / "sprint5-lyapunov-filter-mechanism.json"
    ),
    "lyapunov_filter_source": (
        ROOT / "src" / "q_vla_forge" / "safety" / "lyapunov_filter.py"
    ),
    "robotics_predictor_source": (
        ROOT / "src" / "q_vla_forge" / "safety" / "predictors.py"
    ),
    "robotics_environment_source": (
        ROOT / "src" / "q_vla_forge" / "rl" / "robotics_env.py"
    ),
}

EXPECTED_FROZEN_HASHES = {
    "robotics_constraints": (
        "ce2d4f2895ab60a4dc29c790216ad292" "77810aeff9b4bbe05c3cd6364cb465eb"
    ),
    "lyapunov_foundation": (
        "71bc4cc961ded93af2bbe326ad3814f0" "228fe744f5fb9c82ca2031a679b2a076"
    ),
    "lyapunov_filter_mechanism": (
        "2529c704fddd7cb5fee711a762159e95" "51e6ab3a15b5e0beeba916d2ff788663"
    ),
    "lyapunov_filter_source": (
        "4035be445cbb0b67b764f92e8e17b862" "b4ea141334807155e96a2f214f50e701"
    ),
    "robotics_predictor_source": (
        "29568c29e0feb0974a8603b282bfcea3" "f7141911835daae9decb5fa8bab5742d"
    ),
    "robotics_environment_source": (
        "4da799584500dc841b98169c8d27d5f8" "1a0eb5daa750659647090d0206f70ae5"
    ),
}


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


def main() -> None:
    print("=" * 64)
    print(" Q-VLA FORGE - " "SPRINT 5.9 ROBOTICS CLOSURE READINESS")
    print("=" * 64)
    print()

    _require(
        ROBOTICS_DOMAIN == "robotics",
        "domain locked to robotics",
    )

    _require(
        PRINCIPAL_SEEDS
        == (
            42,
            123,
            456,
        ),
        "principal seeds exactly 42, 123, 456",
    )

    _require(
        SAFETY_EVALUATION_SEEDS
        == tuple(
            range(
                20_000,
                20_020,
            )
        ),
        "evaluation seeds exactly 20000..20019",
    )

    _require(
        EXPECTED_TRAINING_BUDGET == 20_000,
        "training budget locked to 20,000",
    )

    _require(
        EXPECTED_CHECKPOINT_SELECTION == "final_20000_step_policy",
        ("checkpoint selection " "locked to final 20k policy"),
    )

    _require(
        MAX_ROBOTICS_CANDIDATES == 9,
        "robotics candidate ceiling locked to 9",
    )

    for seed in PRINCIPAL_SEEDS:
        checkpoint = (
            ROOT
            / "results"
            / "rl"
            / "checkpoints"
            / (f"robotics-seed-" f"{seed}-final.pt")
        )

        recovery = (
            ROOT
            / "results"
            / "rl"
            / "checkpoint-recovery"
            / (f"robotics-seed-" f"{seed}-recovery.json")
        )

        _require(
            checkpoint.exists(),
            (f"checkpoint exists " f"for seed {seed}"),
        )

        _require(
            recovery.exists(),
            (f"recovery record exists " f"for seed {seed}"),
        )

        checkpoint_hash = _sha256(checkpoint)

        recovery_hash = _sha256(recovery)

        _require(
            checkpoint_hash == EXPECTED_CHECKPOINT_HASHES[seed],
            ("checkpoint SHA matches " f"frozen seed {seed}"),
        )

        _require(
            recovery_hash == EXPECTED_RECOVERY_HASHES[seed],
            ("recovery SHA matches " f"frozen seed {seed}"),
        )

        recovery_payload = json.loads(recovery.read_text(encoding="utf-8"))

        _require(
            recovery_payload["domain"] == "robotics",
            ("recovery domain valid " f"for seed {seed}"),
        )

        _require(
            recovery_payload["seed"] == seed,
            ("recovery seed valid " f"for seed {seed}"),
        )

        _require(
            recovery_payload["training_budget"] == 20_000,
            ("recovery training budget " f"valid for seed {seed}"),
        )

        _require(
            recovery_payload["checkpoint_selection"] == "final_20000_step_policy",
            ("recovery checkpoint selection " f"valid for seed {seed}"),
        )

        _require(
            recovery_payload["historical_trajectory_match"] is True,
            ("historical trajectory " f"reproduced for seed {seed}"),
        )

        _require(
            float(recovery_payload["maximum_absolute_metric_delta"]) == 0.0,
            ("historical trajectory delta " f"zero for seed {seed}"),
        )

    for (
        name,
        path,
    ) in FROZEN_PATHS.items():
        _require(
            path.exists(),
            ("frozen dependency exists: " f"{name}"),
        )

        _require(
            _sha256(path) == EXPECTED_FROZEN_HASHES[name],
            ("frozen dependency SHA " f"matches: {name}"),
        )

    smoke_dir = ROOT / "results" / "safety" / "lyapunov-robotics" / "smoke"

    _require(
        not smoke_dir.exists(),
        ("smoke evidence removed " "from principal evidence scope"),
    )

    runs_dir = ROOT / "results" / "safety" / "lyapunov-robotics" / "runs"

    _require(
        runs_dir.exists(),
        "principal evidence directory exists",
    )

    principal_files = tuple(sorted(runs_dir.glob("*.json")))

    _require(
        len(principal_files) == 3,
        "exactly three principal run artifacts exist",
    )

    expected_names = {(f"robotics-seed-" f"{seed}.json") for seed in PRINCIPAL_SEEDS}

    _require(
        {path.name for path in principal_files} == expected_names,
        ("principal artifact names " "match frozen seeds"),
    )

    runner_path = ROOT / "experiments" / "run_robotics_lyapunov_safety.py"

    evaluation_path = (
        ROOT / "src" / "q_vla_forge" / "evaluation" / "robotics_lyapunov_safety.py"
    )

    runner_source = runner_path.read_text(encoding="utf-8")

    evaluation_source = evaluation_path.read_text(encoding="utf-8")

    for forbidden in (
        "optimizer",
        ".backward(",
        "train_ppo",
        "run_ppo_training",
        "apply_clipping_safety_filter",
        "RobustnessCondition.GAUSSIAN",
    ):
        _require(
            forbidden not in runner_source,
            ("runner excludes " f"{forbidden}"),
        )

    _require(
        "load_frozen_ppo_policy" in runner_source,
        "runner uses frozen PPO loader",
    )

    _require(
        "environment_internal_state" in runner_source,
        ("runner records internal-state " "prediction-context provenance"),
    )

    _require(
        "policy_observation_modified" in runner_source,
        ("runner records policy-observation " "integrity"),
    )

    _require(
        "apply_lyapunov_safety_filter" not in runner_source,
        ("runner delegates filter " "execution through evaluation module"),
    )

    _require(
        "apply_lyapunov_safety_filter" in evaluation_source,
        ("evaluation module uses " "frozen Lyapunov filter"),
    )

    _require(
        "RoboticsPredictionContext" in evaluation_source,
        ("evaluation module uses " "robotics prediction context"),
    )

    _require(
        "env.object_grasped" in evaluation_source,
        ("evaluation reads public " "object_grasped environment state"),
    )

    _require(
        "env._object_grasped" not in evaluation_source,
        ("evaluation does not access " "private grasp state"),
    )

    for forbidden in (
        "optimizer",
        ".backward(",
        "train_ppo",
        "run_ppo_training",
        "apply_clipping_safety_filter",
    ):
        _require(
            forbidden not in evaluation_source,
            ("evaluation module excludes " f"{forbidden}"),
        )

    summary_path = (
        ROOT
        / "results"
        / "safety"
        / "lyapunov-robotics"
        / "sprint5-robotics-lyapunov-summary.json"
    )

    _require(
        summary_path.exists(),
        "Sprint 5.9 summary exists",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    _require(
        summary["effectiveness"]["robotics_lyapunov_empirical_effectiveness_supported"]
        is True,
        ("Sprint 5.9 empirical " "effectiveness criterion passes"),
    )

    _require(
        summary["grasp_context"]["object_grasped_true_steps"] == 0,
        ("summary preserves unexercised " "grasped-state limitation"),
    )

    _require(
        summary["claims"]["grasped_state_branch_empirically_validated"] is False,
        ("grasped-state validation " "claim remains controlled"),
    )

    print()
    print("Protocol freeze: PASS")
    print("Checkpoint provenance: PASS")
    print("Sprint 5.7 mechanism provenance: PASS")
    print("Robotics predictor provenance: PASS")
    print("Robotics environment provenance: PASS")
    print("No training: PASS")
    print("No filter tuning: PASS")
    print("No robustness perturbation: PASS")
    print("Smoke isolation: PASS")
    print("Principal evidence: FROZEN")
    print("Grasp-context provenance: PASS")
    print("Claim controls: PASS")
    print()
    print("SPRINT 5.9 ROBOTICS " "CLOSURE READINESS: PASS")


if __name__ == "__main__":
    main()
