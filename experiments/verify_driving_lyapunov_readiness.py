"""Pre-principal-run readiness gate for Sprint 5.8 driving Lyapunov evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from q_vla_forge.evaluation.driving_lyapunov_safety import (
    DRIVING_DOMAIN,
    MAX_DRIVING_CANDIDATES,
)
from q_vla_forge.evaluation.safety_baseline import (
    EXPECTED_CHECKPOINT_SELECTION,
    EXPECTED_TRAINING_BUDGET,
    PRINCIPAL_SEEDS,
    SAFETY_EVALUATION_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CHECKPOINT_HASHES = {
    42: "671ec3b742dbfdc1468f9ff9944a59dfad35aca7765fa3302e611baf9f533db2",
    123: "af8b7ea9efe3eab81c45589f04a706d7c8b473382ccda3c0f110c6b7a93d4563",
    456: "71e35c17743d5f422a4b50e5dd1d6dea641c3e071713896d5acdd2c2f6a24788",
}

EXPECTED_RECOVERY_HASHES = {
    42: "232e3af1c15a80dfc1616aff35ce2daf2c1916b7a5263f383b3a41d2cacfafe3",
    123: "7fcd18d423b96f485dfdb2a55f18f7c0f9d41b82573d53bea0647b2636e52c56",
    456: "f17dfec31cc0f2ca05467f614fb844a3d61d3c6af761839eb58f6cdf5ba9722b",
}

FROZEN_PATHS = {
    "driving_constraints": (
        ROOT / "src" / "q_vla_forge" / "safety" / "driving_constraints.py"
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
}

EXPECTED_FROZEN_HASHES = {
    "driving_constraints": (
        "82aadb7fab7382cd5ebbd9352066b1c3f876f5710013f9bf4d4a04132a604d1f"
    ),
    "lyapunov_foundation": (
        "71bc4cc961ded93af2bbe326ad3814f0228fe744f5fb9c82ca2031a679b2a076"
    ),
    "lyapunov_filter_mechanism": (
        "2529c704fddd7cb5fee711a762159e9551e6ab3a15b5e0beeba916d2ff788663"
    ),
    "lyapunov_filter_source": (
        "4035be445cbb0b67b764f92e8e17b862b4ea141334807155e96a2f214f50e701"
    ),
}


def _sha256(path: Path) -> str:
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
    print("=" * 60)
    print(" Q-VLA FORGE - SPRINT 5.8 PRINCIPAL RUN READINESS")
    print("=" * 60)
    print()

    _require(
        DRIVING_DOMAIN == "autonomous_driving",
        "domain locked to autonomous_driving",
    )

    _require(
        PRINCIPAL_SEEDS == (42, 123, 456),
        "principal seeds exactly 42, 123, 456",
    )

    _require(
        SAFETY_EVALUATION_SEEDS == tuple(range(20_000, 20_020)),
        "evaluation seeds exactly 20000..20019",
    )

    _require(
        EXPECTED_TRAINING_BUDGET == 20_000,
        "training budget locked to 20,000",
    )

    _require(
        EXPECTED_CHECKPOINT_SELECTION == "final_20000_step_policy",
        "checkpoint selection locked to final 20k policy",
    )

    _require(
        MAX_DRIVING_CANDIDATES == 10,
        "driving candidate ceiling locked to 10",
    )

    for seed in PRINCIPAL_SEEDS:
        checkpoint = (
            ROOT
            / "results"
            / "rl"
            / "checkpoints"
            / f"autonomous_driving-seed-{seed}-final.pt"
        )

        recovery = (
            ROOT
            / "results"
            / "rl"
            / "checkpoint-recovery"
            / f"autonomous_driving-seed-{seed}-recovery.json"
        )

        _require(
            checkpoint.exists(),
            f"checkpoint exists for seed {seed}",
        )

        _require(
            recovery.exists(),
            f"recovery record exists for seed {seed}",
        )

        checkpoint_hash = _sha256(checkpoint)

        recovery_hash = _sha256(recovery)

        _require(
            checkpoint_hash == EXPECTED_CHECKPOINT_HASHES[seed],
            f"checkpoint SHA matches frozen seed {seed}",
        )

        _require(
            recovery_hash == EXPECTED_RECOVERY_HASHES[seed],
            f"recovery SHA matches frozen seed {seed}",
        )

        recovery_payload = json.loads(recovery.read_text(encoding="utf-8"))

        _require(
            recovery_payload["domain"] == "autonomous_driving",
            f"recovery domain valid for seed {seed}",
        )

        _require(
            recovery_payload["seed"] == seed,
            f"recovery seed valid for seed {seed}",
        )

        _require(
            recovery_payload["training_budget"] == 20_000,
            f"recovery training budget valid for seed {seed}",
        )

        _require(
            recovery_payload["checkpoint_selection"] == "final_20000_step_policy",
            f"recovery checkpoint selection valid for seed {seed}",
        )

        _require(
            recovery_payload["historical_trajectory_match"] is True,
            f"historical trajectory reproduced for seed {seed}",
        )

        _require(
            recovery_payload["maximum_absolute_metric_delta"] == 0.0,
            f"historical trajectory delta zero for seed {seed}",
        )

    for name, path in FROZEN_PATHS.items():
        _require(
            path.exists(),
            f"frozen dependency exists: {name}",
        )

        _require(
            _sha256(path) == EXPECTED_FROZEN_HASHES[name],
            f"frozen dependency SHA matches: {name}",
        )

    smoke_dir = ROOT / "results" / "safety" / "lyapunov-driving" / "smoke"

    _require(
        not smoke_dir.exists(),
        "smoke evidence removed from principal evidence scope",
    )

    runs_dir = ROOT / "results" / "safety" / "lyapunov-driving" / "runs"

    if runs_dir.exists():
        principal_files = tuple(runs_dir.glob("*.json"))

        _require(
            len(principal_files) == 0,
            "no prior principal 5.8 run artifacts exist",
        )
    else:
        print("[PASS] no prior principal 5.8 run artifacts exist")

    runner_source = (ROOT / "experiments" / "run_driving_lyapunov_safety.py").read_text(
        encoding="utf-8"
    )

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
            f"runner excludes {forbidden}",
        )

    _require(
        "load_frozen_ppo_policy" in runner_source,
        "runner uses frozen PPO loader",
    )

    _require(
        "apply_lyapunov_safety_filter" not in runner_source,
        ("runner delegates filter execution through " "evaluation module"),
    )

    evaluation_source = (
        ROOT / "src" / "q_vla_forge" / "evaluation" / "driving_lyapunov_safety.py"
    ).read_text(encoding="utf-8")

    _require(
        "apply_lyapunov_safety_filter" in evaluation_source,
        "evaluation module uses frozen Lyapunov filter",
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
            f"evaluation module excludes {forbidden}",
        )

    print()
    print("Protocol freeze: PASS")
    print("Checkpoint provenance: PASS")
    print("Sprint 5.7 mechanism provenance: PASS")
    print("No training: PASS")
    print("No filter tuning: PASS")
    print("No robustness perturbation: PASS")
    print("Smoke isolation: PASS")
    print("Principal evidence directory: CLEAN")
    print()
    print("SPRINT 5.8 PRINCIPAL RUN READINESS: PASS")


if __name__ == "__main__":
    main()
