"""Recover a missing Sprint 4 PPO final checkpoint by deterministic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.ppo_baseline import (
    PPO_EVALUATION_SEEDS,
    training_result_to_dict,
)
from q_vla_forge.rl.driving_env import DrivingRLEnv
from q_vla_forge.rl.ppo import (
    DEFAULT_PPO_CONFIG,
    train_ppo,
)
from q_vla_forge.rl.robotics_env import RoboticsRLEnv

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_DIRECTORY = ROOT / "results" / "rl" / "ppo"

CHECKPOINT_DIRECTORY = ROOT / "results" / "rl" / "checkpoints"

RECOVERY_DIRECTORY = ROOT / "results" / "rl" / "checkpoint-recovery"

FLOAT_TOLERANCE = 1e-10


def sha256_file(
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


def require_equal(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{label} mismatch: " f"actual={actual!r}, expected={expected!r}"
        )


def require_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> float:
    delta = abs(float(actual) - float(expected))

    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise RuntimeError(
            f"{label} mismatch: "
            f"actual={actual:.17g}, "
            f"expected={expected:.17g}, "
            f"delta={delta:.17g}"
        )

    return delta


def verify_recovery(
    *,
    recovered: dict[str, Any],
    historical: dict[str, Any],
) -> float:
    require_equal(
        recovered["domain"],
        historical["domain"],
        label="domain",
    )

    require_equal(
        recovered["seed"],
        historical["seed"],
        label="seed",
    )

    for key in (
        "total_environment_steps",
        "completed_training_episodes",
        "actor_parameters",
        "critic_parameters",
        "total_parameters",
    ):
        require_equal(
            recovered[key],
            historical[key],
            label=key,
        )

    recovered_evaluations = recovered["evaluations"]

    historical_evaluations = historical["evaluations"]

    require_equal(
        len(recovered_evaluations),
        len(historical_evaluations),
        label="evaluation count",
    )

    max_delta = 0.0

    for index, (
        recovered_evaluation,
        historical_evaluation,
    ) in enumerate(
        zip(
            recovered_evaluations,
            historical_evaluations,
            strict=True,
        )
    ):
        prefix = f"evaluation[{index}]"

        for key in (
            "environment_steps",
            "completed_training_episodes",
        ):
            require_equal(
                recovered_evaluation[key],
                historical_evaluation[key],
                label=f"{prefix}.{key}",
            )

        for key in (
            "mean_reward",
            "reward_standard_deviation",
            "success_rate",
        ):
            delta = require_close(
                recovered_evaluation[key],
                historical_evaluation[key],
                label=f"{prefix}.{key}",
            )

            max_delta = max(
                max_delta,
                delta,
            )

    return max_delta


def environment_factory_for(
    domain: str,
):
    if domain == "autonomous_driving":
        return DrivingRLEnv

    if domain == "robotics":
        return RoboticsRLEnv

    raise ValueError(f"unsupported domain: {domain}")


def recover(
    *,
    domain: str,
    seed: int,
) -> None:
    historical_path = HISTORICAL_DIRECTORY / f"{domain}-seed-{seed}.json"

    if not historical_path.exists():
        raise FileNotFoundError(
            f"historical PPO artifact missing: " f"{historical_path}"
        )

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RECOVERY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_checkpoint = CHECKPOINT_DIRECTORY / f"{domain}-seed-{seed}-final.pt"

    temporary_checkpoint = CHECKPOINT_DIRECTORY / f"{domain}-seed-{seed}-final.tmp.pt"

    if temporary_checkpoint.exists():
        temporary_checkpoint.unlink()

    historical = json.loads(historical_path.read_text(encoding="utf-8"))

    print("=" * 64)
    print(" Q-VLA FORGE - PPO CHECKPOINT RECOVERY")
    print("=" * 64)
    print()
    print("Domain:", domain)
    print("Seed:", seed)
    print("Training budget: 20000")
    print(
        "Evaluation seeds:",
        f"{PPO_EVALUATION_SEEDS[0]}" f"..{PPO_EVALUATION_SEEDS[-1]}",
    )
    print(
        "Historical artifact:",
        historical_path.relative_to(ROOT),
    )
    print()
    print("Replaying frozen Sprint 4 PPO protocol...")

    result = train_ppo(
        domain=domain,
        seed=seed,
        environment_factory=(environment_factory_for(domain)),
        evaluation_seeds=(PPO_EVALUATION_SEEDS),
        config=DEFAULT_PPO_CONFIG,
        checkpoint_path=str(temporary_checkpoint),
    )

    recovered = training_result_to_dict(result)

    print()
    print("Comparing complete historical " "evaluation trajectory...")

    try:
        max_delta = verify_recovery(
            recovered=recovered,
            historical=historical,
        )
    except Exception:
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()

        raise

    if final_checkpoint.exists():
        final_checkpoint.unlink()

    temporary_checkpoint.replace(final_checkpoint)

    recovery_payload = {
        "artifact": ("sprint4-ppo-checkpoint-recovery"),
        "recovery_only": True,
        "safety_evaluation_performed": False,
        "original_checkpoint_existed": False,
        "policy_reconstruction_performed": True,
        "domain": domain,
        "seed": seed,
        "training_budget": (DEFAULT_PPO_CONFIG.total_environment_steps),
        "checkpoint_selection": ("final_20000_step_policy"),
        "evaluation_seeds": list(PPO_EVALUATION_SEEDS),
        "historical_artifact": (historical_path.relative_to(ROOT).as_posix()),
        "historical_artifact_sha256": (sha256_file(historical_path)),
        "checkpoint": (final_checkpoint.relative_to(ROOT).as_posix()),
        "checkpoint_sha256": (sha256_file(final_checkpoint)),
        "evaluation_count": len(recovered["evaluations"]),
        "float_tolerance": (FLOAT_TOLERANCE),
        "maximum_absolute_metric_delta": (max_delta),
        "historical_trajectory_match": True,
        "training_seconds_reproduced": False,
        "scientific_interpretation": (
            "Deterministic reconstruction of the "
            "missing final Sprint 4 PPO policy. "
            "The checkpoint is accepted only after "
            "the complete stored evaluation trajectory "
            "matches the frozen historical evidence."
        ),
    }

    recovery_path = RECOVERY_DIRECTORY / f"{domain}-seed-{seed}-recovery.json"

    recovery_path.write_text(
        json.dumps(
            recovery_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Historical trajectory: MATCH")
    print(
        "Maximum absolute metric delta:",
        f"{max_delta:.17g}",
    )
    print(
        "Checkpoint SHA256:",
        recovery_payload["checkpoint_sha256"],
    )
    print(
        "Checkpoint:",
        final_checkpoint.relative_to(ROOT),
    )
    print()
    print("PPO CHECKPOINT RECOVERY: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        required=True,
        choices=(
            "autonomous_driving",
            "robotics",
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=(
            42,
            123,
            456,
        ),
    )

    args = parser.parse_args()

    recover(
        domain=args.domain,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
