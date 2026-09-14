"""Final verification for deterministic Sprint 4 PPO checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIRECTORY = ROOT / "results" / "rl" / "checkpoints"

RECOVERY_DIRECTORY = ROOT / "results" / "rl" / "checkpoint-recovery"

EXPECTED = (
    ("autonomous_driving", 42),
    ("autonomous_driving", 123),
    ("autonomous_driving", 456),
    ("robotics", 42),
    ("robotics", 123),
    ("robotics", 456),
)


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


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main() -> None:
    checkpoint_paths = list(CHECKPOINT_DIRECTORY.glob("*-final.pt"))

    recovery_paths = list(RECOVERY_DIRECTORY.glob("*-recovery.json"))

    require(
        len(checkpoint_paths) == 6,
        "Exactly six recovered checkpoints exist",
    )

    require(
        len(recovery_paths) == 6,
        "Exactly six recovery records exist",
    )

    hashes: dict[str, str] = {}

    for domain, seed in EXPECTED:
        checkpoint_path = CHECKPOINT_DIRECTORY / f"{domain}-seed-{seed}-final.pt"

        recovery_path = RECOVERY_DIRECTORY / f"{domain}-seed-{seed}-recovery.json"

        require(
            checkpoint_path.exists(),
            f"{domain} seed {seed} checkpoint exists",
        )

        require(
            recovery_path.exists(),
            f"{domain} seed {seed} recovery record exists",
        )

        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))

        require(
            recovery["domain"] == domain,
            f"{domain} seed {seed} domain matches",
        )

        require(
            recovery["seed"] == seed,
            f"{domain} seed {seed} training seed matches",
        )

        require(
            recovery["training_budget"] == 20_000,
            f"{domain} seed {seed} budget is 20k",
        )

        require(
            recovery["checkpoint_selection"] == "final_20000_step_policy",
            f"{domain} seed {seed} selection is final policy",
        )

        require(
            recovery["historical_trajectory_match"] is True,
            f"{domain} seed {seed} historical trajectory matches",
        )

        require(
            recovery["maximum_absolute_metric_delta"] == 0.0,
            f"{domain} seed {seed} historical metric delta is zero",
        )

        require(
            recovery["evaluation_count"] == 21,
            f"{domain} seed {seed} has 21 stored evaluations",
        )

        require(
            recovery["original_checkpoint_existed"] is False,
            f"{domain} seed {seed} records missing original persistence",
        )

        require(
            recovery["policy_reconstruction_performed"] is True,
            f"{domain} seed {seed} reconstruction is disclosed",
        )

        checkpoint_hash = sha256_file(checkpoint_path)

        require(
            checkpoint_hash == recovery["checkpoint_sha256"],
            f"{domain} seed {seed} checkpoint SHA256 matches",
        )

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        require(
            payload["policy_family"] == "classical_ppo",
            f"{domain} seed {seed} policy family is PPO",
        )

        require(
            payload["domain"] == domain,
            f"{domain} seed {seed} checkpoint domain matches",
        )

        require(
            payload["seed"] == seed,
            f"{domain} seed {seed} checkpoint seed matches",
        )

        require(
            payload["total_environment_steps"] == 20_000,
            f"{domain} seed {seed} checkpoint is final 20k policy",
        )

        require(
            payload["checkpoint_selection"] == "final_20000_step_policy",
            f"{domain} seed {seed} checkpoint selection metadata matches",
        )

        require(
            "model_state_dict" in payload,
            f"{domain} seed {seed} model state exists",
        )

        hashes[f"{domain}:{seed}"] = checkpoint_hash

    require(
        len(set(hashes.values())) == 6,
        "All six checkpoint hashes are distinct",
    )

    print()
    print("=" * 64)
    print(" Q-VLA FORGE - SPRINT 5.3.5 PPO CHECKPOINT RECOVERY")
    print("=" * 64)
    print()
    print("Recovered policies: 6 / 6")
    print("Historical trajectories: 6 / 6 EXACT")
    print("Checkpoint SHA256 verification: PASS")
    print("Checkpoint loading: PASS")
    print("Training budget: 20000")
    print("Selection: final policy")
    print("Safety evaluation performed: NO")
    print()
    print("SPRINT 5.3.5 PPO CHECKPOINT RECOVERY: PASS")


if __name__ == "__main__":
    main()
