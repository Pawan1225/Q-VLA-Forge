from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from q_vla_forge.utils.reproducibility import DEFAULT_SEEDS


@dataclass(frozen=True)
class BaselineManifest:
    """Frozen description of the Sprint 1 classical baseline."""

    project: str
    baseline_name: str
    baseline_version: str

    seeds: tuple[int, ...]

    vision_dim: int
    language_dim: int
    state_dim: int
    fusion_dim: int
    latent_dim: int
    action_dim: int

    optimizer: str
    scheduler: str
    loss: str

    train_size: int
    validation_size: int
    test_size: int
    epochs: int
    batch_size: int

    driving_action_semantics: tuple[str, ...]
    robotics_action_semantics: tuple[str, ...]

    parameters: int
    fp32_model_size_bytes: int

    evidence_files: tuple[str, ...]
    created_at: str


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"required evidence file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def build_baseline_manifest(
    results_dir: Path,
) -> BaselineManifest:
    """Create the frozen Sprint 1 baseline manifest."""
    summary_path = results_dir / "baseline-validation-summary.json"

    summary = load_json(summary_path)

    if tuple(summary["seeds"]) != tuple(DEFAULT_SEEDS):
        raise ValueError("summary seeds do not match DEFAULT_SEEDS")

    driving = summary["driving"]
    robotics = summary["robotics"]

    if driving["parameters"] != robotics["parameters"]:
        raise ValueError("driving and robotics parameter counts must match")

    if driving["fp32_model_size_bytes"] != robotics["fp32_model_size_bytes"]:
        raise ValueError("driving and robotics FP32 sizes must match")

    required_files = (
        "driving-baseline-seed-42.json",
        "driving-baseline-seed-123.json",
        "driving-baseline-seed-456.json",
        "robotics-baseline-seed-42.json",
        "robotics-baseline-seed-123.json",
        "robotics-baseline-seed-456.json",
        "baseline-validation-summary.json",
    )

    for filename in required_files:
        if not (results_dir / filename).exists():
            raise FileNotFoundError(f"missing baseline evidence: {filename}")

    return BaselineManifest(
        project="Q-VLA Forge",
        baseline_name="shared_vla_fp32",
        baseline_version="1.0",
        seeds=tuple(DEFAULT_SEEDS),
        vision_dim=64,
        language_dim=32,
        state_dim=16,
        fusion_dim=64,
        latent_dim=32,
        action_dim=3,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        loss="MSELoss",
        train_size=512,
        validation_size=128,
        test_size=128,
        epochs=20,
        batch_size=32,
        driving_action_semantics=(
            "steering",
            "acceleration",
            "braking",
        ),
        robotics_action_semantics=(
            "delta_x",
            "delta_y",
            "gripper",
        ),
        parameters=int(driving["parameters"]),
        fp32_model_size_bytes=int(driving["fp32_model_size_bytes"]),
        evidence_files=required_files,
        created_at=(datetime.now(UTC).isoformat()),
    )


def save_baseline_manifest(
    manifest: BaselineManifest,
    output_path: Path,
) -> None:
    """Save the baseline manifest."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
        ),
        encoding="utf-8",
    )
