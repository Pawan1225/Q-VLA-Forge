from __future__ import annotations

import json
from pathlib import Path

import torch

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.evaluation.baseline_manifest import (
    build_baseline_manifest,
    save_baseline_manifest,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import build_tensor_batch
from q_vla_forge.utils.reproducibility import (
    DEFAULT_SEEDS,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

MANIFEST_PATH = RESULTS_DIR / "sprint1-baseline-manifest.json"


def check_forward_pass(
    domain: Domain,
) -> None:
    if domain == Domain.AUTONOMOUS_DRIVING:
        dataset = SyntheticDrivingDataset(
            size=4,
            seed=42,
        )
    else:
        dataset = SyntheticRoboticsDataset(
            size=4,
            seed=42,
        )

    samples = list(dataset)

    batch = build_tensor_batch(
        samples,
        torch.device("cpu"),
    )

    set_seed(42)

    model = SharedVLAModel()
    model.eval()

    with torch.no_grad():
        output = model(
            batch.visual,
            batch.state,
            batch.language_goals,
            domain,
        )

    if output.shape != (4, 3):
        raise RuntimeError(
            f"{domain.value}: invalid output shape " f"{tuple(output.shape)}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(f"{domain.value}: non-finite outputs")


def check_summary_learning() -> None:
    for domain in (
        "driving",
        "robotics",
    ):
        for seed in DEFAULT_SEEDS:
            path = RESULTS_DIR / (f"{domain}-baseline-" f"seed-{seed}.json")

            payload = json.loads(path.read_text(encoding="utf-8"))

            if not (payload["best_validation_mse"] < payload["initial_validation_mse"]):
                raise RuntimeError(f"{path.name}: " "training did not improve")


def main() -> None:
    print()
    print("===== Q-VLA Forge " "Sprint 1 Final Verification =====")

    print()

    print("[1/5] Building baseline manifest...")

    manifest = build_baseline_manifest(RESULTS_DIR)

    save_baseline_manifest(
        manifest,
        MANIFEST_PATH,
    )

    print("PASS")

    print("[2/5] Driving forward pass...")

    check_forward_pass(Domain.AUTONOMOUS_DRIVING)

    print("PASS")

    print("[3/5] Robotics forward pass...")

    check_forward_pass(Domain.ROBOTICS)

    print("PASS")

    print("[4/5] Checking all seed runs learned...")

    check_summary_learning()

    print("PASS")

    print("[5/5] Checking baseline footprint...")

    expected_bytes = manifest.parameters * 4

    if manifest.fp32_model_size_bytes != expected_bytes:
        raise RuntimeError("FP32 footprint is inconsistent " "with parameter count")

    print("PASS")

    print()
    print("===== Sprint 1 Verification PASSED =====")

    print(
        "Baseline:",
        manifest.baseline_name,
    )

    print(
        "Seeds:",
        manifest.seeds,
    )

    print(
        "Parameters:",
        manifest.parameters,
    )

    print(
        "FP32 size:",
        f"{manifest.fp32_model_size_bytes / (1024**2):.4f} MB",
    )

    print(
        "Manifest:",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
