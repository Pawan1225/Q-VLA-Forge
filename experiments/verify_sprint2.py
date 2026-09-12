from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation import (
    build_sprint2_manifest,
    save_sprint2_manifest,
)

ROOT = Path(".")

COMPRESSION_DIR = ROOT / "results" / "compression"

VALIDATION_DIR = COMPRESSION_DIR / "validation"

ABLATION_PATH = COMPRESSION_DIR / "ablation" / "compression-ablation-seed-42.json"

EVIDENCE_PATH = COMPRESSION_DIR / "evidence" / "sprint2-compression-evidence.json"

MANIFEST_PATH = COMPRESSION_DIR / "sprint2-compression-manifest.json"


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _required_artifacts() -> list[Path]:
    return [
        COMPRESSION_DIR / "compression-target-analysis.json",
        COMPRESSION_DIR / "mps-mapping-verification.json",
        COMPRESSION_DIR / "driving-compression-selection.json",
        COMPRESSION_DIR / "robotics-compression-selection.json",
        COMPRESSION_DIR / "cross-domain-compression-selection.json",
        COMPRESSION_DIR / "compression-validation-summary.json",
        COMPRESSION_DIR / "driving-compression-pareto.json",
        COMPRESSION_DIR / "driving-compression-pareto.csv",
        COMPRESSION_DIR / "robotics-compression-pareto.json",
        COMPRESSION_DIR / "robotics-compression-pareto.csv",
        COMPRESSION_DIR / "compression-pareto-summary.json",
        ABLATION_PATH,
        COMPRESSION_DIR / "ablation" / "compression-ablation-summary.csv",
        EVIDENCE_PATH,
        COMPRESSION_DIR / "evidence" / "sprint2-compression-evidence.md",
        ROOT / "figures" / "driving_compression_pareto.png",
        ROOT / "figures" / "robotics_compression_pareto.png",
    ]


def _validate_multiseed_summary() -> None:
    path = COMPRESSION_DIR / "compression-validation-summary.json"

    payload = _load_json(path)

    seeds = tuple(payload["seeds"])

    if seeds != (
        42,
        123,
        456,
    ):
        raise RuntimeError(f"unexpected validation seeds: {seeds}")

    for domain in (
        "driving",
        "robotics",
    ):
        domain_payload = payload[domain]

        for method in (
            "int8",
            "svd",
            "tensor_network",
        ):
            method_payload = domain_payload[method]

            method_seeds = tuple(method_payload["seeds"])

            if method_seeds != seeds:
                raise RuntimeError(f"{domain}/{method} " "seed mismatch")


def _validate_validation_files() -> int:
    files = sorted(VALIDATION_DIR.glob("*.json"))

    if len(files) != 18:
        raise RuntimeError(
            "expected exactly 18 " "validation JSON files, " f"found {len(files)}"
        )

    expected_domains = {
        "autonomous_driving",
        "robotics",
    }

    expected_methods = {
        "int8",
        "svd",
        "tensor_train",
    }

    observed: set[tuple[str, str, int]] = set()

    baseline_mse: dict[
        tuple[str, int],
        set[float],
    ] = {}

    for path in files:
        payload = _load_json(path)

        domain = str(payload["domain"])

        method = str(payload["method"])

        seed = int(payload["seed"])

        if domain not in expected_domains:
            raise RuntimeError(f"unexpected domain: {domain}")

        if method not in expected_methods:
            raise RuntimeError(f"unexpected method: {method}")

        if seed not in (
            42,
            123,
            456,
        ):
            raise RuntimeError(f"unexpected seed: {seed}")

        key = (
            domain,
            method,
            seed,
        )

        if key in observed:
            raise RuntimeError(f"duplicate validation result: {key}")

        observed.add(key)

        metrics = payload["metrics"]

        baseline_key = (
            domain,
            seed,
        )

        baseline_mse.setdefault(
            baseline_key,
            set(),
        ).add(
            round(
                float(metrics["baseline_test_mse"]),
                12,
            )
        )

    if len(observed) != 18:
        raise RuntimeError("validation matrix is incomplete")

    for key, values in baseline_mse.items():
        if len(values) != 1:
            raise RuntimeError(
                "compression methods did not "
                "share the same paired baseline "
                f"for {key}: {values}"
            )

    return len(files)


def _validate_ablation() -> int:
    payload = _load_json(ABLATION_PATH)

    if int(payload["seed"]) != 42:
        raise RuntimeError("ablation must use seed 42")

    points = payload["points"]

    if len(points) != 20:
        raise RuntimeError("expected exactly 20 " "ablation points")

    expected_domains = {
        "autonomous_driving",
        "robotics",
    }

    expected_methods = {
        "svd",
        "tensor_train",
    }

    expected_targets = {
        "fusion",
        "latent",
        "action",
        "fusion_latent",
        "all",
    }

    observed = {
        (
            point["domain"],
            point["method"],
            point["target"],
        )
        for point in points
    }

    expected = {
        (
            domain,
            method,
            target,
        )
        for domain in expected_domains
        for method in expected_methods
        for target in expected_targets
    }

    if observed != expected:
        missing = expected - observed
        extra = observed - expected

        raise RuntimeError(
            "ablation matrix mismatch; " f"missing={missing}, " f"extra={extra}"
        )

    return len(points)


def _validate_evidence() -> None:
    payload = _load_json(EVIDENCE_PATH)

    if tuple(payload["seeds"]) != (
        42,
        123,
        456,
    ):
        raise RuntimeError("evidence seed mismatch")

    if len(payload["methods"]) != 6:
        raise RuntimeError("expected six domain/method " "evidence records")

    if bool(payload["quantum_hardware_used"]):
        raise RuntimeError("Sprint 2 must not claim " "quantum hardware usage")

    if int(payload["ablation_points"]) != 20:
        raise RuntimeError("evidence ablation count mismatch")


def _validate_mps_mapping() -> None:
    payload = _load_json(COMPRESSION_DIR / "mps-mapping-verification.json")

    if bool(
        payload.get(
            "independent_compression_method",
            True,
        )
    ):
        raise RuntimeError(
            "MPS must be represented as "
            "the TT-equivalent tensor-network form, "
            "not an independent benchmark"
        )

    if not bool(
        payload.get(
            "quantum_inspired",
            False,
        )
    ):
        raise RuntimeError("TT/MPS mapping is not marked " "quantum-inspired")

    if bool(
        payload.get(
            "quantum_hardware_used",
            True,
        )
    ):
        raise RuntimeError("Sprint 2 must not claim " "quantum hardware")


def main() -> None:
    print()
    print("===== Q-VLA Forge " "Sprint 2 Final Verification =====")

    print()
    print("[1] Multi-seed summary")

    _validate_multiseed_summary()

    print("    PASS")

    print()
    print("[2] Validation matrix")

    validation_count = _validate_validation_files()

    print(
        "    PASS —",
        validation_count,
        "results",
    )

    print()
    print("[3] Ablation matrix")

    ablation_count = _validate_ablation()

    print(
        "    PASS —",
        ablation_count,
        "points",
    )

    print()
    print("[4] Proposal evidence")

    _validate_evidence()

    print("    PASS")

    print()
    print("[5] TT/MPS taxonomy")

    _validate_mps_mapping()

    print("    PASS")

    print()
    print("[6] Frozen artifacts")

    artifacts = _required_artifacts()

    manifest = build_sprint2_manifest(
        artifact_paths=artifacts,
        validation_result_count=(validation_count),
        ablation_point_count=(ablation_count),
    )

    save_sprint2_manifest(
        manifest,
        MANIFEST_PATH,
    )

    print(
        "    PASS —",
        len(manifest.artifacts),
        "artifacts frozen",
    )

    print()
    print("===== SPRINT 2 VERIFIED =====")

    print()
    print(
        "Manifest:",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
