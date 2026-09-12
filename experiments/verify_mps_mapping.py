"""Generate TT/MPS equivalence evidence from Sprint 2.5 results."""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from q_vla_forge.compression import (
    tt_svd,
    verify_tt_mps_equivalence,
)

RESULTS_DIR = Path("results")

COMPRESSION_DIR = RESULTS_DIR / "compression"

OUTPUT_PATH = COMPRESSION_DIR / "mps-mapping-verification.json"


def _load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_equivalence_checks() -> list[dict[str, object]]:
    """Verify TT/MPS equivalence numerically for ranks 2, 4, and 8."""
    torch.manual_seed(42)

    tensor = torch.randn(
        16,
        16,
        32,
    )

    checks: list[dict[str, object]] = []

    for rank in (
        2,
        4,
        8,
    ):
        decomposition = tt_svd(
            tensor,
            max_rank=rank,
        )

        verification = verify_tt_mps_equivalence(decomposition)

        checks.append(
            {
                "requested_rank": rank,
                "tensor_shape": list(verification.tensor_shape),
                "tt_ranks": list(verification.tt_ranks),
                "mps_bond_dimensions": list(verification.mps_bond_dimensions),
                "tt_parameters": (verification.tt_parameters),
                "mps_parameters": (verification.mps_parameters),
                "same_parameter_count": (verification.same_parameter_count),
                "same_bond_dimensions": (verification.same_bond_dimensions),
                "maximum_absolute_difference": (
                    verification.maximum_absolute_difference
                ),
                "relative_difference": (verification.relative_difference),
                "reconstruction_equivalent": (verification.reconstruction_equivalent),
            }
        )

    return checks


def _map_tt_experiment(
    path: Path,
) -> dict[str, object]:
    """Map one Tensor Train experiment to its MPS interpretation."""
    payload = _load_json(path)

    if payload["method"] != "tensor_train":
        raise ValueError(f"unexpected compression method in {path}")

    tt = payload["tt"]

    mapped_layers: list[dict[str, object]] = []

    for layer in tt["layers"]:
        tt_ranks = tuple(int(value) for value in layer["actual_ranks"])

        compressed_parameters = int(layer["compressed_parameters"])

        mapped_layers.append(
            {
                "name": layer["name"],
                "tensor_shape": (layer["tensor_shape"]),
                "tt_ranks": list(tt_ranks),
                "mps_bond_dimensions": list(tt_ranks),
                "tt_parameters": (compressed_parameters),
                "mps_parameters": (compressed_parameters),
                "equivalent_storage": True,
                "interpretation": ("open_boundary_mps"),
            }
        )

    metrics = payload["metrics"]

    compression_ratio = (
        metrics["baseline_size_bytes"] / metrics["compressed_size_bytes"]
    )

    mse_change_percent = (
        (metrics["compressed_test_mse"] - metrics["baseline_test_mse"])
        / metrics["baseline_test_mse"]
        * 100.0
    )

    return {
        "source_file": path.name,
        "experiment_id": (payload["experiment_id"]),
        "domain": (payload["domain"]),
        "seed": (payload["seed"]),
        "requested_rank": (payload["configuration"]["requested_max_rank"]),
        "compression_ratio": (compression_ratio),
        "mse_change_percent": (mse_change_percent),
        "layers": mapped_layers,
    }


def main() -> None:
    """Generate TT/MPS mapping evidence."""
    files = [
        Path(path)
        for path in sorted(glob.glob(str(COMPRESSION_DIR / "*-tt-rank-*-seed-42.json")))
    ]

    if not files:
        raise FileNotFoundError("no Sprint 2.5 TT result files found")

    experiments = [_map_tt_experiment(path) for path in files]

    synthetic_checks = _synthetic_equivalence_checks()

    if not all(bool(check["reconstruction_equivalent"]) for check in synthetic_checks):
        raise RuntimeError("TT/MPS numerical equivalence check failed")

    if not all(bool(check["same_parameter_count"]) for check in synthetic_checks):
        raise RuntimeError("TT/MPS storage equivalence check failed")

    if not all(bool(check["same_bond_dimensions"]) for check in synthetic_checks):
        raise RuntimeError("TT/MPS bond-dimension equivalence check failed")

    payload = {
        "method": ("tensor_train_open_boundary_mps"),
        "relationship": (
            "Tensor Train and open-boundary Matrix "
            "Product State use the same three-index "
            "tensor-network core structure."
        ),
        "independent_compression_method": False,
        "quantum_inspired": True,
        "quantum_hardware_used": False,
        "synthetic_equivalence_checks": (synthetic_checks),
        "sprint_2_5_experiment_mappings": (experiments),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== Q-VLA Forge " "TT ↔ MPS Verification =====")

    print()

    print(
        "TT experiment files:",
        len(files),
    )

    print(
        "Numerical checks:",
        len(synthetic_checks),
    )

    print()

    for check in synthetic_checks:
        print(
            "Rank",
            check["requested_rank"],
            "| TT ranks:",
            check["tt_ranks"],
            "| MPS bonds:",
            check["mps_bond_dimensions"],
            "| params:",
            check["tt_parameters"],
            "| equivalent:",
            check["reconstruction_equivalent"],
        )

    print()

    print(
        "MPS is recorded as an equivalent "
        "interpretation of the TT representation, "
        "not as an independent compression result."
    )

    print()

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
