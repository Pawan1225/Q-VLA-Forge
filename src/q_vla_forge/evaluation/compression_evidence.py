from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MethodEvidence:
    """Proposal-facing evidence for one compression method."""

    domain: str
    method: str
    configuration: str

    compression_ratio_mean: float
    compression_ratio_std: float

    mse_change_mean: float
    mse_change_std: float

    mae_change_mean: float
    mae_change_std: float

    pilot_feasible_runs: int
    all_runs_pilot_feasible: bool
    pareto_efficient: bool


@dataclass(frozen=True)
class CompressionEvidencePackage:
    """Proposal-facing Sprint 2 evidence package."""

    seeds: tuple[int, ...]
    methods: tuple[MethodEvidence, ...]

    driving_pareto_frontier: tuple[str, ...]
    robotics_pareto_frontier: tuple[str, ...]

    quantum_inspired_method: str
    quantum_hardware_used: bool

    ablation_seed: int
    ablation_points: int

    supported_claims: tuple[str, ...]
    limitations: tuple[str, ...]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _method_evidence(
    *,
    domain: str,
    method_name: str,
    validation: dict[str, Any],
    pareto_frontier: set[str],
) -> MethodEvidence:
    configuration = str(validation["configuration"])

    return MethodEvidence(
        domain=domain,
        method=method_name,
        configuration=configuration,
        compression_ratio_mean=float(validation["compression_ratio"]["mean"]),
        compression_ratio_std=float(validation["compression_ratio"]["std"]),
        mse_change_mean=float(validation["mse_change_percent"]["mean"]),
        mse_change_std=float(validation["mse_change_percent"]["std"]),
        mae_change_mean=float(validation["mae_change_percent"]["mean"]),
        mae_change_std=float(validation["mae_change_percent"]["std"]),
        pilot_feasible_runs=int(validation["pilot_feasible_runs"]),
        all_runs_pilot_feasible=bool(validation["all_runs_pilot_feasible"]),
        pareto_efficient=(configuration in pareto_frontier),
    )


def _claim_for_method(
    evidence: MethodEvidence,
) -> str:
    return (
        f"{evidence.domain}: {evidence.configuration} "
        f"achieved "
        f"{evidence.compression_ratio_mean:.3f} "
        f"± {evidence.compression_ratio_std:.3f}× "
        f"effective whole-model compression with "
        f"{evidence.mse_change_mean:.3f} "
        f"± {evidence.mse_change_std:.3f}% "
        f"relative test-MSE change across three seeds; "
        f"{evidence.pilot_feasible_runs}/3 runs satisfied "
        f"the ≥2× compression and ≤5% MSE-change "
        f"pilot criterion."
    )


def build_compression_evidence(
    *,
    validation_path: Path,
    driving_pareto_path: Path,
    robotics_pareto_path: Path,
    ablation_path: Path,
    mps_mapping_path: Path,
) -> CompressionEvidencePackage:
    """Build proposal-facing evidence from frozen Sprint 2 outputs."""
    validation = _load_json(validation_path)

    driving_pareto = _load_json(driving_pareto_path)

    robotics_pareto = _load_json(robotics_pareto_path)

    ablation = _load_json(ablation_path)

    mps_mapping = _load_json(mps_mapping_path)

    seeds = tuple(int(seed) for seed in validation["seeds"])

    if seeds != (
        42,
        123,
        456,
    ):
        raise ValueError("compression evidence must use seeds " "42, 123, and 456")

    ablation_points = ablation["points"]

    if len(ablation_points) != 20:
        raise ValueError("expected exactly 20 compression " "ablation points")

    driving_frontier = tuple(driving_pareto["pareto_frontier"])

    robotics_frontier = tuple(robotics_pareto["pareto_frontier"])

    driving_frontier_set = set(driving_frontier)

    robotics_frontier_set = set(robotics_frontier)

    methods = (
        _method_evidence(
            domain="autonomous_driving",
            method_name="int8",
            validation=validation["driving"]["int8"],
            pareto_frontier=(driving_frontier_set),
        ),
        _method_evidence(
            domain="autonomous_driving",
            method_name="svd",
            validation=validation["driving"]["svd"],
            pareto_frontier=(driving_frontier_set),
        ),
        _method_evidence(
            domain="autonomous_driving",
            method_name="tensor_train_mps",
            validation=validation["driving"]["tensor_network"],
            pareto_frontier=(driving_frontier_set),
        ),
        _method_evidence(
            domain="robotics",
            method_name="int8",
            validation=validation["robotics"]["int8"],
            pareto_frontier=(robotics_frontier_set),
        ),
        _method_evidence(
            domain="robotics",
            method_name="svd",
            validation=validation["robotics"]["svd"],
            pareto_frontier=(robotics_frontier_set),
        ),
        _method_evidence(
            domain="robotics",
            method_name="tensor_train_mps",
            validation=validation["robotics"]["tensor_network"],
            pareto_frontier=(robotics_frontier_set),
        ),
    )

    claims = [
        (
            "Compression was evaluated over three deterministic "
            "training seeds (42, 123, 456) and reported using "
            "mean ± sample standard deviation."
        ),
        (
            "Classical INT8 quantization, classical truncated SVD, "
            "and quantum-inspired TT/MPS tensor-network compression "
            "were evaluated on the same shared VLA proxy architecture."
        ),
        (
            "Pareto analysis maximized effective whole-model "
            "compression while minimizing relative test-MSE increase."
        ),
        (
            "The target-level ablation compared classical SVD and "
            "quantum-inspired TT/MPS under matched model, domain, "
            "seed, architectural target, and test-data conditions."
        ),
    ]

    claims.extend(_claim_for_method(method) for method in methods)

    if bool(
        mps_mapping.get(
            "independent_compression_method",
            False,
        )
    ):
        raise ValueError(
            "MPS must not be represented as an " "independent compression algorithm"
        )

    limitations = (
        (
            "The experiments use compact synthetic autonomous-driving "
            "and robotics proxy tasks rather than full 7B–10B VLA models."
        ),
        (
            "TT/MPS is quantum-inspired tensor-network compression; "
            "no quantum hardware was used for Sprint 2."
        ),
        (
            "INT8 inference uses FP32-dequantized weights, while SVD "
            "and TT/MPS inference use reconstructed FP32 dense weights; "
            "measured latency is therefore not native compressed-kernel "
            "speedup evidence."
        ),
        (
            "The final compression hyperparameters were selected using "
            "seed-42 exploratory screening; seed 42 also appears in the "
            "three-seed validation set."
        ),
        (
            "Pareto dominance is based on three-seed mean metrics and "
            "does not constitute a statistical-significance test."
        ),
        (
            "The target-level architectural ablation uses seed 42 and "
            "should be interpreted separately from the three-seed "
            "validation results."
        ),
    )

    return CompressionEvidencePackage(
        seeds=seeds,
        methods=methods,
        driving_pareto_frontier=(driving_frontier),
        robotics_pareto_frontier=(robotics_frontier),
        quantum_inspired_method=("TT/open-boundary MPS constructed using TT-SVD"),
        quantum_hardware_used=False,
        ablation_seed=int(ablation["seed"]),
        ablation_points=len(ablation_points),
        supported_claims=tuple(claims),
        limitations=limitations,
    )


def save_compression_evidence(
    package: CompressionEvidencePackage,
    path: Path,
) -> None:
    """Save machine-readable proposal evidence."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(package),
            indent=2,
        ),
        encoding="utf-8",
    )


def evidence_to_markdown(
    package: CompressionEvidencePackage,
) -> str:
    """Render proposal-facing evidence as Markdown."""
    lines = [
        "# Q-VLA Forge — Sprint 2 Compression Evidence",
        "",
        "## Validation Protocol",
        "",
        ("Seeds: " + ", ".join(str(seed) for seed in package.seeds)),
        "",
        "## Validated Results",
        "",
        (
            "| Domain | Method | Configuration | "
            "Compression | MSE Δ | MAE Δ | "
            "Feasible Runs | Pareto |"
        ),
        ("|---|---|---|---:|---:|---:|---:|---:|"),
    ]

    for method in package.methods:
        lines.append(
            "| "
            f"{method.domain} | "
            f"{method.method} | "
            f"{method.configuration} | "
            f"{method.compression_ratio_mean:.3f} "
            f"± {method.compression_ratio_std:.3f}× | "
            f"{method.mse_change_mean:.3f} "
            f"± {method.mse_change_std:.3f}% | "
            f"{method.mae_change_mean:.3f} "
            f"± {method.mae_change_std:.3f}% | "
            f"{method.pilot_feasible_runs}/3 | "
            f"{'Yes' if method.pareto_efficient else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Pareto Frontiers",
            "",
            ("**Autonomous driving:** " + ", ".join(package.driving_pareto_frontier)),
            "",
            ("**Robotics:** " + ", ".join(package.robotics_pareto_frontier)),
            "",
            "## Quantum-Inspired Component",
            "",
            (f"- Method: {package.quantum_inspired_method}"),
            (f"- Quantum hardware used: " f"{package.quantum_hardware_used}"),
            (
                f"- Target-level ablation: "
                f"{package.ablation_points} points "
                f"at seed {package.ablation_seed}"
            ),
            "",
            "## Supported Claims",
            "",
        ]
    )

    lines.extend(f"- {claim}" for claim in package.supported_claims)

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    lines.extend(f"- {limitation}" for limitation in package.limitations)

    return "\n".join(lines) + "\n"
