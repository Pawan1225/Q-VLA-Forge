"""Frozen Phase 1 evidence adapter for the final Q-VLA Forge dashboard.

Sprint 7.9 dashboard code must consume canonical frozen artifacts only.
This module does not train models, execute environments, recompute science,
retune methods, or synthesize metrics across incompatible evidence sources.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

FINAL_VALIDATION_DIR = Path("results/final-validation")

FIGURE_MANIFEST = FINAL_VALIDATION_DIR / "figures" / "figure-manifest.json"

TABLE_MANIFEST = FINAL_VALIDATION_DIR / "tables" / "table-manifest.json"

CLAIM_REGISTRY = FINAL_VALIDATION_DIR / "tables" / "final-claim-registry.csv"

BOTTLENECK_SCORECARD = (
    FINAL_VALIDATION_DIR / "tables" / "challenge-bottleneck-scorecard.csv"
)

FINAL_ABLATION_MATRIX = FINAL_VALIDATION_DIR / "tables" / "final-ablation-matrix.csv"

FULL_SYSTEM_ABLATION = (
    FINAL_VALIDATION_DIR / "full-system-ablation" / "full-system-ablation.json"
)

CHALLENGE_BOTTLENECKS = (
    "compression",
    "training_efficiency",
    "rl_alignment",
    "safety",
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

REQUIRED_SEEDS = (
    42,
    123,
    456,
)

ALGORITHMS_BY_BOTTLENECK = {
    "compression": (
        "FP32",
        "INT8",
        "SVD",
        "TT/MPS",
    ),
    "training_efficiency": (
        "baseline",
        "SVD",
        "TT/MPS",
    ),
    "rl_alignment": (
        "PPO / MLP",
        "matched classical",
        "QML / PQC",
    ),
    "safety": (
        "NONE",
        "CLIPPING",
        "LYAPUNOV",
    ),
}

REQUIRED_FIGURE_IDS = (
    "architecture",
    "compression_pareto",
    "training_convergence",
    "rl_sample_efficiency",
    "safety_violations",
    "cross_domain",
    "ablation_summary",
)

REQUIRED_TABLE_IDS = (
    "compression",
    "training",
    "rl",
    "safety",
    "cross_domain",
)

ABLATION_AREAS = (
    "compression",
    "qml",
    "safety",
    "full_system",
)

MISSING_CANONICAL_EVIDENCE = "MISSING CANONICAL EVIDENCE"


class MissingCanonicalEvidenceError(FileNotFoundError):
    """Raised when required frozen dashboard evidence is unavailable."""


def _require_file(
    root: Path,
    relative_path: Path,
) -> Path:
    path = root / relative_path

    if not path.is_file():
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " f"{relative_path.as_posix()}"
        )

    return path


def _load_json(
    root: Path,
    relative_path: Path,
) -> dict[str, Any]:
    path = _require_file(
        root,
        relative_path,
    )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            f"invalid JSON at "
            f"{relative_path.as_posix()}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            f"{relative_path.as_posix()} "
            "must contain a JSON object"
        )

    return payload


def _load_csv(
    root: Path,
    relative_path: Path,
) -> list[dict[str, str]]:
    path = _require_file(
        root,
        relative_path,
    )

    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            f"unable to read "
            f"{relative_path.as_posix()}"
        ) from exc

    if not rows:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            f"{relative_path.as_posix()} "
            "contains no data rows"
        )

    return rows


def _repo_relative_path(
    value: str,
) -> Path:
    normalized = value.replace(
        "\\",
        "/",
    )

    return Path(normalized)


def _require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    label: str,
) -> None:
    columns = set(rows[0])

    missing = required - columns

    if missing:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            f"{label} missing columns "
            f"{sorted(missing)}"
        )


def _parse_reach(
    value: str,
) -> tuple[int, int]:
    parts = value.split("/")

    if len(parts) != 2:
        raise ValueError("Target-reach value must use " f"'reached/total': {value}")

    return (
        int(parts[0]),
        int(parts[1]),
    )


def load_figure_manifest(
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Load and validate the frozen Sprint 7.7 figure manifest."""
    repo_root = Path(root)

    payload = _load_json(
        repo_root,
        FIGURE_MANIFEST,
    )

    if payload.get("status") != "FROZEN":
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "figure manifest is not FROZEN"
        )

    if payload.get("generated_from_frozen_evidence") is not True:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "figures are not marked as frozen-evidence outputs"
        )

    if payload.get("new_training") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "figure manifest reports new training"
        )

    if payload.get("new_experiments") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "figure manifest reports new experiments"
        )

    if payload.get("new_scientific_results") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "figure manifest reports new scientific results"
        )

    figures = payload.get("figures")

    if not isinstance(
        figures,
        list,
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "figure manifest has no figure list"
        )

    if len(figures) != 7:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "expected seven final figures"
        )

    figure_ids = tuple(str(item["figure_id"]) for item in figures)

    if figure_ids != REQUIRED_FIGURE_IDS:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "final figure IDs differ from "
            "the Sprint 7.7 contract"
        )

    for figure in figures:
        output_path = _repo_relative_path(str(figure["output_path"]))

        _require_file(
            repo_root,
            output_path,
        )

    return payload


def load_table_manifest(
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Load and validate the frozen Sprint 7.8 table manifest."""
    repo_root = Path(root)

    payload = _load_json(
        repo_root,
        TABLE_MANIFEST,
    )

    if payload.get("status") != "FROZEN":
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "table manifest is not FROZEN"
        )

    if payload.get("generated_from_frozen_evidence") is not True:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "tables are not marked as frozen-evidence outputs"
        )

    if payload.get("new_training") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "table manifest reports new training"
        )

    if payload.get("new_experiments") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "table manifest reports new experiments"
        )

    if payload.get("new_scientific_results") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "table manifest reports new scientific results"
        )

    if payload.get("synthetic_metric_composition") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "synthetic metric composition must remain blocked"
        )

    required_ids = tuple(
        payload.get(
            "required_table_ids",
            (),
        )
    )

    if required_ids != REQUIRED_TABLE_IDS:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "final table IDs differ from "
            "the Sprint 7.8 contract"
        )

    tables = payload.get("tables")

    if not isinstance(
        tables,
        list,
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "table manifest has no table metadata"
        )

    if len(tables) != 5:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "expected five final result tables"
        )

    for table in tables:
        output_file = _repo_relative_path(str(table["output_file"]))

        _require_file(
            repo_root,
            output_file,
        )

    return payload


def load_final_tables(
    root: str | Path = ROOT,
) -> dict[str, list[dict[str, str]]]:
    """Load the five frozen canonical Sprint 7.8 result tables."""
    repo_root = Path(root)

    manifest = load_table_manifest(repo_root)

    tables: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for metadata in manifest["tables"]:
        table_id = str(metadata["table_id"])

        relative_path = _repo_relative_path(str(metadata["output_file"]))

        rows = _load_csv(
            repo_root,
            relative_path,
        )

        expected_count = int(metadata["row_count"])

        if len(rows) != expected_count:
            raise MissingCanonicalEvidenceError(
                f"{MISSING_CANONICAL_EVIDENCE}: "
                f"{table_id} table expected "
                f"{expected_count} rows but found "
                f"{len(rows)}"
            )

        tables[table_id] = rows

    if tuple(tables) != REQUIRED_TABLE_IDS:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "loaded final tables differ from "
            "the canonical order"
        )

    return tables


def load_claim_registry(
    root: str | Path = ROOT,
) -> list[dict[str, str]]:
    """Load the final Phase 1 claim registry."""
    repo_root = Path(root)

    rows = _load_csv(
        repo_root,
        CLAIM_REGISTRY,
    )

    _require_columns(
        rows,
        {
            "claim_id",
            "area",
            "status",
            "statement",
            "limitation",
            "evidence",
        },
        "final claim registry",
    )

    return rows


def load_bottleneck_scorecard(
    root: str | Path = ROOT,
) -> list[dict[str, str]]:
    """Load the frozen four-bottleneck scorecard."""
    repo_root = Path(root)

    rows = _load_csv(
        repo_root,
        BOTTLENECK_SCORECARD,
    )

    _require_columns(
        rows,
        {
            "bottleneck",
            "phase1_status",
            "headline",
            "strongest_evidence",
            "boundary",
        },
        "challenge bottleneck scorecard",
    )

    if len(rows) != 4:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "expected four challenge bottlenecks"
        )

    return rows


def load_final_ablation_matrix(
    root: str | Path = ROOT,
) -> list[dict[str, str]]:
    """Load the frozen component-level final ablation matrix."""
    repo_root = Path(root)

    rows = _load_csv(
        repo_root,
        FINAL_ABLATION_MATRIX,
    )

    _require_columns(
        rows,
        {
            "area",
            "comparison",
            "finding",
            "status",
        },
        "final ablation matrix",
    )

    return rows


def load_full_system_ablation(
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Load and validate Sprint 7.6 full-system evidence boundaries."""
    repo_root = Path(root)

    payload = _load_json(
        repo_root,
        FULL_SYSTEM_ABLATION,
    )

    if payload.get("status") != "FROZEN":
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "full-system ablation is not FROZEN"
        )

    if payload.get("new_training") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "full-system package reports new training"
        )

    if payload.get("new_experiments") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "full-system package reports new experiments"
        )

    phase1 = payload.get("phase1")

    if not isinstance(
        phase1,
        dict,
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "full-system Phase 1 evidence is missing"
        )

    if (
        int(
            phase1.get(
                "direct_count",
                -1,
            )
        )
        != 0
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "Phase 1 DIRECT count must remain zero"
        )

    if (
        int(
            phase1.get(
                "component_only_count",
                -1,
            )
        )
        != 16
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "Phase 1 COMPONENT_ONLY count must remain sixteen"
        )

    if (
        int(
            phase1.get(
                "not_evaluated_count",
                -1,
            )
        )
        != 0
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: " "Phase 1 NOT_EVALUATED count changed"
        )

    controls = payload.get("scientific_controls")

    if not isinstance(
        controls,
        dict,
    ):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "full-system scientific controls are missing"
        )

    if controls.get("synthetic_metric_composition_allowed") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "synthetic full-system metrics must remain blocked"
        )

    if controls.get("interaction_effects_estimable_with_component_only") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "interaction effects cannot be estimated "
            "from component-only evidence"
        )

    if controls.get("full_system_end_to_end_metrics_reported") is not False:
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "end-to-end full-system metrics "
            "must remain unreported"
        )

    return payload


def _compression_outcomes(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    output: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        method = row["Method"]

        if method == "FP32":
            continue

        output.setdefault(
            method,
            {},
        )[
            row["Domain"]
        ] = row["Criterion"]

    return output


def _rl_target_reaches(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    totals: dict[
        str,
        dict[str, int],
    ] = {}

    for row in rows:
        policy = row["Policy"]

        reached, total = _parse_reach(row["Target Reaches"])

        record = totals.setdefault(
            policy,
            {
                "reached": 0,
                "total": 0,
            },
        )

        record["reached"] += reached

        record["total"] += total

    return totals


def _pqc_compactness(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    compactness: dict[
        str,
        str,
    ] = {}

    for row in rows:
        if row["Policy"] != "QML / PQC":
            continue

        compactness[row["Domain"]] = row["Parameter Reduction"]

    return compactness


def _safety_outcomes(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    output: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        output.setdefault(
            row["Domain"],
            {},
        )[
            row["Method"]
        ] = row["Violation Rate"]

    return output


def _claim_controls(
    figure_manifest: dict[str, Any],
    table_manifest: dict[str, Any],
    full_system: dict[str, Any],
) -> dict[str, bool]:
    merged: dict[
        str,
        bool,
    ] = {}

    sources = (
        figure_manifest.get(
            "claim_controls",
            {},
        ),
        table_manifest.get(
            "claim_controls",
            {},
        ),
        full_system.get(
            "claim_controls",
            {},
        ),
    )

    for source in sources:
        if not isinstance(
            source,
            dict,
        ):
            continue

        for key, value in source.items():
            boolean_value = bool(value)

            if key in merged and merged[key] != boolean_value:
                raise MissingCanonicalEvidenceError(
                    f"{MISSING_CANONICAL_EVIDENCE}: " f"conflicting claim control {key}"
                )

            merged[str(key)] = boolean_value

    if any(merged.values()):
        raise MissingCanonicalEvidenceError(
            f"{MISSING_CANONICAL_EVIDENCE}: "
            "one or more prohibited dashboard claims "
            "are unexpectedly enabled"
        )

    return merged


def load_dashboard_evidence(
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build the canonical read-only evidence package for Streamlit.

    This function aggregates already-frozen evidence. It does not derive
    new scientific results or combine incompatible metrics.
    """
    repo_root = Path(root)

    figure_manifest = load_figure_manifest(repo_root)

    table_manifest = load_table_manifest(repo_root)

    tables = load_final_tables(repo_root)

    claim_registry = load_claim_registry(repo_root)

    bottleneck_scorecard = load_bottleneck_scorecard(repo_root)

    component_ablation = load_final_ablation_matrix(repo_root)

    full_system = load_full_system_ablation(repo_root)

    claim_controls = _claim_controls(
        figure_manifest,
        table_manifest,
        full_system,
    )

    phase1 = full_system["phase1"]

    return {
        "status": "FROZEN",
        "phase": "Phase 1",
        "generated_from_frozen_evidence": True,
        "new_training": False,
        "new_experiments": False,
        "new_scientific_results": False,
        "synthetic_metric_composition": False,
        "challenge_bottlenecks": CHALLENGE_BOTTLENECKS,
        "domains": DOMAINS,
        "seeds": REQUIRED_SEEDS,
        "algorithms": ALGORITHMS_BY_BOTTLENECK,
        "ablation_areas": ABLATION_AREAS,
        "figures": figure_manifest["figures"],
        "figure_manifest": figure_manifest,
        "tables": tables,
        "table_manifest": table_manifest,
        "claim_registry": claim_registry,
        "bottleneck_scorecard": bottleneck_scorecard,
        "component_ablation": component_ablation,
        "full_system": full_system,
        "claim_controls": claim_controls,
        "criterion_outcomes": {
            "compression": _compression_outcomes(tables["compression"]),
            "training_efficiency": {
                "phase1_result": ("NOT DEMONSTRATED"),
                "threshold": (">=10% robust efficiency improvement"),
            },
            "rl_alignment": {
                "target_reaches": _rl_target_reaches(tables["rl"]),
                "pqc_compactness": _pqc_compactness(tables["rl"]),
            },
            "safety": _safety_outcomes(tables["safety"]),
        },
        "full_system_status": {
            "DIRECT": int(phase1["direct_count"]),
            "COMPONENT_ONLY": int(phase1["component_only_count"]),
            "NOT_EVALUATED": int(phase1["not_evaluated_count"]),
        },
        "provenance": {
            "figure_manifest": FIGURE_MANIFEST.as_posix(),
            "table_manifest": TABLE_MANIFEST.as_posix(),
            "claim_registry": CLAIM_REGISTRY.as_posix(),
            "bottleneck_scorecard": BOTTLENECK_SCORECARD.as_posix(),
            "component_ablation": FINAL_ABLATION_MATRIX.as_posix(),
            "full_system_ablation": FULL_SYSTEM_ABLATION.as_posix(),
        },
    }
