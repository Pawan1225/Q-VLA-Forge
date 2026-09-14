from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_result_files(
    results_dir: str | Path = "results",
) -> list[dict[str, Any]]:
    """Load all valid experiment JSON files from the results directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    records: list[dict[str, Any]] = []

    for file_path in sorted(results_path.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                records.append(data)

        except (json.JSONDecodeError, OSError):
            continue

    return records


def flatten_experiment_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one experiment record for tabular dashboard display."""
    hardware = record.get("hardware") or {}
    metrics = record.get("metrics") or {}

    flattened: dict[str, Any] = {
        "experiment_id": record.get("experiment_id"),
        "domain": record.get("domain"),
        "method": record.get("method"),
        "seed": record.get("seed"),
        "created_at": record.get("created_at"),
        "notes": record.get("notes"),
        "python_version": hardware.get("python_version"),
        "torch_version": hardware.get("torch_version"),
        "cuda_available": hardware.get("cuda_available"),
    }

    for key, value in metrics.items():
        flattened[f"metric_{key}"] = value

    return flattened


def experiments_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert experiment records into a dashboard DataFrame."""
    if not records:
        return pd.DataFrame()

    flattened = [flatten_experiment_record(record) for record in records]

    return pd.DataFrame(flattened)


def load_sprint4_rl_evidence(
    evidence_path: str | Path = ("results/rl/evidence/" "sprint4-rl-evidence.json"),
) -> dict[str, Any]:
    """Load the verified Sprint 4 RL evidence package."""
    path = Path(evidence_path)

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return data
