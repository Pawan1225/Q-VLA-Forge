from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.experiment_record import ExperimentRecord


def save_experiment_record(
    record: ExperimentRecord,
    output_dir: str | Path = "results",
) -> Path:
    """Save an experiment record as a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{record.experiment_id}.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(record.to_dict(), file, indent=2)

    return file_path
