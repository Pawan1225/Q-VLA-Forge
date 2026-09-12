from __future__ import annotations

import csv
import json
from pathlib import Path

INPUT_DIR = Path("results") / "training" / "fp32"

OUTPUT = Path("results") / "training" / "fp32-convergence-summary.csv"


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.json"))

    if len(files) != 6:
        raise RuntimeError(f"expected 6 FP32 files, found {len(files)}")

    rows = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        target = payload["target_reach"]

        rows.append(
            {
                "domain": payload["domain"],
                "seed": payload["seed"],
                "trainable_parameters": payload["trainable_parameters"],
                "best_validation_loss": payload["best_validation_loss"],
                "best_epoch": payload["best_epoch"],
                "final_validation_loss": payload["final_validation_loss"],
                "target_validation_loss": payload["target"]["target_validation_loss"],
                "epoch_to_target": target["epoch_to_target"],
                "steps_to_target": target["steps_to_target"],
                "samples_to_target": target["samples_to_target"],
                "seconds_to_target": target["seconds_to_target"],
                "total_training_seconds": payload["total_training_seconds"],
                "mean_epoch_seconds": payload["mean_epoch_seconds"],
            }
        )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Saved:",
        OUTPUT,
    )

    print(
        "Rows:",
        len(rows),
    )


if __name__ == "__main__":
    main()
