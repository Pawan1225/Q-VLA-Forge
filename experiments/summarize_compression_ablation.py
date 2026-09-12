from __future__ import annotations

import csv
import json
from pathlib import Path

INPUT = (
    Path("results") / "compression" / "ablation" / "compression-ablation-seed-42.json"
)

OUTPUT = (
    Path("results") / "compression" / "ablation" / "compression-ablation-summary.csv"
)


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))

    points = payload["points"]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "domain",
                "method",
                "configuration",
                "target",
                "selected_layer_count",
                "compressed_layer_count",
                "compression_ratio",
                "mse_change_percent",
                "mae_change_percent",
            ]
        )

        for point in points:
            writer.writerow(
                [
                    point["domain"],
                    point["method"],
                    point["configuration"],
                    point["target"],
                    len(point["selected_layers"]),
                    len(point["compressed_layers"]),
                    point["compression_ratio"],
                    point["mse_change_percent"],
                    point["mae_change_percent"],
                ]
            )

    print(
        "Saved:",
        OUTPUT,
    )

    print(
        "Rows:",
        len(points),
    )


if __name__ == "__main__":
    main()
