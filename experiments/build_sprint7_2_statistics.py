"""Build Sprint 7.2 final mean +/- sample-standard-deviation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.evaluation.final_statistics_aggregation import (
    build_final_statistics,
    build_statistics_markdown,
    write_statistics_csv,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "final-validation" / "statistics"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one formatted JSON artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build the complete Sprint 7.2 statistics package."""

    print("=" * 70)
    print(" Q-VLA FORGE - SPRINT 7.2")
    print(" FINAL MEAN +/- SAMPLE STANDARD DEVIATION")
    print("=" * 70)
    print()

    payload = build_final_statistics(ROOT)

    json_path = OUTPUT_DIR / "final-statistics.json"

    csv_path = OUTPUT_DIR / "final-statistics.csv"

    markdown_path = OUTPUT_DIR / "final-statistics.md"

    write_json(
        json_path,
        payload,
    )

    write_statistics_csv(
        csv_path,
        payload,
    )

    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path.write_text(
        build_statistics_markdown(payload),
        encoding="utf-8",
    )

    print("Locked seeds: " f"{payload['statistics_definition']['locked_seeds']}")

    print("Statistical records: " f"{payload['record_count']}")

    print("Fixed-value records: " f"{payload['fixed_value_count']}")

    print(
        "Recomputed from seed values: "
        f"{payload['aggregation_modes']['recomputed_from_seed_values']}"
    )

    print(
        "Validated canonical summaries: "
        f"{payload['aggregation_modes']['validated_canonical_summary']}"
    )

    print()
    print("Artifacts:")

    print("  results/final-validation/statistics/" "final-statistics.json")

    print("  results/final-validation/statistics/" "final-statistics.csv")

    print("  results/final-validation/statistics/" "final-statistics.md")

    print()
    print("-" * 70)
    print("SPRINT 7.2 FINAL STATISTICS: PASS")


if __name__ == "__main__":
    main()
