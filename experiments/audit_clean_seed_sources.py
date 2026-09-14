"""Locate raw clean per-seed safety artifacts for Sprint 5.13B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DIRECTORIES = (
    ROOT / "results" / "safety" / "baseline",
    ROOT / "results" / "safety" / "clipping",
    ROOT / "results" / "safety" / "lyapunov-driving",
    ROOT / "results" / "safety" / "lyapunov-robotics",
)


def _describe_json(
    path: Path,
) -> None:
    payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))

    print()
    print(path.relative_to(ROOT))

    if isinstance(
        payload,
        dict,
    ):
        print(
            "  top keys:",
            sorted(payload.keys()),
        )

        for key in (
            "domain",
            "method",
            "principal_seed",
            "evaluation_seed",
            "episode_count",
            "total_environment_steps",
            "seed_summary",
            "summary",
            "episodes",
        ):
            if key not in payload:
                continue

            value = payload[key]

            print(
                f"  {key}:",
                (
                    value
                    if not isinstance(
                        value,
                        (
                            dict,
                            list,
                        ),
                    )
                    else type(value).__name__
                ),
            )

            if isinstance(
                value,
                dict,
            ):
                print(
                    "    keys:",
                    sorted(value.keys()),
                )

            elif isinstance(
                value,
                list,
            ):
                print(
                    "    length:",
                    len(value),
                )

                if value and isinstance(
                    value[0],
                    dict,
                ):
                    print(
                        "    first keys:",
                        sorted(value[0].keys()),
                    )

    else:
        print(
            "  type:",
            type(payload).__name__,
        )


def main() -> None:
    print("=" * 100)
    print(" SPRINT 5.13B RAW CLEAN SEED SOURCE AUDIT")
    print("=" * 100)

    for directory in DIRECTORIES:
        print()
        print("=" * 100)
        print(directory.relative_to(ROOT))
        print("=" * 100)

        files = sorted(path for path in directory.iterdir() if path.is_file())

        for path in files:
            print(
                " ",
                path.name,
            )

        json_files = [path for path in files if path.suffix.lower() == ".json"]

        for path in json_files:
            _describe_json(path)

    print()
    print("SPRINT 5.13B RAW CLEAN SOURCE AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
