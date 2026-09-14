"""Audit frozen clean safety evidence for Sprint 5.13B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "NONE": (ROOT / "results" / "safety" / "baseline"),
    "CLIPPING": (ROOT / "results" / "safety" / "clipping"),
    "LYAPUNOV_DRIVING": (ROOT / "results" / "safety" / "lyapunov-driving"),
    "LYAPUNOV_ROBOTICS": (ROOT / "results" / "safety" / "lyapunov-robotics"),
}


def _describe_json(
    path: Path,
) -> None:
    payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))

    print()
    print("-" * 100)

    print(path.relative_to(ROOT))

    if isinstance(
        payload,
        dict,
    ):
        print("TYPE: dict")

        print(
            "TOP KEYS:",
            sorted(payload.keys()),
        )

        for candidate in (
            "cells",
            "runs",
            "results",
            "seed_results",
            "per_seed",
            "summary",
            "aggregate",
            "aggregates",
        ):
            if candidate not in payload:
                continue

            value = payload[candidate]

            print(
                f"{candidate}:",
                type(value).__name__,
            )

            if isinstance(
                value,
                list,
            ):
                print(
                    "  length:",
                    len(value),
                )

                if value:
                    first = value[0]

                    if isinstance(
                        first,
                        dict,
                    ):
                        print(
                            "  first keys:",
                            sorted(first.keys()),
                        )

            elif isinstance(
                value,
                dict,
            ):
                print(
                    "  keys:",
                    sorted(value.keys())[:30],
                )

    elif isinstance(
        payload,
        list,
    ):
        print("TYPE: list")

        print(
            "LENGTH:",
            len(payload),
        )

        if payload and isinstance(
            payload[0],
            dict,
        ):
            print(
                "FIRST KEYS:",
                sorted(payload[0].keys()),
            )

    else:
        print(
            "TYPE:",
            type(payload).__name__,
        )


def main() -> None:
    print("=" * 100)
    print(" SPRINT 5.13B CLEAN SAFETY SOURCE AUDIT")
    print("=" * 100)

    for (
        label,
        directory,
    ) in GROUPS.items():
        print()
        print("=" * 100)
        print(label)
        print("=" * 100)

        json_files = sorted(directory.glob("*.json"))

        print(
            "JSON files:",
            len(json_files),
        )

        for path in json_files:
            print(
                " ",
                path.name,
            )

        for path in json_files:
            _describe_json(path)

    print()
    print("SPRINT 5.13B CLEAN SOURCE AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
