"""Audit frozen structured-state robustness evidence for Sprint 5.13D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STATE_ROOT = ROOT / "results" / "safety" / "structured-state-robustness"


def _describe(
    value: Any,
    *,
    prefix: str,
    depth: int = 0,
    max_depth: int = 5,
) -> None:
    indent = "  " * depth

    if isinstance(
        value,
        dict,
    ):
        print(f"{indent}{prefix}: dict")

        for key in sorted(value):
            child = value[key]

            if depth >= max_depth:
                print(f"{indent}  {key}: " f"{type(child).__name__}")
                continue

            _describe(
                child,
                prefix=str(key),
                depth=depth + 1,
                max_depth=max_depth,
            )

        return

    if isinstance(
        value,
        list,
    ):
        print(f"{indent}{prefix}: " f"list[{len(value)}]")

        if value and depth < max_depth:
            _describe(
                value[0],
                prefix="[0]",
                depth=depth + 1,
                max_depth=max_depth,
            )

        return

    print(f"{indent}{prefix}: " f"{type(value).__name__} = {value!r}")


def main() -> None:
    print("=" * 100)
    print(" SPRINT 5.13D STRUCTURED-STATE SOURCE AUDIT")
    print("=" * 100)

    files = sorted(path for path in STATE_ROOT.rglob("*") if path.is_file())

    json_files = [path for path in files if path.suffix.lower() == ".json"]

    print()
    print(
        "Total files:",
        len(files),
    )

    print(
        "JSON files:",
        len(json_files),
    )

    print()
    print("FILES")

    for path in files:
        print(
            " ",
            path.relative_to(ROOT),
        )

    print()
    print("=" * 100)
    print("JSON SCHEMAS")
    print("=" * 100)

    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))

        print()
        print("-" * 100)

        print(path.relative_to(ROOT))

        if not isinstance(
            payload,
            dict,
        ):
            print(
                "TYPE:",
                type(payload).__name__,
            )
            continue

        print(
            "TOP KEYS:",
            sorted(payload.keys()),
        )

        for key in (
            "artifact",
            "condition",
            "domain",
            "method",
            "principal_seed",
            "state_variable",
            "perturbation",
            "magnitude",
            "direction",
            "cells",
            "aggregates",
            "findings",
            "seed_summary",
            "episodes",
        ):
            if key not in payload:
                continue

            _describe(
                payload[key],
                prefix=key,
                max_depth=5,
            )

    print()
    print("SPRINT 5.13D STRUCTURED-STATE SOURCE AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
