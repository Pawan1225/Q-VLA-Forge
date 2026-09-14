"""Inspect nested clean-safety schemas for Sprint 5.13B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FILES = (
    ROOT / "results" / "safety" / "baseline" / "sprint5-no-filter-safety-summary.json",
    ROOT / "results" / "safety" / "clipping" / "sprint5-clipping-safety-summary.json",
    ROOT
    / "results"
    / "safety"
    / "lyapunov-driving"
    / "sprint5-driving-lyapunov-summary.json",
    ROOT
    / "results"
    / "safety"
    / "lyapunov-robotics"
    / "sprint5-robotics-lyapunov-summary.json",
)


def describe(
    value: Any,
    *,
    prefix: str,
    depth: int = 0,
    max_depth: int = 4,
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

            describe(
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
        print(f"{indent}{prefix}: list[{len(value)}]")

        if value and depth < max_depth:
            describe(
                value[0],
                prefix="[0]",
                depth=depth + 1,
                max_depth=max_depth,
            )

        return

    print(f"{indent}{prefix}: " f"{type(value).__name__} = {value!r}")


def main() -> None:
    print("=" * 110)
    print(" SPRINT 5.13B CLEAN SAFETY NESTED SCHEMA AUDIT")
    print("=" * 110)

    for path in FILES:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))

        print()
        print("=" * 110)
        print(path.relative_to(ROOT))
        print("=" * 110)

        for key in (
            "domains",
            "comparison",
            "criterion_definition",
            "seed_level_comparisons",
            "three_method_summary",
            "effectiveness",
            "grasp_context",
            "mechanism_accounting",
        ):
            if key not in payload:
                continue

            describe(
                payload[key],
                prefix=key,
                max_depth=5,
            )

    print()
    print("SPRINT 5.13B NESTED SCHEMA AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
