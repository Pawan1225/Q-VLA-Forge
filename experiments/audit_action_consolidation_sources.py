"""Audit frozen action-robustness evidence for Sprint 5.13E."""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ACTION_ROOT = ROOT / "results" / "safety" / "action-robustness"


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
    print(" SPRINT 5.13E ACTION ROBUSTNESS SOURCE AUDIT")
    print("=" * 100)

    files = sorted(path for path in ACTION_ROOT.rglob("*") if path.is_file())

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

    summary_candidates = [path for path in json_files if "summary" in path.name.lower()]

    print()
    print("SUMMARY CANDIDATES")

    for path in summary_candidates:
        print(
            " ",
            path.relative_to(ROOT),
        )

    for path in summary_candidates:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))

        if not isinstance(
            payload,
            dict,
        ):
            continue

        print()
        print("=" * 100)
        print(path.relative_to(ROOT))
        print("=" * 100)

        print(
            "TOP KEYS:",
            sorted(payload.keys()),
        )

        for key in (
            "condition",
            "cells",
            "aggregates",
            "findings",
            "mechanism",
            "recovery",
            "claims",
        ):
            if key not in payload:
                continue

            _describe(
                payload[key],
                prefix=key,
                max_depth=5,
            )

        if (
            isinstance(
                payload.get("cells"),
                list,
            )
            and payload["cells"]
        ):
            print()
            print("FIRST CELL")
            pprint(
                payload["cells"][0],
                sort_dicts=True,
            )

        if (
            isinstance(
                payload.get("aggregates"),
                list,
            )
            and payload["aggregates"]
        ):
            print()
            print("FIRST AGGREGATE")
            pprint(
                payload["aggregates"][0],
                sort_dicts=True,
            )

    print()
    print("SPRINT 5.13E ACTION SOURCE AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
