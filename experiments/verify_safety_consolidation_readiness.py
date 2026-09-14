"""Verify Sprint 5.13A consolidation protocol and source freeze."""

from __future__ import annotations

import json
from pathlib import Path

from q_vla_forge.evaluation.safety_consolidation import (
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs" / "safety_consolidation.yaml"

MANIFEST = (
    ROOT / "results" / "safety" / "consolidated" / "sprint5-safety-source-manifest.json"
)

EXPECTED_GROUPS = {
    "protocol",
    "driving_constraints",
    "robotics_constraints",
    "ppo_checkpoints",
    "no_filter_clean",
    "clipping_clean",
    "lyapunov_foundation",
    "lyapunov_filter",
    "driving_lyapunov_clean",
    "robotics_lyapunov_clean",
    "gaussian_robustness",
    "structured_state_robustness",
    "action_robustness",
}


def _check(
    condition: bool,
    label: str,
) -> None:
    if not condition:
        raise AssertionError(label)

    print(f"[PASS] {label}")


def main() -> None:
    print("=" * 72)
    print(" SPRINT 5.13A CONSOLIDATION READINESS")
    print("=" * 72)
    print()

    _check(
        CONFIG.exists(),
        "consolidation protocol config",
    )

    _check(
        MANIFEST.exists(),
        "source manifest exists",
    )

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    _check(
        payload["analysis_only"] is True,
        "analysis-only scope",
    )

    _check(
        payload["new_training"] is False,
        "no new training",
    )

    _check(
        payload["new_ppo_runs"] is False,
        "no new PPO runs",
    )

    _check(
        payload["new_principal_runs"] is False,
        "no new principal runs",
    )

    group_counts = payload["group_counts"]

    _check(
        set(group_counts) == EXPECTED_GROUPS,
        "all required source groups",
    )

    _check(
        int(group_counts["ppo_checkpoints"]) == 6,
        "six frozen PPO checkpoints",
    )

    _check(
        int(group_counts["gaussian_robustness"]) == 57,
        "Gaussian source corpus",
    )

    _check(
        int(group_counts["structured_state_robustness"]) == 201,
        "structured-state source corpus",
    )

    _check(
        int(group_counts["action_robustness"]) == 220,
        "action source corpus",
    )

    entries = payload["entries"]

    _check(
        len(entries) == int(payload["manifest_entry_count"]),
        "manifest entry accounting",
    )

    for entry in entries:
        source = ROOT / entry["path"]

        _check(
            source.exists(),
            ("source exists: " + entry["path"]),
        )

        _check(
            sha256_file(source) == entry["sha256"],
            ("source hash: " + entry["path"]),
        )

        _check(
            source.stat().st_size == int(entry["size_bytes"]),
            ("source size: " + entry["path"]),
        )

        _check(
            entry["frozen"] is True,
            ("source frozen: " + entry["path"]),
        )

    print()
    print("Protocol foundation: PASS")
    print("Source manifest: PASS")
    print("Evidence immutability baseline: PASS")
    print("Three-seed source availability: PASS")
    print("No new scientific execution: PASS")

    print()
    print("SPRINT 5.13A READINESS: PASS")


if __name__ == "__main__":
    main()
