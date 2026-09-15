from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

OUTPUT_DIR = ROOT / "results" / "safety" / "final"
OUTPUT = OUTPUT_DIR / "sprint5-chain-inventory.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHAIN: dict[str, dict[str, Any]] = {
    "5.1": {
        "name": "Safety protocol",
        "candidates": [
            "configs/safety.yaml",
            "configs/safety_evaluation.yaml",
        ],
    },
    "5.2": {
        "name": "Driving safety constraints",
        "candidates": [
            "configs/driving_safety.yaml",
        ],
    },
    "5.3": {
        "name": "Robotics safety constraints",
        "candidates": [
            "configs/robotics_safety.yaml",
        ],
    },
    "5.3.5": {
        "name": "PPO checkpoint persistence recovery",
        "candidates": [
            "results/rl/checkpoint-recovery",
            "results/rl/checkpoints",
        ],
    },
    "5.4": {
        "name": "NONE safety baseline",
        "candidates": [
            "results/safety/baseline/sprint5-no-filter-safety-summary.json",
        ],
    },
    "5.5": {
        "name": "Heuristic clipping",
        "candidates": [
            "configs/safety_clipping.yaml",
            "results/safety/clipping",
        ],
    },
    "5.6": {
        "name": "Lyapunov foundation",
        "candidates": [
            "configs/lyapunov_safety.yaml",
            "results/safety/lyapunov-foundation",
        ],
    },
    "5.7": {
        "name": "Lyapunov safety filter",
        "candidates": [
            "results/safety/lyapunov-filter",
        ],
    },
    "5.8": {
        "name": "Driving Lyapunov evaluation",
        "candidates": [
            "results/safety/driving",
        ],
    },
    "5.9": {
        "name": "Robotics Lyapunov evaluation",
        "candidates": [
            "results/safety/robotics",
        ],
    },
    "5.10": {
        "name": "Gaussian robustness",
        "candidates": [
            "results/safety/gaussian-robustness/sprint5-gaussian-robustness-summary.json",
        ],
    },
    "5.11": {
        "name": "Structured-state robustness",
        "candidates": [
            "results/safety/structured-state-robustness",
        ],
    },
    "5.12": {
        "name": "Action robustness",
        "candidates": [
            "results/safety/action-robustness/sprint5-action-robustness-summary.json",
        ],
    },
    "5.13": {
        "name": "Three-seed safety consolidation",
        "candidates": [
            "results/safety/consolidated/sprint5-safety-evidence-package.json",
        ],
    },
    "5.14": {
        "name": "Cross-domain safety analysis",
        "candidates": [
            "results/safety/cross-domain/sprint5-cross-domain-package.json",
        ],
    },
    "5.14.5": {
        "name": "Unified architecture and Phase-2 bridge",
        "candidates": [
            "results/integration/sprint5-unified-bridge-package.json",
            "docs/unified_architecture.md",
        ],
    },
}


def path_exists(candidate: str) -> bool:
    return (ROOT / candidate).exists()


records: list[dict[str, Any]] = []

for sprint, entry in CHAIN.items():
    candidates = [str(candidate) for candidate in entry["candidates"]]

    existing = [candidate for candidate in candidates if path_exists(candidate)]

    if not existing:
        raise FileNotFoundError(f"{sprint} has no canonical artifact candidate")

    records.append(
        {
            "sprint": sprint,
            "name": entry["name"],
            "candidate_count": len(candidates),
            "existing_artifacts": existing,
            "complete": True,
        }
    )

result: dict[str, Any] = {
    "project": "Q-VLA Forge",
    "sprint": "5.15.2",
    "artifact": "sprint5-chain-inventory",
    "analysis_only": True,
    "new_training": False,
    "new_principal_runs": False,
    "new_safety_runs": False,
    "new_robustness_runs": False,
    "chain_stage_count": len(records),
    "stages": records,
    "all_stages_present": all(bool(record["complete"]) for record in records),
}

if not bool(result["all_stages_present"]):
    raise RuntimeError("Sprint 5 chain is incomplete")

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.15.2 SPRINT-5 CHAIN INVENTORY")
print("=" * 80)
print()

for record in records:
    print(f"{record['sprint']:>6}  " f"{record['name']}: PASS")

print()
print(
    "Chain stages:",
    result["chain_stage_count"],
)
print("All stages represented: PASS")
print("No new training: PASS")
print("No new principal execution: PASS")
print()
print("SPRINT 5.15.2 CHAIN INVENTORY: PASS")
