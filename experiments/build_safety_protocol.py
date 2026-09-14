"""Build the frozen Sprint 5.1 safety protocol artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from q_vla_forge.safety.protocol import DEFAULT_SAFETY_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]

HANDOFF_PATH = ROOT / "results" / "rl" / "final" / "sprint4-handoff.json"
CONFIG_PATH = ROOT / "configs" / "safety_evaluation.yaml"

OUTPUT_DIR = ROOT / "results" / "safety" / "protocol"
JSON_PATH = OUTPUT_DIR / "sprint5-safety-protocol.json"
MARKDOWN_PATH = OUTPUT_DIR / "sprint5-safety-protocol.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value

    if isinstance(value, tuple):
        return [enum_value(item) for item in value]

    if isinstance(value, list):
        return [enum_value(item) for item in value]

    if isinstance(value, dict):
        return {key: enum_value(item) for key, item in value.items()}

    return value


def main() -> None:
    if not HANDOFF_PATH.exists():
        raise FileNotFoundError(f"Sprint 4 handoff missing: {HANDOFF_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Safety config missing: {CONFIG_PATH}")

    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if handoff["from_sprint"] != "4":
        raise ValueError("Expected Sprint 4 source handoff.")

    if handoff["to_sprint"] != "5":
        raise ValueError("Expected Sprint 5 target handoff.")

    if handoff["handoff_status"] != "ready":
        raise ValueError("Sprint 4 handoff is not ready.")

    protocol = enum_value(asdict(DEFAULT_SAFETY_PROTOCOL))

    artifact = {
        "artifact": "sprint5-safety-protocol",
        "sprint": "5",
        "sub_sprint": "5.1",
        "protocol_status": "frozen_before_safety_results",
        "protocol": protocol,
        "experimental_design": {
            "primary_policy": "classical_ppo",
            "domains": [
                "autonomous_driving",
                "robotics",
            ],
            "evaluation_episodes_per_cell": 20,
            "primary_metric": "violation_step_rate",
            "secondary_metrics": config["metrics"]["secondary"],
        },
        "action_contract": {
            "proposed_action_preserved": True,
            "executed_action_preserved": True,
            "environment_action_dimension": 3,
            "intervention_definition": (
                "L2(executed_action - proposed_action) " "> intervention_tolerance"
            ),
        },
        "state_contract": {
            "true_state_preserved": True,
            "observed_state_preserved": True,
            "safety_filter_uses_true_state": True,
            "limitation_disclosure_required": True,
        },
        "perturbation_order": config["perturbation_order"],
        "pilot_acceptance": config["pilot_acceptance"],
        "statistics": config["statistics"],
        "claims": config["claims"],
        "claim_boundaries": {
            "formal_certification_claimed": False,
            "production_safety_guarantee_claimed": False,
            "iso_compliance_claimed": False,
            "safety_improvement_claimed_at_5_1": False,
        },
        "sprint4_provenance": {
            "handoff_path": HANDOFF_PATH.relative_to(ROOT).as_posix(),
            "handoff_sha256": sha256_file(HANDOFF_PATH),
            "from_sprint": handoff["from_sprint"],
            "to_sprint": handoff["to_sprint"],
            "handoff_status": handoff["handoff_status"],
            "principal_seeds": handoff["frozen_inputs"]["principal_seeds"],
            "evaluation_seeds": handoff["frozen_inputs"]["evaluation_seeds"],
            "action_dimension": handoff["frozen_inputs"]["action_dimension"],
        },
        "results": {
            "safety_results_present": False,
            "filter_comparison_present": False,
            "robustness_results_present": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    JSON_PATH.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown = f"""# Q-VLA Forge — Sprint 5.1 Safety Protocol

## Status

**FROZEN BEFORE SAFETY RESULTS**

## Scientific question

Can heuristic clipping and Lyapunov-based safety filtering reduce
constraint violations and improve robustness under state/action
perturbations while preserving acceptable task performance?

## Primary policy

`classical_ppo`

## Safety methods

- `none`
- `clipping`
- `lyapunov`

## Principal seeds

- 42
- 123
- 456

## Held-out evaluation seeds

`20000–20019`

## Episodes per evaluation cell

20

## Primary safety metric

`violation_step_rate`

## Pilot acceptance criteria

- Mean violation-step-rate reduction: >= 20%
- Maximum reward degradation: <= 10%
- Maximum success-rate drop: <= 10 percentage points

These are internal pilot thresholds only and are not certification
or Volkswagen production-safety requirements.

## Robustness conditions

- clean
- Gaussian state perturbation
- structured/state perturbation
- action perturbation

Gaussian standard deviations:

`0.00, 0.01, 0.05, 0.10`

## Safety filter information

The primary Sprint 5 pilot safety filter receives simulator
ground-truth current state.

This assumption must remain disclosed in all proposal-facing claims.

## Statistics

- mean
- sample standard deviation
- paired seed deltas
- no inferential significance testing

## Claim state

- Clipping safety improvement: NOT YET TESTED
- Lyapunov safety improvement: NOT YET TESTED
- Lyapunov over clipping: NOT YET TESTED
- Perturbation robustness: NOT YET TESTED
- Cross-domain safety reuse: NOT YET TESTED

## Scientific boundaries

Sprint 5.1 establishes no safety improvement result,
formal safety proof, certification, ISO compliance,
or production safety guarantee.

## Sprint 4 provenance

Handoff status: `{handoff["handoff_status"]}`

SHA256:

`{sha256_file(HANDOFF_PATH)}`
"""

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print("=" * 56)
    print(" Q-VLA FORGE — SPRINT 5.1 SAFETY PROTOCOL")
    print("=" * 56)
    print()
    print("Primary policy: classical_ppo")
    print("Safety methods: none / clipping / lyapunov")
    print("Principal seeds: 42 / 123 / 456")
    print("Evaluation seeds: 20000..20019")
    print("Evaluation episodes per cell: 20")
    print("Primary metric: violation_step_rate")
    print("Minimum violation reduction: 20%")
    print("Maximum reward degradation: 10%")
    print("Maximum success drop: 10 percentage points")
    print("Gaussian std: 0.00 / 0.01 / 0.05 / 0.10")
    print("Safety filter state access: simulator true state")
    print("Significance testing: DISABLED")
    print("Formal certification claim: DISABLED")
    print("Safety results: NOT YET TESTED")
    print()
    print("SPRINT 5.1 PROTOCOL BUILT")


if __name__ == "__main__":
    main()
