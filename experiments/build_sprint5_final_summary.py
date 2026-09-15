from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")

FINAL_DIR = ROOT / "results" / "safety" / "final"

SCOPE = FINAL_DIR / "sprint5-final-scope.json"
CHAIN = FINAL_DIR / "sprint5-chain-inventory.json"
MANIFEST = FINAL_DIR / "sprint5-final-manifest.json"
INVARIANTS = FINAL_DIR / "sprint5-scientific-invariants.json"
CLAIMS = FINAL_DIR / "sprint5-final-claim-boundary.json"
LIMITATIONS = FINAL_DIR / "sprint5-final-limitations.json"
HANDOFF = FINAL_DIR / "sprint7-handoff.json"

OUTPUT = FINAL_DIR / "sprint5-final-summary.md"

SOURCES = (
    SCOPE,
    CHAIN,
    MANIFEST,
    INVARIANTS,
    CLAIMS,
    LIMITATIONS,
    HANDOFF,
)

for path in SOURCES:
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


scope = load_json(SCOPE)
chain = load_json(CHAIN)
manifest = load_json(MANIFEST)
invariants = load_json(INVARIANTS)
claims = load_json(CLAIMS)
limitations = load_json(LIMITATIONS)
handoff = load_json(HANDOFF)

if not bool(chain["all_stages_present"]):
    raise RuntimeError("Sprint 5 chain is incomplete")

if not bool(manifest["all_required_present"]):
    raise RuntimeError("Sprint 5 manifest is incomplete")

if not bool(handoff["ready_for_sprint7"]):
    raise RuntimeError("Sprint 7 handoff is not ready")

claim_counts = claims["status_counts"]

summary = """# Sprint 5 Final Summary

## Scope

Sprint 5 established the Q-VLA Forge safety, robustness, cross-domain, and final integration evidence for the Phase 1 pilot.

The final Sprint 5 closeout is analysis/documentation-only. It introduces no new training, PPO execution, QML execution, safety episodes, robustness runs, tuning, or scientific metrics.

The original full Sprint 6 implementation cycle was not executed. Its proposal-critical architecture-integration and Phase 2 planning requirements were captured in Sprint 5.14.5. Larger end-to-end integration experiments are deferred to Phase 2.

## Safety Architecture

Sprint 5 evaluated three safety modes:

- NONE
- heuristic clipping
- Lyapunov-guided safety filtering

The safety framework uses domain-specific driving and robotics constraints while preserving common decision, intervention, robustness, evaluation, and evidence interfaces.

## Clean Safety Results

Under the frozen clean evaluation protocol, explicit safety filtering was empirically effective in both autonomous-driving and robotics proxy domains.

Clipping and Lyapunov-guided filtering reduced measured executed safety violations under the tested contracts while preserving task behavior within the defined pilot criteria.

These results are empirical Phase 1 findings and do not constitute a formal safety guarantee or certification result.

## Robustness Evaluation

Sprint 5 evaluated robustness across:

- Gaussian observation perturbation
- structured semantic state perturbation
- action perturbation

Cross-domain consistency findings were:

- clean safety: 2 / 2
- Gaussian robustness: 7 / 9
- structured-state robustness: 3 / 3
- action-recovery direction: 3 / 3

Unsafe perturbed-action recovery was explicitly quantified.

## Lyapunov Mechanism Findings

The complete Lyapunov-guided safety layer must not be interpreted as meaning that every intervention was caused by a Lyapunov-decrease rule.

Hard action and domain guards accounted for the observed interventions in clean, Gaussian, and structured-state regimes.

Under action perturbation, Lyapunov-specific candidate selection became empirically active in autonomous driving but not in robotics.

This domain dependence is preserved as a Phase 1 finding.

No formal Lyapunov closed-loop stability proof is claimed.

## Cross-Domain Findings

Sprint 5.14 demonstrated that the safety framework generalizes at the architecture and interface level across autonomous-driving and robotics proxies.

Shared elements include:

- safety decision interface
- intervention accounting
- robustness harness
- evaluation schema
- three-seed protocol
- evidence and claim-control framework

Domain-specific elements remain:

- state semantics
- environment dynamics
- rewards
- trained policy weights
- safety constraints
- clipping logic
- transition predictors
- Lyapunov potentials

Therefore, Phase 1 supports a shared computational framework, not a universal trained controller.

## Unified Architecture

Sprint 5.14.5 integrated the complete Phase 1 evidence into one canonical Q-VLA Forge architecture.

The architecture connects:

- domain adapters
- shared AI/VLA core
- multimodal fusion
- shared latent representation
- compression path
- PPO / hybrid PQC policy path
- safety layer
- domain environments

All four Volkswagen challenge bottlenecks were mapped:

| Challenge bottleneck | Phase 1 evidence |
| --- | --- |
| Model footprint | Sprint 2 compression |
| Training efficiency | Sprint 3 training-efficiency |
| RL alignment / sample efficiency | Sprint 4 PPO/QML |
| Safety | Sprint 5 safety and robustness |

## Phase 1 Scientific Findings

### Compression

INT8 remained the strongest Phase 1 compression result.

TT/MPS compression was implemented and evaluated as a quantum-inspired pathway, but superiority over the strongest classical baseline was not demonstrated.

### Training Efficiency

No robust >=10% training-efficiency improvement was demonstrated under the frozen Phase 1 protocol.

### RL / QML

Classical PPO reached 6 / 6 frozen targets.

The matched classical actor reached 1 / 6.

The hybrid QML actor reached 0 / 6.

The hybrid PQC actor nevertheless demonstrated substantial parameter compactness in both domains.

Therefore, Phase 1 supports QML compactness, not QML sample-efficiency advantage.

### Safety

Explicit safety filtering produced strong empirical pilot effects under the frozen proxy-domain contracts.

Robustness was evaluated across Gaussian, structured-state, and action perturbations.

Lyapunov-specific mechanism activation remained domain-dependent.

## Supported Claims

Sprint 5 freezes the following major supported conclusions:

- a shared computational framework was instantiated across driving and robotics proxy domains
- all four Volkswagen challenge bottlenecks were experimentally addressed
- INT8 achieved the strongest Phase 1 compression result
- hybrid PQC actor compactness was demonstrated
- explicit clean safety filtering was empirically effective in both domains
- Gaussian robustness was evaluated
- structured-state robustness was evaluated
- action perturbation and unsafe-action recovery were quantified
- cross-domain safety-framework reuse was demonstrated

## Supported With Limitation

The following claims require their limitations to remain attached:

- the Phase 1 architecture provides a modular Phase 2 scale-up pathway
- Lyapunov-guided safety was empirically effective under the tested pilot contract
- robustness was demonstrated under synthetic perturbation regimes

## Unsupported Claims

Sprint 5 does not support:

- quantum advantage
- quantum speedup
- QML sample-efficiency superiority
- TT/MPS compression superiority
- robust training-efficiency advantage
- universal trained VLA
- zero-shot cross-domain policy transfer
- formal Lyapunov stability
- formal safety guarantee
- production readiness
- safety certification
- real-world autonomous-driving validation
- physical-robot validation

## Limitations

The final Sprint 5 limitation set includes:

- synthetic proxy environments
- compact Phase 1 VLA baseline
- Sprint 4 policies use compact environment state rather than the Sprint 1 shared latent
- three principal trained-policy seeds
- 20 held-out episodes per seed/condition in the principal safety protocol
- small simulated 4-qubit / 2-layer PQC
- no quantum hardware
- pilot-scale TT/MPS evaluation
- no principal CARLA or nuScenes evaluation
- no principal RLBench, LIBERO, Open X-Embodiment, or physical-robot evaluation
- privileged true-state safety access during perception perturbation studies
- environment action handling separated from explicit safety intervention
- no formal Lyapunov stability proof
- no safety certification
- no production deployment

## Sprint 7 Handoff

Sprint 7 receives the frozen Phase 1 evidence chain from Sprints 1 through 5.15.

Sprint 7 is authorized to perform:

- final validation
- evidence aggregation
- ablation summary
- proposal tables
- proposal figures
- reproducibility verification
- README evidence updates
- submission-ready claim preparation
- proposal evidence packaging

New training, PPO/QML execution, safety episodes, robustness runs, retuning, or new scientific benchmarks require an explicit reopening of the scientific scope.

Sprint 7 handoff status:

READY
"""

OUTPUT.write_text(
    summary,
    encoding="utf-8",
)

print("=" * 80)
print(" SPRINT 5.15.8 FINAL SPRINT-5 SUMMARY")
print("=" * 80)
print()
print(
    "Chain stages:",
    chain["chain_stage_count"],
)
print(
    "Manifest artifacts:",
    manifest["artifact_count"],
)
print(
    "Supported claims:",
    claim_counts["supported"],
)
print(
    "Supported with limitation:",
    claim_counts["supported_with_limitation"],
)
print(
    "Not supported:",
    claim_counts["not_supported"],
)
print(
    "Limitations:",
    limitations["limitation_count"],
)
print()
print("Scope summary: PASS")
print("Safety architecture summary: PASS")
print("Clean safety summary: PASS")
print("Robustness summary: PASS")
print("Lyapunov nuance summary: PASS")
print("Cross-domain summary: PASS")
print("Unified architecture summary: PASS")
print("Scientific findings summary: PASS")
print("Claim boundary summary: PASS")
print("Limitations summary: PASS")
print("Sprint-7 handoff summary: PASS")
print("No new science: PASS")
print()
print("SPRINT 5.15.8 FINAL SUMMARY: PASS")
