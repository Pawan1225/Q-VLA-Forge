# Sprint 4 - Final Scientific Freeze

## Scope

Sprint 4 covers the RL and Hybrid QML pilot across autonomous-driving and robotics proxy environments.

## Protocol

- Principal seeds: 42, 123, 456
- Principal interaction budget: 20,000 steps
- Frozen paired-target evaluation protocol

## Methods

- Classical PPO
- Hybrid QML policy
- Parameter-matched classical control

## Primary Results

- Classical PPO target reach: 6/6
- Matched Classical target reach: 1/6
- Hybrid QML target reach: 0/6

## Compactness

- Driving actor reduction: 95.902883%
- Robotics actor reduction: 95.513748%

## Sample Efficiency

The frozen Sprint 4 evidence does not establish a robust Hybrid QML sample-efficiency advantage.

## Matched-Budget Ablation

- Driving mean-AUC direction: Hybrid QML
- Robotics mean-AUC direction: Matched Classical
- Cross-domain representation direction: not consistent

## Cross-Domain Result

Common hybrid architecture reuse is supported. Shared trained weights, transfer learning, zero-shot transfer, and universal-policy generalization were not demonstrated.

## Reproducibility

- Frozen artifact count: 51
- Frozen manifest groups: 17
- SHA256 scientific freeze manifest generated

## Evidence Inventory

- PPO baseline and frozen targets
- PQC foundation and hybrid policy
- Driving and robotics QML results
- Three-seed validation
- Sample-efficiency analysis
- Matched-budget ablation
- Cross-domain analysis
- Proposal evidence and dashboard package

## Supported Claims

- Classical PPO reached all 6 frozen targets.
- Compact hybrid actors reduced trainable actor parameters by about 95.9% in driving and 95.5% in robotics.
- Common hybrid architecture reuse is supported.

## Unsupported Claims

- Robust QML sample-efficiency advantage
- Quantum computational speedup
- Quantum-hardware advantage
- Universal-policy generalization
- Transfer learning
- Shared trained-weight generalization

## Limitations

- Synthetic proxy environments
- Three principal seeds
- Simulated four-qubit, two-layer PQC
- No quantum hardware
- No formal statistical-significance claim
- No Sprint 4 safety guarantee

## Final Sprint 4 Conclusion

Sprint 4 established a reproducible classical PPO baseline, a four-qubit hybrid PQC actor, a parameter-matched classical control, three-seed evaluation, sample-efficiency analysis, matched representation ablation, and cross-domain comparison across autonomous-driving and robotics proxy environments.

Classical PPO reached all six frozen targets, while the Hybrid QML policy reached none and the matched classical control reached one. The hybrid actors nevertheless reduced actor parameters by approximately 95.9% in driving and 95.5% in robotics.

Matched-budget analysis showed domain-dependent representation behavior rather than a consistent QML or classical compact-policy advantage. Sprint 4 therefore supports hybrid actor compactness and cross-domain architecture reuse, but not robust QML sample-efficiency advantage, quantum speedup, hardware advantage, or universal-policy generalization.

## Sprint 5 Handoff

Handoff status: ready

Sprint 5 will evaluate explicit safety filtering and robustness under clean and perturbed conditions.

Core question: Can heuristic clipping and Lyapunov-based safety filtering reduce constraint violations and improve robustness under state/action perturbations while preserving acceptable task performance?
