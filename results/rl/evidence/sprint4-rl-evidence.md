# Sprint 4 - RL & Hybrid QML Evidence

## Research Question

Can a compact hybrid quantum-classical policy architecture be reused across autonomous-driving and robotics proxy domains while preserving useful RL learning behavior under a matched interaction budget->

## Experimental Protocol

- Principal seeds: 42, 123, 456
- Principal interaction budget: 20,000 environment steps per run
- Primary cross-method metric: normalized learning-curve AUC
- Sprint 4.14 performs evidence packaging only; no new training or metrics were introduced.

## Policy Architectures

- Full PPO: classical actor and classical critic.
- Matched Classical: classical actor matched to the compact QML actor parameter budget.
- Hybrid QML: four-qubit, two-layer variational quantum actor with a classical value critic.

## Primary Target-Reach Results

- Full PPO: 6/6
- Matched Classical: 1/6
- Hybrid QML: 0/6

### Autonomous Driving

| Method | Actor Params | Target Reach | Normalized AUC Mean | AUC SD | Best Progress Mean | Final Progress Mean |
|---|---:|---:|---:|---:|---:|---:|
| Full PPO | 1318 | 3/3 | 0.554811 | 0.264606 | 1.052632 | 0.995282 |
| Matched Classical | 54 | 0/3 | -0.174555 | 0.209167 | 0.700218 | -0.048407 |
| Hybrid QML | 54 | 0/3 | 0.159912 | 0.434197 | 0.441827 | 0.327289 |

### Robotics

| Method | Actor Params | Target Reach | Normalized AUC Mean | AUC SD | Best Progress Mean | Final Progress Mean |
|---|---:|---:|---:|---:|---:|---:|
| Full PPO | 1382 | 3/3 | 0.922914 | 0.032207 | 1.052632 | 1.015904 |
| Matched Classical | 62 | 1/3 | 0.653060 | 0.064081 | 0.901652 | 0.897808 |
| Hybrid QML | 62 | 0/3 | 0.573909 | 0.193446 | 0.758545 | 0.758545 |

## Sample-Efficiency Analysis

Classical PPO reached all six paired frozen targets across both domains. The tested hybrid QML policies reached none within the frozen 20,000-step budget, so no robust hybrid-QML sample-efficiency advantage was demonstrated.

## Matched-Budget Ablation

Autonomous driving: hybrid QML achieved higher mean normalized learning-curve AUC than the parameter-matched classical actor.

Robotics: the parameter-matched classical actor achieved higher mean normalized learning-curve AUC than hybrid QML.

The representation effect therefore changed direction across domains.

## Cross-Domain Comparison

Cross-domain architectural reuse is supported: the same four-qubit, two-layer hybrid QML policy architecture and common PPO evaluation framework were reused across both proxy domains.

Directionally consistent matched-budget representation advantage is not supported.

## Compactness

- Driving: 1318 -> 54 actor parameters (95.90% reduction).
- Robotics: 1382 -> 62 actor parameters (95.51% reduction).

## Reproducibility

- Principal seeds: 42, 123, 456.
- Driving hybrid-QML seed-42 reproduction: PASS.
- Robotics hybrid-QML seed-42 reproduction: PASS.
- Driving matched-classical seed-42 reproduction: PASS.
- Robotics matched-classical seed-42 reproduction: PASS.
- Frozen PPO target and summary hashes remained unchanged.

## Supported Claims

- Classical PPO sample efficiency was evaluated under the frozen Sprint 4 protocol.
- Classical PPO baselines were established for autonomous-driving and robotics proxy domains using principal seeds 42, 123, and 456.
- A hybrid quantum-classical PPO policy with a variational quantum actor and classical value critic was implemented and evaluated in both proxy domains.
- The hybrid PQC actor substantially reduced actor parameter count relative to the classical PPO actor in both proxy domains.
- A common hybrid QML policy architecture and PPO evaluation framework was reused across both autonomous-driving and robotics proxy domains.

## Unsupported Claims

- The pilot does not establish a robust hybrid-QML policy advantage over classical controls.
- The pilot does not establish directionally consistent matched-budget RL representation behavior across the autonomous-driving and robotics proxy domains.
- The pilot does not establish a hybrid-QML sample-efficiency advantage for autonomous driving.
- The pilot does not establish a hybrid-QML sample-efficiency advantage for robotics.
- The pilot does not establish a robust hybrid-QML sample-efficiency advantage across the six principal domain-seed comparisons.
- Sprint 4 does not establish quantum computational speedup.
- Sprint 4 does not establish quantum hardware advantage.
- The pilot does not establish a directionally consistent matched-budget representation advantage across both proxy domains.

## Limitations

- Synthetic proxy environments rather than production autonomous-driving or robotics benchmarks.
- Compact state vectors were used instead of a full production-scale VLA latent input.
- No CARLA, RLBench, or LIBERO deployment was performed in Sprint 4.
- No quantum hardware was used.
- PennyLane default.qubit analytic simulation was used for the PQC.
- The PQC used four qubits.
- The PQC used two variational layers.
- Three principal seeds were evaluated.
- The principal interaction budget was 20,000 environment steps per run.
- No formal statistical significance test was used for the three-seed comparison.
- No transfer learning was evaluated.
- The same trained weights were not reused across both domains.
- No wall-clock quantum computational advantage is claimed.
- Safety filtering is outside Sprint 4 and is evaluated separately in Sprint 5.

## Proposal-Ready Summary

Classical PPO reached all six paired frozen reward targets across three seeds in both proxy domains, whereas the tested hybrid QML policies reached none within the 20,000-step interaction budget.

The hybrid PQC actors reduced trainable actor parameters from 1,318 to 54 in autonomous driving and from 1,382 to 62 in robotics, corresponding to reductions of approximately 95.9% and 95.5%.

Parameter-matched ablation showed domain-dependent representation behavior: hybrid QML achieved higher mean normalized learning-curve AUC than the matched classical control in driving, while the matched classical control achieved higher mean AUC in robotics.

The same four-qubit, two-layer hybrid QML policy architecture was reused across both proxy domains under a common PPO and evaluation protocol.

These pilot results support compact cross-domain architectural reuse, but do not demonstrate robust QML sample-efficiency advantage, quantum computational speedup, or quantum-hardware advantage.
