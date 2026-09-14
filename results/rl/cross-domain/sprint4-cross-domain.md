# Sprint 4.13 — Cross-Domain RL Comparison

Sprint 4.13 performs no new RL training. It compares the frozen Full PPO, matched-classical, and hybrid-QML evidence across the autonomous-driving and robotics proxies.

## Target Reach

| Domain | Full PPO | Matched Classical | Hybrid QML |
|---|---:|---:|---:|
| Autonomous Driving | 3/3 | 0/3 | 0/3 |
| Robotics | 3/3 | 1/3 | 0/3 |

## Normalized Learning-Curve AUC

| Domain | Method | Actor Params | Normalized AUC | Best Progress | Final Progress |
|---|---|---:|---:|---:|---:|
| Autonomous Driving | full_ppo | 1318 | 0.554811 ± 0.264606 | 1.052632 ± 0.000000 | 0.995282 ± 0.053598 |
| Autonomous Driving | matched_classical | 54 | -0.174555 ± 0.209167 | 0.700218 ± 0.248420 | -0.048407 ± 0.589352 |
| Autonomous Driving | hybrid_qml | 54 | 0.159912 ± 0.434197 | 0.441827 ± 0.472874 | 0.327289 ± 0.420983 |
| Robotics | full_ppo | 1382 | 0.922914 ± 0.032207 | 1.052632 ± 0.000000 | 1.015904 ± 0.013050 |
| Robotics | matched_classical | 62 | 0.653060 ± 0.064081 | 0.901652 ± 0.087655 | 0.897808 ± 0.090938 |
| Robotics | hybrid_qml | 62 | 0.573909 ± 0.193446 | 0.758545 ± 0.238557 | 0.758545 ± 0.238557 |

## Matched-Budget Representation Comparison

| Domain | Matched Params | QML Params | Matched AUC | QML AUC | Δ Matched−QML | Direction |
|---|---:|---:|---:|---:|---:|---|
| Autonomous Driving | 54 | 54 | -0.174555 | 0.159912 | -0.334467 | hybrid_qml |
| Robotics | 62 | 62 | 0.653060 | 0.573909 | +0.079151 | matched_classical |

## Actor Compactness

| Domain | Full PPO | Compact Actor | Reduction |
|---|---:|---:|---:|
| Autonomous Driving | 1318 | 54 | 95.902883% |
| Robotics | 1382 | 62 | 95.513748% |

## Architecture Reuse Scorecard

| Property | Driving | Robotics | Cross-Domain |
|---|---|---|---|
| Same 4-qubit PQC core | Yes | Yes | Reused |
| Same 2-layer circuit | Yes | Yes | Reused |
| Same 16 PQC parameters | Yes | Yes | Reused |
| Same 3-D action output | Yes | Yes | Reused |
| Same PPO protocol | Yes | Yes | Reused |
| Same 20k budget | Yes | Yes | Reused |
| Same training seeds | Yes | Yes | Reused |
| Same evaluation seeds | Yes | Yes | Reused |
| Same trained weights | No | No | Not claimed |
| Same observation dimension | No | No | Domain-specific |
| Robust QML target reach | No | No | Not supported |
| Compactness ~95% | Yes | Yes | Consistent |

## Cross-Domain Interpretation

The common hybrid policy architecture was reused across both proxy domains, but the matched-budget representation effect was not directionally consistent.

Driving matched-budget AUC direction: `hybrid_qml`.

Robotics matched-budget AUC direction: `matched_classical`.

Therefore, the pilot does not support a robust cross-domain QML learning advantage or a robust cross-domain matched-classical advantage.

## Domain-Dependent Representation Behavior

At the matched compact actor budget, the representation effect was domain dependent. Hybrid QML achieved higher mean normalized learning-curve AUC than the matched classical control in the autonomous-driving proxy, while the matched classical control achieved higher mean normalized AUC in robotics.

Best normalized progress favored `matched_classical` in driving and `matched_classical` in robotics.

Final normalized progress favored `hybrid_qml` in driving and `matched_classical` in robotics.

These results are interpreted as domain-dependent behavior under the tested proxy environments, not as a universal property of either representation.

## Seed Variability

Hybrid-QML normalized-AUC sample SD was 0.434197 in driving and 0.193446 in robotics.

Under the tested protocol, the observed hybrid-QML learning trajectories showed greater seed-to-seed dispersion in the driving proxy than in the robotics proxy.

This is a descriptive observation from three principal seeds per domain and is not claimed as a general property of autonomous driving versus robotics.

## Scientific Boundary

Cross-domain architectural reuse means that the same conceptual hybrid policy framework, PQC design, PPO protocol, budget, seed set, and validation methodology were used in both domains.

It does not mean that the same trained weights were shared across domains.

No transfer learning, zero-shot transfer, universal-policy claim, quantum speedup, or quantum hardware advantage is claimed.

Cross-domain comparisons use normalized target-relative metrics rather than raw rewards because the two environment reward functions differ.

No formal significance testing is performed because there are only three principal seeds per domain.
