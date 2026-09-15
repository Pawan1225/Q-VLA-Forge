# Q-VLA Forge — Sprint 5 Safety Evidence

## Executive Summary

Sprint 5 packages the frozen safety and robustness evidence for autonomous-driving and robotics proxy domains without new training, policy execution, safety episodes, robustness episodes, retuning, or scientific experimentation.

Three explicit safety modes were evaluated: NONE, heuristic CLIPPING, and the classical Lyapunov-guided safety filter.

Under the frozen synthetic proxy safety contracts, both explicit safety filters reduced measured clean executed violation-step events to zero in both evaluated domains.

This is empirical proxy evidence, not a formal safety guarantee or certification result.

Robustness was evaluated under Gaussian observation noise, structured semantic state perturbations, and structured action perturbations.

The Gaussian and structured-state studies use perturbed policy observations while the explicit safety layer retains true simulator state. They therefore support policy-plus-privileged-state-safety robustness, not real-sensor robustness.

Under structured action perturbation, unsafe-action recovery was explicitly quantified. Environment interface adjustments remain separate from explicit safety-filter interventions.

Lyapunov-specific candidate selection became empirically active under action perturbation in autonomous driving, while robotics remained dominated by action/domain guards. LYAPUNOV_DECREASE reason counts and strict empirical ΔV < 0 counts remain separate.

The cross-domain result supports reuse of a shared safety framework and evaluation interface, not a universal trained controller, zero-shot transfer, formal stability, production readiness, or quantum safety advantage.

## 1. Scope and Protocol

- Principal trained-policy seeds: 42, 123, 456
- Experimental uncertainty: mean ± sample SD across principal seeds
- Evaluation seeds per seed/condition: 20000–20019
- Scientific scope: FROZEN
- New training: NO
- New principal execution: NO
- New scientific experiment: NO

## 2. Safety Architecture

Frozen policy → proposed action → optional perturbation → explicit safety method → executed action → domain environment.

The safety framework shares interfaces, metrics, robustness harnesses, and seed protocol while preserving domain-specific constraints, clipping rules, predictors, and Lyapunov safety potentials.

## 3. Clean Safety Evaluation

| Domain | Method | Violation-step rate | Reward | Success | Intervention |
| --- | --- | ---: | ---: | ---: | ---: |
| Driving | NONE | 0.390841 ± 0.422427 | -9.200230 ± 1.935700 | 0.050000 ± 0.086603 | — |
| Driving | CLIPPING | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| Driving | LYAPUNOV | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| Robotics | NONE | 0.014667 ± 0.018536 | 0.632846 ± 0.028818 | 0.000000 ± 0.000000 | — |
| Robotics | CLIPPING | 0.000000 ± 0.000000 | 0.677900 ± 0.035786 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |
| Robotics | LYAPUNOV | 0.000000 ± 0.000000 | 0.677903 ± 0.035787 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |

Under the frozen synthetic proxy safety contracts, both explicit safety filters reduced measured clean executed violation-step events to zero in both evaluated domains.

This is empirical proxy evidence, not a formal safety guarantee or certification result.

## 4. Gaussian Observation Robustness

Frozen levels: σ = 0.01, 0.05, 0.10, with σ = 0 retained only as a clean reference.

Policy observation: perturbed. Safety state: true simulator state. Privileged safety state: YES.

## 5. Structured-State Robustness

Driving families: lane_offset, obstacle_distance, speed, heading_error.

Robotics families: robot_position, object_position, target_position. Original signed condition identifiers are preserved; positive and negative perturbations are not collapsed.

## 6. Structured-Action Robustness

Driving action families: steering, acceleration, braking. Robotics action families: delta_x, delta_y, gripper.

Environment-interface action adjustment remains separate from explicit CLIPPING or LYAPUNOV intervention.

## 7. Unsafe-Action Recovery

| Domain | Method | Unsafe | Recovered | Unresolved | Recovery rate |
| --- | --- | ---: | ---: | ---: | ---: |
| autonomous_driving | CLIPPING | 33161 | 28609 | 4552 | 0.862730 |
| autonomous_driving | LYAPUNOV | 30663 | 29338 | 1325 | 0.956788 |
| autonomous_driving | NONE | 30380 | 0 | 30380 | 0.000000 |
| robotics | CLIPPING | 5715 | 5715 | 0 | 1.000000 |
| robotics | LYAPUNOV | 5705 | 5705 | 0 | 1.000000 |
| robotics | NONE | 5717 | 0 | 5717 | 0.000000 |

## 8. Lyapunov Mechanism Analysis

| Domain | Regime | ACTION_BOUND | DOMAIN_CONSTRAINT | LYAPUNOV_DECREASE | Strict decrease |
| --- | --- | ---: | ---: | ---: | ---: |
| autonomous_driving | clean | 0 | 2505 | 0 | 0 |
| autonomous_driving | gaussian | 0 | 7037 | 0 | 0 |
| autonomous_driving | structured_state | 0 | 24859 | 0 | 0 |
| robotics | clean | 0 | 88 | 0 | 0 |
| robotics | gaussian | 0 | 227 | 0 | 0 |
| robotics | structured_state | 0 | 1133 | 0 | 0 |
| autonomous_driving | structured_action | 1248 | 30024 | 1584 | 70 |
| robotics | structured_action | 245 | 5705 | 0 | 0 |

`LYAPUNOV_DECREASE` intervention reason and strict empirical `ΔV < 0` are reported separately and must not be treated as equivalent.

## 9. Cross-Domain Safety Analysis

Cross-domain comparison uses within-domain normalized effects rather than treating the two safety contracts as numerically identical.

The shared framework is supported at the interface and infrastructure level. Policy weights, constraints, predictors, and Lyapunov semantics remain domain-specific.

## 10. Proposal-Ready Findings

- Explicit clean safety filtering was empirically effective under both frozen proxy-domain contracts.
- Gaussian, structured-state, and structured-action robustness were evaluated.
- Unsafe-action recovery was quantified separately from environment-interface adjustment.
- Lyapunov-specific action-selection evidence became active under action perturbation in driving but not robotics.
- Cross-domain reuse is supported as a shared framework, not as a universal controller.

### Table A — Clean safety

| Domain | Method | Violation-step rate | Reward | Success | Intervention |
| --- | --- | ---: | ---: | ---: | ---: |
| Driving | NONE | 0.390841 ± 0.422427 | -9.200230 ± 1.935700 | 0.050000 ± 0.086603 | — |
| Driving | CLIPPING | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| Driving | LYAPUNOV | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| Robotics | NONE | 0.014667 ± 0.018536 | 0.632846 ± 0.028818 | 0.000000 ± 0.000000 | — |
| Robotics | CLIPPING | 0.000000 ± 0.000000 | 0.677900 ± 0.035786 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |
| Robotics | LYAPUNOV | 0.000000 ± 0.000000 | 0.677903 ± 0.035787 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |

### Table B — Robustness

| Domain | Regime | Method | Worst violation Delta | Condition | Worst reward Delta | Condition |
| --- | --- | --- | ---: | --- | ---: | --- |
| autonomous_driving | gaussian | CLIPPING | 0.000000 | sigma_0.01 | -0.164254 | sigma_0.1 |
| autonomous_driving | gaussian | LYAPUNOV | 0.000000 | sigma_0.01 | -0.164254 | sigma_0.1 |
| autonomous_driving | gaussian | NONE | -0.003735 | sigma_0.01 | -0.130926 | sigma_0.05 |
| robotics | gaussian | CLIPPING | 0.000000 | sigma_0.01 | -0.000444 | sigma_0.01 |
| robotics | gaussian | LYAPUNOV | 0.000000 | sigma_0.01 | -0.000433 | sigma_0.01 |
| robotics | gaussian | NONE | 0.013667 | sigma_0.1 | -0.016308 | sigma_0.1 |
| autonomous_driving | structured_state | CLIPPING | 0.000000 | heading_error_minus_0p05 | -2.582322 | lane_offset_plus_0p10 |
| autonomous_driving | structured_state | LYAPUNOV | 0.000000 | heading_error_minus_0p05 | -2.582322 | lane_offset_plus_0p10 |
| autonomous_driving | structured_state | NONE | 0.006752 | lane_offset_plus_0p10 | -2.126991 | lane_offset_plus_0p10 |
| robotics | structured_state | CLIPPING | 0.000000 | object_x_minus_0p05 | -0.041698 | robot_x_minus_0p05 |
| robotics | structured_state | LYAPUNOV | 0.000000 | object_x_minus_0p05 | -0.041701 | robot_x_minus_0p05 |
| robotics | structured_state | NONE | 0.020333 | object_x_minus_0p05 | -0.038311 | robot_x_minus_0p05 |
| autonomous_driving | structured_action | NONE | 0.384552 | acceleration_plus_0p25 | -48.982704 | steering_minus_0p25 |
| autonomous_driving | structured_action | CLIPPING | 0.273333 | steering_plus_0p25 | -43.341146 | steering_minus_0p25 |
| autonomous_driving | structured_action | LYAPUNOV | 0.100500 | steering_plus_0p25 | -38.329736 | steering_minus_0p25 |
| robotics | structured_action | NONE | 0.375500 | gripper_plus_0p50 | -0.765698 | gripper_plus_0p50 |
| robotics | structured_action | CLIPPING | 0.000000 | delta_x_minus_0p10 | -0.243419 | delta_x_plus_0p25 |
| robotics | structured_action | LYAPUNOV | 0.000000 | delta_x_minus_0p10 | -0.242889 | delta_x_plus_0p25 |

### Table C — Action recovery

| Domain | Method | Unsafe | Recovered | Unresolved | Recovery rate |
| --- | --- | ---: | ---: | ---: | ---: |
| autonomous_driving | CLIPPING | 33161 | 28609 | 4552 | 0.862730 |
| autonomous_driving | LYAPUNOV | 30663 | 29338 | 1325 | 0.956788 |
| autonomous_driving | NONE | 30380 | 0 | 30380 | 0.000000 |
| robotics | CLIPPING | 5715 | 5715 | 0 | 1.000000 |
| robotics | LYAPUNOV | 5705 | 5705 | 0 | 1.000000 |
| robotics | NONE | 5717 | 0 | 5717 | 0.000000 |

### Table D — Mechanisms

| Domain | Regime | ACTION_BOUND | DOMAIN_CONSTRAINT | LYAPUNOV_DECREASE | Strict decrease |
| --- | --- | ---: | ---: | ---: | ---: |
| autonomous_driving | clean | 0 | 2505 | 0 | 0 |
| autonomous_driving | gaussian | 0 | 7037 | 0 | 0 |
| autonomous_driving | structured_state | 0 | 24859 | 0 | 0 |
| robotics | clean | 0 | 88 | 0 | 0 |
| robotics | gaussian | 0 | 227 | 0 | 0 |
| robotics | structured_state | 0 | 1133 | 0 | 0 |
| autonomous_driving | structured_action | 1248 | 30024 | 1584 | 70 |
| robotics | structured_action | 245 | 5705 | 0 | 0 |

## 11. Claim Boundary

The final frozen Sprint 5 claim boundary is embedded verbatim in the canonical JSON package. Supported claims may be reused exactly or weakened. Supported-with-limitation claims must retain their limitations. Not-supported claims may not be converted into positive Phase 1 conclusions.

## 12. Limitations

- Phase 1 used synthetic proxy environments.
- The shared Phase 1 VLA baseline is compact and does not represent a production-scale pretrained VLA backbone.
- Sprint 4 RL policies consume compact environment state rather than the Sprint 1 shared VLA latent representation.
- The principal trained-policy protocol uses three seeds.
- Safety evaluations use 20 held-out episodes per seed/condition in the principal protocol.
- The Phase 1 hybrid QML pathway uses a small simulated 4-qubit, 2-layer PQC.
- No quantum hardware execution was performed.
- TT/MPS compression was evaluated at pilot-model scale.
- No principal CARLA, nuScenes, or physical-vehicle evaluation was performed.
- No principal RLBench, LIBERO, Open X-Embodiment, or physical-robot evaluation was performed.
- During perception-perturbation experiments, the policy used perturbed/noisy observations while the safety layer retained access to true simulator state.
- Environment action-bound handling is accounted for separately from explicit safety-filter intervention.
- The implemented Lyapunov quantity is an empirical safety potential and does not constitute a formal closed-loop stability proof.
- No functional-safety certification or standards compliance is claimed.
- No production deployment or full-scale embedded VLA validation was performed.

## 13. Evidence Provenance

- **action_summary** — `results/safety/action-robustness/sprint5-action-robustness-summary.json` — SHA256 `a1c6d88f524a72aef6671b2755cdcf1ce6165d615a6ea231faca183827549e9d`
- **claim_boundary** — `results/safety/final/sprint5-final-claim-boundary.json` — SHA256 `c648001903f147347068e27859a8516aa82ca060be9ddfb7bb885d7da0b576d3`
- **clean_summary** — `results/safety/consolidated/sprint5-clean-three-seed-summary.json` — SHA256 `66bd013a96695bb813aa20e10e50245d28c0c4843f90fd6faaa5eac70da7cc42`
- **cross_domain_action** — `results/safety/cross-domain/sprint5-cross-domain-action.json` — SHA256 `86df69e91e4990e91cb8400d5b9aef96cdff394ebb6434e6290aeb0e8fc2d14f`
- **cross_domain_architecture** — `results/safety/cross-domain/sprint5-cross-domain-architecture.json` — SHA256 `75a086532fedfdc401b13f4a7d1b6ed62d9fff800a073d407cf51f13f0620df8`
- **cross_domain_lyapunov** — `results/safety/cross-domain/sprint5-cross-domain-lyapunov-mechanism.json` — SHA256 `bf7e48716f633d1173390ce05bb0106a80afcec0b348c7d72bf6d78ef9ec7d76`
- **cross_domain_package** — `results/safety/cross-domain/sprint5-cross-domain-package.json` — SHA256 `a34f96737db7e9d867138b4df76b99f7d09ec2befccf0adbeb531630f5d85cf5`
- **environment_action_handling** — `results/safety/action-robustness/environment-action-handling.json` — SHA256 `d4219868682a1df447261be3795280ca5ad952c36a3fc3a2e769552f1c73c6a0`
- **freeze_record** — `results/safety/final/sprint5-freeze-record.json` — SHA256 `19926dbcc71b286f3a4e544c06d6a395b6882165c53ad3dc5ab3306b62335236`
- **gaussian_summary** — `results/safety/gaussian-robustness/sprint5-gaussian-robustness-summary.json` — SHA256 `3eb72467a7cf056fe944cae10df85939f299172c4a7308307390241dc3f0b847`
- **limitations** — `results/safety/final/sprint5-final-limitations.json` — SHA256 `530adbb82ad8f9cf9762028a34f2999c087196459a8b1b6cc2fa7ad91899ad34`
- **lyapunov_attribution** — `results/safety/consolidated/sprint5-lyapunov-mechanism-attribution.json` — SHA256 `d38f5ad26b43a8f0186431daf0bf0c5df56ce443bdfd185320e41d35efe10a33`
- **safety_package** — `results/safety/consolidated/sprint5-safety-evidence-package.json` — SHA256 `8afdaa0739aaebb81b6bf1300b0a9d2a88b4304ad3b1b94f7e30b02e5cf60f12`
- **scientific_invariants** — `results/safety/final/sprint5-scientific-invariants.json` — SHA256 `b9ad2e96ca32d37077c89a6880a2c2f17bdc845c7b8fd606d7ed6fc9db03f7c3`
- **structured_state_summary** — `results/safety/structured-state-robustness/sprint5-structured-state-robustness-summary.json` — SHA256 `e50a4f82f2d37acff44b59d490c8f555bcdfae6716c1946ed224e5a0440daaba`
