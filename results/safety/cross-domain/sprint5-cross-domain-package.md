# Sprint 5.14K — Consolidated Cross-Domain Safety Package

## Research question

Which parts of the Q-VLA Forge safety architecture and empirical behavior generalize across autonomous driving and robotics, and which findings remain domain-specific?

## Evidence summary

- Clean consistency: 2/2
- Gaussian consistency: 7/9
- Structured-state consistency: 3/3
- Action-recovery consistency: 3/3
- Lyapunov activation cross-domain consistent: False

## Claim summary

- Supported: 5
- Supported with limitation: 1
- Not supported: 6

## Mechanism summary

- LYAPUNOV_DECREASE interventions: 1584
- Strict Lyapunov decreases: 70
- Selected-lower steps: 4887
- Emergency fallbacks: 0

## Proposal-safe conclusions

- A common safety architecture, intervention protocol, robustness harness, and evaluation schema were reused across autonomous driving and robotics.
- Clean explicit safety filtering was effective under both domain-specific pilot safety contracts.
- Structured-state robustness direction was consistent across the two domains for the tested methods.
- Positive action recovery was observed in both domains for clipping and Lyapunov filtering.
- Gaussian robustness was mostly but not universally directionally consistent, with seven of nine cross-domain comparisons aligned.
- Lyapunov-specific candidate-selection activation occurred in driving but not robotics under the tested action perturbations.
- The evidence supports reuse of the safety framework, not a universal controller, direct policy transfer, formal guarantee, or production deployment claim.

## Domain-specific findings

- Driving and robotics retain different safety constraints and threshold semantics.
- Driving and robotics use separate trained PPO policy weights.
- Driving and robotics use different Lyapunov potentials and domain predictors.
- Lyapunov-specific action-perturbation activation was observed only in autonomous driving.

## Limitations

- Synthetic proxy environments.
- Three principal policy seeds.
- Different domain-specific state contracts.
- Different domain-specific reward scales.
- No direct raw reward ranking across domains.
- No direct raw violation-rate ranking across domains.
- Gaussian and structured-state robustness use noisy or perturbed policy observations while the safety layer retains true simulator state.
- No cross-domain policy-weight transfer experiment.
- No physical vehicle evaluation.
- No physical robot evaluation.
- No formal closed-loop stability proof.
- No formal worst-case robustness proof.
- No certification claim.
- No production generalization claim.
- No quantum safety advantage claim.

## Source manifest

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| results/safety/cross-domain/sprint5-cross-domain-architecture.json | `75a086532fedfdc401b13f4a7d1b6ed62d9fff800a073d407cf51f13f0620df8` | 4644 |
| results/safety/cross-domain/sprint5-cross-domain-clean.json | `c9ce01c2c54e98f5521582f1c8a12fe33c6be0ac6e53bcd28709cb9e2db3ef5a` | 9887 |
| results/safety/cross-domain/sprint5-cross-domain-gaussian.json | `b59dd94960d1eece050851945e983972183848cf4c62ef8518c81f3237b56ad3` | 34289 |
| results/safety/cross-domain/sprint5-cross-domain-structured-state.json | `0d38d712b47d2adf000bc341811cac5a487544f87afdd0d622a7bd7c89f0641e` | 4840 |
| results/safety/cross-domain/sprint5-cross-domain-action.json | `86df69e91e4990e91cb8400d5b9aef96cdff394ebb6434e6290aeb0e8fc2d14f` | 5082 |
| results/safety/cross-domain/sprint5-cross-domain-lyapunov-mechanism.json | `bf7e48716f633d1173390ce05bb0106a80afcec0b348c7d72bf6d78ef9ec7d76` | 11333 |
| results/safety/cross-domain/sprint5-cross-domain-consistency.json | `4f53803896775842c251fc06bab6a114f813de8fb3955a3343b1c1306ff2536b` | 3547 |
| results/safety/cross-domain/sprint5-cross-domain-claim-matrix.json | `a0f277236fc63fcea31dc711959fa336bece781637b1492871d0f51b73e05668` | 7427 |
| results/safety/cross-domain/sprint5-cross-domain-figure-index.json | `f2ae8cbdee39bd3bca20411d11f3070639c871f3b4f74cf6dade8e7e9b72d1ec` | 3312 |
