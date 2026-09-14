# Sprint 5.13H — Consolidated Safety Evidence Package

## Scope

Analysis-only consolidation of frozen Sprint 5 safety evidence. No new training or evaluation was performed.

## Clean safety results

| Domain | Method | Violation rate | Reward | Success |
|---|---|---:|---:|---:|
| autonomous_driving | none | 0.390841 | -9.200230 | 0.050000 |
| autonomous_driving | clipping | 0.000000 | -5.212559 | 0.333333 |
| autonomous_driving | lyapunov | 0.000000 | -5.212559 | 0.333333 |
| robotics | none | 0.014667 | 0.632846 | 0.000000 |
| robotics | clipping | 0.000000 | 0.677900 | 0.000000 |
| robotics | lyapunov | 0.000000 | 0.677903 | 0.000000 |

## Robustness corpus

- Gaussian: 72 seed cells, 24 aggregates.
- Structured state: 198 seed cells, 22 perturbations, 3960 episodes.
- Action: 216 seed cells, 24 perturbations, 4320 episodes.

## Action recovery

- Unsafe perturbed steps: 111341
- Recovered unsafe steps: 69367
- Unresolved unsafe steps: 41974
- Aggregate recovery fraction: 0.623013984

## Lyapunov mechanism attribution

- Active regimes: ['action_perturbation']
- Lyapunov-decrease reasons: 1584
- Strict decrease steps: 70
- Selected-lower steps: 4887
- Emergency fallback steps: 0

## Claim controls

- Supported: 8
- Supported with limitations: 3
- Unsupported / blocked: 5
- Blocked claim IDs: ['S5-C10', 'S5-C11', 'S5-C12', 'S5-C13', 'S5-C14']

## Headline findings

- Clean explicit safety filtering reduced observed violation-step rate to zero in both pilot domains across the three principal policy seeds.
- Gaussian and structured-state evaluations tested policy robustness to observation perturbations while the safety layer retained true simulator state.
- Under direct action perturbation, explicit safety filtering recovered 69367 of 111341 unsafe perturbed steps, corresponding to an aggregate recovery fraction of approximately 0.623.
- The explicit Lyapunov-decrease pathway activated under direct action perturbation, with 1584 Lyapunov-decrease intervention reasons and 70 strict Lyapunov-decrease steps.
- No formal stability, worst-case robustness, production certification, global Lyapunov superiority, ISO compliance, or quantum safety advantage claim is supported.

## Global limitations

- All Sprint 5 evidence is from synthetic pilot environments and finite evaluation corpora.
- The statistical unit is the trained policy seed; only three principal seeds are available.
- No statistical significance testing is claimed.
- The safety layer retains true simulator state in the observation-perturbation evaluations.
- Action perturbations are synthetic and are not validated physical actuator-fault models.
- No formal Lyapunov stability theorem, formal worst-case robustness guarantee, production certification, ISO compliance, or quantum safety advantage is established.
