# Sprint 5.14G — Cross-Domain Lyapunov Mechanism

## Safety architecture decomposition

1. Action-space guard — ACTION_BOUND
2. Domain constraint guard — DOMAIN_CONSTRAINT
3. Lyapunov candidate selection — LYAPUNOV_DECREASE

## Domain mechanism split

| Domain | Action bound | Domain constraint | Lyapunov decrease | Strict decrease | Selected lower | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| autonomous_driving | 1248 | 30024 | 1584 | 70 | 2255 | 0 |
| robotics | 245 | 5705 | 0 | 0 | 2632 | 0 |

## Activation consistency

- Driving activation observed: True
- Robotics activation observed: False
- Cross-domain activation consistent: False

## Global reconstruction

- ACTION_BOUND: 1493
- DOMAIN_CONSTRAINT: 35729
- LYAPUNOV_DECREASE: 1584
- NONE: 105194
- Strict decreases: 70
- Selected-lower steps: 4887
- Emergency fallback steps: 0

## Limitations

- Intervention-reason counts are reconstructed from frozen per-run rates using exact environment-step denominators and integer-integrity checks.
- Lyapunov-specific activation demonstrates use of the candidate-selection mechanism, not a formal closed-loop stability proof.
- Hard-guard and Lyapunov-specific intervention frequencies are empirical properties of the tested synthetic action perturbations.
- Cross-domain consistency refers to observed activation in both domains and does not imply equal activation magnitude.
