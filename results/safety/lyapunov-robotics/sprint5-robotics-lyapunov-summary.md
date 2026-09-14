# Sprint 5.9 — Robotics Lyapunov Safety Summary

## Scope

- Domain: robotics
- Condition: clean
- Policy family: frozen classical PPO
- Principal seeds: 42, 123, 456
- Held-out evaluation seeds: 20000–20019
- Episodes: 60 total
- New training: no
- Policy fine-tuning: no
- Filter tuning: no
- Robustness perturbation: no

## Three-Method Comparison

| Method | Violation-step rate | Reward | Success | Intervention rate |
|---|---:|---:|---:|---:|
| NONE | 0.014667 ± 0.018536 | 0.632846 ± 0.028818 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| CLIPPING | 0.000000 ± 0.000000 | 0.677900 ± 0.035786 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |
| LYAPUNOV | 0.000000 ± 0.000000 | 0.677903 ± 0.035787 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |

## Pre-Registered Effectiveness Criterion

- Seed-level safety requirements: PASS
- Domain mean violation reduction: 1.000000 (PASS)
- Reward degradation fraction: 0.000000 (PASS)
- Success-rate drop: 0.000000 (PASS)

**SPRINT 5.9 ROBOTICS EMPIRICAL EFFECTIVENESS: PASS**

## Mechanism Evidence

- Environment steps: 6000
- Total interventions: 88
- DOMAIN_CONSTRAINT interventions: 88
- LYAPUNOV_DECREASE interventions: 0
- Emergency fallbacks: 0
- Selected hard-guard failures: 0
- Strict Lyapunov-decrease steps: 0 / 6000
- Lyapunov non-increase steps: 6000 / 6000
- Selected lower-than-proposed V steps: 34 / 6000

All principal action-changing interventions were triggered by frozen robotics domain hard guards. No `LYAPUNOV_DECREASE` intervention was observed.

## Grasp-Context Coverage

- `object_grasped=True`: 0 / 6000 steps
- `object_grasped=False`: 6000 / 6000 steps
- Policy observation remained the original 6-D observation.
- `object_grasped` was supplied only to the one-step safety predictor.

**Limitation:** The clean robotics principal evaluation did not enter the `object_grasped=True` regime. Therefore the grasp-aware prediction path is implemented and provenance-controlled, but its grasped-state branch was not empirically exercised by the Sprint 5.9 principal trajectories.

## Interpretation

Under the frozen clean robotics protocol, the complete Lyapunov-guided safety filter reduced the measured executed violation-step rate from the no-filter baseline to zero while preserving the pre-registered reward and success requirements.

The observed action changes were entirely attributable to the frozen domain hard guards rather than the strict Lyapunov-decrease selector. Clipping and the complete Lyapunov-guided filter are therefore compared descriptively only; this Sprint does not support a claim that the Lyapunov method is superior to clipping.

## Claim Controls

- No formal stability claim.
- No formal robotics safety guarantee.
- No production manipulation-safety claim.
- No ISO 10218 claim.
- No IEC 62061 claim.
- No Lyapunov-over-clipping superiority claim.
- No quantum safety advantage claim.
- No empirical validation claim for the grasped-state branch.
