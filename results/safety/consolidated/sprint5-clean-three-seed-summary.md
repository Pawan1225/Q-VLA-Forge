# Sprint 5.13B — Clean Three-Seed Safety Reconstruction

Analysis only. No new policy or environment execution.

| Domain | Method | Violation-step rate | Reward | Success | Intervention |
|---|---|---:|---:|---:|---:|
| autonomous_driving | none | 0.390841 ± 0.422427 | -9.200230 ± 1.935700 | 0.050000 ± 0.086603 | 0.000000 ± 0.000000 |
| autonomous_driving | clipping | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| autonomous_driving | lyapunov | 0.000000 ± 0.000000 | -5.212559 ± 8.335305 | 0.333333 ± 0.577350 | 0.417500 ± 0.419514 |
| robotics | none | 0.014667 ± 0.018536 | 0.632846 ± 0.028818 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| robotics | clipping | 0.000000 ± 0.000000 | 0.677900 ± 0.035786 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |
| robotics | lyapunov | 0.000000 ± 0.000000 | 0.677903 ± 0.035787 | 0.000000 ± 0.000000 | 0.014667 ± 0.018536 |

## Effectiveness

- autonomous_driving / clipping: SUPPORTED
- autonomous_driving / lyapunov: SUPPORTED
- robotics / clipping: SUPPORTED
- robotics / lyapunov: SUPPORTED

## Reporting limitation

Critical and constraint violation rates are not exposed per seed for every clean method by the frozen summary artifacts. They are therefore not fabricated in this three-seed reconstruction.
