# Sprint 4.10 — Three-Seed RL Validation

No new principal RL training was performed.

## Paired Seed Validation

| Domain | Seed | Target | PPO Steps | QML Steps | QML Best | Gap to Target | QML Reached |
|---|---:|---:|---:|---:|---:|---:|---|
| autonomous_driving | 42 | -13.240821 | 12000 | None | -36.908997 | -23.668176 | False |
| autonomous_driving | 123 | -8.172757 | 15000 | None | -48.581392 | -40.408635 | False |
| autonomous_driving | 456 | -5.174480 | 13000 | None | -7.186716 | -2.012236 | False |
| robotics | 42 | 0.563622 | 9000 | None | -1.214293 | -1.777915 | False |
| robotics | 123 | 0.552575 | 9000 | None | -0.025765 | -0.578340 | False |
| robotics | 456 | 0.615590 | 12000 | None | 0.439033 | -0.176557 | False |

## Domain Summary

### autonomous_driving

Classical target reach: 3/3

Hybrid QML target reach: 0/3

Robust >=10% criterion: False

QML best reward: -30.892368 ± 21.343141

QML final reward: -35.444168 ± 18.964421

QML final success: 0.000000 ± 0.000000

QML best-reward target gap: -22.029682 ± 19.250568

### robotics

Classical target reach: 3/3

Hybrid QML target reach: 0/3

Robust >=10% criterion: False

QML best reward: -0.267008 ± 0.852655

QML final reward: -0.267008 ± 0.852655

QML final success: 0.000000 ± 0.000000

QML best-reward target gap: -0.844271 ± 0.833142

## Scientific Boundary

Target gaps are descriptive and are not sample-efficiency measurements.

Reward magnitudes and target gaps are not compared directly across domains because the reward scales differ.

No quantum speedup or hardware quantum advantage is claimed.
