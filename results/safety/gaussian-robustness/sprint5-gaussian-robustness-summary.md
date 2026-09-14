# Sprint 5.10 — Gaussian Observation-Noise Robustness

## Protocol

- Domains: autonomous driving and robotics
- Methods: NONE, CLIPPING, LYAPUNOV
- Principal seeds: 42, 123, 456
- Gaussian sigma: 0.00, 0.01, 0.05, 0.10
- Sigma 0.00 is sourced from frozen clean evidence.
- New noisy episodes: 1080
- Conceptual total including clean references: 1440 episodes.
- PPO receives noisy observation; safety layer receives true simulator state.

## Aggregate robustness

| Domain | Method | Sigma | Violation rate | Delta vs clean | Reward | Reward delta | Success | Intervention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| autonomous_driving | clipping | 0.00 | 0.000000 | +0.000000 | -5.212559 | +0.000000 | 0.333333 | 0.417500 |
| autonomous_driving | clipping | 0.01 | 0.000000 | +0.000000 | -5.227306 | -0.014747 | 0.333333 | 0.412667 |
| autonomous_driving | clipping | 0.05 | 0.000000 | +0.000000 | -5.367814 | -0.155255 | 0.333333 | 0.393000 |
| autonomous_driving | clipping | 0.10 | 0.000000 | +0.000000 | -5.376813 | -0.164254 | 0.333333 | 0.367167 |
| autonomous_driving | lyapunov | 0.00 | 0.000000 | +0.000000 | -5.212559 | +0.000000 | 0.333333 | 0.417500 |
| autonomous_driving | lyapunov | 0.01 | 0.000000 | +0.000000 | -5.227306 | -0.014747 | 0.333333 | 0.412667 |
| autonomous_driving | lyapunov | 0.05 | 0.000000 | +0.000000 | -5.367814 | -0.155255 | 0.333333 | 0.393000 |
| autonomous_driving | lyapunov | 0.10 | 0.000000 | +0.000000 | -5.376813 | -0.164254 | 0.333333 | 0.367167 |
| autonomous_driving | none | 0.00 | 0.390841 | +0.000000 | -9.200230 | +0.000000 | 0.050000 | 0.000000 |
| autonomous_driving | none | 0.01 | 0.387106 | -0.003735 | -9.212793 | -0.012564 | 0.050000 | 0.000000 |
| autonomous_driving | none | 0.05 | 0.367805 | -0.023036 | -9.331156 | -0.130926 | 0.050000 | 0.000000 |
| autonomous_driving | none | 0.10 | 0.340836 | -0.050005 | -9.326114 | -0.125884 | 0.050000 | 0.000000 |
| robotics | clipping | 0.00 | 0.000000 | +0.000000 | 0.677900 | +0.000000 | 0.000000 | 0.014667 |
| robotics | clipping | 0.01 | 0.000000 | +0.000000 | 0.677456 | -0.000444 | 0.000000 | 0.011167 |
| robotics | clipping | 0.05 | 0.000000 | +0.000000 | 0.697552 | +0.019652 | 0.000000 | 0.011833 |
| robotics | clipping | 0.10 | 0.000000 | +0.000000 | 0.682872 | +0.004972 | 0.000000 | 0.017667 |
| robotics | lyapunov | 0.00 | 0.000000 | +0.000000 | 0.677903 | +0.000000 | 0.000000 | 0.014667 |
| robotics | lyapunov | 0.01 | 0.000000 | +0.000000 | 0.677470 | -0.000433 | 0.000000 | 0.010667 |
| robotics | lyapunov | 0.05 | 0.000000 | +0.000000 | 0.697595 | +0.019692 | 0.000000 | 0.011333 |
| robotics | lyapunov | 0.10 | 0.000000 | +0.000000 | 0.683234 | +0.005331 | 0.000000 | 0.015833 |
| robotics | none | 0.00 | 0.014667 | +0.000000 | 0.632846 | +0.000000 | 0.000000 | 0.000000 |
| robotics | none | 0.01 | 0.015167 | +0.000500 | 0.629855 | -0.002991 | 0.000000 | 0.000000 |
| robotics | none | 0.05 | 0.013667 | -0.001000 | 0.645767 | +0.012922 | 0.000000 | 0.000000 |
| robotics | none | 0.10 | 0.028333 | +0.013667 | 0.616538 | -0.016308 | 0.000000 | 0.000000 |

## Mechanism findings

- Noisy LYAPUNOV strict-decrease steps: 0
- Noisy LYAPUNOV reason counts: `{'domain_constraint': 7264, 'none': 28736}`
- Robotics grasped-state steps under noisy LYAPUNOV runs: 357

## Limitations

- Gaussian noise is an uncalibrated synthetic state-coordinate perturbation.
- It is not a production sensor-error model.
- The policy receives noisy observations while the safety layer retains true simulator state.
- No formal robustness, certified safety, or real-sensor robustness claim is made.
