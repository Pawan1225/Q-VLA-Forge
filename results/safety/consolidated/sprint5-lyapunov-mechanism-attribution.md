# Sprint 5.13F — Lyapunov Mechanism Attribution

## Cross-regime result

| Regime | Lyapunov reasons | Strict decreases | Active |
|---|---:|---:|:---:|
| clean | 0 | 0 | False |
| gaussian_state_perturbation | 0 | 0 | False |
| structured_state_perturbation | 0 | 0 | False |
| action_perturbation | 1584 | 70 | True |

## Interpretation

The tested Lyapunov-decrease pathway remained inactive in clean and observation-perturbation regimes, but activated under direct action perturbation.

This is mechanism evidence within the synthetic pilot protocol and is not a formal Lyapunov stability, production-safety, or superiority claim.
