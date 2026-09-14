# Sprint 5.14H — Cross-Domain Safety Consistency

## Summary

- Clean consistency: {'count': 2, 'true': 2, 'false': 0, 'all_consistent': True}
- Gaussian consistency: {'count': 9, 'true': 7, 'false': 2, 'all_consistent': False}
- Structured-state consistency: {'count': 3, 'true': 3, 'false': 0, 'all_consistent': True}
- Action recovery consistency: {'count': 3, 'true': 3, 'false': 0, 'all_consistent': True}
- Lyapunov activation consistency: {'driving_activation_observed': True, 'robotics_activation_observed': False, 'consistent': False}

## Task preservation

| Method | Driving reward Δ | Robotics reward Δ | Driving success Δ | Robotics success Δ | Consistent |
|---|---:|---:|---:|---:|---|
| clipping | 3.987671 | 0.045054 | 0.283333 | 0.000000 | True |
| lyapunov | 3.987671 | 0.045057 | 0.283333 | 0.000000 | True |

## Cross-domain conclusions

- architecture_reuse: supported — The framework is shared, while constraints, predictors, Lyapunov potentials, and trained policy weights remain domain-specific.
- clean_safety_consistency: supported — Raw safety rates are not used to rank the domains because the contracts differ.
- gaussian_robustness_consistency: supported_with_limitation — Two Gaussian comparisons show cross-domain direction differences, and the safety layer uses privileged true state.
- structured_state_consistency: supported — Driving and robotics perturbation features are not semantically equivalent.
- action_recovery_consistency: supported — Positive recovery is observed in both domains for explicit filters, but recovery magnitudes are not required to match.
- lyapunov_activation_consistency: not_supported — Lyapunov-specific action-perturbation activation was observed in driving but not robotics.
