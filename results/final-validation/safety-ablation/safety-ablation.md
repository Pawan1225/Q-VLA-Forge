# Q-VLA Forge - Sprint 7.5 Final Safety Ablation

Frozen Sprint 5 clean-safety evidence only. No new safety episodes, controller tuning, Lyapunov redesign, reward changes, or robustness experiments were performed.

NONE is the descriptive reference. CLIPPING and LYAPUNOV are evaluated configurations. Sprint 7.5 does not define a new binary safety PASS/FAIL threshold.

## Autonomous Driving

| Method | Violation Rate | Reward | Success Rate | Intervention Rate | Role |
|---|---:|---:|---:|---:|---|
| None | 0.390841 +/- 0.422427 | -9.200230 +/- 1.935700 | 0.050000 +/- 0.086603 | Not defined | REFERENCE |
| Clipping | 0.000000 +/- 0.000000 | -5.212559 +/- 8.335305 | 0.333333 +/- 0.577350 | 0.417500 +/- 0.419514 | EVALUATED |
| Lyapunov | 0.000000 +/- 0.000000 | -5.212559 +/- 8.335305 | 0.333333 +/- 0.577350 | 0.417500 +/- 0.419514 | EVALUATED |

### Violation Change Relative to NONE

- Clipping absolute difference vs NONE: -0.390841
- Clipping relative reduction vs NONE: 100.00%
- Lyapunov absolute difference vs NONE: -0.390841
- Lyapunov relative reduction vs NONE: 100.00%

## Robotics

| Method | Violation Rate | Reward | Success Rate | Intervention Rate | Role |
|---|---:|---:|---:|---:|---|
| None | 0.014667 +/- 0.018536 | 0.632846 +/- 0.028818 | 0.000000 +/- 0.000000 | Not defined | REFERENCE |
| Clipping | 0.000000 +/- 0.000000 | 0.677900 +/- 0.035786 | 0.000000 +/- 0.000000 | 0.014667 +/- 0.018536 | EVALUATED |
| Lyapunov | 0.000000 +/- 0.000000 | 0.677903 +/- 0.035787 | 0.000000 +/- 0.000000 | 0.014667 +/- 0.018536 | EVALUATED |

### Violation Change Relative to NONE

- Clipping absolute difference vs NONE: -0.014667
- Clipping relative reduction vs NONE: 100.00%
- Lyapunov absolute difference vs NONE: -0.014667
- Lyapunov relative reduction vs NONE: 100.00%

## Lyapunov Mechanism Observation

The Sprint 5 Lyapunov-guided safety filter is classical. It is not a quantum safety algorithm.

- Autonomous Driving: LYAPUNOV_DECREASE intervention observed: no.
  Under the frozen clean driving protocol, the complete Lyapunov-guided filter achieved the measured safety outcome. Principal action changes were triggered by domain hard guards; no additional LYAPUNOV_DECREASE intervention was observed.
- Robotics: LYAPUNOV_DECREASE intervention observed: no.
  Under the frozen clean robotics protocol, the complete Lyapunov-guided safety filter eliminated all measured executed constraint violations. All observed principal interventions were triggered by frozen robotics domain hard guards. No LYAPUNOV_DECREASE intervention was observed in the principal runs.

## Scientific Conclusion

Under the frozen Phase 1 synthetic clean-safety contracts, both clipping and the classical Lyapunov-guided filter produced zero observed violation-step rate in the evaluated runs, compared with nonzero violation rates for the no-safety reference. These empirical observations do not constitute a formal stability, forward-invariance, certification, production-safety, real-world-safety, or quantum-safety guarantee.

No principal LYAPUNOV_DECREASE intervention was observed in either clean domain evaluation. Observed interventions were attributable to the frozen domain hard-guard mechanisms under the evaluated principal trajectories.

## Privileged-State Limitation

The safety layer operated on true simulator state while the policy could receive perturbed observations. The results therefore apply to the frozen Phase 1 simulator contract and do not establish estimated-state or real-world sensing robustness.

## Claim Controls

- Formal stability guarantee: blocked
- Formal Lyapunov stability proof: blocked
- Forward-invariance guarantee: blocked
- Worst-case safety guarantee: blocked
- ISO 26262 certification claim: blocked
- Production-safety claim: blocked
- Real-world-safety claim: blocked
- Universal safety-controller claim: blocked
- Quantum-safety advantage claim: blocked
- Guaranteed-zero-violations claim: blocked
