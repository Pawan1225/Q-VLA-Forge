# Sprint 5.14I — Cross-Domain Safety Claim Matrix

## Claim status

- Supported: 5
- Supported with limitation: 1
- Not supported: 6

## Claims

| ID | Status | Claim |
|---|---|---|
| S5-CD01 | supported | A common safety architecture was instantiated across autonomous driving and robotics. |
| S5-CD02 | supported | Clean explicit safety filtering was effective in both pilot domains under their respective safety contracts. |
| S5-CD03 | supported_with_limitation | Gaussian robustness behavior was directionally consistent across all tested cross-domain comparisons. |
| S5-CD04 | supported | Structured-state robustness direction was consistent across driving and robotics. |
| S5-CD05 | supported | Positive explicit action recovery was observed in both domains for clipping and Lyapunov filtering. |
| S5-CD06 | not_supported | Lyapunov-specific candidate-selection activation was observed in both domains under action perturbation. |
| S5-CD07 | supported | Clean task-preservation direction was consistent across both domains for clipping and Lyapunov. |
| S5-CD08 | not_supported | The same safety thresholds were used across autonomous driving and robotics. |
| S5-CD09 | not_supported | The same trained policy weights transfer directly between driving and robotics. |
| S5-CD10 | not_supported | A universal cross-domain safety controller was demonstrated. |
| S5-CD11 | not_supported | The results establish a formal cross-domain safety guarantee. |
| S5-CD12 | not_supported | Production generalization across physical vehicles and robots was demonstrated. |

## Blocked claims

- S5-CD06
- S5-CD08
- S5-CD09
- S5-CD10
- S5-CD11
- S5-CD12

## Global limitations

- Synthetic proxy environments.
- Different domain-specific safety contracts.
- Different state dimensions.
- Different constraint semantics.
- Different Lyapunov functions.
- Separate trained policy weights.
- No cross-domain weight transfer.
- No zero-shot safety transfer.
- Three principal policy seeds.
- Privileged true safety state under Gaussian and structured-state observation perturbations.
- No physical vehicle evaluation.
- No physical robot evaluation.
- No formal stability proof.
- No certification claim.
- No production generalization claim.
- No quantum safety advantage claim.

## Proposal-safe statements

- Q-VLA Forge reused a common safety-filter interface, intervention protocol, robustness harness, and evaluation schema across autonomous-driving and robotics proxy environments while retaining domain-specific safety semantics.
- Explicit safety filtering reduced measured clean violations under both domain-specific pilot contracts.
- Action perturbation produced positive explicit recovery in both domains for clipping and Lyapunov filtering.
- Lyapunov-specific candidate-selection activation was observed in driving but not robotics under the tested action perturbations.
