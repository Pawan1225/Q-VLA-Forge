# Sprint 5.13G — Safety Claim Matrix

- Total claims: 16
- Supported: 8
- Supported with limitations: 3
- Unsupported: 5

| ID | Status | Claim |
|---|---|---|
| S5-C01 | supported | The explicit safety filters reduced clean executed violation-step rate to zero in both pilot domains across all three principal seeds. |
| S5-C02 | supported | On the clean driving evaluation, clipping and Lyapunov filtering improved mean reward and success relative to the no-filter baseline. |
| S5-C03 | supported | On the clean robotics evaluation, clipping and Lyapunov filtering reduced violation-step rate to zero without reducing mean reward relative to the no-filter baseline. |
| S5-C04 | supported | The safety layer was evaluated under Gaussian observation noise while retaining access to the true simulator state. |
| S5-C05 | supported | The safety layer was evaluated under structured state perturbations while retaining access to the true simulator state. |
| S5-C06 | supported_with_limitations | Direct action perturbation produced explicit filter-recovery behavior across the tested pilot conditions. |
| S5-C07 | supported_with_limitations | The explicit Lyapunov-decrease intervention pathway activated under direct action perturbation. |
| S5-C08 | supported_with_limitations | Within this pilot, the explicit Lyapunov-decrease pathway was observed under action perturbation but not under clean, Gaussian-noise, or structured-state evaluation. |
| S5-C09 | supported | No emergency fallback was required in the action-perturbation Lyapunov evaluation. |
| S5-C10 | unsupported | The evaluated Lyapunov filter formally guarantees closed-loop stability. |
| S5-C11 | unsupported | The evaluated safety system provides formal worst-case robustness guarantees. |
| S5-C12 | unsupported | The evaluated safety system is certified for production autonomous-driving or robotics deployment. |
| S5-C13 | unsupported | Lyapunov filtering is globally superior to clipping. |
| S5-C14 | unsupported | The safety experiments demonstrate a quantum safety advantage. |
| S5-C15 | supported | The action-robustness evaluation distinguishes explicit safety-filter corrections from environment-interface action adjustments. |
| S5-C16 | supported | The reported three-seed statistics use the principal policy seeds as experimental units rather than treating episodes as independent policy replicates. |

## Global limitations

- All Sprint 5 evidence is from synthetic pilot environments and finite evaluation corpora.
- The statistical unit is the trained policy seed; only three principal seeds are available.
- No statistical significance testing is claimed.
- The safety layer retains true simulator state in the observation-perturbation evaluations.
- Action perturbations are synthetic and are not validated physical actuator-fault models.
- No formal Lyapunov stability theorem, formal worst-case robustness guarantee, production certification, ISO compliance, or quantum safety advantage is established.
