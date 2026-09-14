# Sprint 5.14D — Gaussian Cross-Domain Safety

## Scope

Analysis-only comparison of frozen Gaussian observation-perturbation evidence.

## State separation

- Policy: noisy observation
- Safety layer: true simulator state

Therefore, this analysis characterizes the policy-plus-privileged-state-safety architecture.

## Direction consistency

| Method | Sigma | Driving direction | Robotics direction | Consistent |
|---|---:|---|---|---|
| none | 0.01 | improved | degraded | False |
| none | 0.05 | improved | improved | True |
| none | 0.10 | improved | degraded | False |
| clipping | 0.01 | tie | tie | True |
| clipping | 0.05 | tie | tie | True |
| clipping | 0.10 | tie | tie | True |
| lyapunov | 0.01 | tie | tie | True |
| lyapunov | 0.05 | tie | tie | True |
| lyapunov | 0.10 | tie | tie | True |

## Supported conclusion

The same Gaussian observation-perturbation framework was applied across both pilot domains. Cross-domain interpretation uses changes relative to each domain's own clean reference rather than comparing raw rewards or raw safety rates directly.

## Limitations

- The policy receives noisy observations while the safety layer retains true simulator state.
- This evaluates robustness of the policy-plus-privileged-state-safety architecture, not a safety controller operating under uncertain perception.
- Magnitude equality across domains is neither expected nor required for direction consistency.
- Results use three principal policy seeds and do not establish formal statistical significance.
