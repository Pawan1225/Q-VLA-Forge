# Sprint 5.14E — Structured-State Cross-Domain Safety

## Interpretation

Driving and robotics state perturbations are summarized within each domain. No one-to-one semantic equivalence between features is assumed.

## Domain summaries

| Domain | Method | Conditions | Growth | Zero growth | Worst violation delta | Median violation delta | Worst reward delta | Highest intervention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| autonomous_driving | none | 10 | 4 | 0 | 0.006752 | -0.002475 | -2.126991 | 0.000000 |
| autonomous_driving | clipping | 10 | 0 | 10 | 0.000000 | 0.000000 | -2.582322 | 0.424167 |
| autonomous_driving | lyapunov | 10 | 0 | 10 | 0.000000 | 0.000000 | -2.582322 | 0.424167 |
| robotics | none | 12 | 5 | 0 | 0.020333 | -0.000333 | -0.038311 | 0.000000 |
| robotics | clipping | 12 | 0 | 12 | 0.000000 | 0.000000 | -0.041698 | 0.035000 |
| robotics | lyapunov | 12 | 0 | 12 | 0.000000 | 0.000000 | -0.041701 | 0.034833 |

## Cross-domain direction

| Method | Driving | Robotics | Consistent |
|---|---|---|---|
| none | degraded | degraded | True |
| clipping | tie | tie | True |
| lyapunov | tie | tie | True |

## Limitations

- Driving and robotics perturbation features are not treated as semantically equivalent.
- The policy receives perturbed state observations while the safety layer and environment use true simulator state.
- Worst-case degradation refers only to the tested structured perturbation set.
- Results are based on three principal policy seeds.
