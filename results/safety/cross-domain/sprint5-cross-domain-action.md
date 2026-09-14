# Sprint 5.14F — Action-Recovery Cross-Domain Safety

## Architectural location

Policy → action perturbation → explicit safety filter → environment

## Domain/method recovery

| Domain | Method | Unsafe | Recovered | Unresolved | Recovery rate |
|---|---|---:|---:|---:|---:|
| autonomous_driving | none | 30380 | 0 | 30380 | 0.000000 |
| autonomous_driving | clipping | 33161 | 28609 | 4552 | 0.862730 |
| autonomous_driving | lyapunov | 30663 | 29338 | 1325 | 0.956788 |
| robotics | none | 5717 | 0 | 5717 | 0.000000 |
| robotics | clipping | 5715 | 5715 | 0 | 1.000000 |
| robotics | lyapunov | 5705 | 5705 | 0 | 1.000000 |

## Cross-domain recovery consistency

| Method | Driving recovery | Robotics recovery | Driving positive | Robotics positive | Consistent |
|---|---:|---:|---|---|---|
| none | 0.0 | 0.0 | False | False | True |
| clipping | 0.8627303157323362 | 1.0 | True | True | True |
| lyapunov | 0.9567883116459577 | 1.0 | True | True | True |

## Global reconstruction

- Unsafe perturbed steps: 111341
- Recovered unsafe steps: 69367
- Unresolved unsafe steps: 41974
- Recovery fraction: 0.623013984

## Limitations

- Driving and robotics action channels have different semantics despite sharing a three-dimensional continuous action interface.
- Recovery rate is undefined when a cell or aggregate contains no unsafe perturbed steps.
- Environment-interface adjustments remain distinct from explicit safety-filter interventions.
- The perturbations are synthetic action disturbances and are not calibrated physical actuator-fault models.
