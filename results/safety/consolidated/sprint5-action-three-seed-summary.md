# Sprint 5.13E — Action Robustness Consolidation

Analysis only. No new policy training or environment execution.

## Corpus

- Action perturbations: 24
- Principal cells: 216
- Principal episodes: 4320

## Explicit filter recovery

- Unsafe perturbed steps: 111341
- Recovered unsafe steps: 69367
- Unresolved unsafe steps: 41974
- Overall recovery fraction: 0.623013984

Undefined cell-level recovery rates remain null when no unsafe perturbed steps are present.

## Lyapunov mechanism

- ACTION_BOUND: 1493
- DOMAIN_CONSTRAINT: 35729
- LYAPUNOV_DECREASE: 1584
- NONE: 105194
- Strict decrease steps: 70
- Selected-lower steps: 4887
- Emergency fallback steps: 0
