# Q-VLA Forge - Sprint 7.6 Full-System Ablation

Phase 1 component evidence only. No new training, PPO execution,
QML execution, safety episodes, robustness runs, or matched integrated
factorial experiments were performed in Sprint 7.6.

## Phase 1 Component Evidence

The following independently frozen evidence exists:

- Compression: Sprint 7.3
- QML / RL policy comparison: Sprint 7.4
- Safety filtering: Sprint 7.5

These component results are not combined into synthetic full-system
performance values.

## Full-System Evidence Matrix

| Domain | Config | Compression | QML | Safety | Evidence |
|---|---|---:|---:|---:|---|
| autonomous_driving | baseline | no | no | no | COMPONENT_ONLY |
| autonomous_driving | a | yes | no | no | COMPONENT_ONLY |
| autonomous_driving | b | no | yes | no | COMPONENT_ONLY |
| autonomous_driving | c | no | no | yes | COMPONENT_ONLY |
| autonomous_driving | d | yes | yes | no | COMPONENT_ONLY |
| autonomous_driving | e | yes | no | yes | COMPONENT_ONLY |
| autonomous_driving | f | no | yes | yes | COMPONENT_ONLY |
| autonomous_driving | full | yes | yes | yes | COMPONENT_ONLY |
| robotics | baseline | no | no | no | COMPONENT_ONLY |
| robotics | a | yes | no | no | COMPONENT_ONLY |
| robotics | b | no | yes | no | COMPONENT_ONLY |
| robotics | c | no | no | yes | COMPONENT_ONLY |
| robotics | d | yes | yes | no | COMPONENT_ONLY |
| robotics | e | yes | no | yes | COMPONENT_ONLY |
| robotics | f | no | yes | yes | COMPONENT_ONLY |
| robotics | full | yes | yes | yes | COMPONENT_ONLY |

### Evidence Summary

- DIRECT: 0
- COMPONENT_ONLY: 16
- NOT_EVALUATED: 0

All sixteen Phase 1 factorial cells are classified as COMPONENT_ONLY.

No configuration is permitted to report synthetic end-to-end reward,
MSE, latency, success, violation rate, or other integrated metrics.

## Directly Evaluated vs Not Directly Evaluated

No exact Compression x QML x Safety factorial configuration was directly
executed as one matched end-to-end pipeline.

Sprint 4 RL/QML policies operated on compact environment state rather than
the Sprint 1 shared VLA latent representation. Therefore the independently
evaluated compression, policy, and safety pathways cannot be treated as one
measured integrated system.

## Interaction Effects

The following interaction effects are not established in Phase 1:

- Compression x QML
- Compression x Safety
- QML x Safety
- Compression x QML x Safety

Component-only evidence cannot be used to infer these effects.

## Phase 2 Integrated Factorial Plan

All sixteen domain/configuration cells require matched integrated execution
before they can be promoted from COMPONENT_ONLY to DIRECT evidence.

The Phase 2 plan contains no predicted performance, expected winner, or
fabricated target values.

## Scientific Conclusion

Phase 1 independently evaluated compression, QML policy, and safety-filter components across both proxy domains. Exact end-to-end factorial combinations were not directly executed, so independently measured component results are not combined into synthetic full-system performance estimates. Matched integrated factorial execution remains a Phase 2 validation objective.

## Interaction Conclusion

Compression, QML, and safety main effects or interaction effects cannot be estimated from the Phase 1 full-system factorial matrix because no matched integrated factorial configuration was directly executed.

## Claim Controls

- Quantum advantage claim: blocked
- Full-system superiority claim: blocked
- Integrated factorial validation claim: blocked
- Cross-domain zero-shot transfer claim: blocked
- Production-readiness claim: blocked
- Interaction-effect claims without direct evidence: blocked
