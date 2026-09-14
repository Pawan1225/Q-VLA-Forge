# Sprint 5.14B — Cross-Domain Safety Architecture Reuse

## Scope

Architecture-only analysis. No training, policy execution, safety evaluation, robustness rerun, or threshold change was performed.

## Shared vs domain-specific architecture

| Component | Driving | Robotics | Reuse |
|---|---|---|---|
| policy_interface | PPO | PPO | Yes (interface) |
| action_dimension | 3 | 3 | Yes (interface) |
| safety_decision_contract | SafetyDecision | SafetyDecision | Yes (contract) |
| none_method | shared interface | shared interface | Yes (framework) |
| clipping_method | domain rules | domain rules | Yes (framework_only) |
| lyapunov_method | V_drive | V_robotics | Yes (framework_only) |
| constraint_semantics | driving constraints | robotics constraints | No (domain_specific) |
| predictor | driving one-step predictor | robotics one-step predictor | Yes (framework_only) |
| perturbation_harness | shared harness | shared harness | Yes (framework) |
| metric_schema | shared metrics | shared metrics | Yes (schema) |
| principal_seeds | [42, 123, 456] | [42, 123, 456] | Yes (protocol) |
| trained_policy_weights | driving-specific | robotics-specific | No (domain_specific) |

## Driving-specific components

- State dimension: 4
- State: speed, lane offset, heading error, obstacle distance
- Driving constraint evaluator
- Driving-specific clipping rules
- Driving one-step predictor
- V_drive

## Robotics-specific components

- State dimension: 6
- State: robot position, object position, target position
- Robotics constraint evaluator
- Robotics-specific clipping rules
- Robotics one-step predictor
- V_robotics
- Object-grasped prediction context

## Supported conclusion

A common safety-filter interface, intervention protocol, robustness harness, metric schema, and seed protocol were reused across autonomous-driving and robotics proxy environments while retaining domain-specific constraints, predictors, clipping semantics, and Lyapunov potentials.

## Explicitly unsupported interpretations

- same safety function
- same Lyapunov function
- same safety thresholds
- same trained policy
- zero-shot safety transfer
- universal safety controller
- formal cross-domain safety guarantee
- production generalization
