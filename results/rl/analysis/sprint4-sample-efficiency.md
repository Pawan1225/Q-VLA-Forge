# Sprint 4.11 — Sample-Efficiency Analysis

No new principal RL training, retuning, target changes, environment changes, or policy changes were performed.

## Primary Pre-Registered Metric

Environment steps to the paired frozen PPO-derived reward target.

Classical PPO target reach: **6/6**

Hybrid QML target reach: **0/6**

Failed QML target crossings remain `None`; they are not censored to the 20,000-step budget.

## Secondary Descriptive Metrics

Normalized target progress maps each domain's random-policy reference to 0 and each seed's paired frozen target to 1.

Normalized learning-curve AUC, best normalized progress, and final normalized progress are post-hoc descriptive metrics and do not replace the primary target-reach criterion.

| Domain | Policy | Target Reach | Normalized AUC | Best Progress | Final Progress | Actor Params |
|---|---|---:|---:|---:|---:|---:|
| autonomous_driving | PPO | 3/3 | 0.554811 ± 0.264606 | 1.052632 ± 0.000000 | 0.995282 ± 0.053598 | 1318 |
| autonomous_driving | QML | 0/3 | 0.159912 ± 0.434197 | 0.441827 ± 0.472874 | 0.327289 ± 0.420983 | 54 |
| robotics | PPO | 3/3 | 0.922914 ± 0.032207 | 1.052632 ± 0.000000 | 1.015904 ± 0.013050 | 1382 |
| robotics | QML | 0/3 | 0.573909 ± 0.193446 | 0.758545 ± 0.238557 | 0.758545 ± 0.238557 | 62 |

## Policy Actor Parameter Accounting

| Domain | Classical Actor | Hybrid Actor | Reduction |
|---|---:|---:|---:|
| autonomous_driving | 1318 | 54 | 95.90% |
| robotics | 1382 | 62 | 95.51% |

## Scientific Boundary

Actor parameter accounting refers to the policy actor only; the classical critic remains unchanged within each domain.

Normalized AUC and normalized target progress are descriptive secondary metrics.

No direct raw reward comparison is made between driving and robotics because their reward scales differ.

The analysis does not isolate whether the observed performance gap arises from the PQC representation, the much smaller actor parameter budget, optimization dynamics, or their interaction.

No quantum speedup or quantum hardware advantage is claimed.
