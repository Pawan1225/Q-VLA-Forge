# Sprint 4.12 — Classical vs QML Matched-Budget Ablation

Sprint 4.12 introduces a new parameter-matched classical PPO control while reusing the frozen hybrid QML runs.

## Experimental Control

Driving matched classical and hybrid QML actors both contain 54 trainable actor parameters.

Robotics matched classical and hybrid QML actors both contain 62 trainable actor parameters.

Critic and total trainable parameter counts are also exactly matched within each domain.

## Paired Seed-Level Results

| Domain | Seed | Matched Params | QML Params | Matched Reach | QML Reach | Matched AUC | QML AUC | Delta AUC | Delta Best | Delta Final |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| autonomous_driving | 42 | 54 | 54 | False | False | -0.403328 | 0.074233 | -0.477561 | 0.618567 | 0.198333 |
| autonomous_driving | 123 | 54 | 54 | False | False | -0.127236 | -0.225058 | 0.097821 | 0.451547 | -0.748479 |
| autonomous_driving | 456 | 54 | 54 | False | False | 0.006900 | 0.630561 | -0.623662 | -0.294942 | -0.576944 |
| robotics | 42 | 62 | 62 | False | False | 0.611292 | 0.365377 | 0.245915 | 0.356382 | 0.355701 |
| robotics | 123 | 62 | 62 | False | False | 0.621048 | 0.608842 | 0.012206 | 0.020374 | 0.009523 |
| robotics | 456 | 62 | 62 | True | False | 0.726839 | 0.747508 | -0.020669 | 0.052563 | 0.052563 |

## Aggregate Matched-Budget Results

| Domain | Method | Actor Params | Target Reach | Normalized AUC | Best Progress | Final Progress |
|---|---|---:|---:|---:|---:|---:|
| autonomous_driving | full_ppo | 1318 | 3/3 | 0.554811 ± 0.264606 | 1.052632 ± 0.000000 | 0.995282 ± 0.053598 |
| autonomous_driving | matched_classical | 54 | 0/3 | -0.174555 ± 0.209167 | 0.700218 ± 0.248420 | -0.048407 ± 0.589352 |
| autonomous_driving | qml | 54 | 0/3 | 0.159912 ± 0.434197 | 0.441827 ± 0.472874 | 0.327289 ± 0.420983 |

### autonomous_driving paired deltas

Delta is defined as matched classical minus hybrid QML.

Normalized AUC delta: -0.334467 ± 0.381433

Best-progress delta: 0.258391 ± 0.486422

Final-progress delta: -0.375697 ± 0.504468

| robotics | full_ppo | 1382 | 3/3 | 0.922914 ± 0.032207 | 1.052632 ± 0.000000 | 1.015904 ± 0.013050 |
| robotics | matched_classical | 62 | 1/3 | 0.653060 ± 0.064081 | 0.901652 ± 0.087655 | 0.897808 ± 0.090938 |
| robotics | qml | 62 | 0/3 | 0.573909 ± 0.193446 | 0.758545 ± 0.238557 | 0.758545 ± 0.238557 |

### robotics paired deltas

Delta is defined as matched classical minus hybrid QML.

Normalized AUC delta: 0.079151 ± 0.145355

Best-progress delta: 0.143107 ± 0.185402

Final-progress delta: 0.139263 ± 0.188673

## Scientific Boundary

This is a matched-parameter representation ablation, not a pure test of quantum mechanics.

The classical and PQC cores have equal trainable parameter counts, but their parameterizations and optimization landscapes differ.

Normalized AUC, best normalized progress, and final normalized progress remain descriptive secondary metrics.

No formal statistical significance test is performed because each domain contains only three principal seeds.

No quantum speedup, quantum hardware advantage, or universal claim about quantum versus classical policies is made.

The six hybrid QML runs are reused from the frozen Sprint 4.8/4.9 evidence and were not retrained.
