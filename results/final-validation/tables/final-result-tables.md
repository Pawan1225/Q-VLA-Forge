# Q-VLA Forge — Sprint 7.8 Final Result Tables

Frozen Phase 1 evidence only. No new training, experiments, retuning, or scientific results were introduced.

Statistical protocol: mean ± sample SD across n=3 locked seeds (42, 123, 456) for seed-dependent quantities; deterministic values remain plain values.

Missing-value semantics: `Not reached`, `Not measured`, and `Not applicable` are distinct.

## Compression Results

| Domain | Method | Parameters / Model Size | Compression Ratio | MSE | Relative ΔMSE | Latency | Criterion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Driving | FP32 | Not measured | 1.000× | 0.012068 ± 0.001509 | Not applicable | Not measured | REFERENCE |
| Driving | INT8 | Not measured | 3.846× | 0.012040 ± 0.001503 | -0.225% ± 0.133% | Not measured | PASS |
| Driving | SVD | Not measured | 1.136× | 0.058395 ± 0.009222 | 382.624% ± 15.187% | Not measured | FAIL |
| Driving | TT/MPS | Not measured | 1.801× | 0.168545 ± 0.032997 | 1300.192% ± 254.201% | Not measured | FAIL |
| Robotics | FP32 | Not measured | 1.000× | 0.008105 ± 0.002484 | Not applicable | Not measured | REFERENCE |
| Robotics | INT8 | Not measured | 3.846× | 0.008108 ± 0.002559 | -0.153% ± 1.128% | Not measured | PASS |
| Robotics | SVD | Not measured | 1.296× | 0.147037 ± 0.011154 | 1818.579% ± 519.409% | Not measured | FAIL |
| Robotics | TT/MPS | Not measured | 1.801× | 0.258374 ± 0.020236 | 3270.928% ± 895.501% | Not measured | FAIL |

### Limitations

- Results are from the compact Phase 1 proxy VLA model.
- MSE and MAE are supervised regression metrics; they must not be relabeled as generic accuracy.
- INT8 storage compression does not establish native compressed-runtime speedup.
- Evaluated SVD and TT/MPS configurations did not satisfy the joint frozen Phase 1 criterion.

### Provenance

- `results/final-validation/compression-ablation/compression-ablation.json`
- `results/final-validation/compression-ablation/compression-pareto.csv`
- `results/final-validation/statistics/final-statistics.json`

## Training Efficiency Results

| Domain | Method | Target Reached | Steps to Target | Training Time | Memory | Final Validation Loss / MSE | Efficiency Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Driving | Trainable SVD | 3/3 | 282.667 ± 24.440 | 9.371 ± 0.511 s | Not measured | 0.014642 ± 0.003234 | FAIL |
| Driving | Trainable TT/MPS | 0/3 | Not reached | Not reached | Not measured | 0.040067 ± 0.029402 | FAIL |
| Robotics | Trainable SVD | 0/3 | Not reached | Not reached | Not measured | 0.018958 ± 0.006411 | FAIL |
| Robotics | Trainable TT/MPS | 0/3 | Not reached | Not reached | Not measured | 0.049764 ± 0.007475 | FAIL |

### Limitations

- Phase 1 did not demonstrate a robust >=10% optimizer-step efficiency improvement across both domains and all locked seeds.
- Target-reaching quantities remain missing when the paired target was not reached.
- Memory must be reported as Not measured when the Sprint 3 evidence does not contain it.
- CPU wall-clock timing is descriptive.

### Provenance

- `results/final-validation/training-efficiency/final-training-efficiency-summary.json`
- `results/training/training-validation-summary.json`
- `results/final-validation/statistics/final-statistics.json`

## RL / QML Results

| Domain | Policy | Target Reaches | Steps to Target | Final Reward | Final Success Rate | Actor Parameters | Parameter Reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Driving | Classical PPO / MLP | 3/3 | 13333.333 ± 1527.525 | -9.200 ± 1.936 | 0.050 ± 0.087 | 1318 | Not applicable |
| Driving | QML / PQC | 0/3 | Not reached | -35.444 ± 18.964 | 0.000 ± 0.000 | 54 | 95.90% |
| Robotics | Classical PPO / MLP | 3/3 | 10000.000 ± 1732.051 | 0.633 ± 0.029 | 0.000 ± 0.000 | 1382 | Not applicable |
| Robotics | QML / PQC | 0/3 | Not reached | -0.267 ± 0.853 | 0.000 ± 0.000 | 62 | 95.51% |
| Driving | Matched classical control | 0/3 | Not reached | -51.895 ± 23.971 | 0.000 ± 0.000 | 54 | 95.90% |
| Robotics | Matched classical control | 1/3 | 20000.000 | 0.221 ± 0.351 | 0.000 ± 0.000 | 62 | 95.51% |

### Limitations

- QML target non-attainment remains Not reached and is not replaced with the 20,000-step budget.
- Actor compactness is reported separately from sample efficiency and performance.
- No quantum advantage, quantum speedup, or QML sample-efficiency advantage was demonstrated.

### Provenance

- `results/final-validation/qml-ablation/qml-ablation.json`
- `results/final-validation/qml-ablation/qml-ablation-plot-data.csv`
- `results/final-validation/statistics/final-statistics.json`

## Safety Results

| Domain | Method | Violation Rate | Reward | Success Rate | Activation / Intervention | Evidence Role |
| --- | --- | --- | --- | --- | --- | --- |
| Driving | NONE | 0.391 ± 0.422 | -9.200 ± 1.936 | 0.050 ± 0.087 | Not applicable | Baseline |
| Driving | CLIPPING | 0.000 ± 0.000 | -5.213 ± 8.335 | 0.333 ± 0.577 | 0.417 ± 0.420 | Classical empirical safety filter |
| Driving | LYAPUNOV | 0.000 ± 0.000 | -5.213 ± 8.335 | 0.333 ± 0.577 | 0.417 ± 0.420 | Classical empirical safety filter |
| Robotics | NONE | 0.015 ± 0.019 | 0.633 ± 0.029 | 0.000 ± 0.000 | Not applicable | Baseline |
| Robotics | CLIPPING | 0.000 ± 0.000 | 0.678 ± 0.036 | 0.000 ± 0.000 | 0.015 ± 0.019 | Classical empirical safety filter |
| Robotics | LYAPUNOV | 0.000 ± 0.000 | 0.678 ± 0.036 | 0.000 ± 0.000 | 0.015 ± 0.019 | Classical empirical safety filter |

### Limitations

- Zero violation rate means zero observed violations under the frozen proxy evaluation only.
- The Lyapunov quantity is an empirical classical safety potential, not a formal stability proof.
- Action-perturbation recovery evidence must not be mixed into the primary clean safety-ablation rows.
- During perception perturbations the safety layer retained true simulator state.

### Provenance

- `results/final-validation/safety-ablation/safety-ablation.json`
- `results/final-validation/safety-ablation/safety-ablation-plot-data.csv`
- `results/final-validation/statistics/final-statistics.json`

## Cross-Domain Evidence Summary

| Component / Method | Driving Result | Robotics Result | Shared Across Domains? | Evidence Boundary |
| --- | --- | --- | --- | --- |
| Shared AI/DL architecture | Shared architecture components available | Shared architecture components available | Framework/components: YES; same trained policy weights: NO | Cross-domain reuse is supported at the framework, interface, robustness-harness, metric-schema, and protocol levels, but not as one universal trained policy or universal safety controller. |
| INT8 | PASS | PASS | Same method/protocol family: YES | Storage-compression evidence only; no production runtime claim. |
| SVD | FAIL | FAIL | Same method/protocol family: YES | Evaluated configuration failed the joint criterion. |
| TT/MPS | FAIL | FAIL | Same method/protocol family: YES | Quantum-inspired tensor-network method; no superiority claim. |
| Classical PPO | Target reaches 3/3 | Target reaches 3/3 | Evaluation framework: YES; same learned weights: NO | Separate domain policies and reward/state contracts. |
| QML/PQC | Target reaches 0/3 | Target reaches 0/3 | Evaluation framework/PQC family: YES; same trained weights: NO | No QML sample-efficiency advantage or quantum advantage demonstrated. |
| Clipping | Violation rate 0.000 ± 0.000 | Violation rate 0.000 ± 0.000 | Safety abstraction: YES; constraints/semantics: domain-specific | Zero observed violations are empirical only. |
| Lyapunov | Violation rate 0.000 ± 0.000 | Violation rate 0.000 ± 0.000 | Safety abstraction: YES; domain-specific predictor/constraints | Classical empirical safety potential; no formal stability guarantee. |
| Full-system factorial | COMPONENT_ONLY | COMPONENT_ONLY | Direct integrated evidence: NO | DIRECT = 0; COMPONENT_ONLY = 16; interaction effects remain Phase 2 work. |

### Limitations

- Cross-domain reuse is supported at framework, interface, protocol, and evaluation levels.
- Driving and robotics used separate trained policy weights.
- Domain-specific constraints, predictors, and safety semantics remain distinct.
- No synthetic aggregate score is permitted.
- No matched integrated Compression x QML x Safety factorial pipeline was directly executed in Phase 1.

### Provenance

- `results/final-validation/cross-domain/final-cross-domain-summary.json`
- `results/final-validation/full-system-ablation/full-system-ablation.json`
- `results/final-validation/statistics/final-statistics.json`

## Scientific Controls

- TT/MPS superiority: blocked.
- Robust >=10% training-efficiency improvement: not demonstrated.
- QML sample-efficiency advantage: blocked.
- Quantum advantage / speedup: blocked.
- Formal Lyapunov stability: blocked.
- Guaranteed zero violations: blocked.
- Zero-shot cross-domain transfer: blocked.
- Full-system superiority: blocked.
- Production readiness / certification: blocked.
