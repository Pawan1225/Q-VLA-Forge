# Q-VLA Forge — Sprint 2 Compression Evidence

## Validation Protocol

Seeds: 42, 123, 456

## Validated Results

| Domain | Method | Configuration | Compression | MSE Δ | MAE Δ | Feasible Runs | Pareto |
|---|---|---|---:|---:|---:|---:|---:|
| autonomous_driving | int8 | INT8 | 3.846 ± 0.000× | -0.225 ± 0.133% | 0.354 ± 0.317% | 3/3 | Yes |
| autonomous_driving | svd | SVD-75% | 1.136 ± 0.000× | 382.624 ± 15.187% | 175.887 ± 9.931% | 0/3 | No |
| autonomous_driving | tensor_train_mps | TT-rank-2 | 1.801 ± 0.000× | 1300.192 ± 254.201% | 344.988 ± 4.192% | 0/3 | No |
| robotics | int8 | INT8 | 3.846 ± 0.000× | -0.153 ± 1.128% | -0.067 ± 0.051% | 3/3 | Yes |
| robotics | svd | SVD-50% | 1.296 ± 0.000× | 1818.579 ± 519.409% | 710.097 ± 71.176% | 0/3 | No |
| robotics | tensor_train_mps | TT-rank-2 | 1.801 ± 0.000× | 3270.928 ± 895.501% | 890.445 ± 121.483% | 0/3 | No |

## Pareto Frontiers

**Autonomous driving:** INT8

**Robotics:** INT8

## Quantum-Inspired Component

- Method: TT/open-boundary MPS constructed using TT-SVD
- Quantum hardware used: False
- Target-level ablation: 20 points at seed 42

## Supported Claims

- Compression was evaluated over three deterministic training seeds (42, 123, 456) and reported using mean ± sample standard deviation.
- Classical INT8 quantization, classical truncated SVD, and quantum-inspired TT/MPS tensor-network compression were evaluated on the same shared VLA proxy architecture.
- Pareto analysis maximized effective whole-model compression while minimizing relative test-MSE increase.
- The target-level ablation compared classical SVD and quantum-inspired TT/MPS under matched model, domain, seed, architectural target, and test-data conditions.
- autonomous_driving: INT8 achieved 3.846 ± 0.000× effective whole-model compression with -0.225 ± 0.133% relative test-MSE change across three seeds; 3/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.
- autonomous_driving: SVD-75% achieved 1.136 ± 0.000× effective whole-model compression with 382.624 ± 15.187% relative test-MSE change across three seeds; 0/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.
- autonomous_driving: TT-rank-2 achieved 1.801 ± 0.000× effective whole-model compression with 1300.192 ± 254.201% relative test-MSE change across three seeds; 0/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.
- robotics: INT8 achieved 3.846 ± 0.000× effective whole-model compression with -0.153 ± 1.128% relative test-MSE change across three seeds; 3/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.
- robotics: SVD-50% achieved 1.296 ± 0.000× effective whole-model compression with 1818.579 ± 519.409% relative test-MSE change across three seeds; 0/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.
- robotics: TT-rank-2 achieved 1.801 ± 0.000× effective whole-model compression with 3270.928 ± 895.501% relative test-MSE change across three seeds; 0/3 runs satisfied the ≥2× compression and ≤5% MSE-change pilot criterion.

## Limitations

- The experiments use compact synthetic autonomous-driving and robotics proxy tasks rather than full 7B–10B VLA models.
- TT/MPS is quantum-inspired tensor-network compression; no quantum hardware was used for Sprint 2.
- INT8 inference uses FP32-dequantized weights, while SVD and TT/MPS inference use reconstructed FP32 dense weights; measured latency is therefore not native compressed-kernel speedup evidence.
- The final compression hyperparameters were selected using seed-42 exploratory screening; seed 42 also appears in the three-seed validation set.
- Pareto dominance is based on three-seed mean metrics and does not constitute a statistical-significance test.
- The target-level architectural ablation uses seed 42 and should be interpreted separately from the three-seed validation results.
