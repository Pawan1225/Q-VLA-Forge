# Q-VLA Forge - Sprint 7.3 Final Compression Ablation

Frozen Sprint 2 evidence only. No new compression experiments, retraining, retuning, or rank search.

## Autonomous Driving

| Method | Compression | MSE mean +/- sample SD | Delta MSE mean +/- sample SD (%) | Seeds Passing | Status |
|---|---:|---:|---:|---:|---|
| FP32 | 1.000x | 0.012068 +/- 0.001509 | Reference | - | REFERENCE |
| INT8 | 3.846x | 0.012040 +/- 0.001503 | -0.225 +/- 0.133 | 3/3 | PASS |
| SVD | 1.136x | 0.058395 +/- 0.009222 | 382.624 +/- 15.187 | 0/3 | FAIL |
| TT/MPS | 1.801x | 0.168545 +/- 0.032997 | 1300.192 +/- 254.201 | 0/3 | FAIL |

## Robotics

| Method | Compression | MSE mean +/- sample SD | Delta MSE mean +/- sample SD (%) | Seeds Passing | Status |
|---|---:|---:|---:|---:|---|
| FP32 | 1.000x | 0.008105 +/- 0.002484 | Reference | - | REFERENCE |
| INT8 | 3.846x | 0.008108 +/- 0.002559 | -0.153 +/- 1.128 | 3/3 | PASS |
| SVD | 1.296x | 0.147037 +/- 0.011154 | 1818.579 +/- 519.409 | 0/3 | FAIL |
| TT/MPS | 1.801x | 0.258374 +/- 0.020236 | 3270.928 +/- 895.501 | 0/3 | FAIL |

## Scientific Conclusion

Methods meeting the frozen >=2.0x compression and <=5% relative MSE-degradation criterion in both evaluated domains: INT8.

Methods that do not satisfy the joint criterion remain retained as negative Phase 1 evidence.

## Claim Controls

- TT/MPS superiority: blocked
- Quantum-inspired compression advantage: blocked
- Quantum advantage: blocked
- Production-scale extrapolation: blocked
- 7B VLA generalization: blocked
