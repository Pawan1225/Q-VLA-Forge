# Q-VLA Forge — Sprint 7.3

## Final Frozen Baseline

Baseline: **shared_vla_fp32 v1.0**

Parameters: **76,179**

FP32 storage: **304,716 bytes**

| Domain | Test MSE | Test MAE | Mean Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|
| autonomous_driving | 0.012068 ± 0.001509 | 0.068615 ± 0.003575 | 1.737 ± 0.390 | 2.411 ± 0.702 |
| robotics | 0.008105 ± 0.002484 | 0.037295 ± 0.005609 | 1.989 ± 0.293 | 2.545 ± 0.557 |

All statistics use the frozen three-seed protocol: **42, 123, 456**.

No Sprint 1 baseline values were retrained or recomputed.
