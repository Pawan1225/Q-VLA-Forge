# Q-VLA Forge - Sprint 7.2 Final Statistics

All statistical results use arithmetic mean and sample standard deviation across locked seeds 42, 123, and 456.

Deterministic architectural quantities are reported separately as fixed values and are not represented as artificial +/- 0 statistics.

## Baseline Verification

| Domain | Metric | Mean +/- Sample SD |
|---|---|---:|
| autonomous_driving | test_mse | 0.012068 ± 0.001509 |
| autonomous_driving | test_mae | 0.068615 ± 0.003575 |
| autonomous_driving | mean_latency_ms | 1.736594 ± 0.389927 ms |
| autonomous_driving | p95_latency_ms | 2.411233 ± 0.702197 ms |
| robotics | test_mse | 0.008105 ± 0.002484 |
| robotics | test_mae | 0.037295 ± 0.005609 |
| robotics | mean_latency_ms | 1.988824 ± 0.292819 ms |
| robotics | p95_latency_ms | 2.545300 ± 0.557476 ms |

## Coverage

- Statistical records: 1602
- Fixed-value records: 66
- Recomputed directly from seed values: 1572
- Validated canonical n=3 summaries: 30

## Scientific Boundary

Sprint 7.2 performs statistical aggregation only. It introduces no new training, experiments, retuning, or scientific claims.
