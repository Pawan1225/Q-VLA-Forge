# Q-VLA Forge

A reproducible Phase 1 pilot for the Volkswagen Group Global Quantum + AI Challenge 2026, evaluating a shared lightweight vision-language-action-style framework across autonomous-driving and robotics proxy tasks.

## Phase 1 Scope

Q-VLA Forge experimentally addresses four challenge bottlenecks:

1. Model compression and footprint
2. Training efficiency
3. RL alignment and sample efficiency
4. Safety and robustness

The pilot uses the locked seeds **42, 123, 456**.

## Shared Baseline

- Model: `shared_vla_fp32`
- Parameters: **76,179**

## Final Phase 1 Findings

- **Compression:** INT8 is the only evaluated method on the Pareto frontier in both proxy domains under the frozen criterion, at approximately **3.85× effective compression**.
- **Training efficiency:** No robust ≥10% optimizer-step efficiency improvement was demonstrated across all three locked seeds.
- **RL:** Full PPO reached **6/6 frozen targets**.
- **QML compactness:** PQC/QML actor parameter reduction was approximately **95.90%** for autonomous driving and **95.51%** for robotics.
- **QML performance boundary:** No QML sample-efficiency or computational advantage was demonstrated.
- **Safety:** Explicit clipping and Lyapunov filtering reduced observed clean violation-step rate to **0** in both evaluated proxy domains.
- **Action robustness:** Explicit filtering recovered approximately **62.3%** of unsafe directly perturbed action steps.
- **Cross-domain reuse:** Supported at the framework/interface/protocol level, not as one universal trained policy.

## Challenge Bottleneck Status

| Bottleneck | Phase 1 status |
|---|---|
| model_footprint | demonstrated |
| training_efficiency | not_demonstrated |
| rl_alignment_sample_efficiency | mixed_evidence |
| safety | empirically_supported |

## Scientific Boundaries

Phase 1 does **not** claim quantum advantage, quantum speedup, QML superiority, TT/MPS superiority, formal Lyapunov stability, formal functional-safety certification, production readiness, physical vehicle or robot validation, or zero-shot cross-domain policy transfer.

## Final Evidence

Canonical Sprint 7 evidence is under `results/final-validation/`.

Final proposal-ready figures are under `figures/final/`.

Final proposal-ready CSV tables are under `results/final-validation/tables/`.

Key evidence:

- `results/final-validation/baseline/final-baseline-summary.json`
- `results/final-validation/compression/final-compression-summary.json`
- `results/final-validation/training-efficiency/final-training-efficiency-summary.json`
- `results/final-validation/rl-qml/final-rl-qml-summary.json`
- `results/final-validation/safety-robustness/final-safety-robustness-summary.json`
- `results/final-validation/cross-domain/final-cross-domain-summary.json`
- `results/final-validation/claims/final-claim-registry.json`

## Reproducibility

```powershell
$env:PYTHONPATH="src;."
pytest tests -q
ruff check src tests dashboard
black --check src tests dashboard
```

## Phase 1 Status

**Frozen.** Sprint 7 performs final validation, evidence synthesis, claim control, and submission packaging only. No new principal training is introduced.
