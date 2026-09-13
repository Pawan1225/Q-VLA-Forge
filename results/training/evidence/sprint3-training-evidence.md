# Sprint 3 — Training Efficiency Evidence

Q-VLA Forge evaluates classical trainable SVD and quantum-inspired TT/MPS structured parameterizations on compact autonomous-driving and robotics VLA proxy tasks.

## Experimental Protocol

- Seeds: 42, 123, 456
- Training samples: 512
- Validation samples: 128
- Test samples: 128
- Training budget: 20 epochs
- Batch size: 32
- Optimizer: AdamW
- Scheduler: cosine annealing
- Primary metric: optimizer steps to paired FP32 target
- Robust efficiency criterion: all three seeds reach target and mean optimizer-step reduction is at least 10%

## Three-Seed Results

| Domain | Method | Target Reach | Parameter Reduction | Step Reduction | Test MSE | Robust ≥10% |
|---|---|---:|---:|---:|---:|---:|
| autonomous_driving | trainable_svd | 3/3 | 11.97% | -23.33 ± 11.10% | 0.013267 ± 0.000824 | No |
| autonomous_driving | trainable_tt_mps | 0/3 | 44.46% | N/A | 0.038474 ± 0.029401 | No |
| robotics | trainable_svd | 0/3 | 22.83% | N/A | 0.011115 ± 0.003145 | No |
| robotics | trainable_tt_mps | 0/3 | 44.46% | N/A | 0.050133 ± 0.004061 | No |

## Supported Findings

- Trainable SVD reached the paired FP32 target in all 3 seeds for autonomous driving; the mean optimizer-step reduction was -23.33 ± 11.10%, below the predefined robust 10% criterion.
- Trainable TT/MPS reached the paired FP32 target in 0/3 seeds for autonomous driving; therefore the experiment does not support a robust three-seed training-efficiency claim.
- Trainable SVD reached the paired FP32 target in 0/3 seeds for robotics; therefore the experiment does not support a robust three-seed training-efficiency claim.
- Trainable TT/MPS reached the paired FP32 target in 0/3 seeds for robotics; therefore the experiment does not support a robust three-seed training-efficiency claim.
- Trainable SVD was architecturally reusable across both domains, but did not demonstrate robust training-efficiency improvement in both; target consistency was mixed and efficiency consistency was none.
- Trainable TT/MPS was architecturally reusable across both domains, but did not demonstrate robust training-efficiency improvement in both; target consistency was weak and efficiency consistency was none.

## Classical vs Quantum-Inspired Matched Ablation

- **autonomous_driving**: SVD-25% vs TT-rank-8; parameter-budget difference 1.04%. Neither matched representation reached the paired FP32 target.
- **robotics**: SVD-25% vs TT-rank-8; parameter-budget difference 1.04%. Neither matched representation reached the paired FP32 target.

## Explicitly Unsupported Claims

- Sprint 3 does not establish quantum advantage.
- Sprint 3 does not establish quantum speedup.
- Sprint 3 does not establish native TT/MPS runtime acceleration.
- Sprint 3 does not establish performance on full-scale production 7B-class VLA models.

## Limitations

- Experiments use compact synthetic autonomous-driving and robotics proxy tasks.
- The pilot does not train or evaluate a full 7B-class production VLA model.
- The lightweight language component is a trainable hashed-token embedding encoder, not a large Transformer language model.
- CPU wall-clock timing is descriptive; optimizer steps and samples to target are the primary efficiency measures.
- TT/MPS executes classically and does not constitute a quantum-hardware speedup.
- The matched SVD-vs-TT/MPS ablation uses seed 42 as a controlled isolation experiment.
- Architectural reuse across domains does not imply one universally trained model.
