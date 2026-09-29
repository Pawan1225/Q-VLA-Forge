# Q-VLA Forge — Final Figure Captions

## Figure 1 — Unified Architecture

**Phase 1 architecture and integration boundary.**
Q-VLA Forge reused a common AI/DL architecture, interface conventions,
evaluation contracts, and safety framework across autonomous-driving and
robotics proxy domains while retaining domain-specific trained policy
weights and safety semantics. Dashed integration links represent the
intended system pathway and were not executed as one matched end-to-end
Phase 1 pipeline.

## Figure 2 — Compression Pareto

**Compression Pareto comparison.**
INT8 satisfied the frozen Phase 1 joint criterion of at least 2× storage
compression with no more than 5% relative MSE degradation in both proxy
domains. Evaluated SVD and TT/MPS configurations did not satisfy the joint
criterion. The shaded/threshold region is a frozen Phase 1 acceptance
criterion, not a quantum-advantage region.

## Figure 3 — Training Efficiency

**Training-efficiency evaluation.**
Driving trainable SVD reached its paired FP32 target across all three
locked seeds, but required more optimizer steps on average than the paired
FP32 reference. The evaluated driving TT/MPS, robotics SVD, and robotics
TT/MPS candidates did not reach their paired targets across the locked
three-seed protocol. Phase 1 therefore did not demonstrate a robust
10% or greater optimizer-step efficiency improvement across both domains.
Missing target-reaching quantities remain missing and are not censored to
the final training budget.

## Figure 4 — RL/QML Ablation

**RL/QML target attainment and actor compactness.**
Classical PPO reached the frozen target across all six domain-seed
evaluations, while the evaluated PQC policy reached none and the
parameter-matched classical control reached one. The PQC actor nevertheless
used substantially fewer trainable actor parameters. Compactness is reported
separately from sample efficiency; no quantum advantage or QML
sample-efficiency advantage is claimed.

## Figure 5 — Safety Violations

**Clean safety ablation.**
Clipping and the classical Lyapunov-guided filter produced zero observed
proxy violations under the frozen clean-evaluation contracts in both proxy
domains. These are empirical observations, not formal stability,
certification, production-safety, or real-world safety guarantees. The
safety layer retained privileged true simulator state during perception
perturbation experiments.

## Figure 6 — Cross-Domain Evidence

**Cross-domain Phase 1 comparison.**
The two proxy domains showed the same categorical result for the frozen
INT8, SVD, TT/MPS, PPO target-attainment, QML target-attainment, clipping,
and Lyapunov clean-safety checks. Metrics with different units or semantics
are not normalized into a synthetic overall score.

## Figure 7 — Full-System Ablation Evidence

**Full-system factorial evidence status.**
All sixteen domain/configuration cells in the Compression × QML × Safety
factorial are classified as COMPONENT_ONLY. No exact matched integrated
factorial configuration was directly executed in Phase 1, so no synthetic
full-system performance or interaction effect is reported. Matched
integrated execution remains a Phase 2 validation objective.
