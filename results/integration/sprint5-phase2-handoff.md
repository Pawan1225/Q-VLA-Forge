# Q-VLA Forge Phase 2 Handoff

## Current Phase 1 Status

Phase 1 established a reproducible dual-domain pilot spanning autonomous-driving and robotics proxy tasks. The repository now contains frozen evidence for the shared VLA baseline, classical and quantum-inspired compression, training-efficiency studies, classical PPO, hybrid PQC policies, safety filtering, robustness evaluation, and cross-domain safety analysis.

## What Is Frozen

The following Phase 1 interfaces should be preserved as the starting point for Phase 2:

- data contracts
- configuration system
- reproducibility and seed handling
- result-store schema
- compression interfaces
- policy interfaces
- safety-filter API
- robustness harness
- evidence schema
- claim controls
- dashboard infrastructure

## What Worked

- A common computational framework was instantiated across both domains.
- All four Volkswagen challenge bottlenecks were experimentally addressed.
- INT8 provided the strongest Phase 1 compression result.
- Classical PPO reliably reached the frozen RL targets.
- The hybrid PQC actor provided substantial parameter compactness.
- Explicit safety filtering produced strong empirical pilot safety effects.
- Cross-domain architecture reuse was supported at the framework and interface level.

## What Did Not Establish Advantage

- TT/MPS did not establish superiority over the strongest classical compression result.
- No robust >=10% training-efficiency advantage was demonstrated.
- Hybrid QML did not demonstrate sample-efficiency advantage.
- No quantum advantage or quantum speedup was demonstrated.
- Cross-domain zero-shot policy transfer was not tested.
- No formal safety guarantee or production-readiness claim is supported.

## Critical Phase 1 Limitations

- synthetic proxy environments
- compact shared VLA baseline
- Sprint 4 policies consume compact environment state rather than the Sprint 1 shared latent
- small simulated PQC
- no quantum hardware
- three principal seeds
- no physical driving or robotics validation
- no embedded full-VLA deployment benchmark
- privileged true-state access for the safety layer in perception-perturbation studies
- no formal stability proof or certification

## Phase 2 Priority Sequence

1. Integrate a larger pretrained VLA-style representation.
2. Connect the shared VLA latent directly to PPO and hybrid PQC policy heads.
3. Scale TT/MPS compression to larger VLA matrices.
4. Validate in realistic driving and robotics environments.
5. Evaluate safety using estimated/noisy state.
6. Benchmark embedded deployment latency, memory, and safety overhead.
7. Extend PQC studies to larger/noisy/hardware settings where available.
8. Increase seeds, evaluation corpus size, and statistical rigor.

## Phase 2 Entry Principle

Phase 2 should extend the frozen Phase 1 interfaces rather than rebuild the project from scratch. The main changes should occur in model scale, environment realism, policy integration, safety-state assumptions, and deployment validation.

## Claim Boundary

This handoff is a technical roadmap. It does not claim that Phase 2 scale-up, quantum advantage, hardware speedup, real-world safety, or production deployment has already been demonstrated.
