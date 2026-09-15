# Q-VLA Forge Unified Architecture

## 1. Purpose

Q-VLA Forge is a Phase 1 dual-domain pilot for autonomous driving and robotics. It evaluates a modular computational framework spanning shared AI/VLA representations, compression, training efficiency, reinforcement learning, hybrid quantum machine learning, safety filtering, robustness, and cross-domain analysis.

The Phase 1 objective is not to demonstrate one universal trained VLA model. Instead, the pilot tests which computational interfaces and experimental frameworks can be reused across domains while retaining domain-specific state semantics, policy weights, dynamics, rewards, and safety constraints.

## 2. Canonical Architecture

~~~text
                         Q-VLA FORGE

                  DOMAIN INPUT ADAPTERS
             +-------------+-------------+
             |                           |
             v                           v
   Autonomous Driving                Robotics
  vision/state/language         vision/state/language
             |                           |
             +-------------+-------------+
                           |
                           v
                 SHARED AI / VLA CORE
                           |
              Vision / Language / State
                           |
                 Multimodal Fusion
                           |
               Shared Latent Representation
                           |
             +-------------+-------------+
             |                           |
             v                           v
      COMPRESSION PATH              POLICY PATH
     INT8 / SVD / TT-MPS       PPO / Hybrid PQC
             |                           |
             +-------------+-------------+
                           |
                           v
                    Proposed Action
                           |
                           v
                      SAFETY LAYER
              NONE / CLIPPING / LYAPUNOV
                           |
                           v
                     Executed Action
                           |
             +-------------+-------------+
             v                           v
       Driving Environment         Robotics Environment
~~~

## 3. Shared Components

The shared framework includes:

- data-contract framework
- configuration and reproducibility infrastructure
- vision, language, and state encoder frameworks
- multimodal fusion and shared latent representation
- three-dimensional action interface
- compression interfaces and accounting
- PPO and hybrid-policy interfaces
- safety-decision contract
- robustness harness
- three-seed evaluation protocol
- evidence-generation and claim-control framework
- dashboard infrastructure

## 4. Domain-Specific Components

### Autonomous Driving

- driving state semantics
- driving environment
- driving reward
- driving PPO weights
- driving safety constraints
- driving clipping rules
- driving transition predictor
- driving Lyapunov potential

### Robotics

- robotics state semantics
- robotics environment
- robotics reward
- robotics PPO weights
- robotics safety constraints
- robotics clipping rules
- robotics transition predictor
- robotics Lyapunov potential
- object-grasped context

The architecture is therefore shared at the framework and interface level, not as one identical trained controller.

## 5. Volkswagen Challenge Bottlenecks

| Challenge bottleneck | Q-VLA Forge methods | Primary evidence |
| --- | --- | --- |
| Model footprint | INT8, SVD, TT/MPS | Sprint 2 |
| Training efficiency | AdamW/cosine, trainable SVD, trainable TT/MPS | Sprint 3 |
| RL alignment / sample efficiency | PPO, matched classical actor, hybrid PQC | Sprint 4 |
| Safety | Clipping, Lyapunov-guided filtering, robustness | Sprint 5 |

All four challenge bottlenecks were experimentally addressed. This does not imply that every quantum or quantum-inspired pathway outperformed its classical baseline.

## 6. Classical / Quantum-Inspired / Quantum Components

### Classical

- shared neural baseline
- AdamW
- cosine learning-rate schedule
- INT8
- SVD
- PPO
- matched classical actor
- heuristic clipping
- Lyapunov function
- Lyapunov-guided safety filter

### Quantum-Inspired

- Tensor Train
- Matrix Product State
- TT-SVD

### Quantum / QML

- angle encoding
- four-qubit PQC
- parameterized / variational quantum circuit
- RY / RZ rotations
- CNOT entanglement
- hybrid quantum-classical actor

Lyapunov safety is classical and is not presented as quantum.

## 7. Evidence Mapping

| Architecture block | Sprint | Evidence |
| --- | --- | --- |
| Shared VLA baseline | 1 | dual-domain baseline manifest |
| Compression | 2 | compression evidence package |
| Training efficiency | 3 | training-efficiency evidence package |
| Classical PPO | 4 | RL evidence package |
| Hybrid PQC | 4 | RL evidence + claim controls |
| Safety | 5 | consolidated safety evidence |
| Cross-domain reuse | 5.14 | cross-domain safety package |

## 8. Phase 1 Findings

### Compression

INT8 produced the strongest Phase 1 compression result under the frozen pilot criteria.

Tensor Train and MPS were implemented and evaluated as quantum-inspired compression pathways, but they did not establish superiority over the strongest classical result.

### Training Efficiency

Trainable SVD and TT/MPS were evaluated under the paired convergence protocol.

No robust >=10% training-efficiency advantage was demonstrated.

### RL / QML

Classical PPO reached all 6 of 6 frozen target cases.

The matched classical actor reached 1 of 6.

The hybrid QML actor reached 0 of 6.

The hybrid actor nevertheless provided substantial parameter compactness, approximately 95.90% in autonomous driving and 95.51% in robotics.

The Phase 1 evidence therefore supports QML compactness, not sample-efficiency advantage.

### Safety

Explicit clipping and Lyapunov-guided filtering produced strong empirical pilot safety effects.

Cross-domain direction consistency was:

- clean safety: 2 / 2
- Gaussian robustness: 7 / 9
- structured-state robustness: 3 / 3
- action recovery: 3 / 3

Lyapunov-specific action-selection activation was observed in autonomous driving but not in robotics under the frozen action-perturbation corpus.

## 9. Supported Claims

Phase 1 supports the following conclusions:

- a shared computational framework was instantiated across both proxy domains
- all four Volkswagen challenge bottlenecks were experimentally addressed
- classical, quantum-inspired, and QML alternatives were evaluated under controlled protocols
- common experiment, policy-interface, safety-interface, robustness, and evidence infrastructure was reused across domains
- the architecture provides a modular Phase 2 scale-up pathway, with the limitation that the scale-up itself has not yet been demonstrated

## 10. Unsupported Claims

Phase 1 does not support:

- one universal trained VLA model
- cross-domain zero-shot policy transfer
- quantum advantage
- quantum speedup
- QML sample-efficiency advantage
- TT/MPS superiority over the strongest classical compression result
- robust training-efficiency advantage
- formal safety guarantee
- production deployment readiness
- real-world driving validation
- physical robotics validation

## 11. Phase 1 Limitations

- synthetic proxy environments
- compact shared VLA baseline
- Sprint 4 policies operate on compact environment state rather than the Sprint 1 shared latent
- small simulated PQC
- no quantum hardware
- three principal seeds
- no physical driving validation
- no physical robotics validation
- no embedded full-VLA deployment benchmark
- privileged true-state access for the safety layer in perception-perturbation studies
- no formal stability proof
- no certification

## 12. Phase 2 Roadmap

Phase 2 should prioritize:

1. larger pretrained VLA integration
2. direct shared-latent-to-PPO / hybrid-PQC integration
3. TT/MPS scale-up on larger VLA matrices
4. realistic driving and robotics environments
5. safety under estimated/noisy state
6. embedded deployment benchmarking
7. larger/noisy/hardware PQC evaluation
8. stronger statistical validation

Phase 2 should preserve the frozen Phase 1 interfaces wherever possible and scale the underlying models, environments, policy integration, safety assumptions, and deployment validation rather than rebuild the project from scratch.
