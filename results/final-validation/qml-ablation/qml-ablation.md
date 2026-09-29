# Q-VLA Forge - Sprint 7.4 Final QML Ablation

Frozen Sprint 4 evidence only. No new PPO training, QML training, circuit tuning, target modification, or new experiments.

The primary comparison is PPO + MLP versus PPO + PQC. The matched-classical actor is retained separately as a parameter-matched control.

## Autonomous Driving

| Policy | Target Reach | Steps to Target | Final Reward | Final Success | Actor Params |
|---|---:|---:|---:|---:|---:|
| PPO + MLP | 3/3 | 13333.3 +/- 1527.5 | -9.200 +/- 1.936 | 0.050 +/- 0.087 | 1318 |
| PPO + PQC | 0/3 | Not reached | -35.444 +/- 18.964 | 0.000 +/- 0.000 | 54 |

### Parameter-Matched Classical Control

- Target reach: 0/3
- Steps to target: Not reached
- Actor parameters: 54

### Actor Compactness

- Classical PPO actor: 1318 parameters
- PQC actor: 54 parameters
- Parameter reduction: 95.90%

## Robotics

| Policy | Target Reach | Steps to Target | Final Reward | Final Success | Actor Params |
|---|---:|---:|---:|---:|---:|
| PPO + MLP | 3/3 | 10000.0 +/- 1732.1 | 0.633 +/- 0.029 | 0.000 +/- 0.000 | 1382 |
| PPO + PQC | 0/3 | Not reached | -0.267 +/- 0.853 | 0.000 +/- 0.000 | 62 |

### Parameter-Matched Classical Control

- Target reach: 1/3
- Steps to target: 20000.0
- Actor parameters: 62

### Actor Compactness

- Classical PPO actor: 1382 parameters
- PQC actor: 62 parameters
- Parameter reduction: 95.51%

## Cross-Domain Result

- Classical PPO target attainment: 6/6
- PPO + PQC target attainment: 0/6
- Matched-classical control target attainment: 1/6

## Scientific Conclusion

The evaluated PQC policy used substantially fewer trainable actor parameters than the classical PPO actor, but did not demonstrate a sample-efficiency or performance advantage over the classical PPO reference under the frozen Phase 1 protocol.

Failure to reach the frozen performance target is represented as a null steps-to-target value. The 20,000-step training budget is not treated as a measured convergence value.

Actor compactness is reported as an architecture observation and is not treated as evidence of sample-efficiency or performance superiority.

## Claim Controls

- Quantum advantage: blocked
- Quantum speedup: blocked
- QML sample-efficiency advantage: blocked
- QML performance superiority: blocked
- QML convergence superiority: blocked
- QPU advantage: blocked
- Production VLA superiority: blocked

## Execution Boundary

- Quantum hardware used: false
- Execution backend: simulator
