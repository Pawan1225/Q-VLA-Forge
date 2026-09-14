# Sprint 5.11 — Structured State Robustness

## Protocol

- 10 autonomous-driving structured perturbations.
- 12 robotics structured perturbations.
- NONE, CLIPPING, and LYAPUNOV.
- Principal seeds 42, 123, and 456.
- 20 held-out evaluation episodes per cell.
- 198 principal cells and 3960 principal episodes.
- Policy receives perturbed observation.
- Safety layer and environment receive true simulator state.
- No Gaussian noise, action perturbation, training, or filter tuning.

## Feature sensitivity

### autonomous_driving

- **none** worst violation: `lane_offset_plus_0p10` (+0.006752); worst reward: `lane_offset_plus_0p10` (-2.126991); worst success: `heading_error_minus_0p05` (+0.000000).
- **clipping** worst violation: `heading_error_minus_0p05` (+0.000000); worst reward: `lane_offset_plus_0p10` (-2.582322); worst success: `heading_error_minus_0p05` (+0.000000).
- **lyapunov** worst violation: `heading_error_minus_0p05` (+0.000000); worst reward: `lane_offset_plus_0p10` (-2.582322); worst success: `heading_error_minus_0p05` (+0.000000).

### robotics

- **none** worst violation: `object_x_minus_0p05` (+0.020333); worst reward: `robot_x_minus_0p05` (-0.038311); worst success: `object_x_minus_0p05` (+0.000000).
- **clipping** worst violation: `object_x_minus_0p05` (+0.000000); worst reward: `robot_x_minus_0p05` (-0.041698); worst success: `object_x_minus_0p05` (+0.000000).
- **lyapunov** worst violation: `object_x_minus_0p05` (+0.000000); worst reward: `robot_x_minus_0p05` (-0.041701); worst success: `object_x_minus_0p05` (+0.000000).

## Lyapunov mechanism

- Intervention reason counts: `{'domain_constraint': 25992, 'none': 106008}`
- Strict Lyapunov-decrease steps: 0
- Robotics grasped-state steps in Lyapunov runs: 0

## Limitations

- Structured perturbations are synthetic fixed coordinate offsets.
- They are not calibrated production perception or sensor-fusion faults.
- The safety layer retains privileged access to true simulator state.
- No formal, certified, production, adversarial, ISO, or quantum robustness claim is made.
