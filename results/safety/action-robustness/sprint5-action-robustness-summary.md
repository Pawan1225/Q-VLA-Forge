# Sprint 5.12 — Action Perturbation Robustness

## Protocol

- 24 deterministic structured action perturbations
- 216 principal cells
- 4,320 principal episodes
- Frozen PPO checkpoints
- Methods: NONE, CLIPPING, LYAPUNOV
- Perturbation applied after PPO and before explicit safety filtering
- No pre-filter clipping
- Environment internal clipping accounted separately

## Mechanism Findings

- Unsafe perturbed steps: 111341
- Explicitly recovered unsafe steps: 69367
- Unresolved unsafe steps: 41974
- Overall explicit-filter recovery fraction: 0.623014
- Lyapunov-decrease intervention reasons: 1584
- Strict Lyapunov-decrease steps: 70
- Selected lower than perturbed steps: 4887
- Emergency fallback steps: 0
- Environment-interface adjustment steps: 1579

## Feature Sensitivity

### autonomous_driving / none

- Worst executed-safety delta: acceleration_plus_0p25 (+0.384552)
- Worst reward delta: steering_minus_0p25 (-48.982704)
- Worst success delta: acceleration_plus_0p25 (-0.033333)
- Highest intervention: acceleration_minus_0p10 (0.000000)

### autonomous_driving / clipping

- Worst executed-safety delta: steering_plus_0p25 (+0.273333)
- Worst reward delta: steering_minus_0p25 (-43.341146)
- Worst success delta: braking_plus_0p25 (-0.333333)
- Highest intervention: acceleration_plus_0p25 (0.831500)

### autonomous_driving / lyapunov

- Worst executed-safety delta: steering_plus_0p25 (+0.100500)
- Worst reward delta: steering_minus_0p25 (-38.329736)
- Worst success delta: braking_plus_0p25 (-0.333333)
- Highest intervention: acceleration_plus_0p25 (0.831500)

### robotics / none

- Worst executed-safety delta: gripper_plus_0p50 (+0.375500)
- Worst reward delta: gripper_plus_0p50 (-0.765698)
- Worst success delta: delta_x_minus_0p10 (+0.000000)
- Highest intervention: delta_x_minus_0p10 (0.000000)

### robotics / clipping

- Worst executed-safety delta: delta_x_minus_0p10 (+0.000000)
- Worst reward delta: delta_x_plus_0p25 (-0.243419)
- Worst success delta: delta_x_minus_0p10 (+0.000000)
- Highest intervention: gripper_plus_0p50 (0.392000)

### robotics / lyapunov

- Worst executed-safety delta: delta_x_minus_0p10 (+0.000000)
- Worst reward delta: delta_x_plus_0p25 (-0.242889)
- Worst success delta: delta_x_minus_0p10 (+0.000000)
- Highest intervention: gripper_plus_0p50 (0.391833)

## Interpretation Controls

- A `lyapunov_decrease` intervention reason is not identical to a strict decrease below current V.
- Strict Lyapunov-decrease steps are reported separately.
- Environment-internal clipping is not counted as explicit safety-filter recovery.
- Results are synthetic internal pilot evidence only.
