# Sprint 5.5 - Heuristic Constraint Clipping

Method: `clipping`

Condition: `clean`

New training performed: `false`

Comparison: frozen `NONE` vs `CLIPPING`

Principal seeds: `42, 123, 456`

Evaluation seeds per cell: `20000..20019`

## autonomous_driving

NONE violation-step rate: 0.390841 +/- 0.422427

CLIPPING violation-step rate: 0.000000 +/- 0.000000

Aggregate violation reduction: `100.00%`

NONE reward: -9.200230 +/- 1.935700

CLIPPING reward: -5.212559 +/- 8.335305

Reward delta: 3.987671

NONE success rate: 0.050000

CLIPPING success rate: 0.333333

Success-rate delta: 0.283333

Mean intervention rate: 0.417500

Pilot effectiveness criterion: `SUPPORTED`

## robotics

NONE violation-step rate: 0.014667 +/- 0.018536

CLIPPING violation-step rate: 0.000000 +/- 0.000000

Aggregate violation reduction: `100.00%`

NONE reward: 0.632846 +/- 0.028818

CLIPPING reward: 0.677900 +/- 0.035786

Reward delta: 0.045054

NONE success rate: 0.000000

CLIPPING success rate: 0.000000

Success-rate delta: 0.000000

Mean intervention rate: 0.014667

Pilot effectiveness criterion: `SUPPORTED`

## Claim Boundary

The clipping result is an empirical clean-condition classical safety result.

Lyapunov superiority: `NOT_YET_TESTED`

Perturbation robustness: `NOT_YET_TESTED`

Formal safety guarantee: `NO`

Production safety claim: `NO`

Quantum or quantum-inspired clipping claim: `NO`
