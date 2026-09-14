# Q-VLA Forge — Sprint 5.1 Safety Protocol

## Status

**FROZEN BEFORE SAFETY RESULTS**

## Scientific question

Can heuristic clipping and Lyapunov-based safety filtering reduce
constraint violations and improve robustness under state/action
perturbations while preserving acceptable task performance?

## Primary policy

`classical_ppo`

## Safety methods

- `none`
- `clipping`
- `lyapunov`

## Principal seeds

- 42
- 123
- 456

## Held-out evaluation seeds

`20000–20019`

## Episodes per evaluation cell

20

## Primary safety metric

`violation_step_rate`

## Pilot acceptance criteria

- Mean violation-step-rate reduction: >= 20%
- Maximum reward degradation: <= 10%
- Maximum success-rate drop: <= 10 percentage points

These are internal pilot thresholds only and are not certification
or Volkswagen production-safety requirements.

## Robustness conditions

- clean
- Gaussian state perturbation
- structured/state perturbation
- action perturbation

Gaussian standard deviations:

`0.00, 0.01, 0.05, 0.10`

## Safety filter information

The primary Sprint 5 pilot safety filter receives simulator
ground-truth current state.

This assumption must remain disclosed in all proposal-facing claims.

## Statistics

- mean
- sample standard deviation
- paired seed deltas
- no inferential significance testing

## Claim state

- Clipping safety improvement: NOT YET TESTED
- Lyapunov safety improvement: NOT YET TESTED
- Lyapunov over clipping: NOT YET TESTED
- Perturbation robustness: NOT YET TESTED
- Cross-domain safety reuse: NOT YET TESTED

## Scientific boundaries

Sprint 5.1 establishes no safety improvement result,
formal safety proof, certification, ISO compliance,
or production safety guarantee.

## Sprint 4 provenance

Handoff status: `ready`

SHA256:

`aab1cf3cfeeec866033402555926942a3879239b4051ca234cbebbc7259337d6`
