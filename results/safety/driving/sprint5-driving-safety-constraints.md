# Q-VLA Forge — Sprint 5.2 Driving Safety Constraints

## Status

**FROZEN BEFORE SAFETY-FILTER EVALUATION**

## Domain

Autonomous-driving synthetic proxy.

## State

1. speed
2. lane offset
3. heading error
4. obstacle distance

## Action

1. steering
2. acceleration
3. braking

## Violation categories

1. lane boundary
2. unsafe obstacle distance
3. unsafe speed condition
4. contextual steering risk
5. acceleration/braking conflict

## Lane boundary

Warning violation:

`abs(lane_offset) > 0.75`

Critical:

`abs(lane_offset) > 0.95`

## Obstacle distance

Warning violation:

`obstacle_distance < 0.30`

Critical:

`obstacle_distance < 0.15`

## Pilot speed-distance envelope

`required_safe_distance = 0.15 + 0.35 * max(speed, 0)`

This is a pilot heuristic safety envelope.

It is not a physical stopping-distance model.

## Steering risk

Large steering is only considered a violation when combined
with elevated lane offset or heading error.

## Acceleration/braking conflict

Warning:

`acceleration > 0.25 AND braking > 0.25`

Critical:

`acceleration > 0.60 AND braking > 0.60`

## Scientific boundary

No safety filter has been evaluated in Sprint 5.2.

No safety improvement, collision reduction, certification,
production-vehicle safety, or ISO-compliance claim is made.
