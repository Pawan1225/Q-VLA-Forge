# Q-VLA Forge — Sprint 5.3 Robotics Safety Constraints

## Status

**FROZEN BEFORE SAFETY-FILTER EVALUATION**

## Domain

Synthetic 2-D robotics pick-and-place proxy.

## State

1. robot x
2. robot y
3. object x
4. object y
5. target x
6. target y

## Action

1. delta x
2. delta y
3. gripper

## Violation categories

1. workspace boundary
2. unsafe outward motion
3. unsafe gripper condition
4. object boundary
5. unsafe object/target interaction

## Workspace boundary

Warning:

`abs(coordinate) > 0.90`

Critical:

`abs(coordinate) > 1.00`

## Unsafe motion

Predicted robot position:

`robot_position + 0.08 * action_delta`

Only outward motion into the warning region is considered unsafe.

Inward recovery motion is preserved.

## Gripper safety

Closing command:

`gripper > 0.50`

Unsafe if the robot-object distance exceeds:

`0.12`

Critical if the distance exceeds:

`0.30`

## Object boundary

Warning:

`abs(object coordinate) > 0.90`

Critical:

`abs(object coordinate) > 1.00`

## Object/target interaction

This is a pilot safety heuristic.

A distant target inside the workspace is not automatically unsafe.

Target positions outside the workspace are treated as critical.

## Scientific boundary

No clipping or Lyapunov safety filter has been evaluated.

No safety improvement, industrial robot safety,
production safety, ISO certification, IEC certification,
or formal safety guarantee is claimed.
