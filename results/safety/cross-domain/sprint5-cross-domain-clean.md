# Sprint 5.14C — Clean Cross-Domain Safety Comparison

## Scope

Analysis-only comparison of frozen clean three-seed safety evidence.

## Interpretation rule

Raw driving and robotics violation rates are not treated as directly comparable safety levels because the domains use different safety contracts.

## Cross-domain consistency

### CLIPPING

- Driving effective: True
- Robotics effective: True
- Direction consistent: True

### LYAPUNOV

- Driving effective: True
- Robotics effective: True
- Direction consistent: True

## Supported conclusion

Under each domain's frozen pilot safety contract, the unfiltered PPO policies produced measurable violations in at least part of the three-seed evidence, while explicit clipping and Lyapunov filtering reduced the observed clean executed violation-step rate to zero.

## Limitations

- Raw violation-step rates are not interpreted as cross-domain safety rankings because the two domains use different safety contracts.
- Raw rewards are not compared across domains because the task reward scales differ.
- Relative violation reduction is undefined for a seed whose NONE violation-step rate is zero.
- Results use three principal policy seeds and do not constitute formal statistical significance.
