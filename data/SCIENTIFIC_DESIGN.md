# Scientific design and preregistered reasoning

## Hypothesis

Given identical sensory cues and the same classical average-reward learning algorithm, adding a cue-inferred event boundary and elapsed time since that boundary improves held-out reward rate and survival when action consequences depend on temporal position within an event. The more specific prediction is that event awareness improves over a current-schema-only state representation, not only over an event-blind policy.

## Why current-schema-only is the primary control

A no-event condition omits both current schema and temporal segmentation, so its contrast with event-aware would confound schema identity and event history. `schema_only` receives the same noisy cue stream and decoded current schema as `event_aware`; only event age is absent. The shuffled-boundary control tests whether arbitrary extra timing state is enough. The oracle condition gives privileged true boundaries and is diagnostic only.

## Task mechanics

Every episode begins in forage and moves through warning, attack, and recovery. The warning lasts 12–18 steps in the matched regime; attack lasts 7–10 steps. Food and threat outcomes are independently sampled. Exposed animals face a sudden attack-onset hazard and ongoing attack hazards. A refuge is protective when entered late in the warning, but can be raided if occupied for at least 11 consecutive steps when the attack arrives. This makes the timing problem nontrivial: forage early in warning, enter refuge late, and remain sheltered through attack. An event-age representation can express that policy; a state that only identifies `warning` cannot express a stationary age-dependent action rule.

The occupancy constraint is intentionally explicit and should be viewed as a diagnostic causal manipulation. The flat-refuge scenario removes it. If event awareness still gains similarly in that negative control, the intended mechanism is not isolated and claims must be narrowed.

## Learning and statistics

All learned methods use tabular differential Q-learning, epsilon-greedy exploration, the same alpha/beta schedule and action set. Random event, cue, food, attack, predator and healing streams are independent. The same episode data are used across methods within each training episode; test episode seeds are paired across agents. The inferential unit is the independently trained policy seed (N=30 for the confirmatory profile), with episodes averaged within seed.

Two co-primary endpoints are (1) reward per elapsed decision and (2) proportion of held-out episodes surviving the horizon. Event-aware minus schema-only is tested on the matched regime; two p-values are Holm-adjusted. Bootstrap intervals resample training seeds. Other models/outcomes/scenarios are secondary or diagnostic. No selection among seeds or outcomes is allowed after looking at results.

## Limitations

The event detector is fixed (argmax of a four-dimensional noisy cue), not learned from raw inputs. This study tests the control value of event-structured state features, not autonomous event discovery, neural plausibility, or universal superiority. Because the environment is intentionally constructed to make event age consequential, claims must be bounded to temporally structured ecologies and should be supported by follow-up experiments in distinct task families.
