# Development history and disclosure

This file distinguishes task-development observations from the final result.

1. An early small pilot showed a reward-rate benefit for event-aware state features but no clear survival benefit. It was not treated as confirmatory evidence.
2. A longer exploratory pilot with the initial injury mechanics again showed reward-rate improvement without a reliable survival difference.
3. To better operationalize the stated joint reward/survival hypothesis, the environment was revised so that an attack onset or a raid on a refuge occupied too long can cause multiple injuries, and injury/death costs are explicit and shared across all agents. This was a theory-motivated task revision following exploratory outcomes, not a blind pre-run specification.
4. A subsequent exploratory pilot showed positive reward-rate and survival differences. It was excluded from the final inferential sample.
5. The reported 30-seed run used fresh seed IDs 1000–1029, with the training budget, outcomes and analysis defined before that run. The final run is independent at the seed level, but its task structure was informed by earlier pilots. This should be disclosed in an ICML submission.

The intended claim is narrow: **when the ecology has predictable event timing and a temporal safety window, adding inferred event-age state to a classical RL controller can improve reward collection and survival relative to a controller that sees the current schema but cannot represent event age.** The experiment should not be framed as showing that event awareness always helps, nor that it helps when the temporal structure is absent or cannot be inferred reliably.
