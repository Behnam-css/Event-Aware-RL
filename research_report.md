# E-EARL: When the World Changes, Timing Becomes State
## Event-relative representations for classical average-reward reinforcement learning

**Project status:** completed synthetic benchmark run; results are reproducible from the included code and archives. This report documents one fixed task version, not a claim that the hypothesis is universally established.

> **Central result.** In a synthetic foraging-and-defense ecology with a temporally constrained refuge, a tabular differential Q-learning agent augmented with cue-inferred event age achieved higher held-out reward rate and survival than a matched schema-only agent. The difference was statistically significant on both co-primary outcomes in the matched scenario. The result is bounded to this task, and generalization of the survival effect is incomplete.

---

## 1. Theoretical motivation: perception organizes an unfolding world

An agent encounters a continuous stream of sensory input, but the world it must act in is not best understood as a list of interchangeable instants. A warning is not merely a sensory category; it unfolds. The beginning of a warning, the middle of a warning, and the final moments before an attack can share the same categorical label while differing in the action that is most prudent. The question that motivates E-EARL is therefore not only **which situation is present?** It is also **where in the unfolding situation is the agent now?**

The cognitive science of event perception provides a principled motivation for this distinction. Event-segmentation theories propose that ongoing activity is organized into meaningful events and that transient prediction errors can signal that the current event model should be updated. Event models make the present more coherent and support expectations about what happens next. This motivates a computational idea: a boundary is not merely a descriptive label placed on a sequence after it occurs. It can alter which temporal dependencies are relevant for action.

Reinforcement learning already provides mechanisms for estimating action values under uncertainty. Yet a learner can only condition its policy on what its state representation makes available. In a tabular controller, if two otherwise identical observations map to the same state, the learner cannot select different actions based on an unrepresented distinction. If an action that is appropriate early in a warning becomes dangerous late in the same warning, a representation that stores only the category `warning` forces the agent to blur those temporal positions together. The RL algorithm may be competent, but the state abstraction is not informative enough for the task.

E-EARL tests the narrow, falsifiable claim that **event-relative time can be a useful state variable for control**. It does not assume that event perception replaces model-based RL, cognitive maps, hierarchical reinforcement learning, or world models. Instead, it asks whether the representation of a changing situation can organize an otherwise classical learner's behavior around the appropriate ecological timescale.

### The computational hypothesis

When the environment contains predictable event transitions and action consequences depend on temporal position within those events, an agent that receives a cue-inferred boundary and elapsed time since that boundary should outperform a matched agent that receives the current schema but not event age. The primary prediction is joint: improvement in reward rate **and** survival. A no-event comparison is not sufficient, because it confounds event age with the knowledge of which schema is occurring. The schema-only agent is the critical baseline.

This is a theory of representational utility, not a theory of conscious awareness. “Event-aware” in this implementation means that a policy receives a discrete cue-derived schema and a discretized event clock.

## 2. Experimental ecology: foraging and defense within an event cycle

The environment cycles through four schemas:

1. **Forage** — seek food from one of two replenishing patches.
2. **Warning** — food may still be collected, but an attack is approaching.
3. **Attack** — remaining exposed carries substantial risk.
4. **Recovery** — shelter, rest and health restoration can matter.

The matched regime samples durations of 36–48 decisions for foraging, 12–18 for warning, 7–10 for attack, and 12–18 for recovery. The `wide_warning` stress regime broadens the warning to 10–21 decisions. Each episode has a 420-decision maximum horizon.

### Resource dynamics and actions

Two patches have different initial reward probabilities (0.76 and 0.48). A successful harvest yields +1 reward and depletes the selected patch; inactive patches replenish toward their maxima. The action set is `forage`, `switch_patch`, `seek_shelter`, and `rest_or_defend`. The environment includes small per-step and action costs, hit penalties of 2 reward units, and a terminal death penalty of 12.

### Why event age should matter

The warning phase creates a constrained temporal window. Remaining outside can allow the agent to continue collecting food early in warning. Entering the refuge too early, however, can leave it occupied long enough to be located and raided when the attack arrives. Waiting until attack onset can instead expose the animal to an ambush. In this ecology, the sequence “forage first, shelter later” can be safer and more rewarding than a policy that treats every warning step identically.

At attack onset, an exposed agent or an over-tenured refuge can incur 0–3 injuries (the implementation samples this severity explicitly). During attack, exposed actions also face ongoing risk. Each hit reduces health; death terminates the episode. The ecology is intentionally designed as a diagnostic task to test temporal state representation. It is **not** a claim that all foraging animals face this particular refuge-tenure constraint.

Food draws, event schedules, cue noise, predator draws, onset severity and healing draws use independent random streams. Within a run, methods receive paired environmental streams so the main comparison is not driven simply by different reward or predator realizations.

## 3. From cues to event-relative state

Four cue dimensions encode sensory prototypes for forage, warning, attack and recovery. Each non-oracle agent receives noisy cues, not true event labels. The current schema is decoded by `argmax` over the four cue channels. Whenever that decoded schema differs from the preceding decoded schema, the inferred event clock is reset.

Let:

- `x_t` denote the shared ecological state: location, last patch, resource bins, health, and a refuge-tenure-risk flag;
- `c_t` denote the cue-decoded current schema;
- `b_t = 1[c_t != c_(t-1)]` denote a cue-inferred boundary;
- `tau_t = t - t_boundary` denote steps since the last inferred boundary;
- `phi(tau_t)` denote one of six event-age bins: 0–1, 2–4, 5–7, 8–10, 11–13, and 14+ decisions.

The conditions differ as follows:

| Condition | State representation | Purpose |
|---|---|---|
| No-event | `x_t` | Classical state baseline without explicit schema or event age. |
| Schema-only | `(x_t, c_t)` | Primary control: knows the current schema, but not its age. |
| Event-aware | `(x_t, c_t, phi(tau_t))` | Hypothesis condition: adds inferred event-relative time. |
| Shuffled-event | `(x_t, c_t, phi(tilde_tau_t))` | Negative control using age from an independent, shuffled event stream. |
| Oracle-event | `(x_t, c*_t, phi(tau*_t))` | Privileged true labels and timing; an upper-bound diagnostic, not a fair deployable baseline. |

The event detector is intentionally simple and fixed. It is not trained from observations and does not use the environment's true event label in the event-aware condition. The oracle condition receives true event timing only as a diagnostic. The encoded core state has 192 tabular states; adding the current schema yields 768 states; adding six event-age bins yields 4,608 states for event-aware, shuffled-event and oracle-event conditions (each state has four action values). The shuffled-event control therefore matches the expanded state dimensionality, although it is not a complete control for every learning-capacity effect.

## 4. RL algorithm: one learner, one representation intervention

All conditions use **tabular differential Q-learning with an average-reward objective**. The aim is to learn ongoing reward rate in a cyclical environment rather than to compare neural architectures or discounted return estimators.

For state `s_t`, action `a_t`, reward `r_t`, and next state `s_(t+1)`, define the differential temporal-difference error:

```text
delta_t = r_t - rho_hat_t + max_a Q(s_(t+1), a) - Q(s_t, a_t)
```

Then update the action value and average-gain estimate:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha * delta_t
rho_hat_t   <- rho_hat_t + beta * delta_t   (when the chosen action was greedy)
```

The run uses `alpha = 0.13`, `beta = 0.004`, epsilon-greedy exploration starting at `epsilon = 1.0` and linearly decaying to `0.04` over the first 75% of training. The same learner, action space, reward mechanics and exploration schedule are used across the learned-agent conditions. The event-aware intervention therefore occurs in the state representation rather than in the RL algorithm.

### Why average reward?

Foraging is naturally a repeated-decision problem. Successful behavior should sustain reward collection over elapsed decisions while avoiding injury and death. The average-reward estimate `rho_hat` gives the update a continuing-task interpretation. The exact implementation is a tabular differential TD / R-learning-style gain estimate; this work does not claim a new general-purpose RL algorithm.

## 5. Experimental protocol and inference

The confirmatory profile used:

- 30 independent trained policy seeds, IDs 1000–1029;
- 1,800 training episodes per method per seed;
- 420 decisions maximum per episode;
- 60 held-out episodes per method, seed and test scenario;
- five conditions across four scenarios;
- 270,000 training episode records and 36,000 held-out evaluation records;
- 150 saved tabular-policy checkpoints.

The matched scenario is the primary test. Three additional scenarios probe robustness and mechanism:

- `noisy_cues`: greater sensory cue noise;
- `flat_refuge`: removes the refuge occupancy-raid hazard but leaves warning-to-attack timing intact, so it is a mechanism condition rather than a strict null for all event-age information;
- `wide_warning`: broadens the warning-duration distribution.

### Statistical procedure

Evaluation episodes are averaged within the independently trained seed. The paired seed-level effect is event-aware minus schema-only, on matched ecology. The analysis uses two-sided paired one-sample t-tests of seed-level differences, 20,000-resample percentile bootstrap 95% confidence intervals, and Holm correction across two co-primary outcomes. The inferential sample size is `N = 30` trained policy seeds, not the much larger number of evaluation episodes. Other scenarios and secondary contrasts remain exploratory.

## 6. Primary results

### Matched ecology

| Agent | Reward rate, mean ± SD across seeds | Episode survival, mean ± SD |
|---|---:|---:|
| Event-aware | 0.1680 ± 0.0229 | 0.988 ± 0.015 |
| Schema-only | −0.0352 ± 0.0529 | 0.818 ± 0.130 |
| No-event | −0.0872 ± 0.0464 | 0.544 ± 0.195 |
| Shuffled-event | 0.0927 ± 0.0231 | 0.595 ± 0.095 |
| Oracle-event | 0.1739 ± 0.0265 | 0.979 ± 0.031 |

### Pre-specified co-primary contrasts: event-aware minus schema-only

| Outcome | Paired mean difference | Seed-bootstrap 95% CI | Paired t-test p | Holm-adjusted p | Positive seeds |
|---|---:|---:|---:|---:|---:|
| Reward rate | +0.2032 | [0.1847, 0.2209] | 2.33 × 10⁻¹⁹ | 4.67 × 10⁻¹⁹ | 30/30 |
| Survival probability | +0.1706 (17.1 percentage points) | [0.1278, 0.2178] | 6.04 × 10⁻⁸ | 6.04 × 10⁻⁸ | 29/30 |

Within the matched synthetic ecology, these results support the claim that event-relative time adds control-relevant information beyond the current schema alone. The shuffled-event condition is lower than event-aware on the principal metrics, consistent with the idea that a useful event clock must be aligned to the experienced situation rather than merely adding an arbitrary timing variable. The oracle agent is a privileged diagnostic, and event-aware is near but not above it in the matched condition.

### Scenario sensitivity

Reward-rate benefit persists in noisy-cue and wide-warning regimes. However, the survival difference against schema-only is not statistically distinguishable from zero in these stress settings. The evidence therefore supports a primary matched-condition result, not robust survival superiority across every distribution shift.

The `flat_refuge` condition also retains predictable warning-to-attack structure. As designed, it is not a null environment in which event age must be irrelevant. Similar reward advantage in this condition suggests that event-age information may help time shelter entry even after the specific refuge-occupancy raid penalty is removed. That interpretation requires more targeted controls.

### Visual results

Figures in the website and result archive include: reward rate and survival in the matched condition; training curves; reward-rate and survival comparisons across stress and mechanism scenarios; detector diagnostics; and a separate patch-foraging calibration figure. The matched plots show descriptive means with standard-error bars, whereas inferential confidence intervals are reported separately at the seed level.

## 7. Code framework and outputs

The project is implemented in Python 3.10+ with NumPy, Pandas, SciPy, Matplotlib and Numba. Numba accelerates the step-level simulation and tabular update on CPU. No neural-network framework or GPU is required.

### Main repository layout

```text
E_EARL_Event_Priority_Project/
├── src/eearl_priority/
│   ├── experiment.py       # ecology, state encodings, Q update, training/evaluation,
│   │                       # statistical analysis and plotting
│   └── literature.py       # separate three-patch average-reward calibration
├── tests/test_smoke.py     # lightweight smoke checks
├── run_experiment.py       # local command-line entry point
├── run_eearl_event_priority.py  # convenience runner / project locator
├── pyproject.toml
├── requirements.txt
├── SCIENTIFIC_DESIGN.md
├── analysis_plan.md
├── DEVELOPMENT_HISTORY.md
└── VALIDATION_STATUS.md
```

### Main outputs

The run writes long-form training and held-out evaluation CSVs, seed-level metrics, paired inference, descriptive summaries, compressed Q-table checkpoints (`.npz`), figures, experiment configuration, run manifest, and a Markdown results report. The release archive includes raw episode data and all 150 policy checkpoints so figures can be checked against the underlying data.

### Reproduce locally

```bash
python -m pip install -e .
python run_experiment.py --smoke-test
python run_experiment.py --profile confirmatory --seed-offset 2000 --outdir E_EARL_EventPriority_NewRun
```

Use `--seed-offset 1000` to regenerate the released seed range for reproducibility. Use a fresh seed range for independent replication. The separate literature-inspired three-patch calibration can be invoked with the convenience runner's `--include-paper-baseline` flag; its numerical rewards are not directly comparable to the foraging-defense task.

## 8. Auxiliary literature-inspired patch-foraging calibration

Zeraati and colleagues (CCN 2025) study patch foraging under depletion and replenishment and report an average-reward RL formulation solved with policy iteration, compared with Marginal Value Theorem (MVT) and threshold policies. A separate bundled calibration implements a discretized three-patch model in the same broad model class. In 30 replicates per policy it reports:

| Policy | Mean reward rate | SD |
|---|---:|---:|
| MVT-inspired threshold | 0.5560 | 0.0248 |
| Optimized single threshold | 0.5928 | 0.0214 |
| World-model policy iteration | 0.6220 | 0.0175 |

This is an inspired implementation, not an exact reproduction of the original authors’ unpublished code or all numerical details. It uses a distinct task and reward scale; it is presented as a contextual calibration and is not pooled with the E-EARL primary experiment.

## 9. Interpretation, caveats, and next tests

The event-aware agent significantly outperforms schema-only on both co-primary outcomes within the matched synthetic timed-refuge task. The most important scientific interpretation is that **the temporal position within an event can be a useful control variable when an ecology's hazards and opportunities are temporally structured**.

The result should be interpreted with four safeguards:

1. **Exploratory task refinement.** Early exploratory runs showed reward-rate gains without a reliable survival advantage. The task was then revised to make attack/raid severity and shared injury costs explicit. The final run uses fresh training seeds (1000–1029), but the task design itself was informed by the pilots. This was not a fully blind preregistered task-design process and should be disclosed.
2. **Fixed boundary detector.** The implementation uses cue-category changes to reset the event clock. It does not learn event boundaries from prediction error or raw observations. The present result concerns state augmentation, not autonomous event perception.
3. **Stress-test limits.** Survival benefits do not reliably generalize to noisy cues and wider warning durations. A stronger claim requires more ecologically diverse and distribution-shifted conditions.
4. **State-space growth.** Event-aware state augmentation expands the tabular state space. The shuffled-event negative control is helpful, but future work should more fully match representation size and learning difficulty.

### The decisive next experiments

- Replicate in a second ecology with a different temporal safety or opportunity structure, designed before examining confirmatory outcomes.
- Pre-specify ranges of cue noise, event durations, attack hazards, patch depletion/replenishment and action costs.
- Include a phase-shuffled / boundary-permutation control and an oracle-boundary diagnostic; distinguish current schema, boundary detection, and event age.
- Compare with memory-augmented classical baselines that have matched state capacity, so event structure is tested against generic short-term history rather than merely more states.
- Evaluate risk-sensitive alternatives (e.g., constrained optimization with a minimum survival target) to determine whether reward and survival can improve together without the outcome being tied to a reward trade-off.
- In a later step, make event boundary detection itself learned and test whether perceptual prediction errors produce useful segmentation under uncertain cues.

## 10. Conclusion

E-EARL is a controlled computational test of a cognitive and ecological intuition: situations unfold in time, and the time within a situation can determine the action that is worth taking. By holding the classical average-reward learning rule fixed and manipulating the event structure available in the state, the benchmark makes that intuition testable. The matched-condition findings are promising, with significant improvements in reward rate and survival over a schema-only controller, while the stress results and task-development history clearly delimit the claim. The next scientific challenge is not to make the result look stronger, but to discover whether the advantage survives independent ecologies, harder event inference, and controls for generic temporal memory.

## References

- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds, J. R. (2007). Event perception: A mind–brain perspective. *Psychological Bulletin, 133*(2), 273–293. https://doi.org/10.1037/0033-2909.133.2.273
- Zacks, J. M., & Swallow, K. M. (2007). Event segmentation. *Current Directions in Psychological Science, 16*(2), 80–84. https://doi.org/10.1111/j.1467-8721.2007.00480.x
- Reynolds, J. R., Zacks, J. M., & Braver, T. S. (2007). A computational model of event segmentation from perceptual prediction. *Cognitive Science, 31*(4), 613–643. https://doi.org/10.1080/15326900701399913
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). http://incompleteideas.net/book/the-book-2nd.html
- Zeraati, R., Oña Jodar, T., Safavi, S., Cruz, B., Poo, C., & Dayan, P. (2025). *Optimal Foraging by Learning the World Model*. CCN 2025 abstract. https://2025.ccneuro.org/abstract_pdf/Zeraati_2025_Optimal_Foraging_Learning_World_Model.pdf
