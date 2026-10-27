# E-EARL Event-Priority benchmark (v0.3)

This package tests whether adding an inferred event boundary and elapsed time since that boundary improves **both reward rate and survival** in a controlled, temporally structured foraging-and-defense ecology. All compared learned policies use the same classical tabular differential Q-learning / average-reward TD update. There are no neural networks.

## Main comparison

- `no_event`: shared ecological features, without an explicit current-schema or event-age feature.
- `schema_only`: same ecological features plus current schema decoded from the noisy cue.
- `event_aware`: same as `schema_only` plus elapsed time since the last cue-inferred schema transition (binned event age).
- `shuffled_event`: schema plus event age inferred from an independent shuffled event stream; negative control for arbitrary timing features.
- `oracle_event`: true event schema and boundary timing; privileged diagnostic upper bound, not a deployable baseline.

The primary comparison is **event-aware versus schema-only**, because both receive the same current-schema cue and use identical RL updates. The only primary representation difference is the inferred event-age feature. Reward rate and episode survival are co-primary outcomes; their two paired tests use Holm correction. Independent training seed is the unit of inference, not evaluation episode.

## Run locally (no Google Drive required)

Python 3.10+ is required. From this directory:

```bash
python -m pip install -e .
python run_experiment.py --smoke-test
python run_experiment.py --profile confirmatory --seed-offset 2000 --outdir E_EARL_EventPriority_Results
```

The confirmatory profile trains **30 independent seeds × 1,800 episodes per agent/seed**, with a maximum episode horizon of **420 decisions**. It evaluates 60 held-out episodes per agent, seed, and each of four scenarios. First run includes Numba compilation overhead. All output is local. The run creates CSVs, figures, policy checkpoints, configuration, a statistical summary, and a report.

Use `--seed-offset 1000` to reproduce the seed range in the supplied result archive (seeds 1000–1029). Use a fresh offset for an independent replication. `pilot` is only for software checks and should not be treated as confirmatory evidence.

The external convenience script `run_eearl_event_priority.py` also works when placed alongside `E_EARL_Event_Priority_Project.zip`; it locates/extracts the project automatically and runs locally. Its default seed offset is 2000.

## Task and theoretical prediction

The environment cycles through forage, warning, attack, and recovery. During warning, the agent can forage while exposed, enter a temporary refuge too early (which risks a raid at attack onset), or wait too long and suffer an ambush. The policy-relevant variable is not only *which* event is happening, but *how far into that event* the agent is. The event-aware representation makes this event-relative timing information available to a stationary tabular policy.

This is a deliberately constructed, falsifiable diagnostic ecology—not a claim that every natural environment has a refuge-tenure cost. The `flat_refuge` condition removes refuge-occupancy raids but retains warning-to-attack timing, so it is a mechanism condition rather than a strict null for all event-age information. `noisy_cues` and `wide_warning` test robustness; the supplied results do not show a reliable survival advantage in every stress scenario.

## Statistical safeguards and limitations

The implementation computes statistics from the actual run; it does not force a desired effect or p-value. The two co-primary tests are paired at seed level, use bootstrap confidence intervals, and receive Holm adjustment. Comparisons outside the primary event-aware-versus-schema-only matched condition are secondary/diagnostic.

Development pilots showed reward benefits without clear survival benefits. Task mechanics were subsequently revised to make attack/raid severity and injury costs explicit. The final confirmatory run uses fresh seed IDs 1000–1029, but the task design was informed by exploratory pilot results. This development history is disclosed in `DEVELOPMENT_HISTORY.md` and the experiment report. The findings therefore support the frozen synthetic benchmark, not a fully independent preregistered task-design claim.

The cue-change detector is fixed (argmax over four noisy cue dimensions), not learned from raw observations. The study tests the control value of event-relative state features, not autonomous event discovery, neural plausibility, or universal superiority. A strong ICML claim needs a second independently designed ecology, pre-registered sensitivity ranges, and robust survival gains under distribution shift.

## Paper-inspired foraging baseline

The result archive separately includes a discretized three-patch average-reward policy-iteration calibration inspired by Zeraati et al. (CCN 2025), with MVT-inspired and optimized single-threshold policies. It is not an exact reproduction and is not run by default, because that Python benchmark is slower than the main Numba task. To rerun it, use the external runner with `--include-paper-baseline`. Its reward rates are not directly comparable to the main foraging-defense ecology.
