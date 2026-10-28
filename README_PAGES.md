# E-EARL: Event-Priority Reinforcement Learning 



## Local preview

Open `index.html` in a browser. For a local web server, run from this folder:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Files

- `index.html` — complete long-form research website.
- `technical_results_report.md` — detailed technical report.
- `assets/figures/` — plots from the run plus a hand-authored conceptual schematic in the page.
- `data/` — summary and raw episode-level CSVs, manifest, config, design and development disclosures.
- `downloads/E_EARL_Event_Priority_Project.zip` — source package.
- `downloads/E_EARL_EventPriority_Confirmatory_Results.zip` — complete run output package.
- `downloads/run_eearl_event_priority.py` — convenience runner.

## Scientific-use note

The main reported comparison is event-aware minus schema-only on a synthetic timed-refuge task, with 30 independent training seeds. Task mechanics were refined after exploratory pilots; that history is disclosed. The event detector is a fixed cue-change rule, not learned event discovery. Survival effects do not reliably generalize to every stress scenario. Read the report and analysis plan before using the result as evidence for a broader claim.
