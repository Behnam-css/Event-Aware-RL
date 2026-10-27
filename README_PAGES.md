# E-EARL: Event-Priority Reinforcement Learning — Website Package

This is a static, GitHub Pages–ready research page. It has no build step, no external JavaScript/CSS dependencies, and uses only local images/data/downloads.

## Publish for free with GitHub Pages

1. Create a new GitHub repository (for example, `eearl-event-priority`).
2. Extract this website ZIP locally.
3. Upload **the contents of this folder** (not the enclosing folder itself) to the repository root. `index.html` must sit at the top level.
4. On GitHub, open **Settings → Pages**.
5. Under “Build and deployment,” choose **Deploy from a branch**, select `main` and `/(root)`, then save.
6. Wait for GitHub Pages to publish the site; its URL will appear in Settings → Pages.

The page should work on GitHub Pages, GitLab Pages, Cloudflare Pages, Netlify, or any ordinary static web host. The assets, plots, CSVs, source ZIP, and results ZIP are linked with relative paths.

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
