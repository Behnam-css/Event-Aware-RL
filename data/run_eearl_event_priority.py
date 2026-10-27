#!/usr/bin/env python3
"""Convenience runner for the E-EARL Event-Priority project.

Place this file next to E_EARL_Event_Priority_Project.zip, or run it from an
already-extracted project directory. It writes output locally; no Google Drive
connection is needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys, zipfile

HERE = Path(__file__).resolve().parent


def locate_project(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates += [HERE, HERE / "E_EARL_Event_Priority_Project"]
    for cand in candidates:
        if (cand / "src" / "eearl_priority" / "experiment.py").is_file():
            return cand
    archive = HERE / "E_EARL_Event_Priority_Project.zip"
    if archive.is_file():
        extracted = HERE / ".eearl_event_priority_extracted"
        marker = extracted / "E_EARL_Event_Priority_Project" / "src" / "eearl_priority" / "experiment.py"
        if not marker.is_file():
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
        candidates += [extracted / "E_EARL_Event_Priority_Project", extracted]
        for cand in candidates:
            if (cand / "src" / "eearl_priority" / "experiment.py").is_file():
                return cand
    raise FileNotFoundError(
        "Cannot locate src/eearl_priority/experiment.py. Put this runner next to "
        "E_EARL_Event_Priority_Project.zip, extract the ZIP, or pass --project-root."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--profile", choices=("pilot", "confirmatory"), default="confirmatory")
    parser.add_argument("--outdir", default="E_EARL_EventPriority_NewRun")
    parser.add_argument("--seed-offset", type=int, default=2000,
                        help="Use a fresh seed range for a new replication; default starts at 2000")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--include-paper-baseline", action="store_true",
                        help="Also run the separate, slower three-patch policy-iteration calibration")
    args = parser.parse_args()
    root = locate_project(args.project_root)
    sys.path.insert(0, str(root / "src"))
    from eearl_priority.experiment import run_experiment, smoke_test
    if args.smoke_test:
        print("Smoke test:", smoke_test())
        return
    result = run_experiment(args.outdir, args.profile, args.seed_offset,
                            include_paper_baseline=args.include_paper_baseline)
    print("Run complete.")
    print("Project:", root)
    print("Results:", Path(args.outdir).resolve())
    print("Primary results:")
    for row in result.get("primary_results", []):
        print(f"  {row['metric']}: difference={row['mean_difference']:.4f}, "
              f"95% CI={row['ci95']}, Holm p={row['p_holm']:.4g}")

if __name__ == "__main__":
    main()
