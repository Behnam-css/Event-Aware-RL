#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from eearl_priority.experiment import run_experiment, smoke_test, PROFILES

def main():
    ap = argparse.ArgumentParser(description="Run the non-neural E-EARL event-priority benchmark")
    ap.add_argument("--profile", choices=list(PROFILES), default="confirmatory")
    ap.add_argument("--outdir", default="E_EARL_EventPriority_Results")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--include-paper-baseline", action="store_true", help="Also run the separate patch-foraging policy-iteration/MVT calibration (can take substantially longer; its precomputed results are included in the results archive)")
    args = ap.parse_args()
    if args.smoke_test:
        print("Smoke test:", smoke_test())
        return
    result = run_experiment(args.outdir, args.profile, args.seed_offset, include_paper_baseline=args.include_paper_baseline)
    print(json.dumps(result, indent=2))
    print("\nOutputs:", Path(args.outdir).resolve())

if __name__ == "__main__": main()
