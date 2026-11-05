"""Seed-matched renorm-MLP runs at the confirmatory fresh seeds.

Re-runs the published renorm arm (train_mlp --heads renorm) with the
three fresh seeds 1009 / 20260905 / 31337 -- the same seeds used by
confirm_gated.py for the gated head -- so that every Table-9 system has
seed-matched fresh-seed numbers for renorm, gated and the classical
baselines. Uses the unmodified train_mlp.main() via a SEEDS override.

Usage:
  py -3.12 -X utf8 analysis_revision/renorm_fresh_all.py \
      [--systems fecrmn fecrni ...]

Outputs:
  models/results_renorm_freshseeds_<system>.json (+ pred npz)
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FRESH = [1009, 20260905, 31337]

ap = argparse.ArgumentParser()
ap.add_argument("--systems", nargs="+", default=[
    "fecrmn", "fecrni", "fecrmo", "fecrv", "femnni",
    "fecrnic", "fecrc", "crconi", "crnimn"])
args = ap.parse_args()

import train_mlp as tm
tm.SEEDS = FRESH

for system in args.systems:
    out = os.path.join(ROOT, "models", f"results_renorm_freshseeds_{system}.json")
    print(f"===== [renorm-fresh] system {system} =====", flush=True)
    tm_argv_backup = sys.argv
    sys.argv = ["train_mlp.py", "--heads", "renorm", "--system", system,
                "--out", out]
    try:
        tm.main()
    finally:
        sys.argv = tm_argv_backup
print("[renorm-fresh] done")
