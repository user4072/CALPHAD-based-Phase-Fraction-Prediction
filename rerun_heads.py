"""Rerun all head variants for all five systems with the current code.

Regenerates models/results_heads_<system>.json so that boundary_mae/bulk_mae
use the entropy rule in experiment.py (the stored files predate it) and every
metric in the report comes from one consistent code revision.

Usage:  py -3.12 rerun_heads.py            # all five systems
        py -3.12 rerun_heads.py fecrni     # one system
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
HEADS = ["sigmoid", "softmax", "residue", "renorm", "sig_norm", "sparsemax"]


def main():
    systems = sys.argv[1:] or SYSTEMS
    py = sys.executable
    for system in systems:
        out = os.path.join(HERE, "models", f"results_heads_{system}.json")
        cmd = [py, os.path.join(HERE, "train_mlp.py"), "--system", system,
               "--heads", *HEADS, "--out", out]
        t0 = time.time()
        print(f"\n=== {system} ===\n{' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        print(f"=== {system} done rc={r.returncode} in {(time.time()-t0)/60:.1f} min ===",
              flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
