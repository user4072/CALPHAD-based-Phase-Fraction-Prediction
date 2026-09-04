"""Validate that the probe-driven active phase set is sufficient.

For a random sample of points from the generated dataset, re-run the
equilibrium with the FULL eligible phase set and compare the phase amounts
against the dataset (computed with the active set). Any phase whose amount
differs by more than 1e-3 in any sampled point means the active set was
incomplete and the dataset must be regenerated with the additional phase.
"""
import argparse
import json
import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.config import P
from fe_surrogate.tdb_utils import eligible_phases

FULL_TDB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "databases", "mc_fe_v2.062.tdb")
N_WORKERS = min(8, os.cpu_count() or 4)
_db = None
_elig = None


def _init_worker(tdb_path, elig):
    global _db, _elig
    _db = Database(tdb_path)
    _elig = elig


def run_one(args):
    cfg, point = args
    x1, x2, temperature = point
    cond = {v.T: temperature, v.P: P}
    for sp, val in zip(cfg["comps_species"], [x1, x2]):
        cond[v.X(sp)] = val
    try:
        eq = equilibrium(_db, cfg["elements"], list(_elig), cond)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        amounts = {}
        for i, phase in enumerate(_elig):
            idx = np.where(ph == phase)[0]
            if len(idx) > 0:
                amounts[phase] = float(np.nansum(npf[idx]))
        total = sum(amounts.values())
        return {"ok": abs(total - 1.0) < 1e-6, "amounts": amounts}
    except Exception:
        return {"ok": False, "amounts": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]

    elig = sorted(eligible_phases(FULL_TDB, cfg["comps"]).keys())
    df = pd.read_csv(cfg["dataset"])
    rng = np.random.default_rng(11)
    idx = rng.choice(len(df), min(args.n, len(df)), replace=False)
    points = [(float(df.iloc[i][cfg["comps"][1]]), float(df.iloc[i][cfg["comps"][2]]),
               float(df.iloc[i]["T"])) for i in idx]

    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"], elig)) as pool:
        results = pool.map(run_one, [(cfg, p) for p in points])

    probe_path = os.path.join(os.path.dirname(cfg["dataset"]), f"{args.system}_probe.json")
    with open(probe_path) as f:
        probe = json.load(f)
    active = sorted(probe["active_counts"].keys())
    active_cols = [f"NP_{p}" for p in active]

    max_diff = 0.0
    n_fail = 0
    missing = set()
    n_ok = 0
    for i, r in zip(idx, results):
        if not r["ok"]:
            continue
        n_ok += 1
        for ph in elig:
            full = r["amounts"].get(ph, 0.0)
            if ph in active:
                data_val = float(df.iloc[i][f"NP_{ph}"])
            else:
                data_val = 0.0
            d = abs(full - data_val)
            max_diff = max(max_diff, d)
            if d > 1e-3:
                n_fail += 1
                missing.add(ph)
    print(f"Validated {n_ok} points (full eligible set vs dataset)")
    print(f"max |NP difference| over all phases: {max_diff:.2e}")
    print(f"points with any phase differing >1e-3: {n_fail}")
    if missing:
        print(f"MISSING PHASES (must regenerate with): {sorted(missing)}")
        sys.exit(1)
    print("PHASE SET SUFFICIENT")


if __name__ == "__main__":
    mp.freeze_support()
    main()