"""Characterize the Fe-Cr-Mo validation discrepancies (pooled)."""
import json
import os
import sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.config import P
from fe_surrogate.tdb_utils import eligible_phases

FULL_TDB = os.path.join(HERE, "databases", "mc_fe_v2.062.tdb")
system = "fecrmo"
N_WORKERS = min(8, os.cpu_count() or 4)
_db = None
_elig = None


def _init(tdb, elig):
    global _db, _elig
    _db = Database(tdb)
    _elig = elig


def run_one(args):
    cfg, x1, x2, T, i = args
    cond = {v.T: T, v.P: P}
    for sp, val in zip(cfg["comps_species"], [x1, x2]):
        cond[v.X(sp)] = val
    try:
        eq = equilibrium(_db, cfg["elements"], list(_elig), cond)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        amounts = {}
        for phase in _elig:
            ii = np.where(ph == phase)[0]
            if len(ii):
                amounts[phase] = float(np.nansum(npf[ii]))
        total = sum(amounts.values())
        gm = float(eq.GM.values.flatten()[0])
        return {"ok": abs(total - 1.0) < 1e-6, "amounts": amounts, "gm": gm, "row": i}
    except Exception:
        return {"ok": False, "amounts": {}, "gm": None, "row": i}


def main():
    cfg = SYSTEMS[system]
    elig = sorted(eligible_phases(FULL_TDB, cfg["comps"]).keys())
    df = pd.read_csv(cfg["dataset"])
    rng = np.random.default_rng(11)
    idx = rng.choice(len(df), min(1500, len(df)), replace=False)
    points = [(cfg, float(df.iloc[i][cfg["comps"][1]]), float(df.iloc[i][cfg["comps"][2]]),
               float(df.iloc[i]["T"]), int(i)) for i in idx]
    probe = json.load(open(os.path.join(HERE, "data", "raw", f"{system}_probe.json")))
    active = sorted(probe["active_counts"].keys())
    print("active set:", active, flush=True)

    with mp.Pool(N_WORKERS, initializer=_init, initargs=(cfg["tdb"], elig)) as pool:
        results = pool.map(run_one, points)

    n = 0
    for r in results:
        if not r["ok"]:
            continue
        i = r["row"]
        diffs = {}
        for phase in elig:  # ALL eligible phases, not only the active set
            full = r["amounts"].get(phase, 0.0)
            if phase in active:
                data_val = float(df.iloc[i][f"NP_{phase}"])
            else:
                data_val = 0.0
            d = abs(full - data_val)
            if d > 1e-3:
                diffs[phase] = (data_val, full, d, phase in active)
        if diffs:
            n += 1
            row = df.iloc[i]
            print(f"\nrow {i}: x_{cfg['comps'][1]}={row[cfg['comps'][1]]:.4f}, "
                  f"x_{cfg['comps'][2]}={row[cfg['comps'][2]]:.4f}, T={row['T']:.0f} K, "
                  f"dGM={r['gm'] - float(row['GM']):+.1f} J/mol, in_box={row['in_stainless_box']}",
                  flush=True)
            for ph_, (dv, fv, dd, is_act) in diffs.items():
                print(f"    {ph_:14s}: dataset {dv:.4f} -> full-set {fv:.4f} "
                      f"(|d|={dd:.3f}){' [ACTIVE]' if is_act else ' [EXCLUDED]'}")
    print(f"\ntotal failing points: {n}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
