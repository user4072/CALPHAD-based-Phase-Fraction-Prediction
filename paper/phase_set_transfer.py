"""Phase-set validation with Gibbs-energy capture for the transfer systems.

Mirrors validate_phase_set.py (same full-eligible-set comparison, same
mass-balance gate) but additionally records |dGM| so the extension-table
panel D columns can be filled, and writes the analysis_revision JSON in
the exact schema of phase_set_validation_fecrnic.json. n=1500 matches the
extension rows.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.config import P
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases

N_VAL = 1500
N_WORKERS = min(8, os.cpu_count() or 4)
_db = None


def _init_worker(tdb_path):
    global _db
    _db = Database(tdb_path)


def run_one(args):
    cfg, cond, elig = args
    try:
        eq = equilibrium(_db, cfg["elements"], list(elig), cond,
                         max_iterations=500)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        amounts = {}
        for name in elig:
            idx = np.where(ph == name)[0]
            if len(idx) > 0:
                amounts[name] = float(np.nansum(npf[idx]))
        total = sum(amounts.values())
        gm = float(eq.GM.values.flatten()[0])
        return {"ok": abs(total - 1.0) < 1e-6, "amounts": amounts, "GM": gm}
    except Exception:
        return {"ok": False, "amounts": {}, "GM": np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["crconi", "crnimn"], required=True)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]
    df = pd.read_csv(cfg["dataset"])
    probe = json.load(open(os.path.join(os.path.dirname(cfg["dataset"]),
                                        f"{args.system}_probe.json")))
    elig = sorted(probe["eligible"])
    active = set(probe["active_counts"].keys())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), min(N_VAL, len(df)), replace=False)
    conds = []
    for i in idx:
        r = df.iloc[i]
        cond = {v.T: float(r["T"]), v.P: P}
        for sp, col in zip(cfg["comps_species"], cfg["comps"][1:]):
            cond[v.X(sp)] = float(r[col])
        conds.append(cond)
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"],)) as pool:
        results = pool.map(run_one, [(cfg, c, elig) for c in conds])
    n_ok, n_fail, max_diff, max_ph, max_dgm = 0, 0, 0.0, None, 0.0
    missing = set()
    for i, r in zip(idx, results):
        if not r["ok"]:
            continue
        n_ok += 1
        for ph in elig:
            full = r["amounts"].get(ph, 0.0)
            data_val = float(df.iloc[i][f"NP_{ph}"]) if ph in active else 0.0
            d = abs(full - data_val)
            if d > max_diff:
                max_diff, max_ph = d, ph
            if d > 1e-3:
                n_fail += 1
                missing.add(ph)
        g = abs(r["GM"] - float(df.iloc[i]["GM"]))
        if np.isfinite(g):
            max_dgm = max(max_dgm, g)
    out = {args.system: {
        "system": args.system, "n_points": int(len(idx)), "n_ok": n_ok,
        "n_eligible": len(elig), "n_active": len(active),
        "max_abs_dNP": max_diff, "max_abs_dNP_phase": max_ph,
        "max_abs_dGM_J_per_mol": max_dgm,
        "points_with_dNP_gt_1e-3": n_fail, "missing_phases": sorted(missing)}}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "analysis_revision",
                        f"phase_set_validation_{args.system}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{args.system}: n_ok={n_ok} max_dNP={max_diff:.2e} ({max_ph}) "
          f"max_dGM={max_dgm:.2e} fails={n_fail} -> {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
