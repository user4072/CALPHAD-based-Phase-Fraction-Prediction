"""Phase-set validation with FULL eligible phase set, extended for revision.

Re-solves a sample of dataset points with the full constituent-driven eligible
phase set and compares per-phase amounts AND total molar Gibbs energy (GM)
against the stored dataset (computed with the probe-active set). Same sampling
seed (11) as validate_phase_set.py, so the sampled points are identical to the
original validation run.

Reports per system: n re-solved, max |dNP| per phase, max |dGM|, points with
any |dNP| > 1e-3, and missing phases.

Usage: py -3.12 -X utf8 analysis_revision/validate_phase_set_full.py \
           [--n 1500] [--systems fecrni fecrmn fecrmo fecrv femnni]
"""
import argparse
import json
import os
import sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.config import P
from fe_surrogate.tdb_utils import eligible_phases

FULL_TDB = os.path.join(HERE, "databases", "mc_fe_v2.062.tdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase_set_validation.json")
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
        for phase in _elig:
            idx = np.where(ph == phase)[0]
            if len(idx) > 0:
                # Sum over all composition-set vertices (as the generator does)
                amounts[phase] = float(np.nansum(npf[idx]))
        total = sum(amounts.values())
        gm = float(eq.GM.values.flatten()[0])
        return {"ok": abs(total - 1.0) < 1e-6, "amounts": amounts, "gm": gm}
    except Exception:
        return {"ok": False, "amounts": {}, "gm": None}


def validate_system(system, n, out):
    cfg = SYSTEMS[system]
    elig = sorted(eligible_phases(FULL_TDB, cfg["comps"]).keys())
    df = pd.read_csv(cfg["dataset"])
    rng = np.random.default_rng(11)  # same seed as the original validation
    idx = rng.choice(len(df), min(n, len(df)), replace=False)
    points = [(float(df.iloc[i][cfg["comps"][1]]), float(df.iloc[i][cfg["comps"][2]]),
               float(df.iloc[i]["T"])) for i in idx]

    print(f"[{system}] re-solving {len(points)} points with full eligible set "
          f"({len(elig)} phases)...", flush=True)
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"], elig)) as pool:
        results = pool.map(run_one, [(cfg, p) for p in points])

    probe_path = os.path.join(os.path.dirname(cfg["dataset"]), f"{system}_probe.json")
    with open(probe_path) as f:
        probe = json.load(f)
    active = sorted(probe["active_counts"].keys())

    max_diff = 0.0
    max_phase = None
    max_dgm = 0.0
    n_fail = 0
    missing = set()
    n_ok = 0
    for i, r in zip(idx, results):
        if not r["ok"]:
            continue
        n_ok += 1
        for ph in elig:
            full = r["amounts"].get(ph, 0.0)
            data_val = float(df.iloc[i][f"NP_{ph}"]) if ph in active else 0.0
            d = abs(full - data_val)
            if d > max_diff:
                max_diff, max_phase = d, ph
            if d > 1e-3:
                n_fail += 1
                missing.add(ph)
        dgm = abs(r["gm"] - float(df.iloc[i]["GM"]))
        max_dgm = max(max_dgm, dgm)

    res = {
        "system": system,
        "n_points": len(points),
        "n_ok": n_ok,
        "n_eligible": len(elig),
        "n_active": len(active),
        "max_abs_dNP": max_diff,
        "max_abs_dNP_phase": max_phase,
        "max_abs_dGM_J_per_mol": max_dgm,
        "points_with_dNP_gt_1e-3": n_fail,
        "missing_phases": sorted(missing),
    }
    out[system] = res
    print(f"[{system}] ok={n_ok} max|dNP|={max_diff:.3e} (phase {max_phase}) "
          f"max|dGM|={max_dgm:.3e} J/mol fail>1e-3: {n_fail} missing={sorted(missing)}",
          flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--systems", nargs="+",
                    default=["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"])
    args = ap.parse_args()
    out = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            out = json.load(f)
    for system in args.systems:
        if system in out:
            print(f"[{system}] already validated, skipping", flush=True)
            continue
        validate_system(system, args.n, out)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
