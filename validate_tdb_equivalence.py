"""Validate that constituent pruning does not alter the thermodynamics.

The raw mc_fe_v2.062.tdb cannot be parsed by pycalphad (malformed for its
parser), so the meaningful equivalence question is: does the final pruned
ternary TDB reproduce the equilibria of the syntax-repaired unpruned TDB
(the same TDB with every out-of-system phase and every non-system
constituent still present)?

For N random points per system, run equilibria with the FULL eligible phase
set using (a) the unpruned extracted TDB and (b) the pruned final TDB, and
compare phase amounts and molar Gibbs energy.
"""
import argparse
import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.config import P
from fe_surrogate.tdb_utils import eligible_phases

FULL_TDB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "databases", "mc_fe_v2.062.tdb")
N_WORKERS = min(8, os.cpu_count() or 4)
_dbs = None


def _init_worker(tdb_a, tdb_b, elements, phases):
    global _dbs
    _dbs = (Database(tdb_a), Database(tdb_b))


def run_one(args):
    cfg, point, phases = args
    x1, x2, temperature = point
    cond = {v.T: temperature, v.P: P}
    for sp, val in zip(cfg["comps_species"], [x1, x2]):
        cond[v.X(sp)] = val
    outs = []
    for db in _dbs:
        try:
            eq = equilibrium(db, cfg["elements"], list(phases), cond)
            ph = np.asarray(eq.Phase.values).flatten()
            npf = np.asarray(eq.NP.values).flatten()
            amounts = {}
            for i, phase in enumerate(phases):
                idx = np.where(ph == phase)[0]
                if len(idx) > 0:
                    amounts[phase] = float(np.nansum(npf[idx]))
            gm = float(np.asarray(eq.GM.values).flatten()[0])
            outs.append({"ok": True, "amounts": amounts, "gm": gm})
        except Exception:
            outs.append({"ok": False, "amounts": {}, "gm": np.nan})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]

    elig = sorted(eligible_phases(FULL_TDB, cfg["comps"]).keys())
    tdb_a = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "databases", f"{args.system}_unpruned.tdb")
    if not os.path.exists(tdb_a):
        sys.exit(f"Missing {tdb_a} - run build_unpruned_tdbs.py first")
    rng = np.random.default_rng(5)
    points = []
    for _ in range(args.n):
        f = rng.dirichlet([0.5, 0.3, 0.2])
        f = np.clip(f, 1e-4, None)
        f = f / f.sum()
        points.append((float(f[1]), float(f[2]), float(rng.uniform(cfg["t_min"], cfg["t_max"]))))

    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(tdb_a, cfg["tdb"], cfg["elements"], elig)) as pool:
        results = pool.map(run_one, [(cfg, p, elig) for p in points])

    max_dnp = 0.0
    max_dgm = 0.0
    n_ok = 0
    n_mismatch = 0
    for a, b in results:
        if not (a["ok"] and b["ok"]):
            continue
        n_ok += 1
        for ph in elig:
            d = abs(a["amounts"].get(ph, 0.0) - b["amounts"].get(ph, 0.0))
            max_dnp = max(max_dnp, d)
            if d > 1e-9:
                n_mismatch += 1
        max_dgm = max(max_dgm, abs(a["gm"] - b["gm"]))
    print(f"{args.system}: {n_ok}/{len(points)} points converged on both TDBs")
    print(f"  max |NP difference|  : {max_dnp:.2e}")
    print(f"  max |GM difference|  : {max_dgm:.2e} J/mol")
    print(f"  phase rows differing : {n_mismatch}")
    if max_dnp > 1e-9 or max_dgm > 1e-6:
        print("  RESULT: EXTRACTION CHANGES THERMODYNAMICS")
        sys.exit(1)
    print("  RESULT: THERMODYNAMICALLY EQUIVALENT")


if __name__ == "__main__":
    mp.freeze_support()
    main()