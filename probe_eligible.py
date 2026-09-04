"""Probe a ternary system for actively forming phases (parallel).

Runs equilibria with the FULL constituent-driven eligible phase set (no
hand-picked whitelist) and writes the observed active phases to a sidecar
JSON that the generator then uses. A separate validation step
(validate_phase_set.py) later confirms on the generated data that no
excluded phase was ever stable.
"""
import argparse
import json
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
_db = None


def _init_worker(tdb_path, elements, phases):
    global _db
    _db = Database(tdb_path)


def run_one(args):
    cond, elements, phases = args
    try:
        eq = equilibrium(_db, elements, list(phases), cond)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        out = {}
        total = 0.0
        for i, phase in enumerate(phases):
            idx = np.where(ph == phase)[0]
            if len(idx) > 0:
                s = float(np.nansum(npf[idx]))
                total += s
                if s > 1e-6:
                    out[phase] = s
        return {"ok": abs(total - 1.0) < 1e-6, "active": out}
    except Exception:
        return {"ok": False, "active": {}}


def probe(cfg, phases, n, seed=7):
    rng = np.random.default_rng(seed)
    conds = []
    for _ in range(n):
        fracs = rng.dirichlet([0.5, 0.3, 0.2])
        fracs = np.clip(fracs, 1e-4, None)
        fracs = fracs / fracs.sum()
        cond = {v.T: float(rng.uniform(cfg["t_min"], cfg["t_max"])), v.P: P}
        for sp, frac in zip(cfg["comps_species"], fracs[1:]):
            cond[v.X(sp)] = float(frac)
        conds.append(cond)
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"], cfg["elements"], list(phases))) as pool:
        results = pool.map(run_one, [(c, cfg["elements"], list(phases)) for c in conds])
    counts = {}
    n_ok = 0
    for r in results:
        if r["ok"]:
            n_ok += 1
        for ph in r["active"]:
            counts[ph] = counts.get(ph, 0) + 1
    return n_ok, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]
    elig = sorted(eligible_phases(FULL_TDB, cfg["comps"]).keys())
    print(f"Probing {args.system}: {len(elig)} eligible phases, {args.n} points...")
    n_ok, counts = probe(cfg, elig, args.n)
    out = {"system": args.system, "n_points": args.n, "n_ok": n_ok,
           "eligible": elig, "active_counts": counts,
           "rate": {ph: round(c / max(n_ok, 1), 4) for ph, c in counts.items()}}
    path = os.path.join(os.path.dirname(cfg["dataset"]), f"{args.system}_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{n_ok}/{args.n} mass-balanced | ACTIVE phases (saved to {path}):")
    for ph, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  ACTIVE {ph:14s}: {c} ({100 * c / max(n_ok, 1):.1f}%)")


if __name__ == "__main__":
    mp.freeze_support()
    main()