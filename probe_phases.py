"""Probe a ternary system from the whitelist TDB for actively forming phases."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.config import P

PHASE_WHITELIST = ["LIQUID", "FCC_A1", "BCC_A2", "HCP_A3", "BCC_B2", "SIGMA", "CHI_A12"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    sys_cfg = SYSTEMS[args.system]

    db = Database(sys_cfg["tdb"])
    rng = np.random.default_rng(7)
    t_min, t_max = sys_cfg["t_min"], sys_cfg["t_max"]
    comps = sys_cfg["comps"]
    comps_species = sys_cfg["comps_species"]
    counts = {ph: 0 for ph in PHASE_WHITELIST}
    n_ok = 0
    for _ in range(args.n):
        fracs = rng.dirichlet([0.5, 0.3, 0.2])
        fracs = np.clip(fracs, 1e-4, None)
        fracs = fracs / fracs.sum()
        cond = {v.T: float(rng.uniform(t_min, t_max)), v.P: P}
        for sp, frac in zip(comps_species, fracs[1:]):
            cond[v.X(sp)] = float(frac)
        try:
            eq = equilibrium(db, sys_cfg["elements"], PHASE_WHITELIST, cond)
            ph = np.asarray(eq.Phase.values).flatten()
            npf = np.asarray(eq.NP.values).flatten()
            total = 0.0
            for i, phase in enumerate(PHASE_WHITELIST):
                idx = np.where(ph == phase)[0]
                if len(idx) > 0:
                    s = float(np.nansum(npf[idx]))
                    total += s
                    if s > 1e-6:
                        counts[phase] += 1
            if abs(total - 1.0) < 1e-6:
                n_ok += 1
        except Exception:
            pass
    print(f"System {args.system}: {n_ok}/{args.n} mass-balanced equilibria")
    for ph in PHASE_WHITELIST:
        if counts[ph] > 0:
            print(f"  ACTIVE {ph:10s}: {counts[ph]} ({100 * counts[ph] / n_ok:.1f}%)")


if __name__ == "__main__":
    main()