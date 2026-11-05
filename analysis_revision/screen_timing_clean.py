"""Clean same-hardware per-point solve-cost measurement (R2.5).

The reviewer asked that the screening payback be quoted from measured
constants obtained on the same hardware, not from the literature 2.5 s/point
figure.  The original campaign constants (0.41 s full-set / 0.33 s probe-set
on Fe-Cr-Ni) came from the small validation run; the enlarged audit re-ran
under concurrent training load, inflating wall times (its probe-set timing
even exceeds the full-set timing, proving contention).  This script takes a
dedicated uncontended measurement:

  for each screen system (fecrni @1000 K, fecrnic @1000 K):
    - 14-worker pool, warmup batch (pool init amortised), timed batch,
    - full eligible phase set (28 / 35 phases) and probe set (4 / 9 phases),
    - reports marginal seconds per point (wall) and CPU-seconds per point.

Usage:  py -3.12 -X utf8 analysis_revision/screen_timing_clean.py
Output: models/screen_timing_clean.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "paper"))

import numpy as np
import multiprocessing as mp

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from fe_surrogate.experiment import active_phases as probe_phases
from anchor_eval import _cal_init, _cal_run

WORKERS = 14
WARMUP = 8
N_TIMED = 120
OUT = os.path.join(ROOT, "models", "screen_timing_clean.json")

CASES = [
    ("fecrni", "fecrni_T1000K_step0.001", 1000.0, 1, 2),   # Cr, Ni cols
    ("fecrnic", "fecrnic_T1000K_box", 1000.0, 1, 3),       # Cr, Ni, C cols
]


def measure(system, tag, T_K, c1, c2):
    cfg = SYSTEMS[system]
    scr = np.load(os.path.join(ROOT, "paper", "screen_data",
                               f"screen_{tag}.npz"))
    X = scr["X"]
    rng = np.random.default_rng(11)
    pick = rng.choice(len(X), size=WARMUP + N_TIMED, replace=False)
    warm, timed = pick[:WARMUP], pick[WARMUP:]

    full = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    cols, _ = probe_phases(system)
    probe = sorted(c[len("NP_"):] for c in cols)

    out = {"T_K": T_K, "n_full_phases": len(full), "n_probe_phases": len(probe)}
    for name, phases in (("full", full), ("probe", probe)):
        pool = mp.Pool(WORKERS, initializer=_cal_init,
                       initargs=(cfg["tdb"], cfg["elements"],
                                 cfg["comps_species"], phases))
        for sel, kind in ((warm, "warmup"), (timed, "timed")):
            tasks = [((float(X[i, c1]), float(X[i, c2])), T_K) for i in sel]
            t0 = time.time()
            pool.map(_cal_run, tasks, chunksize=4)
            dt = time.time() - t0
            if kind == "timed":
                out[f"{name}_s_per_point_wall"] = round(dt / len(timed), 3)
                out[f"{name}_s_per_point_cpu"] = round(
                    dt * WORKERS / len(timed), 3)
        pool.close()
        print(f"[{system}] {name} set ({len(phases)} phases): "
              f"{out[f'{name}_s_per_point_wall']} s/point wall, "
              f"{out[f'{name}_s_per_point_cpu']} s/point CPU", flush=True)
    return out


def main():
    res = {"workers": WORKERS, "warmup_points": WARMUP,
           "timed_points": N_TIMED, "systems": {}}
    for system, tag, T_K, c1, c2 in CASES:
        res["systems"][system] = measure(system, tag, T_K, c1, c2)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
