"""Application case A4 (validation): full-set CALPHAD check of the
quaternary screen (35 eligible phases).

Same protocol as screen_validate.py: Pareto-frontier shortlist points
(minimal Ni per Cr bin), a uniform random hit subset (precision), and a
uniform random background sample (miss rate). Query: FCC>=0.90,
carbides<=0.05, SIGMA/LIQUID<=1e-3, Ni<=0.12 at 1000 K.

Outputs (NEW files only):
  paper/screen_data/validate_fecrnic_T1000K_box.npz + .json

Usage:
    py -3.12 paper/screen_validate_q.py --n-hits 400 --n-bg 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from anchor_eval import _cal_run, _cal_init
import multiprocessing as mp

SYSTEM = "fecrnic"
TAG = "fecrnic_T1000K_box"
Q_FCC_MIN, Q_CARB_MAX = 0.90, 0.05
Q_SIGMA_MAX, Q_LIQ_MAX, Q_NI_MAX = 1e-3, 1e-3, 0.12
CARBIDES = ("M23C6", "M7C3", "CEMENTITE", "GRAPHITE")
N_CR_BINS = 35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hits", type=int, default=400)
    ap.add_argument("--n-bg", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe-timing-n", type=int, default=30)
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, "screen_data", f"screen_{TAG}.npz"))
    X, ens, hit_idx = d["X"], d["ensemble"], d["hit_idx"]
    names = list(d["phases"])
    T_K = float(X[0, 4])
    rng = np.random.default_rng(7)

    cr = X[hit_idx, 1]
    bins = np.floor((cr - 0.001) / 0.01).astype(int)
    frontier = []
    for b in sorted(set(bins.tolist())):
        sel = hit_idx[bins == b]
        frontier.append(sel[np.argmin(X[sel, 2])])
    frontier = np.array(sorted(set(int(f) for f in frontier)))
    rest = np.setdiff1d(hit_idx, frontier)
    sub_hits = rng.choice(rest, size=min(args.n_hits, len(rest)),
                          replace=False) if len(rest) else np.array([], int)
    nonhit = np.setdiff1d(np.arange(len(X)), hit_idx)
    sub_bg = rng.choice(nonhit, size=min(args.n_bg, len(nonhit)),
                        replace=False)
    print(f"[validate-q] hits={len(hit_idx):,} frontier={len(frontier)} "
          f"sub-hits={len(sub_hits)} background={len(sub_bg)}")

    cfg = SYSTEMS[SYSTEM]
    phases = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    print(f"[validate-q] full set: {len(phases)} eligible phases, "
          f"{args.workers} workers")
    pool = mp.Pool(args.workers, initializer=_cal_init,
                   initargs=(cfg["tdb"], cfg["elements"],
                             cfg["comps_species"], phases))

    import time

    def run_points(sel):
        tasks = [((float(X[i, 1]), float(X[i, 2]), float(X[i, 3])), T_K)
                 for i in sel]
        t0 = time.time()
        out = pool.map(_cal_run, tasks, chunksize=4)
        return out, time.time() - t0

    order = np.concatenate([frontier, sub_hits, sub_bg])
    kinds = (["frontier"] * len(frontier) + ["hit"] * len(sub_hits)
             + ["bg"] * len(sub_bg))
    res, t_cal = run_points(order)
    print(f"[validate-q] {len(order)} full-set solves in {t_cal:.0f}s "
          f"({t_cal / len(order):.2f} s/point wall)")

    def frac(r, phase):
        return r["amounts"].get(phase, 0.0) if r["ok"] else np.nan

    F = np.array([frac(r, "FCC_A1") for r in res])
    C = np.array([sum(frac(r, p) for p in CARBIDES) for r in res])
    S = np.array([frac(r, "SIGMA") for r in res])
    L = np.array([frac(r, "LIQUID") for r in res])
    ok = np.array([r["ok"] for r in res])
    confirmed = (ok & (F >= Q_FCC_MIN) & (C <= Q_CARB_MAX)
                 & (S <= Q_SIGMA_MAX) & (L <= Q_LIQ_MAX))

    out = {}
    for k in ("frontier", "hit", "bg"):
        m = np.array(kinds) == k
        out[k] = {
            "n": int(m.sum()),
            "n_ok": int((ok[m]).sum()),
            "n_confirmed": int((confirmed[m]).sum()),
            "precision": (round(float(confirmed[m].sum() / ok[m].sum()), 4)
                          if ok[m].sum() else None),
        }
    out["calphad_wall_s"] = round(t_cal, 1)
    out["calphad_s_per_point_wall"] = round(t_cal / len(order), 3)

    from fe_surrogate.experiment import active_phases as probe_phases
    cols, _ = probe_phases(SYSTEM)
    probe_set = sorted(c[len("NP_"):] for c in cols)
    pool2 = mp.Pool(args.workers, initializer=_cal_init,
                    initargs=(cfg["tdb"], cfg["elements"],
                              cfg["comps_species"], probe_set))
    sample = rng.choice(np.arange(len(X)),
                        size=min(args.probe_timing_n, len(X)), replace=False)
    tasks = [((float(X[i, 1]), float(X[i, 2]), float(X[i, 3])), T_K)
             for i in sample]
    t0 = time.time()
    out2 = pool2.map(_cal_run, tasks, chunksize=4)
    t_probe = time.time() - t0
    out["probe_set"] = {
        "phases": probe_set, "n": len(sample),
        "s_per_point_wall": round(t_probe / len(sample), 3),
        "n_ok": int(sum(r["ok"] for r in out2)),
    }
    print(f"[validate-q] probe-set ({len(probe_set)} phases): "
          f"{t_probe / len(sample):.2f} s/point wall")

    pool.close()
    pool2.close()

    np.savez_compressed(
        os.path.join(HERE, "screen_data", f"validate_{TAG}.npz"),
        order=order, kinds=np.array(kinds),
        FCC=F, CARB=C, SIGMA=S, LIQUID=L, ok=ok, confirmed=confirmed)
    with open(os.path.join(HERE, "screen_data",
                           f"validate_{TAG}.json"), "w") as f:
        json.dump({"tag": TAG, "T_K": T_K,
                   "query": {"FCC_min": Q_FCC_MIN, "CARB_max": Q_CARB_MAX,
                             "SIGMA_max": Q_SIGMA_MAX,
                             "LIQUID_max": Q_LIQ_MAX, "Ni_max": Q_NI_MAX},
                   **out}, f, indent=2)
    print("[validate-q] precision:", out["hit"]["precision"],
          "| frontier:", out["frontier"]["precision"])
    print(f"[validate-q] wrote validate_{TAG}.npz/.json")


if __name__ == "__main__":
    main()
