"""B1: validate EVERY quaternary shortlist hit with full-set CALPHAD.

Validates the shortlist points not covered by screen_validate_q.py
(20 frontier + 400 subset already done): full 35-phase set, single-T
(1000 K) solves, mass-balance gate. Chunked with per-chunk saves ->
resumable: re-running continues from the partial file.

Outputs (NEW files only):
  paper/screen_data/validate_all_hits_fecrnic_T1000K_box_partial.npz
  paper/screen_data/validate_all_hits_fecrnic_T1000K_box.npz + .json

Usage:
    py -3.12 paper/screen_validate_all_q.py --workers 8 --chunks 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from anchor_eval import _cal_init, _cal_run
import multiprocessing as mp

SYSTEM = "fecrnic"
TAG = "fecrnic_T1000K_box"
Q_FCC_MIN, Q_CARB_MAX = 0.90, 0.05
Q_SIGMA_MAX, Q_LIQ_MAX, Q_NI_MAX = 1e-3, 1e-3, 0.12
CARBIDES = ("M23C6", "M7C3", "CEMENTITE", "GRAPHITE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=10)
    args = ap.parse_args()

    SD = os.path.join(HERE, "screen_data")
    scr = np.load(os.path.join(SD, f"screen_{TAG}.npz"))
    val = np.load(os.path.join(SD, f"validate_{TAG}.npz"))
    X = scr["X"]
    T_K = float(X[0, 4])
    hit_idx = np.array(scr["hit_idx"])
    kinds = np.array([k for k in val["kinds"]])
    done = np.array(val["order"])[kinds != "bg"]
    todo = np.setdiff1d(hit_idx, done)
    print(f"[all-q] hits={len(hit_idx):,} already validated={len(done)} "
          f"remaining={len(todo):,}", flush=True)

    partial = os.path.join(SD, f"validate_all_hits_{TAG}_partial.npz")
    got = {}
    if os.path.exists(partial):
        p = np.load(partial, allow_pickle=True)
        for i, r in zip(p["order"], p["res"].tolist()):
            got[int(i)] = r
        print(f"[all-q] resumed {len(got)} completed points", flush=True)
    todo = np.array([i for i in todo if int(i) not in got])
    n_pool = len(todo) + len(got)
    print(f"[all-q] to run: {len(todo):,}", flush=True)

    cfg = SYSTEMS[SYSTEM]
    phases = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    pool = mp.Pool(args.workers, initializer=_cal_init,
                   initargs=(cfg["tdb"], cfg["elements"],
                             cfg["comps_species"], phases))
    t_all = time.time()
    try:
        for ci, blk in enumerate(np.array_split(todo, args.chunks)):
            if not len(blk):
                continue
            tasks = [((float(X[i, 1]), float(X[i, 2]), float(X[i, 3])), T_K)
                     for i in blk]
            t0 = time.time()
            out = pool.map(_cal_run, tasks, chunksize=4)
            for i, r in zip(blk, out):
                got[int(i)] = {"ok": bool(r["ok"]),
                               "amounts": r["amounts"],
                               "total": r["total"], "err": r["err"]}
            np.savez_compressed(
                partial, order=np.array(sorted(got)),
                res=np.array([got[i] for i in sorted(got)], dtype=object))
            dt = time.time() - t0
            print(f"[all-q] chunk {ci + 1}: {len(blk)} pts in {dt:.0f}s "
                  f"({dt / len(blk):.2f} s/pt wall), "
                  f"total done {len(got)}/{n_pool}", flush=True)
    finally:
        pool.close()

    order = np.array(sorted(got))
    F = np.array([got[i]["amounts"].get("FCC_A1", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    C = np.array([sum(got[i]["amounts"].get(p, 0.0) for p in CARBIDES)
                  if got[i]["ok"] else np.nan for i in order])
    S = np.array([got[i]["amounts"].get("SIGMA", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    L = np.array([got[i]["amounts"].get("LIQUID", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    ok = np.array([got[i]["ok"] for i in order])
    conf = (ok & (F >= Q_FCC_MIN) & (C <= Q_CARB_MAX)
            & (S <= Q_SIGMA_MAX) & (L <= Q_LIQ_MAX))
    t_wall = time.time() - t_all
    summary = {
        "tag": TAG, "n": int(len(order)), "n_ok": int(ok.sum()),
        "n_confirmed": int(conf.sum()),
        "precision": round(float(conf.sum() / ok.sum()), 4)
        if ok.sum() else None,
        "query": {"FCC_min": Q_FCC_MIN, "CARB_max": Q_CARB_MAX,
                  "SIGMA_max": Q_SIGMA_MAX, "LIQUID_max": Q_LIQ_MAX,
                  "Ni_max": Q_NI_MAX},
        "wall_s": round(t_wall, 1),
        "s_per_point_wall": round(t_wall / max(len(order), 1), 3),
    }
    np.savez_compressed(os.path.join(SD, f"validate_all_hits_{TAG}.npz"),
                        order=order, FCC=F, CARB=C, SIGMA=S, LIQUID=L,
                        ok=ok, confirmed=conf)
    with open(os.path.join(SD, f"validate_all_hits_{TAG}.json"),
              "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[all-q] precision {summary['precision']} "
          f"({summary['n_confirmed']}/{summary['n_ok']}) "
          f"in {t_wall:.0f}s wall")
    print(f"[all-q] wrote validate_all_hits_{TAG}.npz/.json")


if __name__ == "__main__":
    main()
