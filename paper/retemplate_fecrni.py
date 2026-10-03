"""Re-templated Fe-Cr-Ni dataset: the sampling-coverage gold standard.

The five ternaries share one seed-42 sampling template, so no analysis
of the published splits can separate topology effects from template
effects. This script generates a second Fe-Cr-Ni dataset under a
DIFFERENT template -- new RNG seed (1007) plus a shifted sigma-focus
box -- with everything else identical: same TDB (fecrni_ternary.tdb),
same probe-active phase basis (no re-probing: the basis is an oracle
property, not a sampling property), same acceptance rule, same
stainless-box tag definition (kept identical for comparability).

Template delta (documented, arbitrary-but-fixed):
  seed 42 -> 1007 (all random draws + task shuffle)
  sigma_focus low  [0.15, 0.50, 700]  -> [0.10, 0.45, 700]
  sigma_focus high [0.02, 0.25, 1250] -> [0.05, 0.30, 1250]

Outputs (NEW files only; fecrni artifacts untouched):
  data/raw/dataset_fecrni_rt.csv
  data/raw/checkpoint_fecrni_rt.csv
  data/raw/fecrni_rt_active.json

Usage: py -3.12 paper/retemplate_fecrni.py [--limit N]  (smoke test)
       py -3.12 paper/retemplate_fecrni.py --base femnni  (second system)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import generate_data as G
from fe_surrogate.systems import SYSTEMS
from generate_data import (run_eq, _init_worker, N_WORKERS, ACTIVE_THRESHOLD,
                           NONZERO_LEVEL, MASS_BALANCE_TOL)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retemplate")

SYSID = "fecrni_rt"
SEED = 1007
BASE = "fecrni"
FOCUS_LOW = [0.10, 0.45, 700.0]
FOCUS_HIGH = [0.05, 0.30, 1250.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", type=str, default=BASE,
                    help="base system id in SYSTEMS (same TDB/probe basis)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--suffix", type=str, default="rt")
    ap.add_argument("--focus-low", type=float, nargs=3,
                    default=FOCUS_LOW)
    ap.add_argument("--focus-high", type=float, nargs=3,
                    default=FOCUS_HIGH)
    args = ap.parse_args()
    sysid, base, seed = f"{args.base}_{args.suffix}", args.base, args.seed

    cfg = dict(SYSTEMS[base])
    cfg["dataset"] = os.path.join(
        HERE, "..", "data", "raw", f"dataset_{sysid}.csv")
    cfg["dataset"] = os.path.normpath(cfg["dataset"])
    cfg["checkpoint"] = os.path.join(
        os.path.dirname(cfg["dataset"]), f"checkpoint_{sysid}.csv")
    cfg["sigma_focus"] = {"low": list(args.focus_low),
                          "high": list(args.focus_high)}
    # box tag definition intentionally UNCHANGED (comparability)
    probe = json.load(open(os.path.join(os.path.dirname(cfg["dataset"]),
                                        f"{base}_probe.json")))
    phases = sorted(probe["active_counts"].keys())
    phase_cols = [f"NP_{p}" for p in phases]
    logger.info(f"system {sysid}: {len(phases)} probe-active phases "
                f"(reused {base} basis)")

    tasks = G.generate_tasks(cfg, seed=seed)
    logger.info(f"Total tasks: {len(tasks)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)
    if args.limit:
        tasks = tasks[:args.limit]

    done = {}
    if os.path.exists(cfg["checkpoint"]):
        existing = pd.read_csv(cfg["checkpoint"])
        if set(phase_cols).issubset(existing.columns):
            for _, r in existing.iterrows():
                key = tuple(float(r[c]) for c in cfg["comps"][1:]) + (
                    float(r["T"]),)
                done[key] = r.to_dict()
            logger.info(f"Resuming: {len(done)} rows already present")

    t0 = time.time()
    rows = list(done.values())
    batch_size = 400
    task_args = [{"cfg": cfg, "names": phases, "cols": phase_cols,
                  "point": t} for t in tasks if t not in done]
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"],)) as pool:
        total = len(task_args)
        for i in range(0, total, batch_size):
            batch = task_args[i:i + batch_size]
            results = pool.map(run_eq, batch)
            ok = [r for r in results if r["converged"]]
            rows.extend(ok)
            df_batch = pd.DataFrame(ok)
            df_batch.to_csv(
                cfg["checkpoint"], mode="a",
                header=(not os.path.exists(cfg["checkpoint"])
                        or os.path.getsize(cfg["checkpoint"]) == 0),
                index=False)
            elapsed = time.time() - t0
            rate = len(rows) / elapsed if elapsed > 0 else 0
            logger.info(f"  {i + len(batch)}/{total} | accepted "
                        f"{len(ok)}/{len(batch)} | total {len(rows)} | "
                        f"{rate:.1f}/s | ETA "
                        f"{(total - i) / max(rate, 0.01):.0f}s")

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=cfg["comps"] + ["T"], keep="first")
    logger.info(f"Dedup (features): {n_before} -> {len(df)}")
    sums = df[phase_cols].sum(axis=1)
    df = df[np.abs(sums - 1.0) < MASS_BALANCE_TOL].copy()
    logger.info(f"Mass-balance filter: {len(df)} rows")

    box = cfg["box"]
    mask = np.ones(len(df), dtype=bool)
    for col, val in box.items():
        if col == "Fe":
            mask &= df[col] >= val
        else:
            mask &= df[col] <= val
    df["in_stainless_box"] = mask.astype(int)
    logger.info(f"Stainless-box rows: "
                f"{100 * df['in_stainless_box'].mean():.1f}%")

    occ = {c: float((df[c] > NONZERO_LEVEL).mean()) for c in phase_cols}
    active_cols = [c for c in phase_cols if occ[c] >= ACTIVE_THRESHOLD]
    sidecar = {"system": sysid, "n_rows": int(len(df)),
               "template": {"seed": seed, "sigma_focus": cfg["sigma_focus"],
                            "box_unchanged": True},
               "eligible_phases": phase_cols,
               "active_phases": active_cols}
    with open(os.path.join(os.path.dirname(cfg["dataset"]),
                           f"{sysid}_active.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    df.to_csv(cfg["dataset"], index=False)
    logger.info(f"Saved {len(df)} samples to {cfg['dataset']}")
    for c in active_cols:
        logger.info(f"  {c}: {100 * occ[c]:.1f}% nonzero, "
                    f"max={df[c].max():.4f}")


if __name__ == "__main__":
    main()
