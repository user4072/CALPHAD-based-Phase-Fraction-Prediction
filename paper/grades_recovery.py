"""N3': commercial-grade recovery check (Fe-Cr-Ni projection).

Tests whether the lean-Ni screen rediscovers real stainless grades:
  304    Fe-19Cr-9Ni   (wt% nominal, minors folded into Fe) -> expect HIT
  430    Fe-17Cr-0.3Ni (ferritic; minors folded into Fe)   -> expect REJECT
  duplex S31803 composition projected onto Fe-Cr-Ni (Mo/N/C dropped,
    stated) -> out-of-scope control: the projection is unfaithful by
    construction; whatever the numbers say is reported as such.

Per grade: wt% -> mole fractions (Fe 55.845, Cr 51.996, Ni 58.693),
renormalize modeled Fe+Cr+Ni, surrogate 3-seed renorm ensemble at
1000 K + shortlist membership under the ternary query, full-set
fecrni CALPHAD (28 phases) at the projected point.

Output (NEW): paper/screen_data/grades.json

Usage: py -3.12 paper/grades_recovery.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from fe_surrogate.experiment import SEEDS, renorm
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from train_mlp import MLP4
from anchor_eval import _cal_init, _cal_run
import multiprocessing as mp

AW = {"Fe": 55.845, "Cr": 51.996, "Ni": 58.693}
T_K = 1000.0
CKPT_DIR = os.path.join(HERE, "anchor_data", "anchor_ckpts")
Q_FCC_MIN, Q_SIGMA_MAX, Q_LIQ_MAX, Q_NI_MAX = 0.99, 1e-3, 1e-3, 0.12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRADES = {
    # (Cr wt%, Ni wt%) nominal; Fe = balance (minors folded in, stated)
    "AISI 304": (19.0, 9.0, "expect HIT (fully austenitic)"),
    "AISI 430": (17.0, 0.3, "expect REJECT (ferritic)"),
    # S31803 minus Mo/N/C (projection unfaithful by construction)
    "S31803-projected": (22.07, 5.68, "out-of-scope control (Mo dropped)"),
}


def wt_to_mol_fecrni(cr_wt, ni_wt):
    fe_wt = 100.0 - cr_wt - ni_wt
    mol = {e: w / AW[e] for e, w in
           [("Fe", fe_wt), ("Cr", cr_wt), ("Ni", ni_wt)]}
    tot = sum(mol.values())
    return {e: v / tot for e, v in mol.items()}


def load_bundle(seed):
    ck = torch.load(
        os.path.join(CKPT_DIR, f"fecrni_renorm_s{seed}.pt"),
        map_location=DEVICE, weights_only=False)
    model = MLP4(n_phases=ck["n_phases"], head=ck["model_head"],
                 hidden=tuple(ck["hidden"]), n_in=ck["n_in"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return (model, ck["scaler_mean"].cpu().numpy(),
            ck["scaler_scale"].cpu().numpy(), ck["phase_names"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    bundles = [load_bundle(s) for s in SEEDS]
    names = bundles[0][3]
    i_fcc = names.index("FCC_A1")
    i_sig = names.index("SIGMA")
    i_liq = names.index("LIQUID")

    cfg = SYSTEMS["fecrni"]
    phases = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    pool = mp.Pool(args.workers, initializer=_cal_init,
                   initargs=(cfg["tdb"], cfg["elements"],
                             cfg["comps_species"], phases))
    out = {"T_K": T_K, "grades": {}}
    try:
        for grade, (cr_wt, ni_wt, expect) in GRADES.items():
            mole = wt_to_mol_fecrni(cr_wt, ni_wt)
            x = np.array([[mole["Fe"], mole["Cr"], mole["Ni"], T_K]])
            preds = []
            for model, mu, sd, _ in bundles:
                with torch.no_grad():
                    p = model(torch.tensor(
                        (x - mu) / sd, dtype=torch.float32,
                        device=DEVICE)).cpu().numpy()
                preds.append(renorm(p)[0])
            ens = np.mean(preds, axis=0)
            hit = bool((ens[i_fcc] >= Q_FCC_MIN)
                       and (ens[i_sig] <= Q_SIGMA_MAX)
                       and (ens[i_liq] <= Q_LIQ_MAX)
                       and (mole["Ni"] <= Q_NI_MAX))
            r = pool.map(_cal_run,
                         [((mole["Cr"], mole["Ni"]), T_K)])[0]
            cal = {p: (r["amounts"].get(p, 0.0) if r["ok"] else None)
                   for p in ("FCC_A1", "SIGMA", "LIQUID", "BCC_A2")}
            out["grades"][grade] = {
                "wt_pct": {"Cr": cr_wt, "Ni": ni_wt},
                "mole_frac": {k: round(v, 4) for k, v in mole.items()},
                "expectation": expect,
                "surrogate_FCC": round(float(ens[i_fcc]), 4),
                "surrogate_SIGMA": float(ens[i_sig]),
                "shortlist_hit": hit,
                "calphad_ok": bool(r["ok"]),
                "calphad": {k: (round(v, 4) if v is not None else None)
                            for k, v in cal.items()},
            }
            print(f"[{grade}] mole={mole} surrFCC={ens[i_fcc]:.4f} "
                  f"hit={hit} cal={cal} ok={r['ok']}", flush=True)
    finally:
        pool.close()
    with open(os.path.join(HERE, "screen_data", "grades.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote paper/screen_data/grades.json")


if __name__ == "__main__":
    main()
