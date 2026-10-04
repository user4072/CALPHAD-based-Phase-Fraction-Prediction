r"""Ideal-form region-matched band control: removes the ~15% training-size
confound of the published region-matched reanalysis (Table 16).

The published ratio compares the band-holdout model (band rows removed from
training, ~15% fewer rows) against the main-study model on band rows; it also
uses different test sets (all band rows vs main-test band rows). This script
runs the ideal three-way comparison on IDENTICAL test rows per seed
(te intersection band):

  main  : main-study model, band present, full training size N    [stored npz]
  arm I : band present, training size matched to arm H by dropping a random
          non-band subset D of size |tr intersection band|                  [retrained]
  arm H : band absent, training = tr \ band  (size N - |tr intersection band|) [retrained]

Decomposition on the same rows:
  I / main : the training-size effect alone (the old residual confound)
  H / I     : the band-presence effect at fixed training size (the quantity
              the old analysis wanted)
  H / main  : (H/I)(I/main), the old ratio re-expressed on identical rows

Both arms share identical validation rows (va \ band) and identical seeded
initialisation per (system, seed, model), so the only difference is which
training rows they see. Models: MLP renorm, RF+renorm, XGBoost+renorm (the
Table 16 columns), 3 seeds, five ternary systems.

Output: models/region_matched_ideal.json (written incrementally per system).

Usage: py -3.12 analysis_revision/region_matched_ideal.py
       py -3.12 analysis_revision/region_matched_ideal.py --system fecrni
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import torch

from fe_surrogate.config import MODELS_DIR
from fe_surrogate.experiment import (load_data, active_phases, cluster_split,
                                     evaluate, renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS
from holdout_eval import fit_tree, T_LO, T_HI, X2_LO, X2_HI
from train_mlp import MLP4, train_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS3 = ["mlp_renorm", "rf_renorm", "xgb_renorm"]
OUT = os.path.join(MODELS_DIR, "region_matched_ideal.json")


def band_masks_df(df, cfg):
    """Global boolean band masks from the dataset (same cut as holdout_eval)."""
    x2 = cfg["comps"][1]
    t = ((df["T"] >= T_LO) & (df["T"] <= T_HI)).values
    x = ((df[x2] >= X2_LO) & (df[x2] <= X2_HI)).values
    return {"T_band": t, "X2_band": x}


def mean_mae_subset(y_true, y_pred, mask):
    """Per-phase MAE then mean over phases (same convention as metrics())."""
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]), axis=0)
    return float(np.mean(mae)), int(mask.sum())


def main_ref(npz, bm_global):
    """Main-study model MAE on te intersection band from stored predictions."""
    d = np.load(npz)
    m = bm_global[d["te_idx"]]
    return mean_mae_subset(d["y_true"], d["y_pred"], m)


def npz_path(system, model, seed):
    tag = (f"{model}_192x192x192_huber" if model.startswith("mlp")
           else f"{model}_std")
    return os.path.join(MODELS_DIR, f"pred_{system}_{tag}_s{seed}.npz")


def run_system(system, results):
    cfg = SYSTEMS[system]
    X, Y, df = load_data(system)
    _, names = active_phases(system)
    box = df["in_stainless_box"].values.astype(bool)
    bands = band_masks_df(df, cfg)
    print(f"\n{'=' * 70}\n{system}: {len(X)} rows, {len(names)} phases "
          f"{names} | device {DEVICE}\n{'=' * 70}", flush=True)

    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        for band_name, bm in bands.items():
            te_b = te[bm[te]]
            va_keep = va[~bm[va]]
            tr_b = tr[bm[tr]]
            tr_nonband = tr[~bm[tr]]
            n_drop = len(tr_b)
            rng = np.random.default_rng(3000 + seed +
                                        (0 if band_name == "T_band" else 7))
            dropped = rng.choice(tr_nonband, n_drop, replace=False)
            arms = {
                "hold": tr[~bm[tr]],
                "incl": tr[~np.isin(tr, dropped)],
            }
            assert len(arms["hold"]) == len(arms["incl"]) == len(tr) - n_drop
            for model in MODELS3:
                refs = main_ref(npz_path(system, model, seed), bm)
                recs = {}
                for arm, tr_arm in arms.items():
                    t0 = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if model.startswith("mlp"):
                        net = MLP4(n_phases=len(names), head=model[4:],
                                   n_in=X.shape[1]).to(DEVICE)
                        net, _, scaler = train_mlp(net, X, Y, tr_arm, va_keep,
                                                   seed)
                        net.eval()
                        with torch.no_grad():
                            pred = net(torch.tensor(scaler.transform(X[te_b]),
                                                    dtype=torch.float32,
                                                    device=DEVICE)
                                       ).cpu().numpy()
                        pred = renorm(pred)
                    else:
                        pred = renorm(fit_tree(model.split("_")[0], X, Y,
                                               tr_arm, va_keep, te_b, seed))
                    res = evaluate(Y[te_b], pred, box[te_b])
                    recs[arm] = (float(res["mean_mae"]),
                                 int(len(tr_arm)),
                                 round(time.time() - t0, 1))
                hold_mae, n_tr, t_h = recs["hold"]
                incl_mae, n_tr_i, t_i = recs["incl"]
                main_mae, n_te = refs
                key = f"{system}_{band_name}_{model}_s{seed}"
                results[key] = {
                    "system": system, "band": band_name, "model": model,
                    "seed": int(seed),
                    "n_test_band": n_te,
                    "n_train_hold": n_tr, "n_train_incl": n_tr_i,
                    "n_train_main": int(len(tr)),
                    "n_dropped": n_drop,
                    "mae_hold": hold_mae, "mae_incl": incl_mae,
                    "mae_main": main_mae,
                    "ratio_h_i": hold_mae / incl_mae,
                    "ratio_i_main": incl_mae / main_mae,
                    "ratio_h_main": hold_mae / main_mae,
                    "time_s": round(t_h + t_i, 1),
                }
                print(f"  [{key}] H={hold_mae:.4f} I={incl_mae:.4f} "
                      f"main={main_mae:.4f} | H/I={hold_mae / incl_mae:.2f} "
                      f"I/main={incl_mae / main_mae:.2f} "
                      f"| n {n_tr}={n_tr_i}<{len(tr)} te={n_te} "
                      f"| {t_h + t_i:.0f}s", flush=True)


def aggregate(results):
    """Ratio-of-means aggregates per (system, band, model) over records."""
    agg = {}
    keys = sorted({(r["system"], r["band"], r["model"])
                   for r in results.values()})
    for system, band, model in keys:
        rows = [r for r in results.values()
                if r["system"] == system and r["band"] == band
                and r["model"] == model]
        if not rows:
            continue
        h = float(np.mean([r["mae_hold"] for r in rows]))
        i = float(np.mean([r["mae_incl"] for r in rows]))
        m = float(np.mean([r["mae_main"] for r in rows]))
        agg[f"{system}_{band}_{model}"] = {
            "n_seeds": len(rows),
            "mae_hold_mean": h, "mae_incl_mean": i, "mae_main_mean": m,
            "ratio_h_i_of_means": h / i,
            "ratio_i_main_of_means": i / m,
            "ratio_h_main_of_means": h / m,
            "ratio_h_i_mean": float(np.mean([r["ratio_h_i"] for r in rows])),
            "ratio_h_i_std": float(np.std([r["ratio_h_i"] for r in rows])),
            "ratio_i_main_mean": float(np.mean([r["ratio_i_main"] for r in rows])),
            "ratio_i_main_std": float(np.std([r["ratio_i_main"] for r in rows])),
            "n_train_main": rows[0]["n_train_main"],
            "n_train_arm": rows[0]["n_train_hold"],
            "n_dropped_mean": float(np.mean([r["n_dropped"] for r in rows])),
            "n_test_band_mean": float(np.mean([r["n_test_band"] for r in rows])),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), action="append")
    args = ap.parse_args()
    systems = args.system or ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]

    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
        print(f"loaded {len(results)} existing records from {OUT}")

    for system in systems:
        # skip fully-completed systems (resume support)
        have = [k for k, v in results.items()
                if isinstance(v, dict) and v.get("system") == system]
        expected = len(SEEDS) * 2 * len(MODELS3)
        if len(have) == expected:
            print(f"{system}: {expected} records already present, skipping")
            continue
        t0 = time.time()
        run_system(system, results)
        results["__meta__"] = {"note": "ideal-form region-matched control",
                               "protocol": "te∩band test; va∩band val trimmed "
                                           "in both arms; I drops |tr∩band| "
                                           "random non-band rows; seeded init "
                                           "before every fit"}
        results["__agg__"] = aggregate(
            {k: v for k, v in results.items()
             if isinstance(v, dict) and "system" in v and "band" in v})
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {system} done in {(time.time() - t0) / 60:.1f} min "
              f"(saved {OUT})", flush=True)

    print("\n== aggregates (ratio of means) ==")
    for k, v in sorted(results.get("__agg__", {}).items()):
        print(f"  {k:24s} H/I={v['ratio_h_i_of_means']:.2f}x "
              f"I/main={v['ratio_i_main_of_means']:.2f}x "
              f"H/main={v['ratio_h_main_of_means']:.2f}x "
              f"(n {v['n_train_main']}->{v['n_train_arm']})")


if __name__ == "__main__":
    main()
