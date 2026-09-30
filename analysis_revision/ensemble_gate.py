"""Ensemble gate + uncertainty/OOD signals from stored predictions.

Per system:
  (a) 3-seed ensemble of mlp_sigmoid (after projection) and of the stored
      mlp_renorm predictions -> ensemble MAE vs mean single-seed MAE (% gain).
      Test splits differ per seed, so results are reported on the union of
      test rows (partial ensembles: mean over the seeds covering a row) and on
      the rows covered by all 3 seeds (fully paired).
  (b) U1 = per-row std across the 3 seeds (mean over phases), rows covered by
      >= 2 seeds. Spearman(U1, per-row MAE), AUROC for row MAE > 0.05, and the
      coverage-risk curve (retain the lowest-uncertainty rows).
  (c) U2 = nearest-neighbour distance from each test row to the TRAINING rows
      of the same seed (features = comps + T, each column divided by its
      dataset std, scipy cKDTree). Same three metrics as (b).
  (d) Band masks on ALL dataset rows: T in [1200,1400] and comps[1] in
      [0.25,0.35]; mean U1 / U2 / test MAE inside vs outside (test rows only).

Output: models/ensemble_gate.json

Usage: py -3.12 -X utf8 analysis_revision/ensemble_gate.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fe_surrogate.systems import SYSTEMS

MODELS = os.path.join(HERE, "models")
OUT = os.path.join(MODELS, "ensemble_gate.json")

SYSTEM_LIST = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
SEEDS = [42, 123, 2024]
SIG_TAG = "mlp_sigmoid_192x192x192_huber"
REN_TAG = "mlp_renorm_192x192x192_huber"
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ERR_THR = 0.05
T_LO, T_HI = 1200.0, 1400.0
X2_LO, X2_HI = 0.25, 0.35


def f(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def project(pred):
    """clip to [0,1], divide by row sum (zero rows guarded) == experiment.renorm."""
    p = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    return p / np.where(s > 1e-12, s, 1.0)


def row_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=-1)


def signal_metrics(scores, errors):
    """Spearman corr, AUROC (error > threshold), coverage-risk curve."""
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    ok = np.isfinite(scores) & np.isfinite(errors)
    scores, errors = scores[ok], errors[ok]
    out = {"n": int(ok.sum())}
    if len(scores) < 3:
        out.update({"spearman": None, "auroc": None, "coverage_risk": {}})
        return out

    rho, pval = spearmanr(scores, errors)
    out["spearman"] = f(rho)
    out["spearman_p"] = f(pval)

    labels = (errors > ERR_THR).astype(int)
    if labels.min() == labels.max():
        out["auroc"] = None
        out["n_pos"] = int(labels.sum())
        out["n_neg"] = int((1 - labels).sum())
    else:
        out["auroc"] = f(roc_auc_score(labels, scores))
        out["n_pos"] = int(labels.sum())
        out["n_neg"] = int((1 - labels).sum())
        out["pos_rate"] = f(labels.mean())

    order = np.argsort(scores, kind="mergesort")  # ascending uncertainty
    n = len(scores)
    cr = {}
    for c in COVERAGES:
        k = int(round(c * n))
        k = max(k, 1)
        cr[f"{c:.1f}"] = f(errors[order[:k]].mean())
    out["coverage_risk"] = cr
    out["coverage_risk_note"] = ("retain the lowest-uncertainty rows "
                                 "(ascending score) at each coverage")
    return out


def load_system(system):
    per = {}
    for seed in SEEDS:
        ps = os.path.join(MODELS, f"pred_{system}_{SIG_TAG}_s{seed}.npz")
        pr = os.path.join(MODELS, f"pred_{system}_{REN_TAG}_s{seed}.npz")
        if not (os.path.exists(ps) and os.path.exists(pr)):
            raise FileNotFoundError(f"missing {ps} or {pr}")
        ds, dr = np.load(ps), np.load(pr)
        te = np.asarray(ds["te_idx"], dtype=np.int64)
        y_true = np.asarray(ds["y_true"], dtype=np.float64)
        if not np.array_equal(te, np.asarray(dr["te_idx"], dtype=np.int64)):
            raise ValueError(f"{system} s{seed}: sigmoid/renorm test rows differ")
        per[seed] = {
            "te_idx": te,
            "y_true": y_true,
            "y_sig": project(ds["y_pred"]),          # projected sigmoid
            "y_sig_raw": np.asarray(ds["y_pred"], dtype=np.float64),
            "y_ren": np.asarray(dr["y_pred"], dtype=np.float64),
            "features": np.asarray(ds["features"], dtype=np.float64),
        }
    return per


def align(per):
    """Union of test rows across seeds -> (n_union,) row ids + (S,n,k) tensors."""
    all_rows = np.unique(np.concatenate([per[s]["te_idx"] for s in SEEDS]))
    k = per[SEEDS[0]]["y_true"].shape[1]
    S = len(SEEDS)
    n = len(all_rows)
    sig = np.full((S, n, k), np.nan)
    ren = np.full((S, n, k), np.nan)
    true = np.full((n, k), np.nan)
    ycons = 0.0
    for si, seed in enumerate(SEEDS):
        idx = np.searchsorted(all_rows, per[seed]["te_idx"])
        sig[si, idx] = per[seed]["y_sig"]
        ren[si, idx] = per[seed]["y_ren"]
        if np.isnan(true[idx]).any():
            true[idx] = per[seed]["y_true"]
        else:
            ycons = max(ycons, float(np.abs(true[idx] - per[seed]["y_true"]).max()))
    present = ~np.isnan(sig[:, :, 0])
    count = present.sum(axis=0)
    mae_sig = np.where(present, row_mae(true[None, :, :], np.nan_to_num(sig)), np.nan)
    mae_ren = np.where(present, row_mae(true[None, :, :], np.nan_to_num(ren)), np.nan)
    return all_rows, sig, ren, true, present, count, mae_sig, mae_ren, ycons


def ensemble_block(sig, ren, true, present, count, mae_sig, mae_ren):
    """Ensemble MAE vs mean single-seed MAE for both prediction variants."""
    out = {}
    union = count >= 1
    common = count == len(SEEDS)
    for name, arr, mae_arr in (("mlp_sigmoid_projected", sig, mae_sig),
                               ("mlp_renorm_stored", ren, mae_ren)):
        with np.errstate(invalid="ignore"):
            ens = np.nanmean(arr, axis=0)
        ens_mae_row = row_mae(true, ens)
        ens_union = float(np.mean(ens_mae_row[union]))
        ens_common = float(np.mean(ens_mae_row[common])) if common.any() else None
        per_seed = [float(np.nanmean(mae_arr[si])) for si in range(len(SEEDS))]
        pooled = float(np.nanmean(mae_arr))
        mean_single = float(np.mean(per_seed))
        single_common = [float(np.nanmean(mae_arr[si][common]))
                         for si in range(len(SEEDS))] if common.any() else []
        mean_single_common = float(np.mean(single_common)) if single_common else None
        out[name] = {
            "single_seed_mae": [f(v) for v in per_seed],
            "mean_single_seed_mae": f(mean_single),
            "pooled_single_mae_union_rows": f(pooled),
            "ensemble_mae_union_rows": f(ens_union),
            "gain_pct_vs_mean_single": f(100.0 * (mean_single - ens_union) / mean_single),
            "gain_pct_vs_pooled": f(100.0 * (pooled - ens_union) / pooled),
            "n_union_rows": int(union.sum()),
            "n_common_rows": int(common.sum()),
            "single_seed_mae_common_rows": [f(v) for v in single_common],
            "mean_single_seed_mae_common_rows": f(mean_single_common),
            "ensemble_mae_common_rows": f(ens_common),
            "gain_pct_vs_mean_single_common_rows": (
                f(100.0 * (mean_single_common - ens_common) / mean_single_common)
                if (ens_common is not None and mean_single_common) else None),
        }
    return out


def uncertainty_u1(sig, ren, true, count, mae_sig, mae_ren):
    """Per-row std across seeds (mean over phases); rows with >= 2 seeds."""
    rows = count >= 2
    out = {"n_rows": int(rows.sum()), "min_seeds_per_row": 2}
    if not rows.any():
        return {"U1": {"n": 0}, "U1_renorm": {"n": 0}}
    idx = np.where(rows)[0]
    for name, arr, mae_arr in (("U1", sig, mae_sig), ("U1_renorm", ren, mae_ren)):
        sub = arr[:, idx, :]
        with np.errstate(invalid="ignore"):
            u1 = np.nanstd(sub, axis=0).mean(axis=1)
            err = np.nanmean(mae_arr[:, idx], axis=0)
        out[name] = signal_metrics(u1, err)
    return out


def uncertainty_u2(system, per, mae_by_seed):
    """NN distance from each test row to that seed's training rows."""
    cfg = SYSTEMS[system]
    df = pd.read_csv(cfg["dataset"])
    X = df[cfg["comps"] + ["T"]].values.astype(np.float64)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = X / std
    scores, errors, rows = [], [], []
    feat_check = 0.0
    train_frac = []
    for si, seed in enumerate(SEEDS):
        te = per[seed]["te_idx"]
        tr_mask = np.ones(len(X), dtype=bool)
        tr_mask[te] = False
        tree = cKDTree(Xs[tr_mask])
        d, _ = tree.query(Xs[te], k=1)
        scores.append(d)
        errors.append(mae_by_seed[si])
        rows.append(te)
        train_frac.append(f(tr_mask.mean()))
        feat_check = max(feat_check,
                         float(np.abs(X[te] - per[seed]["features"]).max()))
    scores = np.concatenate(scores)
    errors = np.concatenate(errors)
    rows = np.concatenate(rows)
    out = signal_metrics(scores, errors)
    out["n_pairs"] = int(len(scores))
    out["mean_distance"] = f(scores.mean())
    out["max_distance"] = f(scores.max())
    out["feature_reconstruction_max_abs_diff"] = f(feat_check)
    out["train_fraction_per_seed"] = [f(v) for v in train_frac]
    return out, rows, scores, errors


def bands(system, all_rows, u1_rows, u1_vals, pair_rows, pair_u2, pair_mae):
    cfg = SYSTEMS[system]
    df = pd.read_csv(cfg["dataset"])
    t_all = (df["T"].values >= T_LO) & (df["T"].values <= T_HI)
    x_all = (df[cfg["comps"][1]].values >= X2_LO) & (df[cfg["comps"][1]].values <= X2_HI)
    masks = {"T_band": t_all, "X2_band": x_all}
    out = {}
    for name, mask_all in masks.items():
        u1_in = mask_all[u1_rows]
        p_in = mask_all[pair_rows]
        blk = {
            "n_all_rows_in": int(mask_all.sum()),
            "n_all_rows_out": int((~mask_all).sum()),
            "n_test_rows_u1": {"in": int(u1_in.sum()), "out": int((~u1_in).sum())},
            "n_test_pairs": {"in": int(p_in.sum()), "out": int((~p_in).sum())},
            "U1_mean_in": f(np.mean(u1_vals[u1_in])) if u1_in.any() else None,
            "U1_mean_out": f(np.mean(u1_vals[~u1_in])) if (~u1_in).any() else None,
            "U2_mean_in": f(np.mean(pair_u2[p_in])) if p_in.any() else None,
            "U2_mean_out": f(np.mean(pair_u2[~p_in])) if (~p_in).any() else None,
            "test_mae_mean_in": f(np.mean(pair_mae[p_in])) if p_in.any() else None,
            "test_mae_mean_out": f(np.mean(pair_mae[~p_in])) if (~p_in).any() else None,
        }
        blk["U2_ratio_in_over_out"] = (
            f(blk["U2_mean_in"] / blk["U2_mean_out"])
            if blk["U2_mean_in"] and blk["U2_mean_out"] else None)
        blk["mae_ratio_in_over_out"] = (
            f(blk["test_mae_mean_in"] / blk["test_mae_mean_out"])
            if blk["test_mae_mean_in"] and blk["test_mae_mean_out"] else None)
        out[name] = blk
    return out


def analyse_system(system):
    per = load_system(system)
    all_rows, sig, ren, true, present, count, mae_sig, mae_ren, ycons = align(per)
    res = {
        "n_dataset_rows": int(len(pd.read_csv(SYSTEMS[system]["dataset"]))),
        "n_union_test_rows": int(len(all_rows)),
        "seed_test_sizes": {str(s): int(len(per[s]["te_idx"])) for s in SEEDS},
        "coverage_hist": {str(int(c)): int((count == c).sum())
                          for c in np.unique(count)},
        "y_true_consistency_max_abs_diff": f(ycons),
        "ensemble": ensemble_block(sig, ren, true, present, count, mae_sig, mae_ren),
        "uncertainty_u1": uncertainty_u1(sig, ren, true, count, mae_sig, mae_ren),
    }
    mae_by_seed = [row_mae(per[s]["y_true"], per[s]["y_sig"]) for s in SEEDS]
    u2, pair_rows, pair_u2, pair_mae = uncertainty_u2(system, per, mae_by_seed)
    res["uncertainty_u2"] = u2

    u1_rows = all_rows[count >= 2]
    with np.errstate(invalid="ignore"):
        u1_vals = np.nanstd(sig[:, count >= 2, :], axis=0).mean(axis=1)
    res["bands"] = bands(system, all_rows, u1_rows, u1_vals,
                         pair_rows, pair_u2, pair_mae)
    res["bands"]["_u1_rows_basis"] = "rows covered by >=2 seeds"
    res["bands"]["_pair_basis"] = "(row, seed) test pairs"
    res["bands"]["_test_mae_model"] = "mlp_sigmoid after projection"
    return res


def main():
    results = {}
    for system in SYSTEM_LIST:
        results[system] = analyse_system(system)

    # cross-system averages of the signal metrics
    def avg(path_getter):
        vals = [path_getter(results[s]) for s in SYSTEM_LIST]
        vals = [v for v in vals if v is not None]
        return f(np.mean(vals)) if vals else None

    cross = {
        "U1_spearman_mean": avg(lambda r: r["uncertainty_u1"]["U1"].get("spearman")),
        "U1_auroc_mean": avg(lambda r: r["uncertainty_u1"]["U1"].get("auroc")),
        "U1_renorm_spearman_mean": avg(lambda r: r["uncertainty_u1"]["U1_renorm"].get("spearman")),
        "U1_renorm_auroc_mean": avg(lambda r: r["uncertainty_u1"]["U1_renorm"].get("auroc")),
        "U2_spearman_mean": avg(lambda r: r["uncertainty_u2"].get("spearman")),
        "U2_auroc_mean": avg(lambda r: r["uncertainty_u2"].get("auroc")),
        "ensemble_gain_pct_sigmoid_mean": avg(
            lambda r: r["ensemble"]["mlp_sigmoid_projected"]["gain_pct_vs_mean_single"]),
        "ensemble_gain_pct_renorm_mean": avg(
            lambda r: r["ensemble"]["mlp_renorm_stored"]["gain_pct_vs_mean_single"]),
        "ensemble_gain_pct_sigmoid_common_mean": avg(
            lambda r: r["ensemble"]["mlp_sigmoid_projected"]["gain_pct_vs_mean_single_common_rows"]),
    }
    for band in ("T_band", "X2_band"):
        for key in ("U1_mean_in", "U1_mean_out", "U2_mean_in", "U2_mean_out",
                    "test_mae_mean_in", "test_mae_mean_out", "U2_ratio_in_over_out",
                    "mae_ratio_in_over_out"):
            cross[f"{band}_{key}"] = avg(lambda r, b=band, k=key: r["bands"][b].get(k))

    payload = {
        "meta": {
            "script": "analysis_revision/ensemble_gate.py",
            "systems": SYSTEM_LIST,
            "seeds": SEEDS,
            "ensemble_members": {"mlp_sigmoid": SIG_TAG + " (projected)",
                                 "mlp_renorm": REN_TAG + " (stored)"},
            "alignment": "test splits differ across seeds; union rows use the "
                         "mean over the seeds that cover the row, common rows "
                         "are covered by all 3 seeds",
            "U1": "per-row std across seeds, mean over phases (mlp_sigmoid "
                  "after projection; _renorm variant from stored mlp_renorm)",
            "U2": "cKDTree NN distance to the same seed's training rows; "
                  "features = comps + T divided by the dataset column std",
            "error_threshold": ERR_THR,
            "coverages": COVERAGES,
            "bands": {"T_band": [T_LO, T_HI], "X2_band": [X2_LO, X2_HI],
                      "X2": "second composition column (comps[1])"},
        },
        "per_system": results,
        "cross_system_mean": cross,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {OUT}")

    # compact per-system summary table
    def _n(v):
        return float("nan") if v is None else float(v)

    print("\n=== per-system summary ===")
    print(f"{'system':8s} {'ens%sig':>8s} {'ens%ren':>8s} {'nCom':>5s} "
          f"{'rho_U1':>7s} {'AU_U1':>6s} {'rho_U2':>7s} {'AU_U2':>6s} "
          f"{'U2_in':>7s} {'U2_out':>7s} {'mae_in':>8s} {'mae_out':>8s}")
    for system in SYSTEM_LIST:
        r = results[system]
        e = r["ensemble"]
        u1 = r["uncertainty_u1"]["U1"]
        u2 = r["uncertainty_u2"]
        tb = r["bands"]["T_band"]
        print(f"{system:8s} "
              f"{_n(e['mlp_sigmoid_projected']['gain_pct_vs_mean_single']):8.2f} "
              f"{_n(e['mlp_renorm_stored']['gain_pct_vs_mean_single']):8.2f} "
              f"{e['mlp_sigmoid_projected']['n_common_rows']:5d} "
              f"{_n(u1.get('spearman')):7.3f} {_n(u1.get('auroc')):6.3f} "
              f"{_n(u2.get('spearman')):7.3f} {_n(u2.get('auroc')):6.3f} "
              f"{_n(tb['U2_mean_in']):7.4f} {_n(tb['U2_mean_out']):7.4f} "
              f"{_n(tb['test_mae_mean_in']):8.5f} {_n(tb['test_mae_mean_out']):8.5f}")
    print("\n=== cross-system means ===")
    for k, v in cross.items():
        print(f"  {k:42s} {v}")


if __name__ == "__main__":
    main()
