"""Revision analyses computed from stored predictions and datasets (no retraining).

Answers the reviewer requests that can be addressed from stored artefacts:
  1. Region-matched band controls: MAE on band rows inside the main
     interpolation test set (band data in training) vs the band-holdout MAE
     (band data removed) -- same region, so the ratio isolates the regional
     shift far better than the random-control ratio.
  2. Seen/unseen phase structure of the two extrapolation protocols +
     per-phase far-side MAE and the constant near-side-mean predictor.
  3. Gibbs-phase-rule violations (>3 phases above threshold) in stored
     predictions vs ground truth.
  4. Active-cells-only MAE and tail statistics (P95) on the main test set.
  5. Macro-F1 excluding zero-positive phases + per-phase AUPRC
     (average precision) for rare phases.
  6. F1 under the two spatial band holdouts for all five systems.
  7. Fe-Cr-V random-control per-seed audit.
  8. Sparsemax per-seed spread (variance claim check).
  9. renorm == sigmoid + post-hoc projection check (identical weights).
 10. Rejected-candidate reconstruction: rebuild the deterministic design,
     diff against the dataset, locate the solver rejections.
 11. MLP forward-pass latency (fresh 192x3x192 architecture; latency is
     weight-independent) vs the CALPHAD 2.5 s/point reference.

Usage: py -3.12 -X utf8 analysis_revision/revision_analyses.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd

from fe_surrogate.experiment import (load_data, active_phases, evaluate,
                                     phase_classification, renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS

MODELS = os.path.join(HERE, "models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "revision_analyses.json")
T_LO, T_HI = 1200.0, 1400.0
X2_LO, X2_HI = 0.25, 0.35
X2_TRAIN_HI, X2_TEST_LO = 0.25, 0.35
T_TRAIN_HI, T_TEST_LO = 1200.0, 1400.0
HOLDOUT_MODELS = ["mlp_renorm", "mlp_sig_norm", "rf_renorm", "xgb_renorm"]
CALPHAD_S_PER_POINT = 2.5

R = {}


def npz_path(system, tag, seed):
    return os.path.join(MODELS, f"pred_{system}_{tag}_s{seed}.npz")


def load_npz(system, tag, seed):
    return np.load(npz_path(system, tag, seed))


def band_masks(feat):
    """feat columns: [Fe, x2, x3, T]; band variable x2 = comps[1] = col 1."""
    t_band = (feat[:, 3] >= T_LO) & (feat[:, 3] <= T_HI)
    x_band = (feat[:, 1] >= X2_LO) & (feat[:, 1] <= X2_HI)
    return t_band, x_band


def mean_mae_subset(y_true, y_pred, mask):
    """Same convention as metrics(): per-phase MAE then mean over phases."""
    if mask.sum() == 0:
        return float("nan"), 0
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]), axis=0)
    return float(np.mean(mae)), int(mask.sum())


# ----------------------------------------------------------------------
# 1. Region-matched band controls
# ----------------------------------------------------------------------
def region_matched_bands():
    print("== 1. Region-matched band controls ==")
    out = {}
    for system in sorted(SYSTEMS):
        blk = json.load(open(os.path.join(MODELS, f"block_holdout_{system}.json")))
        rows = {}
        for model in HOLDOUT_MODELS:
            for band, sel in [("T_band", "t"), ("X2_band", "x")]:
                per_seed = []
                for seed in SEEDS:
                    d = load_npz(system, f"{model}_192x192x192_huber"
                                 if model.startswith("mlp") else f"{model}_std", seed)
                    t_band, x_band = band_masks(d["features"])
                    m = t_band if sel == "t" else x_band
                    in_dist, n_in = mean_mae_subset(d["y_true"], d["y_pred"], m)
                    hold = blk[f"{band}_{model}_s{seed}"]["mean_mae"]
                    per_seed.append({
                        "seed": int(seed), "n_band_test_rows": n_in,
                        "band_in_distribution_mae": in_dist,
                        "band_holdout_mae": hold,
                        "region_matched_ratio": hold / in_dist,
                        "random_ctrl_mae": blk[f"random_ctrl_{model}_s{seed}"]["mean_mae"],
                        "random_ctrl_ratio": hold / blk[f"random_ctrl_{model}_s{seed}"]["mean_mae"],
                    })
                ms = [p["region_matched_ratio"] for p in per_seed]
                rc = [p["random_ctrl_ratio"] for p in per_seed]
                hold_mean = float(np.mean([p["band_holdout_mae"] for p in per_seed]))
                ind_mean = float(np.mean([p["band_in_distribution_mae"] for p in per_seed]))
                rows[f"{band}_{model}"] = {
                    "per_seed": per_seed,
                    "band_in_distribution_mae_mean": ind_mean,
                    "band_holdout_mae_mean": hold_mean,
                    "region_matched_ratio_of_means": hold_mean / ind_mean,
                    "region_matched_ratio_mean": float(np.mean(ms)),
                    "region_matched_ratio_std": float(np.std(ms)),
                    "random_ctrl_ratio_mean": float(np.mean(rc)),
                    "random_ctrl_ratio_std": float(np.std(rc)),
                }
        out[system] = rows
        for k, v in rows.items():
            print(f"  {system:7s} {k:22s} region-matched(ratio-of-means) {v['region_matched_ratio_of_means']:.2f}x"
                  f" (was random-ctrl {v['random_ctrl_ratio_mean']:.2f}x)")
    R["region_matched_bands"] = out


# ----------------------------------------------------------------------
# 2. Seen/unseen structure of the extrapolation protocols
# ----------------------------------------------------------------------
def extrap_seen_unseen():
    print("== 2. Extrapolation seen/unseen structure ==")
    out = {}
    for system in sorted(SYSTEMS):
        cfg = SYSTEMS[system]
        X, Y, df = load_data(system)
        _, names = active_phases(system)
        x2 = cfg["comps"][1]
        res = {"phases": names}

        for tag, near_m, far_m in [
                ("T_extrap", (df["T"] <= T_TRAIN_HI).values, (df["T"] > T_TEST_LO).values),
                ("X2_extrap", (df[x2] <= X2_TRAIN_HI).values, (df[x2] > X2_TEST_LO).values)]:
            near_pres = (Y[near_m] > 1e-3).mean(axis=0)
            far_pres = (Y[far_m] > 1e-3).mean(axis=0)
            far_mean = Y[far_m].mean(axis=0)
            # Constant near-side-mean predictor evaluated on the far side
            near_mean = Y[near_m].mean(axis=0)
            const_mae = np.mean(np.abs(Y[far_m] - near_mean[None, :]).mean(axis=0))
            res[tag] = {
                "n_near": int(near_m.sum()), "n_far": int(far_m.sum()),
                "near_presence": {n: float(p) for n, p in zip(names, near_pres)},
                "far_presence": {n: float(p) for n, p in zip(names, far_pres)},
                "far_mean_fraction": {n: float(p) for n, p in zip(names, far_mean)},
                "unseen_phases": [n for n, p in zip(names, near_pres) if p == 0.0
                                  and far_pres[names.index(n)] > 0.0],
                "constant_near_mean_far_mae": float(const_mae),
            }
            # per-phase far-side MAE from the stored extrapolation JSONs
            ej = json.load(open(os.path.join(MODELS, f"holdout_extrap_{system}.json")))
            per_model = {}
            for model in HOLDOUT_MODELS:
                maes = []
                for seed in SEEDS:
                    k = f"{tag}_{model}_s{seed}"
                    maes.append(ej[k]["mae"])
                per_model[model] = {
                    "far_mean_mae": float(np.mean([ej[f"{tag}_{model}_s{s}"]["mean_mae"]
                                                   for s in SEEDS])),
                    "per_phase_mae": {n: float(np.mean([m[i] for m in maes]))
                                      for i, n in enumerate(names)},
                }
            res[tag]["models"] = per_model
        out[system] = res
        for tag in ("T_extrap", "X2_extrap"):
            t = res[tag]
            print(f"  {system:7s} {tag:10s} unseen={t['unseen_phases']} "
                  f"const-near-mean-MAE={t['constant_near_mean_far_mae']:.4f} "
                  f"vs model MAEs " + ", ".join(
                      f"{m}={v['far_mean_mae']:.4f}" for m, v in t["models"].items()))
    R["extrap_seen_unseen"] = out


# ----------------------------------------------------------------------
# 3. Gibbs-phase-rule violations (>3 phases above threshold)
# ----------------------------------------------------------------------
def gibbs_violations():
    print("== 3. Gibbs-phase-rule violations (>3 phases above threshold) ==")
    out = {}
    for system in sorted(SYSTEMS):
        _, names = active_phases(system)
        tags = ["mlp_renorm_192x192x192_huber", "mlp_sigmoid_192x192x192_huber",
                "mlp_softmax_192x192x192_huber", "mlp_sparsemax_192x192x192_huber",
                "rf_raw_std", "rf_renorm_std", "xgb_renorm_std", "knn_renorm_std"]
        rows = {}
        for thr in (1e-3, 5e-3):
            gt = []
            for seed in SEEDS:
                d = load_npz(system, tags[0], seed)
                gt.append(float(((d["y_true"] > thr).sum(axis=1) > 3).mean()))
            rows["ground_truth"] = {"gt": float(np.mean(gt))}
            for tag in tags:
                if not os.path.exists(npz_path(system, tag, 42)):
                    continue
                vals = []
                for seed in SEEDS:
                    if not os.path.exists(npz_path(system, tag, seed)):
                        continue
                    d = load_npz(system, tag, seed)
                    vals.append(float(((d["y_pred"] > thr).sum(axis=1) > 3).mean()))
                if vals:
                    rows[tag] = {"frac": float(np.mean(vals))}
            out[f"{system}_thr{thr:g}"] = rows
        print(f"  {system}: gt(1e-3)={out[f'{system}_thr0.001']['ground_truth']['gt']:.4f} "
              f"mlp_renorm={out[f'{system}_thr0.001'].get('mlp_renorm_192x192x192_huber', {}).get('frac', float('nan')):.4f} "
              f"rf_renorm={out[f'{system}_thr0.001'].get('rf_renorm_std', {}).get('frac', float('nan')):.4f}")
    R["gibbs_violations"] = out


# ----------------------------------------------------------------------
# 4. Active-cells MAE and tail statistics (main test set)
# ----------------------------------------------------------------------
def active_cells_and_tail():
    print("== 4. Active-cells MAE and tail statistics ==")
    out = {}
    for system in sorted(SYSTEMS):
        rows = {}
        for tag in ["mlp_renorm_192x192x192_huber", "rf_renorm_std", "rf_raw_std"]:
            acc, p95c, p95r, mxa, allm = [], [], [], [], []
            for seed in SEEDS:
                d = load_npz(system, tag, seed)
                err = np.abs(d["y_pred"] - d["y_true"])
                act = d["y_true"] > 5e-3
                acc.append(float(err[act].mean()))
                p95c.append(float(np.percentile(err, 95)))
                p95r.append(float(np.percentile(err.mean(axis=1), 95)))
                mxa.append(float(err.max()))
                allm.append(float(err.mean()))
            rows[tag] = {
                "all_cell_mae": float(np.mean(allm)),
                "active_cell_mae": float(np.mean(acc)),
                "p95_cell_abs_err": float(np.mean(p95c)),
                "p95_row_mean_err": float(np.mean(p95r)),
                "max_cell_abs_err": float(np.mean(mxa)),
            }
        out[system] = rows
        r = rows["mlp_renorm_192x192x192_huber"]
        f = rows["rf_renorm_std"]
        print(f"  {system}: MLP all={r['all_cell_mae']:.4f} active={r['active_cell_mae']:.4f} "
              f"P95={r['p95_cell_abs_err']:.4f} | RF all={f['all_cell_mae']:.4f} "
              f"active={f['active_cell_mae']:.4f} P95={f['p95_cell_abs_err']:.4f}")
    R["active_cells_and_tail"] = out


# ----------------------------------------------------------------------
# 5. Macro-F1 excluding zero-positive phases + AUPRC for rare phases
# ----------------------------------------------------------------------
def detection_recomputed():
    print("== 5. Macro-F1 (zero-positive phases excluded) + AUPRC ==")
    from sklearn.metrics import average_precision_score
    out = {}
    for system in sorted(SYSTEMS):
        _, names = active_phases(system)
        rows = {}
        for tag in ["mlp_renorm_192x192x192_huber", "mlp_sig_norm_192x192x192_huber",
                    "rf_renorm_std", "xgb_renorm_std", "knn_renorm_std"]:
            if not os.path.exists(npz_path(system, tag, 42)):
                continue
            f1_incl, f1_excl, zero_pos, ap = [], [], [], {n: [] for n in names}
            for seed in SEEDS:
                if not os.path.exists(npz_path(system, tag, seed)):
                    continue
                d = load_npz(system, tag, seed)
                y_t, y_p = d["y_true"], d["y_pred"]
                f1, _ = phase_classification(y_t, y_p)
                pos = y_t.sum(axis=0) > 0  # phases with >=1 positive row
                zero_pos.append([names[i] for i in range(len(names)) if not pos[i]])
                f1_incl.append(float(f1.mean()))
                if pos.sum() > 0:
                    f1_excl.append(float(f1[pos].mean()))
                for i, n in enumerate(names):
                    if pos[i]:
                        ap[n].append(float(average_precision_score(
                            (y_t[:, i] > 1e-3).astype(int), y_p[:, i])))
            rows[tag] = {
                "macro_f1_with_zero_pos": float(np.mean(f1_incl)),
                "macro_f1_zero_pos_excluded": (float(np.mean(f1_excl))
                                               if f1_excl else None),
                "zero_positive_phases_by_seed": zero_pos[0],
                "per_phase_AUPRC": {n: (float(np.mean(v)) if v else None)
                                    for n, v in ap.items()},
            }
        out[system] = rows
        a = rows.get("mlp_renorm_192x192x192_huber", {})
        b = rows.get("rf_renorm_std", {})
        print(f"  {system}: MLP F1 {a.get('macro_f1_with_zero_pos'):.3f} -> "
              f"excl {a.get('macro_f1_zero_pos_excluded'):.3f} | "
              f"RF F1 {b.get('macro_f1_with_zero_pos'):.3f} -> "
              f"excl {b.get('macro_f1_zero_pos_excluded'):.3f} "
              f"| zero-pos: {a.get('zero_positive_phases_by_seed')}")
    R["detection_recomputed"] = out


# ----------------------------------------------------------------------
# 6. F1 under the spatial band holdouts (all systems)
# ----------------------------------------------------------------------
def f1_under_shift():
    print("== 6. F1 under band holdouts ==")
    out = {}
    for system in sorted(SYSTEMS):
        blk = json.load(open(os.path.join(MODELS, f"block_holdout_{system}.json")))
        rows = {}
        for model in ["mlp_renorm", "rf_renorm"]:
            interp = []
            for seed in SEEDS:
                tag = (f"{model}_192x192x192_huber" if model.startswith("mlp")
                       else f"{model}_std")
                d = load_npz(system, tag, seed)
                f1, _ = phase_classification(d["y_true"], d["y_pred"])
                interp.append(float(f1.mean()))
            rows[model] = {
                "interp_macro_f1": float(np.mean(interp)),
                "T_band_macro_f1": float(np.mean(
                    [blk[f"T_band_{model}_s{s}"]["mean_f1"] for s in SEEDS])),
                "X2_band_macro_f1": float(np.mean(
                    [blk[f"X2_band_{model}_s{s}"]["mean_f1"] for s in SEEDS])),
            }
        out[system] = rows
        for model, v in rows.items():
            print(f"  {system:7s} {model:10s} interp F1={v['interp_macro_f1']:.3f} "
                  f"T-band={v['T_band_macro_f1']:.3f} X2-band={v['X2_band_macro_f1']:.3f}")
    R["f1_under_shift"] = out


# ----------------------------------------------------------------------
# 7. Fe-Cr-V random-control per-seed audit
# ----------------------------------------------------------------------
def fecrv_ctrl_audit():
    print("== 7. Fe-Cr-V random-control audit ==")
    blk = json.load(open(os.path.join(MODELS, "block_holdout_fecrv.json")))
    out = {}
    for model in HOLDOUT_MODELS:
        per_seed = {str(s): {
            "mean_mae": blk[f"random_ctrl_{model}_s{s}"]["mean_mae"],
            "consistency": blk[f"random_ctrl_{model}_s{s}"]["consistency"],
        } for s in SEEDS}
        out[model] = per_seed
        print(f"  {model}: " + " ".join(
            f"s{s}={per_seed[str(s)]['mean_mae']:.4f}" for s in SEEDS))
    R["fecrv_random_ctrl_per_seed"] = out


# ----------------------------------------------------------------------
# 8. Sparsemax per-seed spread
# ----------------------------------------------------------------------
def sparsemax_spread():
    print("== 8. Sparsemax per-seed spread ==")
    out = {}
    for system in sorted(SYSTEMS):
        vals = []
        for seed in SEEDS:
            p = npz_path(system, "mlp_sparsemax_192x192x192_huber", seed)
            if not os.path.exists(p):
                continue
            d = np.load(p)
            vals.append(float(np.mean(np.abs(d["y_pred"] - d["y_true"]))))
        if vals:
            out[system] = {"per_seed_mean_mae": vals,
                            "mean": float(np.mean(vals)), "std": float(np.std(vals))}
            print(f"  {system}: {[round(v, 4) for v in vals]} "
                  f"mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    R["sparsemax_per_seed"] = out


# ----------------------------------------------------------------------
# 9. renorm == sigmoid + post-hoc projection check
# ----------------------------------------------------------------------
def renorm_identity_check():
    print("== 9. renorm identity check ==")
    out = {}
    for system in ["fecrni", "femnni"]:
        for seed in SEEDS:
            ds = load_npz(system, "mlp_sigmoid_192x192x192_huber", seed)
            dr = load_npz(system, "mlp_renorm_192x192x192_huber", seed)
            projected = renorm(ds["y_pred"])
            diff = float(np.abs(projected - dr["y_pred"]).max())
            out[f"{system}_s{seed}"] = diff
            print(f"  {system} s{seed}: max|renorm(sigmoid) - renorm_pred| = {diff:.2e}")
    R["renorm_identity_max_diff"] = out


# ----------------------------------------------------------------------
# 10. Rejected-candidate reconstruction
# ----------------------------------------------------------------------
def rejected_candidates():
    print("== 10. Rejected candidates ==")
    out = {}
    for system in sorted(SYSTEMS):
        cfg = SYSTEMS[system]
        rng = np.random.default_rng(42)
        tasks = []
        tags = []
        skipped = {}
        sf = cfg["sigma_focus"]

        def add(c1, c2, T, tag):
            tasks.append((c1, c2, T))
            tags.append(tag)

        for _ in range(6000):
            if rng.uniform() < 0.6:
                c1 = rng.uniform(0.001, 0.45)
                c2 = rng.uniform(0.001, 0.45)
            else:
                c1 = rng.uniform(0.001, 0.90)
                c2 = rng.uniform(0.001, 0.90)
            if c1 + c2 > 0.995:
                skipped["S1_LHS"] = skipped.get("S1_LHS", 0) + 1
                continue
            add(c1, c2, float(rng.uniform(cfg["t_min"], cfg["t_max"])), "S1_LHS")
        lo1, hi1, t_lo = sf["low"]
        lo2, hi2, t_hi = sf["high"]
        for _ in range(2000):
            c1 = rng.uniform(lo1, hi1)
            c2 = rng.uniform(lo2, hi2)
            if c1 + c2 > 0.995:
                skipped["S2_sigma"] = skipped.get("S2_sigma", 0) + 1
                continue
            add(c1, c2, float(rng.uniform(t_lo, t_hi)), "S2_sigma")
        for T in [900, 1100, 1300, 1500, 1700]:
            grid = np.linspace(0.01, 0.85, 12)
            for c1 in grid:
                for c2 in grid:
                    if 0.01 < c1 + c2 < 0.98:
                        add(float(c1), float(c2), float(T), "S3_isothermal")
                    else:
                        skipped["S3_isothermal"] = skipped.get("S3_isothermal", 0) + 1
        for _ in range(1500):
            c1 = rng.uniform(0.001, 0.85)
            c2 = rng.uniform(0.001, 0.85)
            if 0.70 <= c1 + c2 <= 0.995:
                add(c1, c2, float(rng.uniform(1300, 2000)), "S4_liquidus")
            else:
                skipped["S4_liquidus"] = skipped.get("S4_liquidus", 0) + 1
        for _ in range(1000):
            c1 = rng.uniform(0.0, 0.04)
            c2 = rng.uniform(0.0, 0.04)
            add(c1, c2, float(rng.uniform(700, 1811)), "S5_pureFe")

        rng.shuffle(tasks)  # same order as the generator
        df = pd.read_csv(cfg["dataset"])
        accepted = set()
        for _, r in df.iterrows():
            accepted.add((round(r[cfg["comps"][1]], 9), round(r[cfg["comps"][2]], 9),
                          round(r["T"], 6)))
        cand_keys = {(round(c1, 9), round(c2, 9), round(T, 6))
                     for c1, c2, T in tasks}
        n_acc_not_in_cand = len(accepted - cand_keys)
        out[system] = {"n_accepted_not_in_candidates": n_acc_not_in_cand}
        if n_acc_not_in_cand > 5:
            print(f"  !! {system}: {n_acc_not_in_cand} accepted rows not found in "
                  "reconstructed candidates -- reconstruction FAILED, "
                  "numbers below are unreliable")

        rej = {"T": [], "tag": [], "c1": [], "c2": []}
        dup_of_accepted = 0
        seen = {}
        n_dup = 0
        for (c1, c2, T), tag in zip(tasks, tags):
            key = (round(c1, 9), round(c2, 9), round(T, 6))
            seen[key] = seen.get(key, 0) + 1
            if key in accepted:
                continue
            if seen[key] > 1:
                n_dup += 1
                dup_of_accepted += 1
                continue
            rej["T"].append(T)
            rej["tag"].append(tag)
            rej["c1"].append(c1)
            rej["c2"].append(c2)

        n_tasks = len(tasks)
        n_rej = len(rej["T"])
        tbins = [(700, 900), (900, 1100), (1100, 1300), (1300, 1500),
                 (1500, 1700), (1700, 2001)]
        rej_by_T = {f"{a}-{b}": 0 for a, b in tbins}
        acc_by_T = {f"{a}-{b}": 0 for a, b in tbins}
        for T in rej["T"]:
            for a, b in tbins:
                if a <= T < b:
                    rej_by_T[f"{a}-{b}"] += 1
        for T in df["T"]:
            for a, b in tbins:
                if a <= T < b:
                    acc_by_T[f"{a}-{b}"] += 1
        rej_rate_by_T = {k: (rej_by_T[k] / max(rej_by_T[k] + acc_by_T[k], 1))
                         for k in rej_by_T}
        rej_by_strategy = {}
        for t in rej["tag"]:
            rej_by_strategy[t] = rej_by_strategy.get(t, 0) + 1
        strat_counts = {}
        for t in tags:
            strat_counts[t] = strat_counts.get(t, 0) + 1

        out[system] = {
            "n_accepted_not_in_candidates": n_acc_not_in_cand,
            "n_candidates": n_tasks,
            "n_raw_design_draws": 11220,
            "design_filter_exclusions_by_strategy": skipped,
            "n_design_exclusions_total": int(sum(skipped.values())),
            "solver_rejected_points": [{"c1": c1, "c2": c2, "T": T, "strategy": t}
                                        for c1, c2, T, t in zip(rej["c1"], rej["c2"],
                                                                rej["T"], rej["tag"])],
            "n_accepted": int(len(df)),
            "n_unique_rejected": n_rej,
            "n_duplicate_task_instances": n_dup,
            "rejection_rate": n_rej / n_tasks,
            "rejection_rate_by_T_bin": rej_rate_by_T,
            "rejected_by_T_bin": rej_by_T,
            "accepted_by_T_bin": acc_by_T,
            "rejected_by_strategy": rej_by_strategy,
            "rejected_rate_by_strategy": {
                k: rej_by_strategy.get(k, 0) / strat_counts[k] for k in strat_counts},
            "low_T_share_of_rejections": (float(np.mean([t < 1100 for t in rej["T"]]))
                                           if rej["T"] else None),
        }
        r = out[system]
        print(f"  {system}: {n_tasks} valid candidates (of 11,220 raw draws, "
              f"{r['n_design_exclusions_total']} design-filter exclusions) -> "
              f"{len(df)} accepted ({n_rej} solver rejections)")
        if rej["T"]:
            for c1, c2, T, t in zip(rej["c1"], rej["c2"], rej["T"], rej["tag"]):
                print(f"    rejected: x=({c1:.3f}, {c2:.3f}) T={T:.0f} [{t}]")
        print(f"    design exclusions by strategy: {skipped}")
    R["rejected_candidates"] = out


# ----------------------------------------------------------------------
# 11. MLP forward-pass latency
# ----------------------------------------------------------------------
def inference_timing():
    print("== 11. MLP forward-pass latency ==")
    import torch
    from train_mlp import MLP4
    out = {}
    for n_phases in (4, 7):
        model = MLP4(n_phases=n_phases, head="sigmoid").eval()
        x = torch.randn(1780, 4)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            t0 = time.perf_counter()
            n_rep = 50
            for _ in range(n_rep):
                model(x)
            batch_us = (time.perf_counter() - t0) / n_rep / x.shape[0] * 1e6
            x1 = torch.randn(1, 4)
            for _ in range(10):
                model(x1)
            t0 = time.perf_counter()
            for _ in range(200):
                model(x1)
            single_us = (time.perf_counter() - t0) / 200 * 1e6
        out[f"{n_phases}_phase_head_cpu"] = {
            "batch_inference_us_per_row": batch_us,
            "single_query_us": single_us,
            "speedup_vs_2.5s_calphad_batch": CALPHAD_S_PER_POINT / (batch_us / 1e6),
            "speedup_vs_2.5s_calphad_single": CALPHAD_S_PER_POINT / (single_us / 1e6),
        }
        print(f"  {n_phases}-phase head: {batch_us:.1f} us/row (batch), "
              f"{single_us:.0f} us (single query) | speedup "
              f"{CALPHAD_S_PER_POINT / (batch_us / 1e6):.2e} (batch), "
              f"{CALPHAD_S_PER_POINT / (single_us / 1e6):.2e} (single)")
    if torch.cuda.is_available():
        print("  (CUDA present but CPU numbers reported as the conservative "
              "deployment reference)")
    R["inference_timing"] = out


def main():
    funcs = [region_matched_bands, extrap_seen_unseen, gibbs_violations,
             active_cells_and_tail, detection_recomputed, f1_under_shift,
             fecrv_ctrl_audit, sparsemax_spread, renorm_identity_check,
             rejected_candidates, inference_timing]
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"!! {fn.__name__} failed: {e}")
            traceback.print_exc()
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
