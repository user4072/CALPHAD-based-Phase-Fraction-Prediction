"""Rare-phase detection analysis for the NEW fecrnic (Fe-Cr-Ni-C) system.

Reuses the exact protocol of analysis_revision/revision_analyses.py::
detection_recomputed (section 5), applied to the fecrnic prediction npz
files:

  * positive label: y_true > 1e-3, per phase
  * AUPRC: sklearn average_precision_score per phase, phases with zero
    positive rows in a seed excluded from that seed's macro
  * macro AUPRC (zero-positive excluded): mean over phases of their
    seed-mean AUPRC (also reported: mean over seeds of the per-seed macro)
  * F1: fe_surrogate.experiment.phase_classification at threshold 1e-3,
    macro with all phases and macro with zero-positive phases excluded

Families evaluated (npz tags as stored in models/):
  mlp_sigmoid, mlp_renorm, mlp_sig_norm, rf/xgb/knn raw+renorm,
  weighted, gated (train_remedy.py variants).

KEY question: does `gated` beat rf/xgb/knn on macro AUPRC restricted to
the carbide phases (M23C6, M7C3, CEMENTITE, GRAPHITE)?

Usage: py -3.12 -X utf8 analysis_revision/detection_fecrnic.py
Output: analysis_revision/detection_fecrnic.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from sklearn.metrics import average_precision_score

from fe_surrogate.experiment import active_phases, phase_classification, SEEDS

MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(HERE, "detection_fecrnic.json")
SYSTEM = "fecrnic"
POS_THRESH = 1e-3
CARBIDES = ["M23C6", "M7C3", "CEMENTITE", "GRAPHITE"]

FAMILIES = {
    "mlp_sigmoid": "mlp_sigmoid_192x192x192_huber",
    "mlp_renorm": "mlp_renorm_192x192x192_huber",
    "mlp_sig_norm": "mlp_sig_norm_192x192x192_huber",
    "rf_raw": "rf_raw_std",
    "rf_renorm": "rf_renorm_std",
    "xgb_raw": "xgb_raw_std",
    "xgb_renorm": "xgb_renorm_std",
    "knn_raw": "knn_raw_std",
    "knn_renorm": "knn_renorm_std",
    "weighted": "weighted_192x192x192",
    "gated": "gated_192x192x192",
}


def npz_path(tag, seed):
    return os.path.join(MODELS, f"pred_{SYSTEM}_{tag}_s{seed}.npz")


def evaluate_family(tag, names):
    """Exact protocol of revision_analyses.detection_recomputed."""
    if not os.path.exists(npz_path(tag, 42)):
        return None
    ap = {n: [] for n in names}
    f1_incl, f1_excl, zero_pos, per_seed = [], [], [], []
    for seed in SEEDS:
        p = npz_path(tag, seed)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        y_t, y_p = d["y_true"], d["y_pred"]
        f1, _ = phase_classification(y_t, y_p)          # threshold 1e-3
        pos = y_t.sum(axis=0) > 0                       # >= 1 positive row
        zp = [names[i] for i in range(len(names)) if not pos[i]]
        zero_pos.append(zp)
        seed_ap = {}
        for i, n in enumerate(names):
            if pos[i]:
                v = float(average_precision_score(
                    (y_t[:, i] > POS_THRESH).astype(int), y_p[:, i]))
                ap[n].append(v)
                seed_ap[n] = v
        f1_incl.append(float(f1.mean()))
        if pos.sum() > 0:
            f1_excl.append(float(f1[pos].mean()))
        per_seed.append({
            "seed": int(seed),
            "macro_f1_with_zero_pos": float(f1.mean()),
            "macro_f1_zero_pos_excluded": (float(f1[pos].mean())
                                           if pos.sum() > 0 else None),
            "macro_auprc": (float(np.mean(list(seed_ap.values())))
                            if seed_ap else None),
            "per_phase_auprc": seed_ap,
        })
    if not per_seed:
        return None
    per_phase = {n: (float(np.mean(v)) if v else None) for n, v in ap.items()}
    seed_macros = [s["macro_auprc"] for s in per_seed
                   if s["macro_auprc"] is not None]
    phase_means = [v for v in per_phase.values() if v is not None]
    carb = [per_phase[c] for c in CARBIDES
            if c in per_phase and per_phase[c] is not None]
    return {
        "n_seeds": len(per_seed),
        "macro_auprc": float(np.mean(seed_macros)) if seed_macros else None,
        "macro_auprc_phase_means": (float(np.mean(phase_means))
                                    if phase_means else None),
        "macro_f1_with_zero_pos": float(np.mean(f1_incl)),
        "macro_f1_zero_pos_excluded": (float(np.mean(f1_excl))
                                       if f1_excl else None),
        "zero_positive_phases": zero_pos[0],
        "per_phase_auprc": per_phase,
        "carbide_macro_auprc": float(np.mean(carb)) if carb else None,
        "per_seed": per_seed,
    }


def main():
    _, names = active_phases(SYSTEM)
    missing_carbides = [c for c in CARBIDES if c not in names]
    if missing_carbides:
        raise SystemExit(f"carbide phases absent from targets: {missing_carbides}")

    families, missing = {}, []
    for fam, tag in FAMILIES.items():
        res = evaluate_family(tag, names)
        if res is None:
            missing.append(tag)
            continue
        res["npz_tag"] = tag
        families[fam] = res
        print(f"[{fam:14s}] macroAUPRC={res['macro_auprc']:.4f} "
              f"carbideAUPRC={res['carbide_macro_auprc']:.4f} "
              f"F1excl={res['macro_f1_zero_pos_excluded']:.4f} "
              f"seeds={res['n_seeds']}", flush=True)

    # ---- KEY question: gated vs rf/xgb/knn on the carbide phases ----
    carb_macro = {f: r["carbide_macro_auprc"] for f, r in families.items()
                  if r["carbide_macro_auprc"] is not None}
    trees = ["rf_raw", "rf_renorm", "xgb_raw", "xgb_renorm",
             "knn_raw", "knn_renorm"]
    gated = carb_macro.get("gated")
    vs = {}
    for t in trees:
        if t in carb_macro and gated is not None:
            vs[t] = {"gated_minus_family": float(gated - carb_macro[t]),
                     "gated_wins": bool(gated > carb_macro[t])}
    key = {
        "carbide_phases": CARBIDES,
        "carbide_macro_auprc_by_family": carb_macro,
        "gated_vs_baselines": vs,
        "gated_beats_all_tree_knn": (bool(all(v["gated_wins"] for v in vs.values()))
                                     if vs else None),
        "gated_beats_rf_xgb_knn_renorm": bool(
            all(vs.get(f, {}).get("gated_wins", False)
                for f in ["rf_renorm", "xgb_renorm", "knn_renorm"])),
    }
    print("KEY question: gated beats all tree/knn on carbide macro AUPRC: "
          f"{key['gated_beats_all_tree_knn']}", flush=True)

    out = {
        "system": SYSTEM,
        "protocol": {
            "source": "analysis_revision/revision_analyses.py::detection_recomputed",
            "positive_threshold": POS_THRESH,
            "auprc": "sklearn average_precision_score, per phase, "
                     "zero-positive phases excluded from macros",
            "f1": "phase_classification threshold 1e-3",
            "seeds": SEEDS,
            "families": FAMILIES,
        },
        "phases": names,
        "families": families,
        "missing_tags": missing,
        "key_question": key,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
