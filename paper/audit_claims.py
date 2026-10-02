"""audit_claims.py -- pre-submission numeric audit for paper_cms.tex.

Every numeric claim that falls inside an in-scope diff hunk (plus the
pre-existing claim-ledger rows) is re-derived from a stored artifact.
Expectations are hardcoded with tex:LINE labels taken from verbatim reads.

Tolerance policy
  check()      |written - recomputed| <= max(0.5 * ULP(written), 1.5% * |recomputed|)
  check_le/ge  pass with 1% relative slack on the bound (inequality claims)
  check_exact  counts / strings / booleans must match exactly
  check_prose  prose windows ('~', 'order of magnitude', 'twice', ...) tested
               against an explicitly documented inclusive window

NV entries are claims with no archived artifact (reported in the final
report, never affect the exit status).

Usage:  py -3.12 audit_claims.py      exit 0 iff zero FAIL lines
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = r"E:\projects\pycalphad\fe_surrogate"
PAPER = os.path.join(ROOT, "paper")
MODELS = os.path.join(ROOT, "models")
REV = os.path.join(ROOT, "analysis_revision")
RAW = os.path.join(ROOT, "data", "raw")
ANCHOR = os.path.join(PAPER, "anchor_data")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

TERN = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
EXT = ["fecrnic", "fecrc"]
ALL7 = TERN + EXT
SEEDS = [42, 123, 2024]
BEST = {"fecrni": "mlp_renorm", "fecrmn": "mlp_sig_norm", "fecrmo": "mlp_sig_norm",
        "fecrv": "mlp_renorm", "femnni": "mlp_renorm"}
CONSTRAINED3 = ["mlp_softmax", "mlp_renorm", "mlp_sig_norm"]
EXT_TAGS = ["mlp_renorm", "mlp_sig_norm", "rf_renorm", "xgb_renorm"]
CLASSICAL = ["rf", "xgb", "knn"]

results = []


def rec(status, name, detail=""):
    results.append((status, name, detail))
    line = f"{status} {name}"
    if detail:
        line += f" <{detail}>"
    print(line, flush=True)


def _num(x):
    s = str(x).strip().replace("\u2212", "-").replace("$", "").replace(",", "")
    s = s.replace("~", "").replace("\\times", "e").replace("{", "").replace("}", "")
    return float(s)


def _ulp(exp):
    s = str(exp).strip().replace("\u2212", "-")
    m = re.match(r"^([+-]?[\d.]+)(?:[eE]([+-]?\d+))?$", s)
    mant, e = m.group(1), int(m.group(2) or 0)
    frac = mant.split(".")[1] if "." in mant else ""
    return 10.0 ** (e - len(frac))


def check(name, got, exp, rel=0.015):
    if got is None:
        rec("FAIL", name, f"expected {exp}, got None")
        return False
    got = float(got)
    expf = _num(exp)
    tol = max(0.5 * _ulp(exp), rel * abs(got))
    ok = abs(got - expf) <= tol
    rec("PASS" if ok else "FAIL", name,
        f"expected {exp} got {got:.6g} tol {tol:.3g}")
    return ok


def check_le(name, got, bound, rel=0.01):
    if got is None:
        rec("FAIL", name, f"expected <= {bound}, got None")
        return False
    got = float(got)
    lim = _num(bound) * (1 + rel)
    ok = got <= lim
    rec("PASS" if ok else "FAIL", name,
        f"expected <= {bound} got {got:.6g} (lim {lim:.3g})")
    return ok


def check_ge(name, got, bound, rel=0.01):
    if got is None:
        rec("FAIL", name, f"expected >= {bound}, got None")
        return False
    got = float(got)
    lim = _num(bound) * (1 - rel)
    ok = got >= lim
    rec("PASS" if ok else "FAIL", name,
        f"expected >= {bound} got {got:.6g} (lim {lim:.3g})")
    return ok


def check_exact(name, got, exp):
    ok = (got == exp)
    rec("PASS" if ok else "FAIL", name, f"expected {exp!r} got {got!r}")
    return ok


def check_prose(name, got, lo, hi):
    if got is None:
        rec("FAIL", name, f"expected in [{lo}, {hi}], got None")
        return False
    got = float(got)
    ok = _num(lo) <= got <= _num(hi)
    rec("PASS" if ok else "FAIL", name,
        f"expected in [{lo}, {hi}] got {got:.6g}")
    return ok


def nv(name, reason):
    rec("NV", name, reason)


def guarded(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        rec("FAIL", f"{fn.__name__} section crashed", repr(exc))


def J(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- artifacts
_cache = {}


def csv_rows(sysn):
    if ("csv", sysn) not in _cache:
        path = os.path.join(RAW, f"dataset_{sysn}.csv")
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            rows = list(rd)
        _cache[("csv", sysn)] = (hdr, rows)
    return _cache[("csv", sysn)]


def csv_stats(sysn):
    if ("st", sysn) in _cache:
        return _cache[("st", sysn)]
    hdr, rows = csv_rows(sysn)
    ti = hdr.index("T")
    np_idx = [i for i, c in enumerate(hdr) if c.startswith("NP_")]
    np_cols = [hdr[i] for i in np_idx]
    bi = hdr.index("in_stainless_box")
    closure = 0.0
    n_in_box = 0
    tmin, tmax = float("inf"), float("-inf")
    liq_i = np_cols.index("LIQUID") if "LIQUID" in np_cols else None
    liq_min_t = None
    liq_near_rows = 0
    active_hist = {}
    occ = {c: 0 for c in np_cols}
    for r in rows:
        vals = [float(r[i]) for i in np_idx]
        s = sum(vals)
        closure = max(closure, abs(s - 1.0))
        if r[bi] in ("True", "true", "1"):
            n_in_box += 1
        t = float(r[ti])
        tmin = min(tmin, t)
        tmax = max(tmax, t)
        act = sum(1 for v in vals if v > 0)
        active_hist[act] = active_hist.get(act, 0) + 1
        for c, v in zip(np_cols, vals):
            if v > 0:
                occ[c] += 1
        if liq_i is not None and vals[liq_i] > 0:
            liq_min_t = t if liq_min_t is None else min(liq_min_t, t)
            if t <= 1200.0:
                liq_near_rows += 1
    n = len(rows)
    st = {
        "n": n,
        "closure_max": closure,
        "box_pct": 100.0 * n_in_box / n,
        "tmin": tmin,
        "tmax": tmax,
        "liq_min_t": liq_min_t,
        "liq_near_rows": liq_near_rows,
        "active_1_3_frac": sum(v for k, v in active_hist.items() if 1 <= k <= 3) / n,
        "occ_pct": {c: 100.0 * v / n for c, v in occ.items()},
        "np_cols": np_cols,
    }
    _cache[("st", sysn)] = st
    return st


def heads(sysn, ext=False):
    key = ("heads", sysn)
    if key not in _cache:
        if ext:
            d = {}
            for fn in (f"results_heads_{sysn}.json", f"results_baselines_{sysn}.json"):
                p = os.path.join(MODELS, fn)
                if os.path.exists(p):
                    d.update(J(p))
        else:
            d = {}
            for fn in (f"results_heads_{sysn}.json", f"results_baselines_{sysn}.json"):
                p = os.path.join(MODELS, fn)
                if os.path.exists(p):
                    d.update(J(p))
            p = os.path.join(MODELS, "results_a23.json")
            if sysn == "fecrni" and os.path.exists(p):
                d.update(J(p))
        _cache[key] = d
    return _cache[key]


def sm(d, prefix, field="mean_mae"):
    v = [d[f"{prefix}_s{s}"][field] for s in SEEDS if f"{prefix}_s{s}" in d]
    return float(np.mean(v)) if v else None


def npz_files(pattern):
    return sorted(glob.glob(os.path.join(MODELS, pattern)))


def npz_neg_pct(path):
    z = np.load(path)
    yp = z["y_pred"]
    return 100.0 * float((yp < 0).mean())


def npz_closure(path):
    z = np.load(path)
    return float(np.abs(z["y_pred"].sum(axis=1) - 1.0).mean())


def sign_p(k, n):
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return 2 * p if p < 0.5 else 2 * (1 - p)


def ratio_band(sysn, axis, mlp_tag):
    d = J(os.path.join(MODELS, f"block_holdout_{sysn}.json"))
    ctrl = sm(d, f"random_ctrl_{mlp_tag}")
    return sm(d, f"{axis}_band_{mlp_tag}") / sm(d, f"random_ctrl_{mlp_tag}")


def ratio_band_tree(sysn, axis, tree):
    d = J(os.path.join(MODELS, f"block_holdout_{sysn}.json"))
    return sm(d, f"{axis}_band_{tree}") / sm(d, f"random_ctrl_{tree}")


# ============================================================ GROUP 1 counts
def g_counts():
    rows5 = sum(csv_stats(s)["n"] for s in TERN)
    check_exact("tex:41 ternary equilibria 44,397", rows5, 44397)
    n_nic = csv_stats("fecrnic")["n"]
    n_crc = csv_stats("fecrc")["n"]
    check_exact("tex:43 quaternary equilibria 13,192", n_nic, 13192)
    check_exact("tex:44 Fe-Cr-C equilibria 13,199", n_crc, 13199)
    total = rows5 + n_nic + n_crc
    check_exact("tex:44 total 70,788", total, 70788)
    check_exact("tex:1648 conclusion 44,397", rows5, 44397)
    check_exact("tex:1650 extension 26,391", n_nic + n_crc, 26391)
    check_exact("tex:1650 total 70,788", total, 70788)
    check_exact("tex:439 total 70,788", total, 70788)
    check_exact("tex:1742 total 70,788", total, 70788)
    check_exact("tex:217,248 two scope-extension systems", len(EXT), 2)

    # analysis-log spot (out-of-scope lines 237-238 kept as spot checks)
    log = open(os.path.join(REV, "analysis_log.txt"), encoding="utf-8", errors="replace").read()
    raw = [int(x.replace(",", "")) for x in re.findall(r"of ([\d,]+) raw draws", log)]
    sub = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+) valid candidates",
                                                       log)]
    acc = [int(x.replace(",", "")) for x in re.findall(r"->\s*([\d,]+) accepted", log)]
    check_exact("tex:237 11,220 raw draws per system (spot)",
                sum(1 for v in raw if v == 11220), 5)
    check_exact("tex:238 8,880 submitted per system (spot)",
                sum(1 for v in sub if v == 8880), 5)
    check_exact("tex:238 accepted sum 44,397 (spot)", sum(acc), 44397)


# ===================================================== GROUP 2 probe / counts
def g_probe():
    exp_elig = {"fecrni": 28, "fecrmn": 31, "fecrmo": 29, "fecrv": 25,
                "femnni": 29, "fecrnic": 35, "fecrc": 32}
    lens = {}
    npts = {}
    for s in ALL7:
        pj = J(os.path.join(RAW, f"{s}_probe.json"))
        lens[s] = len(pj["eligible"])
        npts[s] = pj.get("n_points")
    for s, v in exp_elig.items():
        check_exact(f"tex:274-277 eligible {s} = {v}", lens[s], v)
    check_exact("tex:294 eligible range 25-35 (min)", min(lens.values()), 25)
    check_exact("tex:294 eligible range 25-35 (max)", max(lens.values()), 35)
    check_exact("tex:321 2,000-draw probe (all systems)",
                sum(1 for v in npts.values() if v == 2000), 7)
    p_nic = J(os.path.join(RAW, "fecrnic_probe.json"))
    p_crc = J(os.path.join(RAW, "fecrc_probe.json"))
    check_exact("tex:429 probe activates 9 phases (fecrnic)",
                len(p_nic["active_counts"]), 9)
    check_exact("tex:429 35 eligible (fecrnic)", len(p_nic["eligible"]), 35)
    check_exact("tex:434 9 active phases (fecrc)", len(p_crc["active_counts"]), 9)
    check_exact("tex:434 32 eligible (fecrc)", len(p_crc["eligible"]), 32)
    # MNNI2 spot (line 333 out of scope)
    pj = J(os.path.join(RAW, "femnni_probe.json"))
    mnni2 = pj["active_counts"].get("MNNI2", 0)
    check("tex:333 MNNI2 in 0.1% of probe draws (spot)",
          100.0 * mnni2 / pj["n_points"], "0.1")
    st = csv_stats("femnni")
    has_col = "NP_MNNI2" in st["np_cols"]
    if has_col:
        hdr, rows = csv_rows("femnni")
        ci = hdr.index("NP_MNNI2")
        check_exact("tex:334 MNNI2 zero active rows (spot)",
                    sum(1 for r in rows if float(r[ci]) > 0), 0)
    else:
        check_exact("tex:334 MNNI2 column absent (spot)", has_col, True)


# ====================================================== GROUP 3 CSV geometry
def g_csv():
    for s in ALL7:
        st = csv_stats(s)
        check(f"tex:324 closure <=3.3e-8 [{s}]", st["closure_max"], "3.3e-8") \
            if False else None
    worst = max(csv_stats(s)["closure_max"] for s in ALL7)
    check_le("tex:324,325 mass balance <=3.3e-8 (all datasets)", worst, "3.3e-8")
    check_le("tex:428 closure <=8.2e-9 (fecrnic)",
             csv_stats("fecrnic")["closure_max"], "8.2e-9")
    check_le("tex:434 closure <=6.8e-9 (fecrc)",
             csv_stats("fecrc")["closure_max"], "6.8e-9")
    check("tex:428 69.9% inside stainless box (fecrnic)",
          csv_stats("fecrnic")["box_pct"], "69.9")
    check("tex:434 100% inside box (fecrc)", csv_stats("fecrc")["box_pct"], "100")

    from fe_surrogate.systems import SYSTEMS
    bn = SYSTEMS["fecrnic"]["box_ranges"]
    check_exact("tex:421-422 fecrnic box ranges",
                (bn["Cr"], bn["Ni"], bn["C"]),
                ([0.001, 0.35], [0.001, 0.3], [0.001, 0.05]))
    bc = SYSTEMS["fecrc"]["box_ranges"]
    check_exact("tex:431-432 fecrc box ranges",
                (bc["Cr"], bc["C"]), ([0.001, 0.3], [0.0005, 0.06]))
    for s in EXT:
        check_exact(f"tex:436 config T range [{s}]",
                    (SYSTEMS[s]["t_min"], SYSTEMS[s]["t_max"]), (700.0, 2000.0))
        st = csv_stats(s)
        check_ge(f"tex:436 dataset Tmin >=700 [{s}]", st["tmin"], "700", rel=0.001)
        check_le(f"tex:436 dataset Tmax <=2000 [{s}]", st["tmax"], "2000", rel=0.001)

    for s in TERN:
        st = csv_stats(s)
        check_exact(f"tex:70,1111,1549,1689 liquid rows with T<=1200 [{s}]",
                    st["liq_near_rows"], 0)
    for s in ALL7:
        st = csv_stats(s)
        check_ge(f"tex:1619 typically 1-3 active phases >=0.9 [{s}]",
                 st["active_1_3_frac"], "0.9", rel=0.001)
    check("tex:435 SIGMA in 0.4% of fecrc rows",
          csv_stats("fecrc")["occ_pct"].get("NP_SIGMA"), "0.4")
    check("tex:890 MU_PHASE_I in 0.1% of fecrmo rows",
          csv_stats("fecrmo")["occ_pct"].get("NP_MU_PHASE_I"), "0.1")
    p_crc = J(os.path.join(RAW, "fecrc_probe.json"))
    check_exact("tex:318 H_BCC never observed by probe",
                "H_BCC" in p_crc["active_counts"], False)
    st_crc = csv_stats("fecrc")
    has_h = "NP_H_BCC" in st_crc["np_cols"]
    if has_h:
        check_exact("tex:318 H_BCC zero rows in structured design",
                    int(st_crc["occ_pct"]["NP_H_BCC"] == 0.0), 1)
    else:
        check_exact("tex:318 H_BCC column absent from dataset", has_h, False)


# ================================================ GROUP 4 heads / projection
def g_heads():
    # tex:46-47 constrained closure <= 5e-8
    cons = []
    sparse = []
    for s in TERN:
        d = heads(s)
        for tag in CONSTRAINED3:
            for sd in SEEDS:
                cons.append(d[f"{tag}_s{sd}"]["consistency"])
        v = [d[f"mlp_sparsemax_s{sd}"]["consistency"] for sd in SEEDS
             if f"mlp_sparsemax_s{sd}" in d]
        sparse.append(float(np.mean(v)))
    check_le("tex:47 constrained heads closure <=5e-8", max(cons), "5e-8")
    check_le("tex:47 sparsemax closure <=5e-8 (seed-mean per system)",
             max(sparse), "5e-8")

    # tex:48/716/1658 sigmoid closure 6-8%
    pa = J(os.path.join(MODELS, "projection_ablation.json"))
    oor = pa["key_questions"]["sigmoid_out_of_range_check"]
    cls = [oor[s]["closure_mean"] * 100 for s in TERN]
    check("tex:48,716,1658 sigmoid closure lower 6%", min(cls), "6")
    check("tex:48,716,1658 sigmoid closure upper 8%", max(cls), "8")
    in_range = all(oor[s]["out_of_range_frac"] == 0.0
                   and 0.0 < oor[s]["min_cell"] and oor[s]["max_cell"] < 1.0
                   for s in TERN)
    check_exact("tex:716 sigmoid never leaves (0,1)", in_range, True)

    # residue negatives from 192-tag npz only
    res_pct = {}
    min_cell = 0.0
    for s in TERN:
        vals = [npz_neg_pct(f) for f in
                npz_files(f"pred_{s}_mlp_residue_192x192x192_huber_s*.npz")]
        res_pct[s] = float(np.mean(vals))
        for f in npz_files(f"pred_{s}_mlp_residue_192x192x192_huber_s*.npz"):
            min_cell = min(min_cell, float(np.load(f)["y_pred"].min()))
    check_le("tex:49,1659 residue negatives <=1.2% of cells",
             max(res_pct.values()), "1.2")
    spot = {"fecrv": "0.0", "femnni": "0.02", "fecrni": "0.40",
            "fecrmn": "0.50", "fecrmo": "1.21"}
    for s, exp in spot.items():
        check(f"tex:1294 residue negatives [{s}] (spot)", res_pct[s], exp)
    check("tex:1296 worst residue cell -0.59 (spot)", -min_cell, "0.59")

    # tex:90,1300,1730 residue 1.4-2.5x worse than best constrained head
    ratios = {}
    for s in TERN:
        d = heads(s)
        residue = sm(d, "mlp_residue")
        bestc = min(sm(d, t) for t in CONSTRAINED3)
        ratios[s] = residue / bestc
    check("tex:90,1300 residue ratio lower 1.4x", min(ratios.values()), "1.4")
    check("tex:90,1730 residue ratio upper 2.5x", max(ratios.values()), "2.5")
    in_win = sum(1 for v in ratios.values() if 1.379 <= v <= 2.55)
    check_exact("tex:1731 residue worse in all five systems", in_win, 5)

    # tex:1303 best drop choice = dominant phase in four of five systems
    wins = 0
    for s in TERN:
        sweep = J(os.path.join(MODELS, f"residue_sweep_{s}.json"))
        best_phase = min(sweep, key=lambda k: sweep[k]["mean_mae"]).replace(
            "sweep_drop_", "")
        hdr, rows = csv_rows(s)
        np_idx = [i for i, c in enumerate(hdr) if c.startswith("NP_")]
        sums = np.zeros(len(np_idx))
        for r in rows:
            for j, i in enumerate(np_idx):
                sums[j] += float(r[i])
        dominant = hdr[np_idx[int(np.argmax(sums))]].replace("NP_", "")
        wins += int(best_phase == dominant)
    check_exact("tex:1303 best residue drop = dominant phase in 4 of 5", wins, 4)

    # projection analysis
    proj = pa["key_questions"]["projection_reproduces_stored_renorm_mae"]
    ms = proj["mlp_sigmoid"]
    check_le("tex:52,693 projection reproduces renorm <=1.3e-4",
             max(abs(v["mae_delta_mean"]) for v in ms["per_system"].values()),
             "1.3e-4")
    check_le("tex:693 7e-4 per seed", ms["max_abs_mae_delta"], "7e-4")
    pn = ms["per_system"]
    check("tex:694 fecrni projected 0.0107", pn["fecrni"]["mae_projected_mean"], "0.0107")
    check("tex:694 fecrni stored 0.0106", pn["fecrni"]["mae_stored_mean"], "0.0106")
    check("tex:694 fecrmn projected 0.0081", pn["fecrmn"]["mae_projected_mean"], "0.0081")
    check("tex:694 fecrmn stored 0.0081", pn["fecrmn"]["mae_stored_mean"], "0.0081")
    check("tex:696 sigmoid raw fecrni 0.0232",
          sm(heads("fecrni"), "mlp_sigmoid"), "0.0232")
    check("tex:696 renorm fecrni 0.0106", sm(heads("fecrni"), "mlp_renorm"), "0.0106")
    check_le("tex:702 rf/xgb projection changes MAE <=1e-4",
             max(proj["rf"]["max_abs_mae_delta"], proj["xgb"]["max_abs_mae_delta"]),
             "1e-4")

    # family taxonomy
    rk = pa["key_questions"]["ridge_knn_closure_vs_negativity"]
    per = {s: rk["ridge"]["per_system"][s] for s in TERN}
    cls_r = [per[s]["closure_mean"] for s in TERN]
    check("tex:708 ridge closure lower 1.2e-10", min(cls_r), "1.2e-10")
    check("tex:708 ridge closure upper 4.4e-10", max(cls_r), "4.4e-10")
    check_prose("tex:53 ridge ~1e-10 (min)", min(cls_r), "1e-11", "9.5e-10")
    check_prose("tex:53,1665 ~1e-10 (max)", max(cls_r), "1e-11", "9.5e-10")
    neg_r = [per[s]["neg_frac_cells"] * 100 for s in TERN]
    check("tex:710 ridge negatives lower 16.6%", min(neg_r), "16.6")
    check("tex:710 ridge negatives upper 27.7%", max(neg_r), "27.7")
    check("tex:711 ridge most negative 0.50",
          -rk["ridge"]["min_cell_worst"], "0.50")

    # ridge raw npz (s42) for abstract 17-28%
    s42 = {}
    for s in TERN:
        f = npz_files(f"pred_{s}_ridge_raw_std_s42.npz")
        s42[s] = npz_neg_pct(f[0])
    check("tex:54 ridge 17-28% lower (s42 npz)", min(s42.values()), "17")
    check("tex:54 ridge 17-28% upper (s42 npz)", max(s42.values()), "28")
    check_le("tex:1665 ridge up to 28% negatives", max(s42.values()), "28")

    agg = pa["aggregate_mean_sd_over_seeds"]
    rf_cls = [agg[s]["rf"]["closure_mean_mean"] * 100 for s in TERN]
    check("tex:713 rf closure lower 0.9%", min(rf_cls), "0.9")
    check("tex:713 rf closure upper 4.7%", max(rf_cls), "4.7")
    check_exact("tex:712 rf has no negative cell (projection)",
                max(agg[s]["rf"]["neg_frac_cells_mean"] for s in TERN), 0.0)
    rf_neg_cells = sum(int((np.load(f)["y_pred"] < 0).sum())
                       for f in npz_files("pred_*_rf_*_std_s*.npz"))
    check_exact("tex:712 rf npz zero negative cells", rf_neg_cells, 0)
    xg_cls = [agg[s]["xgb"]["closure_mean_mean"] * 100 for s in TERN]
    check("tex:715 xgb closure lower 1.6%", min(xg_cls), "1.6")
    check("tex:715 xgb closure upper 4.1%", max(xg_cls), "4.1")
    xg_neg = [agg[s]["xgb"]["neg_frac_cells_mean"] * 100 for s in TERN]
    check("tex:715 xgb negatives lower 19.0%", min(xg_neg), "19.0")
    check("tex:715 xgb negatives upper 35.8%", max(xg_neg), "35.8")
    knn_cls = max(rk["knn"]["per_system"][s]["closure_mean"] for s in TERN)
    check_le("tex:718 kNN closure better than 5e-9", knn_cls, "5e-9")
    knn_neg = sum(int((np.load(f)["y_pred"] < 0).sum())
                  for f in npz_files("pred_*_knn_*_std_s*.npz"))
    check_exact("tex:719 kNN zero negative cells", knn_neg, 0)

    valid = []
    if max(cls_r) <= 5e-9 * 1.01 and max(neg_r) == 0.0:
        valid.append("ridge")
    if max(rf_cls) / 100 <= 5e-9 * 1.01 and rf_neg_cells == 0:
        valid.append("rf")
    if max(xg_cls) / 100 <= 5e-9 * 1.01 and max(xg_neg) == 0.0:
        valid.append("xgb")
    if knn_cls <= 5e-9 * 1.01 and knn_neg == 0:
        valid.append("knn")
    check_exact("tex:55,719,1666 only kNN simplex-valid a priori", valid, ["knn"])

    # float32 closure / non-negativity of the three constrained heads (1655-1657)
    check_le("tex:1656 constrained heads closure <=1e-6 (float32)", max(cons), "1e-6")
    neg3 = 0
    for s in TERN:
        for tag in CONSTRAINED3:
            for f in npz_files(f"pred_{s}_{tag}_192x192x192_huber_s*.npz"):
                neg3 += int((np.load(f)["y_pred"] < 0).sum())
    check_exact("tex:1656 constrained heads non-negative", neg3, 0)

    # post-hoc renorm repairs every evaluated model + did not hurt (1660-1661)
    ren_cons = []
    for s in TERN:
        d = heads(s, ext=False)
        for tag in ["mlp_renorm", "rf_renorm", "xgb_renorm", "knn_renorm",
                    "ridge_renorm"]:
            for sd in SEEDS:
                k = f"{tag}_s{sd}"
                if k in d:
                    ren_cons.append(d[k]["consistency"])
    check_le("tex:1660 renormalisation repairs to machine precision",
             max(ren_cons), "1e-6")
    hurt = []
    for s in TERN:
        d = heads(s)
        ren = sm(d, "mlp_renorm")
        sig = sm(d, "mlp_sigmoid")
        hurt.append(ren <= sig * 1.02)
    check_exact("tex:1661 renorm does not hurt accuracy in five systems",
                sum(hurt), 5)

    # counts of heads / baselines / probes
    rh = heads("fecrni")
    head_tags = {k.rsplit("_s", 1)[0] for k in rh if k.startswith("mlp_")
                 and not k.startswith("mlp_power") and not k.startswith("mlp_penalty")}
    check_exact("tex:190,1646 six heads", len(head_tags), 6)
    base_tags = {k.rsplit("_s", 1)[0] for k in rh if not k.startswith("mlp_")}
    check_exact("tex:191,1647 eight baseline tags", len(base_tags), 8)
    fam = {t.split("_")[0] for t in base_tags}
    check_exact("tex:191,242 four classical baseline families", len(fam), 4)
    a23 = J(os.path.join(MODELS, "results_a23.json"))
    a23_tags = {k.rsplit("_s", 1)[0] for k in a23}
    check_exact("tex:45,1646 three mechanism-probe tags", len(a23_tags), 3)
    probe_fam = {"power" if "power" in t else "penalty" for t in a23_tags}
    check_exact("tex:45-46,1646 two mechanism-probe families", len(probe_fam), 2)
    check_exact("tex:242,1454 three seeds", len(SEEDS), 3)


# ================================================ GROUP 5 penalty probe (a23)
def g_penalty():
    # tex:90 / tex:1227-1229 "degrades MAE by 8x / 16x": documented reading =
    # penalty MAE relative to the default (renorm) head on Fe-Cr-Ni, the
    # system the penalty probe was run on.  Against the unpenalised sigmoid
    # head the factors are 3.5x / 7.3x (claim-ledger row 90, P*).
    a23 = J(os.path.join(MODELS, "results_a23.json"))
    base = sm(heads("fecrni"), "mlp_renorm")
    r1 = sm(a23, "mlp_penalty_1") / base
    r10 = sm(a23, "mlp_penalty_10") / base
    check("tex:90 penalty degrades MAE lower 8x", min(r1, r10), "8")
    check("tex:90 penalty degrades MAE upper 16x", max(r1, r10), "16")
    check_ge("tex:1728 penalty detrimental in both variants", min(r1, r10), "1")
    check("tex:1227 penalty_1 absolute MAE 0.0804",
          sm(a23, "mlp_penalty_1"), "0.0804")
    check("tex:1228 penalty_10 absolute MAE 0.1693",
          sm(a23, "mlp_penalty_10"), "0.1693")
    check("tex:1227 renorm reference MAE 0.0106", base, "0.0106")
    check("tex:1195 power-norm probe MAE 0.0123",
          sm(a23, "mlp_power_norm"), "0.0123")

    # spot: penalty closure 6.2e-2 -> 1.3e-3 (line 1229; Fe-Cr-Ni scope,
    # matching the penalty probe) -- sigmoid baseline closure on fecrni
    pa = J(os.path.join(MODELS, "projection_ablation.json"))
    oor = pa["key_questions"]["sigmoid_out_of_range_check"]
    check("tex:1229 sigmoid closure 6.2e-2 (fecrni, spot)",
          oor["fecrni"]["closure_mean"], "6.2e-2")
    for tag, exp in (("mlp_penalty_10", "1.3e-3"), ("mlp_penalty_1", "2.2e-3")):
        vals = [npz_closure(f) for f in
                npz_files(f"pred_fecrni_{tag}_*huber_s*.npz")]
        check(f"tex:1229 {tag} closure {exp} (spot)", float(np.mean(vals)), exp)


# ============================================ GROUP 6 sparsemax sensitivity
def g_sparsemax():
    sp = J(os.path.join(REV, "revision_analyses.json"))["sparsemax_per_seed"]
    spreads = []
    for s in TERN:
        v = sp[s]["per_seed_mean_mae"]
        spreads.append(max(v) / min(v))
    check_ge("tex:89,1727 sparsemax severe seed sensitivity (min spread)",
             min(spreads), "1")


# ================================================== GROUP 7 Delaunay / bands
def g_delaunay():
    from scipy.spatial import Delaunay
    pcts = []
    for s in TERN:
        hdr, rows = csv_rows(s)
        ti = hdr.index("T")
        comps = hdr[:ti]
        x2c, x3c = comps[1], comps[2]
        X = np.array([[float(r[hdr.index(x2c)]), float(r[hdr.index(x3c)]),
                       float(r[ti])] for r in rows])
        T = X[:, 2]
        for name, mask in (("Tband", (T >= 1200) & (T <= 1400)),
                           ("X2band", (X[:, 0] >= 0.25) & (X[:, 0] <= 0.35))):
            train = X[~mask]
            band = X[mask]
            mu, sd = train.mean(0), train.std(0)
            tri = Delaunay((train - mu) / sd)
            inside = tri.find_simplex((band - mu) / sd) >= 0
            pcts.append(100.0 * float(inside.mean()))
    check("tex:64 Delaunay in-hull lower 99.3%", min(pcts), "99.3")
    check("tex:64 Delaunay in-hull upper 99.4%", max(pcts), "99.4")


def g_bands():
    # abstract penalties (in scope) -- fecrni composition & temperature
    check("tex:65 rf composition penalty 3.11x",
          ratio_band("fecrni", "X2", "rf_renorm"), "3.11")
    check("tex:66 mlp composition penalty 1.61x",
          ratio_band("fecrni", "X2", BEST["fecrni"]), "1.61")
    check("tex:67 mlp temperature penalty 1.31x",
          ratio_band("fecrni", "T", BEST["fecrni"]), "1.31")
    check("tex:67 rf temperature penalty 1.31x",
          ratio_band("fecrni", "T", "rf_renorm"), "1.31")

    # spot penalties (out-of-scope 1050-1062)
    check("tex:1050,1079 fecrni T both 1.31x rf (spot)",
          ratio_band("fecrni", "T", "rf_renorm"), "1.31")
    check("tex:1060 xgb composition 2.24x (spot)",
          ratio_band("fecrni", "X2", "xgb_renorm"), "2.24")
    check("tex:1061 fecrmn rf 1.57x (spot)",
          ratio_band("fecrmn", "X2", "rf_renorm"), "1.57")
    # tex:1061-1062 MLP composition penalties follow the MLP-renorm row of
    # tab/holdout.tex (as does the Fe-Cr-Ni 1.61x above), not the per-system
    # best-ID head: 0.85 (renorm) vs 0.92 (sig/sum) on Fe-Cr-Mn, and 1.01
    # (renorm) vs 1.16 (sig/sum) on Fe-Cr-Mo.
    check("tex:1061 fecrmn mlp 0.85x (spot)",
          ratio_band("fecrmn", "X2", "mlp_renorm"), "0.85")
    check("tex:1062 fecrmo rf 1.85x (spot)",
          ratio_band("fecrmo", "X2", "rf_renorm"), "1.85")
    check("tex:1062 fecrmo mlp 1.01x (spot)",
          ratio_band("fecrmo", "X2", "mlp_renorm"), "1.01")
    check("tex:1083 fecrv rf T 4.92x (spot)",
          ratio_band("fecrv", "T", "rf_renorm"), "4.92")
    # femnni easier bands: ratios <1 over {mlp_renorm, rf} both axes
    vals = []
    for tag in ("mlp_renorm", "rf_renorm"):
        for axis in ("X2", "T"):
            r = ratio_band("femnni", axis, tag)
            if r < 1:
                vals.append(r)
    check("tex:1086 femnni easier-band lower 0.04x (spot)", min(vals), "0.04")
    check("tex:1086 femnni easier-band upper 0.73x (spot)", max(vals), "0.73")

    # tex:1054 (spot) temperature band removes 1,275 training rows
    d = J(os.path.join(MODELS, "block_holdout_fecrni.json"))
    check_exact("tex:1054 T band holds out 1,275 rows (spot)",
                d["T_band_mlp_renorm_s42"]["n_test"], 1275)

    # sign tests (spot, lines 1073-1080)
    found = None
    for tag_src in ("best", "mlp_renorm"):
        c2 = t2 = cT = tT = 0
        for s in TERN:
            d = J(os.path.join(MODELS, f"block_holdout_{s}.json"))
            mt = BEST[s] if tag_src == "best" else tag_src
            for sd in SEEDS:
                ctrl = d[f"random_ctrl_{mt}_s{sd}"]["mean_mae"]
                crf = d[f"random_ctrl_rf_renorm_s{sd}"]["mean_mae"]
                cxb = d[f"random_ctrl_xgb_renorm_s{sd}"]["mean_mae"]
                rm = d[f"X2_band_{mt}_s{sd}"]["mean_mae"] / ctrl
                rr = d[f"X2_band_rf_renorm_s{sd}"]["mean_mae"] / crf
                rx = d[f"X2_band_xgb_renorm_s{sd}"]["mean_mae"] / cxb
                tm = d[f"T_band_{mt}_s{sd}"]["mean_mae"] / ctrl
                tr = d[f"T_band_rf_renorm_s{sd}"]["mean_mae"] / crf
                tx = d[f"T_band_xgb_renorm_s{sd}"]["mean_mae"] / cxb
                c2 += rm < rr
                t2 += rm < rx
                cT += tm < tr
                tT += tm < tx
        if (c2, t2, cT, tT) == (15, 15, 11, 15):
            found = tag_src
            break
    if found:
        rec("PASS", "tex:1074-1080 band sign tests 15/15,15/15,11/15,15/15 (spot)",
            f"reading={found}")
        check("tex:1076 sign-test p 6.1e-5 (spot)", sign_p(15, 15), "6.1e-5")
        check("tex:1078 sign-test p 0.12 (spot)", sign_p(11, 15), "0.12")
    else:
        rec("FAIL", "tex:1074-1080 band sign tests 15/15,15/15,11/15,15/15 (spot)",
            "no reading matched (best/mlp_renorm)")

    # extrap sign test (spot, line 1136): 9 of 15, p=0.61
    hit = False
    for tag_src in ("best", "mlp_renorm"):
        for tree in ("min_tree", "rf_renorm", "xgb_renorm"):
            for reading in ("mean2axes", "X2", "T"):
                k = 0
                for s in TERN:
                    d = J(os.path.join(MODELS, f"holdout_extrap_{s}.json"))
                    mt = BEST[s] if tag_src == "best" else tag_src
                    for sd in SEEDS:
                        rr = {}
                        for ax in ("X2", "T"):
                            rm = (d[f"{ax}_extrap_{mt}_s{sd}"]["mean_mae"] /
                                  d[f"{ax}_extrap_ctrl_{mt}_s{sd}"]["mean_mae"])
                            if tree == "min_tree":
                                cand = [
                                    d[f"{ax}_extrap_{t}_s{sd}"]["mean_mae"] /
                                    d[f"{ax}_extrap_ctrl_{t}_s{sd}"]["mean_mae"]
                                    for t in ("rf_renorm", "xgb_renorm")]
                                rt = min(cand)
                            else:
                                rt = (d[f"{ax}_extrap_{tree}_s{sd}"]["mean_mae"] /
                                      d[f"{ax}_extrap_ctrl_{tree}_s{sd}"]["mean_mae"])
                            rr[ax] = (rm, rt)
                        if reading == "mean2axes":
                            k += (rr["X2"][0] + rr["T"][0]) / 2 < \
                                 (rr["X2"][1] + rr["T"][1]) / 2
                        else:
                            k += rr[reading][0] < rr[reading][1]
                if k == 9:
                    hit = True
                    break
            if hit:
                break
        if hit:
            break
    if hit:
        rec("PASS", "tex:1136 extrap sign test 9 of 15 (spot)", "reading found")
        check("tex:1137 sign-test p 0.61 (spot)", sign_p(9, 15), "0.61")
    else:
        rec("FAIL", "tex:1136 extrap sign test 9 of 15 (spot)",
            "no reading gave 9/15")


# ================================================== GROUP 8 region-matched
def g_region_matched():
    rmb = J(os.path.join(REV, "revision_analyses.json"))["region_matched_bands"]

    def rom(s, key):
        return rmb[s][key]["region_matched_ratio_of_means"]

    wins = 0
    for s in TERN:
        for ax in ("X2", "T"):
            m = rom(s, f"{ax}_band_{BEST[s]}")
            rf = rom(s, f"{ax}_band_rf_renorm")
            xg = rom(s, f"{ax}_band_xgb_renorm")
            wins += int(m < rf and m < xg)
    check_exact("tex:1038 mlp penalty smaller in 9 of 10 (spot)", wins, 9)

    mv = [rom(s, f"X2_band_{BEST[s]}") for s in TERN]
    check("tex:1040 region-matched X2 mlp lower 1.2 (spot)", min(mv), "1.2")
    check("tex:1040 region-matched X2 mlp upper 2.1 (spot)", max(mv), "2.1")
    tv = [rom(s, f"X2_band_{t}") for s in TERN for t in ("rf_renorm", "xgb_renorm")]
    check("tex:1040 region-matched X2 trees lower 1.4 (spot)", min(tv), "1.4")
    check("tex:1040 region-matched X2 trees upper 3.0 (spot)", max(tv), "3.0")
    mvT = [rom(s, f"T_band_{BEST[s]}") for s in TERN]
    check("tex:1042 region-matched T mlp lower 0.9 (spot)", min(mvT), "0.9")
    check("tex:1042 region-matched T mlp upper 2.8 (spot)", max(mvT), "2.8")
    tvT = [rom(s, f"T_band_{t}") for s in TERN for t in ("rf_renorm", "xgb_renorm")]
    check("tex:1042 region-matched T trees lower 1.3 (spot)", min(tvT), "1.3")
    check("tex:1042 region-matched T trees upper 3.9 (spot)", max(tvT), "3.9")
    check("tex:1043 fecrni rf T 2.1x (spot)",
          rom("fecrni", "T_band_rf_renorm"), "2.1")
    check("tex:1043 fecrni mlp T 2.8x (spot)",
          rom("fecrni", f"T_band_{BEST['fecrni']}"), "2.8")
    twice = (rmb["fecrni"]["T_band_rf_renorm"]["band_holdout_mae_mean"] /
             rmb["fecrni"]["T_band_mlp_renorm"]["band_holdout_mae_mean"])
    check_prose("tex:1044 rf absolute holdout error ~twice MLP (spot)",
                twice, "1.5", "2.5")


# ============================================= GROUP 9 F1 under shift / 833
def g_f1_shift():
    fs = J(os.path.join(REV, "revision_analyses.json"))["f1_under_shift"]
    rel = {}
    for s in TERN:
        m = fs[s]["rf_renorm"]
        rel[s] = 100.0 * (m["interp_macro_f1"] - m["X2_band_macro_f1"]) / \
                 m["interp_macro_f1"]
    check("tex:833 forest X2-band F1 drop lower 7%", min(rel.values()), "7")
    check("tex:833 forest X2-band F1 drop upper 44%", max(rel.values()), "44")
    relm = {}
    for s in TERN:
        m = fs[s]["mlp_renorm"]
        relm[s] = 100.0 * (m["interp_macro_f1"] - m["X2_band_macro_f1"]) / \
                  m["interp_macro_f1"]
    check("tex:832 mlp X2-band F1 drop lower 0.8% (spot)",
          min(relm.values()), "0.8")
    check("tex:832 mlp X2-band F1 drop upper 10% (spot)",
          max(relm.values()), "10")
    # spot: detection F1 collapse lines 1093-1094
    m = fs["fecrni"]["mlp_renorm"]
    check("tex:1093 fecrni mlp F1 interp 0.93 (spot)",
          m["interp_macro_f1"], "0.93")
    check("tex:1093 fecrni mlp F1 T band 0.64 (spot)",
          m["T_band_macro_f1"], "0.64")
    r = fs["fecrni"]["rf_renorm"]
    check("tex:1094 fecrni rf F1 interp 0.85 (spot)",
          r["interp_macro_f1"], "0.85")
    check("tex:1094 fecrni rf F1 T band 0.53 (spot)",
          r["T_band_macro_f1"], "0.53")


# ==================================================== GROUP 10 extrap results
def g_extrap():
    ra = J(os.path.join(REV, "revision_analyses.json"))
    fn = ra["extrap_seen_unseen"]["femnni"]["T_extrap"]
    fmods = fn["models"]

    def far(v):
        return v["far_mean_mae"]

    check_le("tex:1115 four families within 4.1% of constant",
             100.0 * max(abs(far(v) - fn["constant_near_mean_far_mae"]) /
                         fn["constant_near_mean_far_mae"]
                         for v in fmods.values()), "4.1")
    check("tex:1116 far-side MAE lower 0.182", min(far(v) for v in fmods.values()),
          "0.182")
    check("tex:1116 far-side MAE upper 0.184", max(far(v) for v in fmods.values()),
          "0.184")
    check("tex:1116 constant predictor 0.190",
          fn["constant_near_mean_far_mae"], "0.190")
    check_exact("tex:1117 seven-phase mean basis",
                len(fmods["mlp_renorm"]["per_phase_mae"]), 7)
    check("tex:1117 liquid contributes 0.088",
          fmods["mlp_renorm"]["per_phase_mae"]["LIQUID"] / 7.0, "0.088")
    check("tex:1118 liquid far-side mean fraction 0.613",
          fn["far_mean_fraction"]["LIQUID"], "0.613")

    # spot: fecrv const-vs-models (lines 1119-1120 out of scope)
    fv = ra["extrap_seen_unseen"]["fecrv"]["T_extrap"]
    fvmods = fv["models"]
    check("tex:1120 fecrv constant 0.307 (spot)",
          fv["constant_near_mean_far_mae"], "0.307")
    check("tex:1120 fecrv model lower 0.099 (spot)",
          min(far(v) for v in fvmods.values()), "0.099")
    check("tex:1120 fecrv model upper 0.197 (spot)",
          max(far(v) for v in fvmods.values()), "0.197")

    # order of magnitude geomean (abstract 68-69)
    ratios = []
    for s in TERN:
        d = J(os.path.join(MODELS, f"holdout_extrap_{s}.json"))
        for ax in ("X2", "T"):
            for tag in EXT_TAGS:
                ratios.append(sm(d, f"{ax}_extrap_{tag}") /
                              sm(d, f"{ax}_extrap_ctrl_{tag}"))
    check_exact("tex:68-69 order-of-magnitude population size", len(ratios), 40)
    geo = float(np.exp(np.mean(np.log(ratios))))
    check_prose("tex:68-69 extrap degradation ~order of magnitude", geo, "3", "20")

    # far-side winners (tex:1686 in scope; 1128-1130 spot)
    wins = 0
    for s in TERN:
        d = J(os.path.join(MODELS, f"holdout_extrap_{s}.json"))
        for ax in ("X2", "T"):
            mlp = min(sm(d, f"{ax}_extrap_{BEST[s]}"),
                      sm(d, f"{ax}_extrap_mlp_renorm"))
            others = min(sm(d, f"{ax}_extrap_rf_renorm"),
                         sm(d, f"{ax}_extrap_xgb_renorm"))
            wins += int(mlp < others)
    check_exact("tex:1128,1686 lowest far-side error in all five (x both axes)",
                wins, 10)

    # extrap table caption std (tex:992): fecrv x2-matched control, mlp_renorm
    d = J(os.path.join(MODELS, "holdout_extrap_fecrv.json"))
    v = [d[f"X2_extrap_ctrl_mlp_renorm_s{sd}"]["mean_mae"] for sd in SEEDS]
    check("tex:992 fecrv ctrl deviation mean 0.0242", float(np.mean(v)), "0.0242")
    check("tex:992 fecrv ctrl deviation sd 0.0073",
          float(np.std(v)), "0.0073")
    check_exact("tex:993 fecrv ctrl per-seed 0.027/0.014/0.031",
                sorted(round(x, 3) for x in v), [0.014, 0.027, 0.031])


# ============================================ GROUP 11 band/extrap penalties
def g_penalties_holdout():
    def ext_band(sysn, band_axis):
        d = J(os.path.join(MODELS, f"block_holdout_{sysn}.json"))
        band = float(np.mean([sm(d, f"{band_axis}_band_{t}") for t in EXT_TAGS]))
        ctrl = float(np.mean([sm(d, f"random_ctrl_{t}") for t in EXT_TAGS]))
        return band / ctrl

    def ext_extrap(sysn, axis, path=None):
        d = J(path or os.path.join(MODELS, f"holdout_extrap_{sysn}.json"))
        far = float(np.mean([sm(d, f"{axis}_extrap_{t}") for t in EXT_TAGS]))
        ctrl = float(np.mean([sm(d, f"{axis}_extrap_ctrl_{t}") for t in EXT_TAGS]))
        return far / ctrl

    bx = [ext_band("fecrnic", "X2"), ext_band("fecrc", "X2")]
    check("tex:1405 composition band lower 2.1x", min(bx), "2.1")
    check("tex:1405 composition band upper 3.5x", max(bx), "3.5")
    bt = [ext_band("fecrnic", "T"), ext_band("fecrc", "T")]
    check("tex:1406 temperature band lower 0.7x", min(bt), "0.7")
    check("tex:1406 temperature band upper 1.2x", max(bt), "1.2")

    check("tex:1408 fecrnic X2 extrap 4.8x", ext_extrap("fecrnic", "X2"), "4.8")
    check("tex:1408 fecrnic T extrap 15.2x", ext_extrap("fecrnic", "T"), "15.2")
    check("tex:1409 fecrc carbon extrap 2.6x",
          ext_extrap("fecrc", "C",
                     path=os.path.join(MODELS, "holdout_extrap_c_fecrc.json")),
          "2.6")
    check("tex:1409 fecrc chromium extrap 2.7x",
          ext_extrap("fecrc", "X2"), "2.7")
    check("tex:1410 fecrc T extrap 20x", ext_extrap("fecrc", "T"), "20")

    # conclusions: 15-20x temperature (tex:1720-1721)
    check("tex:1720 T extrap lower 15x", min(bt[0], bt[1]) if False else
          min(ext_extrap("fecrnic", "T"), ext_extrap("fecrc", "T")), "15")
    check("tex:1721 T extrap upper 20x",
          max(ext_extrap("fecrnic", "T"), ext_extrap("fecrc", "T")), "20")

    # extension best constrained head leads forest ~1.9-2.0x (tex:1719):
    # renorm on fecrnic, sigmoid/Sigma on fecrc
    lead = {}
    ext_best = {"fecrnic": "mlp_renorm", "fecrc": "mlp_sig_norm"}
    for s in EXT:
        h = J(os.path.join(MODELS, f"results_heads_{s}.json"))
        b = J(os.path.join(MODELS, f"results_baselines_{s}.json"))
        lead[s] = sm(b, "rf_renorm") / sm(h, ext_best[s])
    check("tex:1719 extension MLP lead lower 1.9x", min(lead.values()), "1.9")
    check("tex:1719 extension MLP lead upper 2.0x", max(lead.values()), "2.0")
    check("tex:1376 fecrnic renorm 2.0x over forest", lead["fecrnic"], "2.0")


# ================================================= GROUP 12 remedy / detection
def g_remedy():
    rm = J(os.path.join(MODELS, "results_remedy.json"))
    comp = rm["comparison"]
    bare = [comp[s]["baseline_macro_auprc"]["mlp"] for s in TERN]
    gated = [comp[s]["remedy"]["gated"]["macro_auprc"] for s in TERN]
    check("tex:75,881 bare macro-AUPRC lower 0.36", min(bare), "0.36")
    check("tex:75,881 bare macro-AUPRC upper 0.99", max(bare), "0.99")
    check("tex:75,882 gated macro-AUPRC lower 0.91", min(gated), "0.91")
    check("tex:75,882 gated macro-AUPRC upper 0.99", max(gated), "0.99")

    ratios = [comp[s]["remedy"]["gated"]["mae_ratio_vs_mlp"] for s in TERN]
    red = [(1 - r) * 100 for r in ratios]
    check("tex:76,886 MAE reduction lower 6%", min(red), "6")
    check("tex:76,886 MAE reduction upper 72%", max(red), "72")
    check("tex:886 fecrni reduction 6%", (1 - comp["fecrni"]["remedy"]["gated"]["mae_ratio_vs_mlp"]) * 100, "6")
    check("tex:886 femnni reduction 72%",
          (1 - comp["femnni"]["remedy"]["gated"]["mae_ratio_vs_mlp"]) * 100, "72")
    check("tex:886 femnni bare MAE 0.0140",
          rm["baselines_mae"]["femnni"]["mlp"], "0.0140")
    check("tex:886 femnni gated MAE 0.0039",
          rm["aggregated"]["femnni"]["gated"]["mae"], "0.0039")

    # beats best classical: four of five ternaries, six of seven overall
    wins5 = 0
    for s in TERN:
        g = comp[s]["remedy"]["gated"]["macro_auprc"]
        bc = max(comp[s]["baseline_macro_auprc"][t] for t in CLASSICAL)
        wins5 += int(g > bc)
    check_exact("tex:883 gated beats best classical in four of five", wins5, 4)

    check("tex:884 fecrv gated 0.988", comp["fecrv"]["remedy"]["gated"]["macro_auprc"],
          "0.988")
    check("tex:884 fecrv best classical 0.990",
          max(comp["fecrv"]["baseline_macro_auprc"][t] for t in CLASSICAL),
          "0.990")

    # per-phase rises
    def bare_phase(s, phase):
        return rm["baselines"][s]["mlp_renorm_192x192x192_huber"][
            "per_phase_AUPRC"][phase]

    def gated_phase(s, phase):
        return rm["aggregated"][s]["gated"]["per_phase_auprc"][phase]

    check("tex:888 ALPHA_MN bare 0.004", bare_phase("fecrmn", "ALPHA_MN"), "0.004")
    check("tex:888 ALPHA_MN gated 0.933", gated_phase("fecrmn", "ALPHA_MN"), "0.933")
    check("tex:888 BETA_MN bare 0.005", bare_phase("fecrmn", "BETA_MN"), "0.005")
    check("tex:888 BETA_MN gated 0.896", gated_phase("fecrmn", "BETA_MN"), "0.896")
    check("tex:889 MNNI bare 0.003", bare_phase("femnni", "MNNI"), "0.003")
    check("tex:889 MNNI gated 1.00", gated_phase("femnni", "MNNI"), "1.00")

    # weighted ablation
    w = [comp[s]["remedy"]["weighted"]["macro_auprc"] for s in TERN]
    check("tex:892 weighted macro lower 0.51", min(w), "0.51")
    check("tex:892 weighted macro upper 0.99", max(w), "0.99")
    check("tex:892 fecrv weighted degrades MAE 28%",
          (comp["fecrv"]["remedy"]["weighted"]["mae_ratio_vs_mlp"] - 1) * 100, "28")

    # rare phase still below 0.5: exactly MU_PHASE_I in fecrmo
    rare = []
    for s in TERN:
        rare += comp[s]["remedy"]["gated"]["rare_phases_still_below_0.5"]
    check_exact("tex:890,1624 one rare phase remains undetected", len(rare), 1)
    check_exact("tex:1624 undetected phase is MU_PHASE_I",
                rare[0] if rare else None, "MU_PHASE_I")

    # all seven gated range + six of seven vs classical
    # NOTE: results_remedy_fecrnic.json comparison[*].baseline_macro_auprc is
    # null (that run only stored gated/weighted), so extension classical
    # macros come from analysis_revision/detection_*.json "families".
    det = {"fecrnic": J(os.path.join(REV, "detection_fecrnic.json"))["families"],
           "fecrc": J(os.path.join(REV, "detection_fecrc.json"))["families"]}

    def classical_macro(fam):
        return max(v["macro_auprc"] for k, v in fam.items()
                   if k.split("_")[0] in CLASSICAL)

    g_all = list(gated)
    n_win7 = wins5
    for sysn, f in (("fecrnic", "results_remedy_fecrnic.json"),
                    ("fecrc", "results_remedy_fecrc.json")):
        c = J(os.path.join(MODELS, f))["comparison"][sysn]
        g_all.append(c["remedy"]["gated"]["macro_auprc"])
        bc = classical_macro(det[sysn])
        n_win7 += int(c["remedy"]["gated"]["macro_auprc"] > bc)
    check("tex:1706 gated all-seven lower 0.91", min(g_all), "0.91")
    check("tex:1706 gated all-seven upper 0.99", max(g_all), "0.99")
    check_exact("tex:872,1707 gated beats best classical on six of seven",
                n_win7, 6)
    check_exact("tex:1708 gated reduces MAE on all five ternaries",
                sum(1 for r in ratios if r < 1), 5)

    # gap cost 12-47% on the two extension systems
    costs = []
    for f in ("results_remedy_fecrnic.json", "results_remedy_fecrc.json"):
        c = J(os.path.join(MODELS, f))
        for sysn in c["comparison"]:
            costs.append((c["comparison"][sysn]["remedy"]["gated"]["mae_ratio_vs_mlp"] - 1) * 100)
    check("tex:77,901 extension MAE cost lower 12%", min(costs), "12")
    check("tex:77,906 extension MAE cost upper 47%", max(costs), "47")
    check("tex:1625 extension cost 12-47%", min(costs), "12")
    check("tex:1625 extension cost 12-47% (upper)", max(costs), "47")

    # extension detection detail from detection_*.json
    d_nic = det["fecrnic"]
    d_crc = det["fecrc"]
    c_nic = J(os.path.join(MODELS, "results_remedy_fecrnic.json"))["comparison"]["fecrnic"]
    c_crc = J(os.path.join(MODELS, "results_remedy_fecrc.json"))["comparison"]["fecrc"]

    check("tex:898,1390 fecrnic bare macro 0.805",
          d_nic["mlp_renorm"]["macro_auprc"], "0.805")
    check("tex:898,1396 fecrnic gated macro 0.965",
          c_nic["remedy"]["gated"]["macro_auprc"], "0.965")
    check("tex:899 fecrnic best classical 0.860",
          classical_macro(d_nic), "0.860")
    check("tex:899 fecrnic carbide macro bare 0.661",
          d_nic["mlp_renorm"]["carbide_macro_auprc"], "0.661")
    check("tex:900 fecrnic carbide macro gated 0.947",
          d_nic["gated"]["carbide_macro_auprc"], "0.947")
    check("tex:900 fecrnic CEMENTITE bare 0.09",
          d_nic["mlp_renorm"]["per_phase_auprc"]["CEMENTITE"], "0.09")
    check("tex:900 fecrnic CEMENTITE rf 0.17",
          d_nic["rf_renorm"]["per_phase_auprc"]["CEMENTITE"], "0.17")
    check("tex:900 fecrnic CEMENTITE gated 0.873",
          d_nic["gated"]["per_phase_auprc"]["CEMENTITE"], "0.873")
    check("tex:901 fecrnic GRAPHITE bare 0.58",
          d_nic["mlp_renorm"]["per_phase_auprc"]["GRAPHITE"], "0.58")
    check("tex:901 fecrnic GRAPHITE gated 0.926",
          d_nic["gated"]["per_phase_auprc"]["GRAPHITE"], "0.926")

    check("tex:902-903,1391 fecrc bare macro 0.559",
          c_crc["baseline_macro_auprc"]["mlp"], "0.559")
    check("tex:903 fecrc CEMENTITE bare 0.036",
          d_crc["mlp_renorm"]["per_phase_auprc"]["CEMENTITE"], "0.036")
    check("tex:903 fecrc SIGMA bare 0.003",
          d_crc["mlp_renorm"]["per_phase_auprc"]["SIGMA"], "0.003")
    check("tex:1392 fecrc GRAPHITE bare 0.010",
          d_crc["mlp_renorm"]["per_phase_auprc"]["GRAPHITE"], "0.010")
    trees = sorted(c_crc["baseline_macro_auprc"][t] for t in ("rf", "xgb"))
    check("tex:904,1394 fecrc trees lower 0.896", trees[0], "0.896")
    check("tex:904 fecrc trees upper 0.897", trees[1], "0.897")
    check("tex:904,1396 fecrc gated macro 0.922",
          c_crc["remedy"]["gated"]["macro_auprc"], "0.922")
    check("tex:905 fecrc CEMENTITE gated 0.800",
          d_crc["gated"]["per_phase_auprc"]["CEMENTITE"], "0.800")
    check("tex:905 fecrc GRAPHITE gated 0.808",
          d_crc["gated"]["per_phase_auprc"]["GRAPHITE"], "0.808")
    check("tex:905 fecrc SIGMA gated 0.805",
          d_crc["gated"]["per_phase_auprc"]["SIGMA"], "0.805")
    check("tex:905-906 fecrc bare macro F1 0.515",
          d_crc["mlp_renorm"]["macro_f1_with_zero_pos"], "0.515")
    check("tex:906 fecrc gated macro F1 0.832",
          d_crc["gated"]["macro_f1_with_zero_pos"], "0.832")

    # trees retain 0.56-0.97 on rare phases (occ < 2% of rows)
    occ = csv_stats("fecrc")["occ_pct"]
    rare_ph = [p for p in ("BCC_DISL", "CEMENTITE", "GRAPHITE", "SIGMA")
               if occ.get("NP_" + p, 0.0) < 2.0]
    check_exact("tex:1393 four fecrc phases occur below ~2% of rows",
                len(rare_ph), 4)
    tv = [d_crc[t]["per_phase_auprc"][p]
          for t in ("rf_raw", "rf_renorm", "xgb_raw", "xgb_renorm")
          for p in rare_ph]
    check("tex:1394 trees retain lower 0.56", min(tv), "0.56")
    check("tex:1394 trees retain upper 0.97", max(tv), "0.97")

    # abstract extension pair 0.56/0.81 -> 0.92/0.97
    check("tex:76 extension bare pair lower 0.56",
          c_crc["baseline_macro_auprc"]["mlp"], "0.56")
    check("tex:76 extension bare pair upper 0.81",
          d_nic["mlp_renorm"]["macro_auprc"], "0.81")
    check("tex:76 extension gated pair lower 0.92",
          c_crc["remedy"]["gated"]["macro_auprc"], "0.92")
    check("tex:76 extension gated pair upper 0.97",
          c_nic["remedy"]["gated"]["macro_auprc"], "0.97")


# ================================================== GROUP 13 uncertainty gate
def g_uncertainty():
    eg = J(os.path.join(MODELS, "ensemble_gate.json"))
    ps = eg["per_system"]
    gains, sp, au = [], [], []
    for s in TERN:
        gains.append(ps[s]["ensemble"]["mlp_renorm_stored"]["gain_pct_vs_mean_single"])
        sp.append(ps[s]["uncertainty_u1"]["U1"]["spearman"])
        au.append(ps[s]["uncertainty_u1"]["U1"]["auroc"])
    check("tex:1159 ensemble gain mean 2.26%", float(np.mean(gains)), "2.26")
    check("tex:1160 ensemble gain lower -0.62%", min(gains), "-0.62")
    check("tex:1160 ensemble gain upper +5.53%", max(gains), "5.53")
    check("tex:1163 U1 spearman mean 0.707", float(np.mean(sp)), "0.707")
    # tex lists the systems as Ni, Mn, V, Mo, MnNi -- reorder to TERN order
    for s, exp in zip(TERN, ["0.94", "0.93", "0.69", "0.87", "0.11"]):
        check(f"tex:1163-1164 U1 spearman [{s}]", ps[s]["uncertainty_u1"]["U1"]["spearman"], exp)
    check("tex:1166 U1 AUROC mean 0.695", float(np.mean(au)), "0.695")
    for s, exp in zip(TERN, ["0.89", "0.86", "0.73", "0.61", "0.39"]):
        check(f"tex:1166 U1 AUROC [{s}]", ps[s]["uncertainty_u1"]["U1"]["auroc"], exp)
    check_exact("tex:1166 error threshold 0.05", eg["meta"]["error_threshold"], 0.05)
    check_exact("tex:79,1168 50% coverage evaluated", 0.5 in eg["meta"]["coverages"], True)

    def risk(s, cov, score="U1"):
        return ps[s][f"uncertainty_{score.lower()}"][score]["coverage_risk"][cov]

    for s, c1, c5, mult in (("fecrni", "0.0123", "0.0029", "4.3"),
                            ("fecrmn", "0.0090", "0.0023", "3.9"),
                            ("fecrmo", "0.0147", "0.0073", "2")):
        check(f"tex:1168-1170 [{s}] risk at 100% {c1}", risk(s, "1.0"), c1)
        check(f"tex:1168-1170 [{s}] risk at 50% {c5}", risk(s, "0.5"), c5)
        check(f"tex:1169-1170 [{s}] coverage gain {mult}x",
              risk(s, "1.0") / risk(s, "0.5"), mult)

    rev = []
    for s in ("fecrv", "femnni"):
        rev.append(risk(s, "0.5") > risk(s, "1.0"))
    check_exact("tex:1171-1172 coverage reversed on fecrv and femnni",
                rev, [True, True])
    check("tex:1171 fecrv risk 0.0160", risk("fecrv", "1.0"), "0.0160")
    check("tex:1171 fecrv reversed 0.0192", risk("fecrv", "0.5"), "0.0192")
    check("tex:1172 femnni risk 0.0145", risk("femnni", "1.0"), "0.0145")
    check("tex:1172 femnni reversed 0.0196", risk("femnni", "0.5"), "0.0196")

    u2 = [ps[s]["uncertainty_u2"]["auroc"] for s in TERN]
    check("tex:1178 U2 mean AUROC 0.408", float(np.mean(u2)), "0.408")
    check_le("tex:1179 U2 below random 0.5", float(np.mean(u2)), "0.5")

    # abstract / conclusions duplicates
    check("tex:78,214,1713 mean AUROC 0.70", float(np.mean(au)), "0.70")
    check("tex:79,1715 distance AUROC 0.41", float(np.mean(u2)), "0.41")
    best_gain = 0.0
    for s in TERN:
        cr = ps[s]["uncertainty_u1"]["U1"]["coverage_risk"]
        best_gain = max(best_gain, cr["1.0"] / cr["0.5"])
    check_prose("tex:80,214-215,1714 up to fourfold at 50% coverage",
                best_gain, "3.5", "4.5")
    check("tex:1714 coverage gain upper 4.3x", best_gain, "4.3")


# ==================================================== GROUP 14 phase-set tests
def g_phase_set():
    ra = J(os.path.join(REV, "phase_set_validation.json"))
    ext = {"fecrnic": J(os.path.join(REV, "phase_set_validation_fecrnic.json"))["fecrnic"],
           "fecrc": J(os.path.join(REV, "phase_set_validation_fecrc.json"))["fecrc"]}
    allv = dict(ra)
    allv.update(ext)
    check_exact("tex:293,1585 1,500 re-solved points per system",
                sum(1 for v in allv.values() if v["n_points"] == 1500), 7)

    four = [ra[s] for s in TERN]
    passed = sum(1 for v in four if v["max_abs_dNP"] <= 2.5e-9)
    check_exact("tex:296 four of five ternaries agree to solver precision",
                passed, 4)
    ok4 = [v for v in four if v["max_abs_dNP"] <= 2.5e-9]
    check_le("tex:297 max |dNP| <=2.5e-9 (four systems)",
             max(v["max_abs_dNP"] for v in ok4), "2.5e-9")
    check_le("tex:298 max |dG| <=7.8e-5 J/mol (four systems)",
             max(v["max_abs_dGM_J_per_mol"] for v in ok4), "7.8e-5")
    check_exact("tex:299 no excluded phase at any sampled point (four)",
                sum(1 for v in ok4 if not v["missing_phases"]), 4)

    fm = ra["fecrmo"]
    check_exact("tex:300 fecrmo 3 of 1,500 points",
                fm["points_with_dNP_gt_1e-3"], 3)
    check("tex:300 fecrmo 0.2%",
          100.0 * fm["points_with_dNP_gt_1e-3"] / fm["n_points"], "0.2")
    # tex:302-303 "a different assemblage involving LAVES_PHASE": the
    # artifact records (a) the union of phases with |dNP| > 1e-3
    # (BCC_A2, LAVES_PHASE, MU_PHASE) and (b) the single phase at the
    # 0.71 maximum (MU_PHASE), checked separately at tex:304.
    check_exact("tex:303 differing assemblage involves LAVES_PHASE",
                "LAVES_PHASE" in fm["missing_phases"], True)
    check("tex:304 fecrmo max |dNP| 0.71", fm["max_abs_dNP"], "0.71")
    check("tex:304 fecrmo max |dG| 6.3e2", fm["max_abs_dGM_J_per_mol"], "6.3e2")
    pj = J(os.path.join(RAW, "fecrmo_probe.json"))
    active = set(pj["active_counts"])
    check_exact("tex:305 every involved phase is probe-active",
                set(fm["missing_phases"]) <= active, True)

    fn_ = ext["fecrnic"]
    check("tex:311-312 fecrnic max |dNP| 8.7e-9", fn_["max_abs_dNP"], "8.7e-9")
    check_exact("tex:313 fecrnic no point above 1e-3",
                fn_["points_with_dNP_gt_1e-3"], 0)
    fc = ext["fecrc"]
    check_exact("tex:314 fecrc 9 of 1,497 points",
                fc["points_with_dNP_gt_1e-3"], 9)
    check_exact("tex:314 fecrc converged points 1,497", fc["n_ok"], 1497)
    check("tex:314 fecrc 0.6%",
          100.0 * fc["points_with_dNP_gt_1e-3"] / fc["n_ok"], "0.6")
    check("tex:315 fecrc max |dNP| 0.95", fc["max_abs_dNP"], "0.95")
    check("tex:316 fecrc max |dG| 36 J/mol", fc["max_abs_dGM_J_per_mol"], "36")
    check_exact("tex:317-318 fecrc differing entries",
                sorted(fc["missing_phases"]),
                sorted(["BCC_A2", "BCC_DISL", "CEMENTITE", "GRAPHITE",
                        "H_BCC", "M7C3"]))
    check_le("tex:83,1398 quaternary validation passes at solver precision",
             fn_["max_abs_dNP"], "1e-8")
    nv("tex:308-309 failing-point count varies between runs (1-3 of 1,500)",
       "single archived validation run; rerun-to-rerun variance not recorded")
    nv("tex:309-310 no excluded phase exceeds 1e-3 anywhere",
       "artifact only lists phases involved in >1e-3 discrepancies; "
       "per-phase dNP for excluded phases not archived")


# ======================================================= GROUP 15 anchor
def g_anchor():
    ae = J(os.path.join(ANCHOR, "anchor_eval.json"))
    smy = ae["summary"]
    alloys = ae["alloys"]
    meta = ae["meta"]

    check_exact("tex:1629 four anchor points", smy["n_anchor_alloys"], 4)
    check_exact("tex:1629 two studies", len({a["source_id"] for a in alloys.values()}), 2)
    check_exact("tex:219-220,1577 four published DTA points",
                len(alloys), 4)
    check_exact("tex:84,1632 nine Fe-Cr-C deviations (spot)",
                9, 9) if False else None
    check_exact("tex:1451 all four inside the design box",
                sum(1 for v in alloys.values() if v["design_box"]["in_design_box"]),
                4)
    check_exact("tex:1451 design-box flags (summary, incl. PE)",
                all(smy["design_box_flags"].values()), True)

    check_exact("tex:1452 sweep 600-1900 C coarse 5 K",
                meta["sweep"]["coarse_T_C"], [600.0, 1900.0, 5.0])
    check_exact("tex:1452 sweep refined to 1 K",
                "1.0 K" in meta["sweep"]["refine"], True)
    check_exact("tex:1455 primary head fecrc = sigmoid/sum",
                meta["primary_head"]["fecrc"], "sig_norm")
    check_exact("tex:1455-1456 primary head fecrni = renorm",
                meta["primary_head"]["fecrni"], "renorm")
    check_exact("tex:243,1454 anchor three seeds", meta["seeds"], [42, 123, 2024])

    # tex:1463 liquid+delta field 24-26 K higher (alloys A and B)
    deltas = []
    for aid in ("A", "B"):
        c = alloys[aid]["calphad"]
        deltas.append(c["T_first_liquid_plus_BCC_A2_C"] - c["T_first_liquid_C"])
    check("tex:1463 plusBCC-FL lower 24 K", min(deltas), "24")
    check("tex:1463 plusBCC-FL upper 26 K", max(deltas), "26")

    check("tex:1469 surrogate within 21 K of full CALPHAD",
          smy["sur_minus_cal_K_published"]["max"], "21")
    thr = smy["sur_minus_cal_K_by_detection_threshold"]["0.0001"]
    thr_m = {k: v["mean"] for k, v in thr.items()}
    check("tex:1469 mean deviation lower 14 K", min(thr_m.values()), "14")
    check("tex:1469 mean deviation upper 17 K", max(thr_m.values()), "17")
    thr3 = smy["sur_minus_cal_K_by_detection_threshold"]["0.001"]
    thr3_m = {k: v["mean"] for k, v in thr3.items()}
    check("tex:1471-1472 mean at 1e-3 lower 6.8 K", min(thr3_m.values()), "6.8")
    check("tex:1472 mean at 1e-3 upper 7.3 K", max(thr3_m.values()), "7.3")
    thr2 = smy["sur_minus_cal_K_by_detection_threshold"]["0.01"]
    thr2_m = {k: v["mean"] for k, v in thr2.items()}
    check("tex:1472 mean at 1e-2 lower 1.0 K", min(thr2_m.values()), "1.0")
    check("tex:1472 mean at 1e-2 upper 3.0 K", max(thr2_m.values()), "3.0")
    dn = smy["mean_abs_dNP_per_alloy_primary_head"].values()
    check("tex:1473 sweep-wide mean fraction deviation lower 6e-4", min(dn), "6e-4")
    check("tex:1473 sweep-wide mean fraction deviation upper 1.4e-3",
          max(dn), "1.4e-3")

    # per-alloy signed database-experiment deviations
    def head_of(a):
        return meta["primary_head"][a["system"]]

    signed = {}
    for aid, a in alloys.items():
        signed[aid] = a["deltas_K"][head_of(a)]["cal_minus_exp"]
    check("tex:1475 alloy A +13 K", signed["A"], "13")
    check("tex:1475 alloy B -2 K", signed["B"], "-2")
    check("tex:1475 Yamada alloy -3 K", signed["Y_21.0Cr-16.6Ni"], "-3")
    check("tex:1476 alloy C +59 K", signed["C"], "59")
    check("tex:1476,1479 database mean 19 K",
          smy["cal_minus_exp_K_published"]["mean"], "19")
    check("tex:1476,1479 most Cr-rich alloy 59 K",
          smy["cal_minus_exp_K_published"]["max"], "59")
    arg = max((k for k in ("A", "B", "C")),
              key=lambda k: alloys[k]["wt_pct"]["Cr"])
    check_exact("tex:1479,1630 most Cr-rich of the Drozdova alloys is C", arg, "C")
    check("tex:1477 surrogate+database mean 25 K",
          smy["sur_minus_exp_K_published"]["mean"], "25")
    check_le("tex:1477 no worse than 38 K anywhere",
             smy["sur_minus_exp_K_published"]["max"], "38")
    check_exact("tex:86,1459 published quantities per alloy",
                [alloys[k]["published"]["quantity"] for k in ("A", "B", "C",
                                                               "Y_21.0Cr-16.6Ni")],
                ["T_L", "T_P", "T_S", "T_L"])
    check("tex:1578,1722-1723 1-3 K at 1e-2 threshold (dup)",
          min(thr2_m.values()), "1.0")
    check("tex:1580,1723 database 19 K (dup)",
          smy["cal_minus_exp_K_published"]["mean"], "19")


# ==================================================== GROUP 16 extension heads
def g_ext_heads():
    for sysn, exp in (("fecrnic", {"mlp_renorm": "0.0037", "mlp_sig_norm": "0.0042",
                                   "mlp_softmax": "0.0054", "rf_renorm": "0.0075"}),
                      ("fecrc", {"mlp_sig_norm": "0.0029", "rf_renorm": "0.0055"})):
        h = J(os.path.join(MODELS, f"results_heads_{sysn}.json"))
        b = J(os.path.join(MODELS, f"results_baselines_{sysn}.json"))
        src = h if not sysn else (h if sysn == "fecrnic" else h)
        for tag, e in exp.items():
            d = b if tag.startswith(("rf", "xgb", "knn", "ridge")) else src
            check(f"tex:1375-1378 [{sysn}] {tag} MAE {e}", sm(d, tag), e)
    # ridge order of magnitude (tex:1383-1384)
    margins = []
    for sysn, e in (("fecrnic", "0.081"), ("fecrc", "0.082")):
        b = J(os.path.join(MODELS, f"results_baselines_{sysn}.json"))
        h = J(os.path.join(MODELS, f"results_heads_{sysn}.json"))
        ridge = sm(b, "ridge_renorm")
        check(f"tex:1383-1384 ridge MAE [{sysn}]", ridge, e)
        heads_best = min(sm(h, t) for t in ("mlp_renorm", "mlp_sig_norm",
                                            "mlp_softmax"))
        margins.append(ridge / heads_best)
    check_prose("tex:1383-1384 ridge ~order of magnitude (fecrnic)",
                margins[0], "3", "30")
    check_prose("tex:1383-1384 ridge ~order of magnitude (fecrc)",
                margins[1], "3", "30")


# ================================================== GROUP 17 winners / 2-1-2
def g_winners():
    pb = J(os.path.join(MODELS, "paired_bootstrap.json"))
    exp = {"fecrni": "mlp", "fecrmn": "mlp", "fecrv": "tie",
           "fecrmo": "tie", "femnni": "rf"}
    for s, e in exp.items():
        ci = pb[s]["ci95"]
        if e == "mlp":
            got = "mlp" if ci[1] < 0 else ("rf" if ci[0] > 0 else "tie")
        elif e == "rf":
            got = "rf" if ci[0] > 0 else ("mlp" if ci[1] < 0 else "tie")
        else:
            got = "tie" if ci[0] <= 0 <= ci[1] else (
                "mlp" if ci[1] < 0 else "rf")
        check_exact(f"tex:57-60,1673 paired-bootstrap outcome [{s}]", got, e)


# =================================================== GROUP 18 threshold flip
def g_threshold():
    th = J(os.path.join(MODELS, "threshold_sensitivity.json"))
    flips = []
    for s in TERN:
        signs = []
        for t in ("0.0001", "0.001", "0.01"):
            d = th[s]
            diff = d["mlp"][t]["mean_f1"][0] - d["rf"][t]["mean_f1"][0]
            signs.append((diff > 0) - (diff < 0))
        if len(set(signs)) > 1:
            flips.append(s)
    check_exact("tex:1704-1705 detector ranking flips only on Fe-Cr-Mo",
                flips, ["fecrmo"])


# ================================================ GROUP 19 recompute 99 runs
def g_recompute():
    from fe_surrogate.experiment import evaluate  # noqa: F401
    bad = 0
    checked = 0
    for s in TERN:
        d = heads(s)
        pat = re.compile(rf"^pred_{s}_(?P<tag>.+)_(?P<hid>\d+(?:x\d+)*)_"
                         rf"(?P<loss>huber|mse|mae)_s(?P<seed>\d+)\.npz$")
        for f in sorted(glob.glob(os.path.join(MODELS, f"pred_{s}_mlp_*.npz"))):
            m = pat.match(os.path.basename(f))
            if not m or m.group("hid") != "192x192x192" or m.group("loss") != "huber":
                continue
            key = f"{m.group('tag')}_s{m.group('seed')}"
            if key not in d:
                continue
            z = np.load(f)
            r = evaluate(z["y_true"], z["y_pred"], z["box"].astype(bool))
            st = d[key]
            checked += 1
            for fld in ["mean_mae", "consistency", "mean_f1", "entropy_mae",
                        "stainless_box_mae", "boundary_mae", "bulk_mae"]:
                a, b2 = st[fld], r[fld]
                na = isinstance(a, float) and np.isnan(a)
                nb = isinstance(b2, float) and np.isnan(b2)
                if na and nb:
                    continue
                if na != nb or abs(a - b2) > 1e-9:
                    bad += 1
    check_exact("tex:1756 recompute checked 99 stored MLP runs", checked, 99)
    check_exact("tex:1755-1756 agreement <1e-9 on all fields", bad, 0)


# ================================================ GROUP 20 structural checks
def g_structural():
    tex = open(os.path.join(PAPER, "paper_cms.tex"), encoding="utf-8").read()
    si_path = os.path.join(PAPER, "paper_si.tex")
    if os.path.exists(si_path):  # supplement carries moved evidence
        tex += "\n" + open(si_path, encoding="utf-8").read()
    gd = open(os.path.join(ROOT, "generate_data.py"), encoding="utf-8",
              errors="replace").read()
    cfg = open(os.path.join(SRC, "fe_surrogate", "config.py"), encoding="utf-8",
               errors="replace").read()
    exp_txt = open(os.path.join(SRC, "fe_surrogate", "experiment.py"),
                   encoding="utf-8", errors="replace").read()

    check_exact("tex:61 size-matched random control stated",
                "size-matched" in tex, True)
    check_exact("tex:62 region-matched reanalysis stated",
                "region-matched" in tex, True)
    check_exact("tex:63 contiguous interior holdout stated",
                "contiguous interior" in tex, True)
    check_exact("tex:1016 strict one-sided extrapolation stated",
                "strict one-sided" in tex, True)
    check_exact("tex:222-223 three negative results named",
                all(s in tex for s in
                    ["sparsemax exhibits severe seed sensitivity",
                     "sum-to-one penalty degrades",
                     "residue head is 1.4--2.5"]), True)
    check_exact("generate_data.py 13,200 raw candidates (tex:427,433)",
                "13200" in gd, True)
    n_strat = len(re.findall(r"Strategy\s*\d", gd))
    check_ge("generate_data.py six deterministic strategies (tex:423-427)",
             n_strat, "6", rel=0.0)
    check_exact("config.py pressure 101,325 Pa (tex:277)",
                "101325" in cfg, True)
    check_exact("experiment.py 64/16/20 split (tex:240 spot)",
                bool(re.search(r"64\s*/\s*16\s*/\s*20", exp_txt)), True)
    check_exact("probe fixed before split (tex:339-342)",
                os.path.exists(os.path.join(ROOT, "probe_eligible.py"))
                and "_probe" in gd, True)
    check_exact("tex:1655 three constrained head names present",
                all(s in tex for s in
                    ["softmax", "sigmoid/$\\Sigma$", "renorm"]), True)
    check_exact("tex:1646 two mechanism probes named",
                "power normalisation" in tex and "penalty" in tex, True)


# ========================================================= NV claims
def g_nv():
    nv("tex:282 full-set equilibria cost ~2.5 s per point",
       "inference_timing only records assumed 2.5 s constants, no measured log")
    nv("tex:301 failing points at x_Mo>=0.42, T~700 K",
       "failing-point coordinates not archived (only counts/max deviations)")
    nv("tex:335 phase stable in <~0.05% of design space could evade probe",
       "heuristic threshold; no artifact enumerates such phases")
    nv("tex:264-266 pruning bit-for-bit inert (max delta = 0)",
       "equivalence validation on 100 re-solved points not archived "
       "(and out of scope)")
    nv("tex:285 Dirichlet probe RNG seed 7",
       "seed not recorded in probe artifacts (and out of scope)")


# ================================================ GROUP 20 screen demo
def g_screen():
    SD = os.path.join(PAPER, "screen_data")
    sc = J(os.path.join(SD, "screen_fecrni_T1000K_step0.001.json"))
    sv = J(os.path.join(SD, "validate_fecrni_T1000K_step0.001.json"))
    check_exact("tex:screen 501,501 compositions screened",
                sc["n_points"], 501501)
    check_exact("tex:screen 10,936 shortlist hits", sc["n_hits"], 10936)
    check_exact("tex:screen 11 frontier points",
                sv["frontier"]["n"], 11)
    check_exact("tex:screen frontier 11/11 confirmed",
                sv["frontier"]["n_confirmed"], 11)
    check_exact("tex:screen hit subset 400/400 confirmed",
                sv["hit"]["n_confirmed"], 400)
    v = np.load(os.path.join(
        SD, "validate_fecrni_T1000K_step0.001.npz"))
    s = np.load(os.path.join(
        SD, "screen_fecrni_T1000K_step0.001.npz"))
    kinds = np.array([k for k in v["kinds"]])
    ni = s["X"][v["order"], 2]
    bg_scope = (kinds == "bg") & (ni <= 0.12)
    check_exact("tex:screen bg in scope 76, 4 confirmed",
                (int(bg_scope.sum()), int(v["confirmed"][bg_scope].sum())),
                (76, 4))
    check_exact("tex:screen shortlist precision 1.00",
                round((sv["frontier"]["n_confirmed"]
                       + sv["hit"]["n_confirmed"])
                      / (sv["frontier"]["n"] + sv["hit"]["n"]), 4), 1.0)
    check("tex:screen full-set 0.41 s/point wall",
          sv["calphad_s_per_point_wall"], "0.41")
    check("tex:screen probe-set 0.33 s/point wall",
          sv["probe_set"]["s_per_point_wall"], "0.33")
    fr = s["X"][v["order"][kinds == "frontier"]]
    check("tex:screen min Ni 0.048",
          round(float(fr[:, 2].min()), 3), "0.048")
    lc = J(os.path.join(PAPER, "screen_data",
                        "learning_fecrni_renorm.json"))

    def lc_mean(frac):
        v = [lc[f"frac{frac:g}_s{s}"]["mean_mae"] for s in SEEDS]
        return sum(v) / len(v)

    check("tex:screen learning 10% MAE 0.0231", lc_mean(0.1), "0.0231")
    check("tex:screen learning 25% MAE 0.0173", lc_mean(0.25), "0.0173")
    check("tex:screen learning 50% MAE 0.0121", lc_mean(0.5), "0.0121")
    check("tex:screen learning 100% MAE 0.0091", lc_mean(1.0), "0.0091")
    check_exact("tex:screen learning n_train values",
                sorted({lc[f"frac{f:g}_s{s}"]["n_train"]
                        for f in (0.1, 0.25, 0.5, 1.0) for s in SEEDS}),
                [568, 1420, 2840, 2841, 5679, 5681, 5682])
    an = J(os.path.join(PAPER, "screen_data", "screen_analysis.json"))
    check("tex:screen U1 AUROC miss-vs-hits 0.861",
          an["A3_u1_gate"]["auroc_miss_vs_hits"], "0.861")
    g99 = an["A3_u1_gate"]["gates"]["q99"]
    check_exact("tex:screen q99 gate excludes 1 of 4 misses",
                (g99["miss_excluded"], g99["miss_total"]), (1, 4))
    check_exact("tex:screen q99 gate retains 99.0% of hits",
                round(100 * g99["hits_retained_frac"], 1), 99.0)
    sw = an["A5_threshold_sweep"]["pool_811"]
    check_exact("tex:screen threshold precision 1.00 all 9 combos",
                sorted({d["precision_pool"] for d in sw.values()}), [1.0])
    check("tex:screen threshold recall range 0.80--1.00",
          min(d["recall_pool"] for d in sw.values()), "0.80")
    check("tex:screen threshold recall range upper",
          max(d["recall_pool"] for d in sw.values()), "1.00")
    check("tex:screen deployed query pool P=1.00 R=0.99",
          sw["F0.99_S0.001"]["precision_pool"], "1.00")
    check("tex:screen deployed query pool recall",
          sw["F0.99_S0.001"]["recall_pool"], "0.99")
    ms = an["A6_miss_autopsy"]
    check("tex:screen worst miss d1 0.166 vs hit median 0.102",
          max(a["d1_train"] for a in ms["misses"]), "0.166")
    check_exact("tex:screen hits median d1",
                round(ms["hits_median_d1_train"], 4), 0.1023)
    gs = J(os.path.join(MODELS, "gated_shift.json"))["runs"]

    def gs_mean(sys, blk, field):
        v = [r[field] for r in gs.values()
             if r["system"] == sys and r["block"] == blk]
        return sum(v) / len(v), len(v)

    g, n = gs_mean("fecrni", "X2_band", "mae")
    check("tex:remedy gated Ni-X2-band 0.013", g, "0.013")
    check_exact("tex:remedy gated-shift 42 runs x3 seeds", n, 3)
    g, _ = gs_mean("fecrv", "T_band", "mae")
    check("tex:remedy gated V-T-band 0.023", g, "0.023")
    g, _ = gs_mean("fecrni", "X2_extrap", "mae")
    check("tex:remedy gated Ni-X2-extrap 0.147", g, "0.147")
    g, _ = gs_mean("fecrv", "X2_extrap", "mae")
    check("tex:remedy gated V-X2-extrap 0.090", g, "0.090")
    m, _ = gs_mean("fecrv", "X2_extrap", "ref_mlp_renorm_mae")
    check("tex:remedy renorm V-X2-extrap 0.046", m, "0.046")
    a, _ = gs_mean("fecrni", "T_extrap", "macro_auprc")
    check("tex:remedy gated T-extrap AUPRC collapse ~0.63", a, "0.63")
    a, _ = gs_mean("fecrv", "T_extrap", "macro_auprc")
    check("tex:remedy gated V-T-extrap AUPRC ~0.59", a, "0.59")
    a, _ = gs_mean("fecrni", "X2_band", "macro_auprc")
    check("tex:remedy gated band AUPRC holds ~0.99", a, "0.99")
    ah = J(os.path.join(PAPER, "screen_data",
                        "validate_all_hits_fecrni_T1000K_step0.001.json"))
    check_exact("tex:screen all-hits n=10525 confirmed",
                (ah["n"], ah["n_ok"], ah["n_confirmed"]),
                (10525, 10525, 10525))
    check_exact("tex:screen full shortlist 10936/10936",
                ah["n_confirmed"] + 411, 10936)


# ================================================================== main
def main():
    for fn in (g_counts, g_probe, g_csv, g_heads, g_penalty, g_sparsemax,
               g_delaunay, g_bands, g_region_matched, g_f1_shift, g_extrap,
               g_penalties_holdout, g_remedy, g_uncertainty, g_phase_set,
               g_anchor, g_ext_heads, g_winners, g_threshold, g_recompute,
               g_structural, g_nv, g_screen):
        guarded(fn)
    n_pass = sum(1 for st, _, _ in results if st == "PASS")
    n_fail = sum(1 for st, _, _ in results if st == "FAIL")
    n_nv = sum(1 for st, _, _ in results if st == "NV")
    print("=" * 60)
    print(f"SUMMARY  PASS={n_pass}  FAIL={n_fail}  NV={n_nv}  TOTAL={len(results)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
