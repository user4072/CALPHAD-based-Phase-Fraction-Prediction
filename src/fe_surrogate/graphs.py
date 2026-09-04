"""Graph construction for Fe-Cr-Ni surrogates.

Follows the representation proven in the Al-Ni-Ti project (pycalphad1 common.py):
four nodes (one per element + temperature), 11 node descriptors, 3 pairwise
edge descriptors. Node/edge physics are re-targeted from Ni-base to steel
chemistry, and every physical quantity appears exactly once (the lesson from
the 14-vs-11 node feature study).

Edge mixing enthalpies are Miedema-scheme values (liquid, kJ/mol) from the
Takeuchi-Inoue tabulation (Intermetallics 18 (2010) 1779), the same source the
previous project used for Al-Ni-Ti.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

PERIODIC_TABLE = {
    "Fe": {"Z": 26, "chi": 1.83, "radius": 126, "valence": 8, "mp": 1811, "group": 8, "period": 4, "d_electrons": 6},
    "Cr": {"Z": 24, "chi": 1.66, "radius": 128, "valence": 6, "mp": 2180, "group": 6, "period": 4, "d_electrons": 5},
    "Ni": {"Z": 28, "chi": 1.91, "radius": 124, "valence": 10, "mp": 1728, "group": 10, "period": 4, "d_electrons": 8},
}

# Steel-specific alloy descriptors (replaces the Ni-base PHYSICS_4):
#   [fcc_stabilizer, bcc_stabilizer, relative_atomic_volume, sigma_former]
PHYSICS_4 = {
    "Fe": [0.0, 1.0, 1.00, 0.6],
    "Cr": [0.0, 1.0, 1.02, 1.0],
    "Ni": [1.0, 0.0, 0.98, 0.0],
}

# Miedema-scheme dHmix at equiatomic composition, liquid (kJ/mol).
MIXING_H = {("Cr", "Fe"): -1.0, ("Fe", "Ni"): -2.0, ("Cr", "Ni"): 7.0}


def build_node_11(comp, temp):
    total = sum(comp.values()) + 1e-8
    f = np.zeros((4, 11), dtype=np.float32)
    for i, e in enumerate(["Fe", "Cr", "Ni"]):
        pt = PERIODIC_TABLE[e]
        pf = PHYSICS_4[e]
        f[i, 0] = comp[e] / total
        f[i, 1] = pt["Z"] / 100
        f[i, 2] = pt["valence"] / 10
        f[i, 3] = pt["mp"] / 4000
        f[i, 4] = pt["group"] / 18
        f[i, 5] = pt["period"] / 7
        f[i, 6] = pt["d_electrons"] / 10
        f[i, 7] = pf[0]
        f[i, 8] = pf[1]
        f[i, 9] = pf[2]
        f[i, 10] = pf[3]
    t = 3
    f[t, 0] = (temp - 700) / 1000
    f[t, 1] = temp / 4000
    f[t, 3] = temp / 4000
    return f


def build_edge_3(comp):
    all_n = ["Fe", "Cr", "Ni", "T"]
    n = len(all_n)
    ne = 3
    edges, attrs = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edges.append([i, j])
            e1, e2 = all_n[i], all_n[j]
            if i < ne and j < ne:
                pair = (min(e1, e2), max(e1, e2))
                p1, p2 = PERIODIC_TABLE[e1], PERIODIC_TABLE[e2]
                mh = MIXING_H.get(pair, 0.0) * min(comp[e1], comp[e2]) / -50.0
                rm = abs(p1["radius"] - p2["radius"]) / ((p1["radius"] + p2["radius"]) / 2 + 1e-8)
                cd = abs(p1["chi"] - p2["chi"])
            else:
                mh, rm, cd = 0.0, 0.0, 0.0
            attrs.append([mh, rm, cd])
    return np.array(edges, dtype=np.int64).T, np.array(attrs, dtype=np.float32)


def build_graphs(X, Y, use_edges=True):
    """X: (n, 4) rows [Fe, Cr, Ni, T]; Y: (n, 4) phase fractions or None."""
    out = []
    for k, row in enumerate(X):
        comp = {"Fe": float(row[0]), "Cr": float(row[1]), "Ni": float(row[2])}
        temp = float(row[3])
        nf = torch.tensor(build_node_11(comp, temp), dtype=torch.float32)
        ei, ea = build_edge_3(comp)
        kw = dict(x=nf, edge_index=torch.tensor(ei, dtype=torch.long))
        if use_edges:
            kw["edge_attr"] = torch.tensor(ea, dtype=torch.float32)
        d = Data(**kw)
        if Y is not None:
            d.y = torch.tensor(Y[k].reshape(1, -1), dtype=torch.float32)
        out.append(d)
    return out


def build_flat_80(X):
    """Flattened node+edge features, the 'same information without structure'
    baseline used in the Al-Ni-Ti study."""
    F = np.zeros((len(X), 80), dtype=np.float32)
    for k, row in enumerate(X):
        comp = {"Fe": float(row[0]), "Cr": float(row[1]), "Ni": float(row[2])}
        temp = float(row[3])
        nf = build_node_11(comp, temp).flatten()
        _, ea = build_edge_3(comp)
        F[k] = np.concatenate([nf, ea.flatten()])
    return F