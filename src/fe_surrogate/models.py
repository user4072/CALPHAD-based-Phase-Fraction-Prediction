"""Models for the Fe-Cr-Ni surrogate: GATv2 (graph), MLP (flat-80), XGBoost.

GATv2 and MLP reproduce the Al-Ni-Ti project's architectures at matched
capacity (GATv2 ~192-dim hidden, MLP-80 256 hidden -> comparable parameter
counts), because the pycalphad1 revision showed capacity-matched comparison
is the only defensible one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2Net(nn.Module):
    """4-node graph GATv2, 3 conv layers, 6 heads, mean pool, per-phase heads."""

    def __init__(self, node_dim=11, edge_dim=3, hid=192, heads=6, layers=3, n_phases=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(node_dim, hid)
        self.convs = nn.ModuleList(
            [GATv2Conv(hid, hid // heads, heads=heads, dropout=dropout, residual=True,
                       edge_dim=edge_dim if edge_dim > 0 else None)
             for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.heads_out = nn.ModuleList(
            [nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
             for _ in range(n_phases)])
        self.output_scales = nn.Parameter(torch.ones(n_phases))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        h = self.proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index, edge_attr=edge_attr))
            h = F.silu(h)
            h = self.dropout(h)
        h_global = global_mean_pool(h, batch)
        preds = []
        for i, head in enumerate(self.heads_out):
            scale = torch.sigmoid(self.output_scales[i])
            preds.append(head(h_global).sigmoid() * scale)
        return torch.cat(preds, dim=-1)


class MLP80(nn.Module):
    """Flat MLP on the 80-dim flattened features (the 'structure removed' arm)."""

    def __init__(self, input_dim=80, hidden=256, n_phases=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
        )
        self.heads_out = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
             for _ in range(n_phases)])
        self.output_scales = nn.Parameter(torch.ones(n_phases))

    def forward(self, x):
        h = self.net(x)
        preds = []
        for i, head in enumerate(self.heads_out):
            scale = torch.sigmoid(self.output_scales[i])
            preds.append(head(h).sigmoid() * scale)
        return torch.cat(preds, dim=-1)