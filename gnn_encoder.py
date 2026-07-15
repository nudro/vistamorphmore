"""
PyG GCN encoder: two modality graphs -> 48-D each -> fused 144-D (same contract as SLIC_GAN RAG).
"""

from __future__ import annotations

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Diffusion_B_VMM_Github requires torch-geometric. Install with:\n"
        "  pip install torch-geometric"
    ) from e

from slic_features import (
    GRAPH_DESC_PER_MODALITY,
    GRAPH_FUSED_DIM,
    diffslic_dense_label_maps,
    gaussian_blur_gray_bchw,
    luma_norm_p99_bchw,
)
from graph_builder import build_superpixel_data


def _labels_diff(Sa: torch.Tensor, Sb: torch.Tensor, diff_slic: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """DiffSLIC labels only; call inside ``torch.no_grad()``."""
    B = Sa.size(0)
    cand = int(diff_slic.candidate_radius)
    x_in = torch.cat([Sa, Sb], dim=0)
    feats, p2s, _ = diff_slic(x_in)
    La = diffslic_dense_label_maps(feats[:B], p2s[:B], cand)
    Lb = diffslic_dense_label_maps(feats[B:], p2s[B:], cand)
    return La, Lb


def _labels_single_diff(Sa: torch.Tensor, diff_slic: nn.Module) -> torch.Tensor:
    """DiffSLIC labels for one luma batch ``Sa`` (B,1,H,W); call inside ``torch.no_grad()``."""
    B = Sa.size(0)
    cand = int(diff_slic.candidate_radius)
    x_in = torch.cat([Sa, Sa], dim=0)
    feats, p2s, _ = diff_slic(x_in)
    return diffslic_dense_label_maps(feats[:B], p2s[:B], cand)


def _labels_skimage(
    Sa: torch.Tensor,
    Sb: torch.Tensor,
    n_segments: int,
    compactness: float,
    slic_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from skimage.segmentation import slic

    B = Sa.size(0)
    La_list = []
    Lb_list = []
    sig = float(slic_sigma) if slic_sigma > 0 else 0.0
    kw = dict(n_segments=n_segments, compactness=compactness, sigma=sig, start_label=1)

    def _run_slic(S_np):
        try:
            return slic(S_np, **kw, channel_axis=None)
        except TypeError:
            return slic(S_np, **kw, multichannel=False)

    for i in range(B):
        Sa_np = Sa[i, 0].detach().float().cpu().numpy()
        Sb_np = Sb[i, 0].detach().float().cpu().numpy()
        la = _run_slic(Sa_np)
        lb = _run_slic(Sb_np)
        dev = Sa.device
        La_list.append(torch.from_numpy(la).to(device=dev, dtype=torch.long))
        Lb_list.append(torch.from_numpy(lb).to(device=dev, dtype=torch.long))
    La = torch.stack(La_list, dim=0)
    Lb = torch.stack(Lb_list, dim=0)
    return La, Lb


def _labels_skimage_single(
    S: torch.Tensor,
    n_segments: int,
    compactness: float,
    slic_sigma: float,
) -> torch.Tensor:
    """SLIC labels for one Sobel batch ``S`` (B,1,H,W); call inside ``torch.no_grad()``."""
    from skimage.segmentation import slic

    B = S.size(0)
    L_list: list[torch.Tensor] = []
    sig = float(slic_sigma) if slic_sigma > 0 else 0.0
    kw = dict(n_segments=n_segments, compactness=compactness, sigma=sig, start_label=1)
    dev = S.device

    def _run_slic(S_np):
        try:
            return slic(S_np, **kw, channel_axis=None)
        except TypeError:
            return slic(S_np, **kw, multichannel=False)

    for i in range(B):
        S_np = S[i, 0].detach().float().cpu().numpy()
        lab = _run_slic(S_np)
        L_list.append(torch.from_numpy(lab).to(device=dev, dtype=torch.long))
    return torch.stack(L_list, dim=0)


class SuperpixelGNNEncoder(nn.Module):
    """Shared GCN over superpixel graphs; outputs GRAPH_FUSED_DIM fused vector per batch."""

    def __init__(
        self,
        *,
        in_dim: int = 7,
        hidden: int = 64,
        num_layers: int = 3,
        out_dim: int = GRAPH_DESC_PER_MODALITY,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = float(dropout)
        self.pool = pool if pool in ("mean", "max") else "mean"
        convs: list[GCNConv] = []
        dims = [in_dim] + [hidden] * num_layers
        for i in range(num_layers):
            convs.append(GCNConv(dims[i], dims[i + 1]))
        self.convs = nn.ModuleList(convs)
        self.readout = nn.Linear(hidden, out_dim)

    def _encode_batch(self, graphs: list[Data]) -> torch.Tensor:
        """graphs length B; returns (B, out_dim)."""
        device = next(self.parameters()).device
        B = len(graphs)
        if B == 0:
            return torch.zeros(0, self.out_dim, device=device)
        batch = Batch.from_data_list(graphs)
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pool == "max":
            xg = global_max_pool(x, batch_idx)
        else:
            xg = global_mean_pool(x, batch_idx)
        return self.readout(xg)

    def forward_graphs(
        self,
        Sa: torch.Tensor,
        Sb: torch.Tensor,
        La: torch.Tensor,
        Lb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sa, Sb: (B,1,H,W); La, Lb: (B,H,W) long (detached indices).
        Returns (B, GRAPH_FUSED_DIM).
        """
        B, _, H, W = Sa.shape
        graphs_a: list[Data] = []
        graphs_b: list[Data] = []
        for i in range(B):
            graphs_a.append(build_superpixel_data(Sa[i, 0], La[i]))
            graphs_b.append(build_superpixel_data(Sb[i, 0], Lb[i]))
        g_a = self._encode_batch(graphs_a)
        g_b = self._encode_batch(graphs_b)
        fused = torch.cat([g_a, g_b, torch.abs(g_a - g_b)], dim=1)
        assert fused.shape[1] == GRAPH_FUSED_DIM, fused.shape
        return fused

    def forward_training(
        self,
        img_A: torch.Tensor,
        src: torch.Tensor,
        *,
        slic_backend: str,
        diff_slic: nn.Module | None,
        slic_n_segments: int,
        slic_compactness: float,
        slic_sigma: float,
    ) -> torch.Tensor:
        """Luma (p99-norm) maps allow grad; SLIC / skimage labels computed in ``no_grad``."""
        img_A = img_A.float()
        src = src.float()
        Sa = luma_norm_p99_bchw(img_A)
        Sb = luma_norm_p99_bchw(src)
        if slic_sigma > 0:
            Sa = gaussian_blur_gray_bchw(Sa, float(slic_sigma))
            Sb = gaussian_blur_gray_bchw(Sb, float(slic_sigma))
        with torch.no_grad():
            if slic_backend == "diff":
                if diff_slic is None:
                    raise ValueError("diff_slic required for slic_backend=diff")
                La, Lb = _labels_diff(Sa, Sb, diff_slic)
            else:
                La, Lb = _labels_skimage(Sa, Sb, slic_n_segments, slic_compactness, slic_sigma)
        return self.forward_graphs(Sa, Sb, La.detach(), Lb.detach())

    def forward_eo_modality_48(
        self,
        img: torch.Tensor,
        *,
        slic_backend: str,
        diff_slic: nn.Module | None,
        slic_n_segments: int,
        slic_compactness: float,
        slic_sigma: float,
    ) -> torch.Tensor:
        """
        Single EO branch: luma p99(img) -> SLIC labels (no grad) -> GCN readout -> (B, GRAPH_DESC_PER_MODALITY).
        """
        img = img.float()
        S = luma_norm_p99_bchw(img)
        if slic_sigma > 0:
            S = gaussian_blur_gray_bchw(S, float(slic_sigma))
        with torch.no_grad():
            if slic_backend == "diff":
                if diff_slic is None:
                    raise ValueError("diff_slic required for slic_backend=diff")
                L = _labels_single_diff(S, diff_slic)
            else:
                L = _labels_skimage_single(S, slic_n_segments, slic_compactness, slic_sigma)
        B = S.size(0)
        graphs: list[Data] = []
        for i in range(B):
            graphs.append(build_superpixel_data(S[i, 0], L[i].detach()))
        return self._encode_batch(graphs)
