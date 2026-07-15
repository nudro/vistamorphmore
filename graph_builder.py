"""
Build PyTorch Geometric ``Data`` graphs from Sobel magnitude ``S`` and superpixel labels ``L`` (GPU).
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def _label_to_node_map(L: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    L: (H, W) long, labels >= 1 for valid superpixels.
    Returns node_img (H, W) with values in [-1, N-1] (-1 = ignore), and N = num nodes.
    """
    max_lab = int(L.max().item())
    if max_lab < 1:
        return torch.full_like(L, -1), 0
    positives = torch.unique(L[L > 0])
    n = int(positives.numel())
    if n == 0:
        return torch.full_like(L, -1), 0
    lut = torch.full((max_lab + 1,), -1, dtype=torch.long, device=L.device)
    lut[positives] = torch.arange(n, device=L.device, dtype=torch.long)
    clipped = L.clamp(0, max_lab)
    node_img = torch.where(L > 0, lut[clipped], torch.full_like(L, -1))
    return node_img, n


def _rag_edge_index(L: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Undirected superpixel adjacency from 4-connectivity on label map L (raw labels)."""
    if num_nodes <= 0:
        return torch.zeros(2, 0, dtype=torch.long, device=L.device)
    H, W = L.shape
    max_lab = int(L.max().item())
    if max_lab < 1:
        return torch.zeros(2, 0, dtype=torch.long, device=L.device)
    lut = torch.full((max_lab + 1,), -1, dtype=torch.long, device=L.device)
    uniq = torch.unique(L[L > 0])
    lut[uniq] = torch.arange(int(uniq.numel()), device=L.device, dtype=torch.long)

    left = L[:, :-1].reshape(-1)
    right = L[:, 1:].reshape(-1)
    up = L[:-1, :].reshape(-1)
    down = L[1:, :].reshape(-1)
    a = torch.cat([left, up])
    b = torch.cat([right, down])
    m = (a > 0) & (b > 0) & (a != b)
    a = a[m].clamp(0, max_lab)
    b = b[m].clamp(0, max_lab)
    ui = lut[a]
    vi = lut[b]
    m2 = (ui >= 0) & (vi >= 0)
    ui = ui[m2]
    vi = vi[m2]
    if ui.numel() == 0:
        ei = torch.arange(num_nodes, device=L.device, dtype=torch.long).repeat(2, 1)
        return ei
    e = torch.stack([torch.cat([ui, vi]), torch.cat([vi, ui])], dim=0)
    e = torch.unique(e, dim=1)
    # self-loops for GCN stability
    sl = torch.arange(num_nodes, device=L.device, dtype=torch.long).repeat(2, 1)
    e = torch.cat([e, sl], dim=1)
    e = torch.unique(e, dim=1)
    return e


def _scatter_mean_1d(idx: torch.Tensor, val: torch.Tensor, n: int) -> torch.Tensor:
    """idx (P,) long in [0,n-1], val (P,) float -> (n,) means."""
    dev, dt = val.device, val.dtype
    sum_v = torch.zeros(n, device=dev, dtype=dt)
    cnt = torch.zeros(n, device=dev, dtype=dt)
    sum_v.index_add_(0, idx, val)
    ones = torch.ones_like(val)
    cnt.index_add_(0, idx, ones)
    return sum_v / cnt.clamp(min=1.0)


def _node_features_xy(
    S_hw: torch.Tensor,
    node_img: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """S_hw (H,W) float [0,1]; node_img (H,W) long in [-1, n-1]. Returns x (N, 7)."""
    if num_nodes <= 0:
        return torch.zeros(0, 7, device=S_hw.device, dtype=S_hw.dtype)
    H, W = S_hw.shape
    yy = torch.arange(H, device=S_hw.device, dtype=S_hw.dtype).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=S_hw.device, dtype=S_hw.dtype).view(1, W).expand(H, W)
    flat_s = S_hw.reshape(-1)
    flat_y = yy.reshape(-1) / max(H - 1, 1)
    flat_x = xx.reshape(-1) / max(W - 1, 1)
    flat_n = node_img.reshape(-1)
    m = flat_n >= 0
    idx = flat_n[m]
    mean_s = _scatter_mean_1d(idx, flat_s[m], num_nodes)
    # std
    sum_sq = torch.zeros(num_nodes, device=S_hw.device, dtype=S_hw.dtype)
    sum_sq.index_add_(0, idx, (flat_s[m] - mean_s[idx]) ** 2)
    cnt = torch.bincount(idx, minlength=num_nodes).to(S_hw.dtype).clamp(min=1.0)
    std_s = torch.sqrt((sum_sq / cnt).clamp(min=0.0))
    cy = _scatter_mean_1d(idx, flat_y[m], num_nodes)
    cx = _scatter_mean_1d(idx, flat_x[m], num_nodes)
    area = cnt / float(H * W)
    max_s = torch.full((num_nodes,), float("-inf"), device=S_hw.device, dtype=S_hw.dtype)
    min_s = torch.full((num_nodes,), float("inf"), device=S_hw.device, dtype=S_hw.dtype)
    max_s.scatter_reduce_(0, idx, flat_s[m], reduce="amax", include_self=False)
    min_s.scatter_reduce_(0, idx, flat_s[m], reduce="amin", include_self=False)
    max_s = torch.where(torch.isfinite(max_s), max_s, mean_s)
    min_s = torch.where(torch.isfinite(min_s), min_s, mean_s)
    x = torch.stack([cx, cy, area, mean_s, std_s, max_s, min_s], dim=1)
    return x


def build_superpixel_data(S_hw: torch.Tensor, L_hw: torch.Tensor) -> Data:
    """
    S_hw: (H, W) float32 Sobel magnitude in [0,1].
    L_hw: (H, W) int64 superpixel labels (>=1).
    """
    S_hw = S_hw.float()
    L_hw = L_hw.long()
    node_img, num_nodes = _label_to_node_map(L_hw)
    if num_nodes == 0:
        return Data(
            x=torch.zeros(1, 7, device=S_hw.device, dtype=S_hw.dtype),
            edge_index=torch.tensor([[0], [0]], device=S_hw.device, dtype=torch.long),
        )
    edge_index = _rag_edge_index(L_hw, num_nodes)
    x = _node_features_xy(S_hw, node_img, num_nodes)
    return Data(x=x, edge_index=edge_index)


GRAPH_COND_CH = 4


def _scatter_mean_channels(feat_chw: torch.Tensor, node_img_hw: torch.Tensor, num_nodes: int) -> torch.Tensor:
    c, _, _ = feat_chw.shape
    flat_n = node_img_hw.reshape(-1)
    m = flat_n >= 0
    idx = flat_n[m]
    if num_nodes <= 0 or idx.numel() == 0:
        return torch.zeros(max(num_nodes, 1), c, device=feat_chw.device, dtype=feat_chw.dtype)
    out = torch.zeros(num_nodes, c, device=feat_chw.device, dtype=feat_chw.dtype)
    flat_f = feat_chw.reshape(c, -1)
    for ci in range(c):
        vals = flat_f[ci][m]
        sum_v = torch.zeros(num_nodes, device=feat_chw.device, dtype=feat_chw.dtype)
        cnt = torch.zeros(num_nodes, device=feat_chw.device, dtype=feat_chw.dtype)
        sum_v.index_add_(0, idx, vals)
        cnt.index_add_(0, idx, torch.ones_like(vals))
        out[:, ci] = sum_v / cnt.clamp(min=1.0)
    return out


def build_superpixel_feature_data(feat_chw: torch.Tensor, labels_hw: torch.Tensor) -> Data:
    """feat (C,H,W), labels (H,W) int64 -> PyG Data with x (N,C)."""
    feat_chw = feat_chw.float()
    labels_hw = labels_hw.long()
    node_img, num_nodes = _label_to_node_map(labels_hw)
    c = int(feat_chw.size(0))
    if num_nodes <= 0:
        return Data(
            x=torch.zeros(1, c, device=feat_chw.device, dtype=feat_chw.dtype),
            edge_index=torch.tensor([[0], [0]], device=feat_chw.device, dtype=torch.long),
        )
    x = _scatter_mean_channels(feat_chw, node_img, num_nodes)
    edge_index = _rag_edge_index(labels_hw, num_nodes)
    return Data(x=x, edge_index=edge_index)


def luminance_01(bchw_m11: torch.Tensor) -> torch.Tensor:
    t = ((bchw_m11.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    return 0.2989 * t[:, 0:1] + 0.5870 * t[:, 1:2] + 0.1140 * t[:, 2:3]


def labels_skimage_batch(
    gray_b1hw: torch.Tensor,
    *,
    n_segments: int = 98,
    compactness: float = 10.0,
    slic_sigma: float = 0.0,
) -> torch.Tensor:
    import numpy as np
    from skimage.segmentation import slic

    b = gray_b1hw.size(0)
    sig = float(slic_sigma) if slic_sigma > 0 else 0.0
    kw = dict(n_segments=n_segments, compactness=compactness, sigma=sig, start_label=1)
    dev = gray_b1hw.device
    out: list[torch.Tensor] = []

    def _run(s_np: np.ndarray) -> np.ndarray:
        try:
            return slic(s_np, **kw, channel_axis=None)
        except TypeError:
            return slic(s_np, **kw, multichannel=False)

    for i in range(b):
        s_np = gray_b1hw[i, 0].detach().float().cpu().numpy()
        out.append(torch.from_numpy(_run(s_np)).to(device=dev, dtype=torch.long))
    return torch.stack(out, dim=0)


def boundary_overlay_rgb(gray_01_b1hw: torch.Tensor, labels_bhw: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from skimage.segmentation import mark_boundaries

    b, _, h, w = gray_01_b1hw.shape
    rows: list[torch.Tensor] = []
    for i in range(b):
        g = gray_01_b1hw[i, 0].detach().float().cpu().numpy()
        lab = labels_bhw[i].detach().cpu().numpy()
        rgb = np.stack([g, g, g], axis=-1)
        marked = mark_boundaries(rgb, lab, mode="thick").astype(np.float32)
        t = torch.from_numpy(np.transpose(marked, (2, 0, 1)))
        rows.append(t * 2.0 - 1.0)
    return torch.stack(rows, dim=0).to(device=gray_01_b1hw.device)


def _centroids_from_node_map(node_img_hw, num_nodes: int):
    import numpy as np

    h, w = node_img_hw.shape
    cent = np.zeros((max(num_nodes, 1), 2), dtype=np.float64)
    flat_n = node_img_hw.reshape(-1)
    ys, xs = np.mgrid[0:h, 0:w]
    flat_y = ys.reshape(-1).astype(np.float64)
    flat_x = xs.reshape(-1).astype(np.float64)
    m = flat_n >= 0
    idx = flat_n[m]
    for ni in range(num_nodes):
        sel = idx == ni
        if not np.any(sel):
            continue
        cent[ni, 0] = flat_y[m][sel].mean()
        cent[ni, 1] = flat_x[m][sel].mean()
    return cent


def rag_overlay_rgb(gray_01_b1hw: torch.Tensor, labels_bhw: torch.Tensor) -> torch.Tensor:
    import numpy as np
    from skimage.draw import line as sk_line

    b, _, h, w = gray_01_b1hw.shape
    rows: list[torch.Tensor] = []
    for i in range(b):
        g = gray_01_b1hw[i, 0].detach().float().cpu().numpy()
        lab_np = labels_bhw[i].detach().cpu().numpy().astype(np.int64)
        rgb = np.stack([g, g, g], axis=-1).copy()
        lab_t = torch.from_numpy(lab_np)
        _, num_nodes = _label_to_node_map(lab_t)
        if num_nodes <= 0:
            t = torch.from_numpy(np.transpose(rgb.astype(np.float32), (2, 0, 1)))
            rows.append(t * 2.0 - 1.0)
            continue
        node_img, _ = _label_to_node_map(lab_t)
        node_np = node_img.numpy()
        cent = _centroids_from_node_map(node_np, num_nodes)
        ei = _rag_edge_index(lab_t, num_nodes)
        ui = ei[0].cpu().numpy()
        vi = ei[1].cpu().numpy()
        for u_idx, v_idx in zip(ui, vi):
            if int(u_idx) == int(v_idx):
                continue
            y0, x0 = cent[int(u_idx)]
            y1, x1 = cent[int(v_idx)]
            rr, cc = sk_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            rr = np.clip(rr, 0, h - 1)
            cc = np.clip(cc, 0, w - 1)
            rgb[rr, cc, 1] = np.maximum(rgb[rr, cc, 1], 0.9)
            rgb[rr, cc, 2] = np.maximum(rgb[rr, cc, 2], 0.9)
        for ni in range(num_nodes):
            cy, cx = cent[ni]
            r0, r1 = max(0, int(cy) - 1), min(h, int(cy) + 2)
            c0, c1 = max(0, int(cx) - 1), min(w, int(cx) + 2)
            rgb[r0:r1, c0:c1, 0] = 1.0
        t = torch.from_numpy(np.transpose(rgb.astype(np.float32), (2, 0, 1)))
        rows.append(t * 2.0 - 1.0)
    return torch.stack(rows, dim=0).to(device=gray_01_b1hw.device)


def _node_raster_chw(feat_chw: torch.Tensor, labels_hw: torch.Tensor) -> torch.Tensor:
    data = build_superpixel_feature_data(feat_chw, labels_hw)
    node_img, num_nodes = _label_to_node_map(labels_hw)
    h, w = labels_hw.shape
    raster = torch.zeros(1, h, w, device=feat_chw.device, dtype=feat_chw.dtype)
    if num_nodes > 0:
        m = node_img >= 0
        node_norm = data.x.norm(dim=1).to(dtype=raster.dtype)
        raster[0, m] = node_norm[node_img[m]]
        mx = raster.max().clamp(min=1e-6)
        raster = raster / mx * 2.0 - 1.0
    return raster


@torch.no_grad()
def build_graph_cond(
    img_bchw: torch.Tensor,
    labels_bhw: torch.Tensor | None = None,
    *,
    n_segments: int = 98,
    compactness: float = 10.0,
    slic_sigma: float = 0.0,
) -> torch.Tensor:
    """4ch SLIC/RAG/centroid raster cond (B,4,H,W) in same dtype as ``img_bchw``."""
    gray = luminance_01(img_bchw)
    if labels_bhw is None:
        labels_bhw = labels_skimage_batch(
            gray,
            n_segments=n_segments,
            compactness=compactness,
            slic_sigma=slic_sigma,
        )
    bnd = boundary_overlay_rgb(gray, labels_bhw)
    rag = rag_overlay_rgb(gray, labels_bhw)
    ch0 = bnd.mean(dim=1, keepdim=True)
    ch1 = rag[:, 1:2]
    ch2 = rag[:, 0:1]
    node_rows = [_node_raster_chw(img_bchw[i], labels_bhw[i]) for i in range(img_bchw.size(0))]
    ch3 = torch.stack(node_rows, dim=0)
    return torch.cat([ch0, ch1, ch2, ch3], dim=1).to(dtype=img_bchw.dtype)


@torch.no_grad()
def graph_cond_spatial(
    img_bchw: torch.Tensor,
    h_lat: int,
    w_lat: int,
    *,
    n_segments: int = 98,
    compactness: float = 10.0,
    slic_sigma: float = 0.0,
) -> torch.Tensor:
    """SLIC graph cond downsampled to latent resolution (B,4,h_lat,w_lat)."""
    import torch.nn.functional as F

    full = build_graph_cond(
        img_bchw,
        n_segments=n_segments,
        compactness=compactness,
        slic_sigma=slic_sigma,
    )
    return F.adaptive_avg_pool2d(full.float(), (h_lat, w_lat)).to(dtype=img_bchw.dtype)
