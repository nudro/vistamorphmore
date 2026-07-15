"""Image / DiffSLIC helpers required by Diffusion_B GNN registration + struct-B loss.

Extracted from the former ``SLIC_GAN.slic_graph`` module — only symbols used by
Diffusion_B (constants, luma/Sobel helpers, DiffSLIC dense labels). Omits unused
hand-crafted RAG descriptor / ``compute_g_fused*`` / preview paths.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

GRAPH_DESC_PER_MODALITY = 48
GRAPH_FUSED_DIM = GRAPH_DESC_PER_MODALITY * 3


def linear_rgb_to_01(img_bchw: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [-1,1] or [0,1] -> [0,1]."""
    if float(img_bchw.min()) < -0.01:
        return ((img_bchw + 1.0) * 0.5).clamp(0.0, 1.0)
    return img_bchw.clamp(0.0, 1.0)


def luma_bt601_bchw(rgb_01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B,1,H,W) BT.601 luma."""
    r = rgb_01[:, 0:1]
    g = rgb_01[:, 1:2]
    b = rgb_01[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


def sobel_mag_gray_bchw(gray_b1hw: torch.Tensor) -> torch.Tensor:
    """Sobel magnitude on (B,1,H,W) float."""
    dev, dt = gray_b1hw.device, gray_b1hw.dtype
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=dev,
        dtype=dt,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    x = F.pad(gray_b1hw, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def normalize_mag_p99_bchw(mag_b1hw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-batch-element p99 normalize to [0,1]. mag: (B,1,H,W)."""
    flat = mag_b1hw.float().flatten(1)
    p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return (mag_b1hw / p99.to(mag_b1hw.dtype).clamp_min(eps)).clamp(0.0, 1.0)


def sobel_mag_norm_bchw(img_bchw: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) EO/IR tensor (~[-1,1]) -> (B,1,H,W) normalized Sobel magnitude [0,1]."""
    rgb = linear_rgb_to_01(img_bchw)
    lum = luma_bt601_bchw(rgb)
    mag = sobel_mag_gray_bchw(lum)
    return normalize_mag_p99_bchw(mag)


def luma_norm_p99_bchw(img_bchw: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) EO/IR (~[-1,1]) -> (B,1,H,W) BT.601 luma, p99-normalized to [0,1]."""
    rgb = linear_rgb_to_01(img_bchw)
    lum = luma_bt601_bchw(rgb)
    return normalize_mag_p99_bchw(lum)


def gaussian_blur_gray_bchw(gray_b1hw: torch.Tensor, sigma: float) -> torch.Tensor:
    """Light Gaussian smooth on (B,1,H,W); sigma in pixels (skimage-like)."""
    if sigma <= 0:
        return gray_b1hw
    k = max(5, 2 * int(4.0 * float(sigma) + 0.5) + 1)
    if k % 2 == 0:
        k += 1
    return gaussian_blur(gray_b1hw, kernel_size=[k, k], sigma=[sigma, sigma])


def diffslic_dense_label_maps(
    feats: torch.Tensor,
    p2s_assign: torch.Tensor,
    candidate_radius: int,
) -> torch.Tensor:
    """
    feats: (B, C, h_s, w_s); p2s_assign: (B, K, H, W) with K = (2*candidate_radius+1)**2.
    Returns (B, H, W) int64 labels in 1..(h_s*w_s) for RAG.
    """
    from diffslic_upstream import spixel_upsampling

    B, K, H, W = p2s_assign.shape
    h_s, w_s = feats.shape[-2:]
    nr = 2 * candidate_radius + 1
    if K != nr * nr:
        raise ValueError(
            f"p2s_assign K={K} != (2*r+1)^2={nr*nr} for candidate_radius={candidate_radius}"
        )
    hard = F.one_hot(p2s_assign.argmax(dim=1), num_classes=K).permute(0, 3, 1, 2).float()
    ids = (
        torch.arange(h_s * w_s, device=feats.device, dtype=feats.dtype)
        .view(1, 1, h_s, w_s)
        .expand(B, 1, h_s, w_s)
    )
    lbl = spixel_upsampling(ids, hard, candidate_radius=candidate_radius)
    return lbl.squeeze(1).long() + 1
