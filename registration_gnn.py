"""
Affine-only STN + PyG GNN graph embedding (no dense flow UNet — removes residual waviness).

Phase 3 (``training_phase >= 3``): ViT sees ``concat(real_A, fake_A1, spatial(graph_real_A))`` where
``graph_real_A`` is the EO-only GNN readout (48-D) upsampled to H×W planes before the Kornia ViT.
``fc_loc`` still concatenates ViT tokens with the **fused** pair graph (144-D) as in phases 1–2.

Bundled for **VMM/Diffusion** (same weights as TRES registration).
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.contrib as K

from slic_features import GRAPH_DESC_PER_MODALITY, GRAPH_FUSED_DIM

from gnn_encoder import SuperpixelGNNEncoder


class LocalizerVITGraphSpatial(nn.Module):
    """Kornia ViT on ``concat(EO, cycle_visible)`` plus bilinear-upsampled spatial channels from EO-only GNN (48-D)."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        patch_size: int,
        *,
        graph_extra_spatial: int = 4,
    ):
        super().__init__()
        channels, self.h, self.w = img_shape
        self.patch_size = int(patch_size)
        if self.h % self.patch_size != 0 or self.w % self.patch_size != 0:
            raise ValueError(
                f"image size ({self.h}, {self.w}) must be divisible by patch_size={self.patch_size} for ViT"
            )
        self.ph = self.h // self.patch_size
        self.pw = self.w // self.patch_size
        self.base_ch = channels * 2
        self.graph_extra = int(graph_extra_spatial)
        in_ch = self.base_ch + self.graph_extra
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=self.patch_size, in_channels=in_ch)
        )
        self.graph_patch_proj = nn.Linear(GRAPH_DESC_PER_MODALITY, self.graph_extra * self.ph * self.pw)

    def forward(self, x_pair: torch.Tensor, g_eo: torch.Tensor | None) -> torch.Tensor:
        """``x_pair`` (B,6,H,W); ``g_eo`` (B,48) or None → zeros for extra planes."""
        B, _, H, W = x_pair.shape
        if g_eo is None:
            extra = x_pair.new_zeros(B, self.graph_extra, H, W)
        else:
            t = self.graph_patch_proj(g_eo.to(dtype=x_pair.dtype))
            t = t.view(B, self.graph_extra, self.ph, self.pw)
            extra = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=True)
        return self.vit(torch.cat([x_pair, extra], dim=1))


def kornia_vit_flat_dim(
    channels: int,
    height: int,
    width: int,
    patch_size: int,
    *,
    graph_extra_spatial: int = 4,
) -> int:
    loc = LocalizerVITGraphSpatial((channels, height, width), patch_size, graph_extra_spatial=graph_extra_spatial)
    loc.eval()
    with torch.no_grad():
        x = torch.zeros(1, channels * 2, height, width)
        y = loc(x, None)
    return int(y.reshape(1, -1).numel())


class DeformableRegistrationNetWithGNNAffineOnly(nn.Module):
    """
    ViT + GNN -> affine theta; ``warped = grid_sample(src, affine_grid(theta))`` only.
    Returns ``flow`` as zeros ``(B,2,H,W)`` for API compatibility with trainers that unpack three values.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        patch_size: int = 64,
        *,
        use_superpixel_graph: bool = True,
        slic_backend: str = "diff",
        slic_n_segments: int = 98,
        slic_compactness: float = 10.0,
        slic_sigma: float = 0.0,
        diffslic_n_iter: int = 5,
        diffslic_tau: float = 0.01,
        diffslic_candidate_radius: int = 1,
        diffslic_stable: bool = True,
        diffslic_normalize: bool = False,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.0,
        graph_pool: str = "mean",
        vit_graph_extra_spatial: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.use_superpixel_graph = use_superpixel_graph
        self.slic_backend = slic_backend if slic_backend in ("skimage", "diff") else "diff"
        self.slic_n_segments = slic_n_segments
        self.slic_compactness = slic_compactness
        self.slic_sigma = slic_sigma
        self.graph_dim = GRAPH_FUSED_DIM
        self.vit_graph_extra_spatial = int(vit_graph_extra_spatial)

        self.diff_slic: nn.Module | None = None
        if use_superpixel_graph and self.slic_backend == "diff":
            from diffslic_upstream import DiffSLIC

            self.diff_slic = DiffSLIC(
                n_spixels=slic_n_segments,
                n_iter=diffslic_n_iter,
                tau=diffslic_tau,
                candidate_radius=diffslic_candidate_radius,
                normalize=diffslic_normalize,
                stable=diffslic_stable,
            )

        self.graph_encoder: SuperpixelGNNEncoder | None = None
        if use_superpixel_graph:
            self.graph_encoder = SuperpixelGNNEncoder(
                hidden=gnn_hidden,
                num_layers=gnn_layers,
                dropout=gnn_dropout,
                pool=graph_pool,
            )

        input_shape = (channels, height, width)
        self.localization = LocalizerVITGraphSpatial(
            input_shape,
            patch_size=patch_size,
            graph_extra_spatial=self.vit_graph_extra_spatial,
        )
        vit_flat = kornia_vit_flat_dim(
            channels, height, width, patch_size, graph_extra_spatial=self.vit_graph_extra_spatial
        )
        self.vit_flat_dim = vit_flat
        in_lin = vit_flat + self.graph_dim
        self.fc_loc = nn.Sequential(
            nn.Linear(in_lin, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 3 * 2),
        )
        self.fc_loc[2].bias.data.zero_()
        self.register_buffer("identity_flat", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), persistent=False)

    def eo_readout_alignment_loss(
        self,
        real_A: torch.Tensor,
        fake_eo: torch.Tensor,
        *,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        """Align GNN EO readout of ``fake_eo`` (phase-2 ``fake_A2``) to ``real_A``."""
        if not self.use_superpixel_graph or self.graph_encoder is None:
            return real_A.new_tensor(0.0)
        ge = self.graph_encoder
        kw = dict(
            slic_backend=self.slic_backend,
            diff_slic=self.diff_slic,
            slic_n_segments=self.slic_n_segments,
            slic_compactness=self.slic_compactness,
            slic_sigma=self.slic_sigma,
        )
        z_real = ge.forward_eo_modality_48(real_A, **kw)
        z_fake = ge.forward_eo_modality_48(fake_eo, **kw)
        z_t = z_real.detach()
        lt = (loss_type or "l2").lower()
        if lt == "huber":
            return F.smooth_l1_loss(z_fake, z_t)
        if lt == "cosine":
            return (1.0 - F.cosine_similarity(z_fake, z_t, dim=1, eps=1e-8)).mean()
        return F.mse_loss(z_fake, z_t)

    def _graph_kw(self) -> dict:
        return dict(
            slic_backend=self.slic_backend,
            diff_slic=self.diff_slic,
            slic_n_segments=self.slic_n_segments,
            slic_compactness=self.slic_compactness,
            slic_sigma=self.slic_sigma,
        )

    def stn_phi(
        self,
        img_input: torch.Tensor,
        img_A: torch.Tensor,
        img_b_graph: torch.Tensor,
        *,
        training_phase: int = 3,
    ) -> torch.Tensor:
        g_eo_vit: torch.Tensor | None = None
        if (
            training_phase >= 3
            and self.use_superpixel_graph
            and self.graph_encoder is not None
        ):
            g_eo_vit = self.graph_encoder.forward_eo_modality_48(img_A, **self._graph_kw()).to(
                device=img_input.device, dtype=img_input.dtype
            )

        xs = self.localization(img_input, g_eo_vit)
        xs = xs.reshape(xs.size(0), -1)
        if self.use_superpixel_graph and self.graph_encoder is not None:
            g = self.graph_encoder.forward_training(
                img_A,
                img_b_graph,
                **self._graph_kw(),
            ).to(device=xs.device, dtype=xs.dtype)
        else:
            g = torch.zeros(xs.size(0), GRAPH_FUSED_DIM, device=xs.device, dtype=xs.dtype)
        h = torch.cat([xs.float(), g], dim=1)
        theta = self.fc_loc(h)
        return theta.view(-1, 2, 3)

    def forward(
        self,
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        src: torch.Tensor,
        *,
        training_phase: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        img_input = torch.cat((img_A, img_B), dim=1)
        dtheta = self.stn_phi(img_input, img_A, img_B, training_phase=training_phase)
        Bsz = img_A.size(0)
        flat = dtheta.reshape(Bsz, 6) + self.identity_flat.to(dtype=dtheta.dtype, device=dtheta.device).unsqueeze(0)
        theta = flat.view(Bsz, 2, 3)

        grid_a = F.affine_grid(theta, src.size(), align_corners=True)
        warped = F.grid_sample(src, grid_a, mode="bicubic", padding_mode="border", align_corners=True)
        B, _, H, W = warped.shape
        flow = warped.new_zeros(B, 2, H, W)
        return warped, flow, None
