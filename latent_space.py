"""Light conv encoder / decoder: EO latent with pooled LWIR + EO reference at latent resolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EOEncoder(nn.Module):
    """``real_A`` (B,3,H,W) -> latent ``z_0`` (B, C_lat, H/f, W/f), f=4."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 8, base: int = 64):
        super().__init__()
        self.latent_ch = latent_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, latent_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EODecoder(nn.Module):
    """Decode ``z`` + pooled LWIR + pooled EO ref at latent res -> EO image ``[-1,1]`` (same cond as ε-UNet)."""

    def __init__(self, latent_ch: int = 8, out_ch: int = 3, base: int = 64):
        super().__init__()
        cin = latent_ch + 3 + 3
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 3, 1, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, ir_lat: torch.Tensor, a_lat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, ir_lat, a_lat], dim=1)
        return self.net(x)


def image_to_latent_spatial(img_bchw: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Downsample (B,3,H,W) to (B,3,h_lat,w_lat)."""
    return F.adaptive_avg_pool2d(img_bchw, (h_lat, w_lat))


def _radial_freq_masks(
    h: int,
    w: int,
    radius_frac: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial low/high masks on normalized fftfreq radius; ``radius_frac`` in (0, 1]."""
    rf = float(radius_frac)
    rf = min(max(rf, 1e-4), 1.0)
    yy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    xx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(gy * gy + gx * gx)
    r_norm = r / r.max().clamp(min=1e-8)
    low = (r_norm <= rf).to(dtype)
    high = 1.0 - low
    return low, high


def latent_freq_split(
    z_bchw: torch.Tensor,
    radius_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split latent (B,C,H,W) into complementary low/high radial bands via fft2."""
    orig_dtype = z_bchw.dtype
    z = z_bchw.float()
    _, _, h, w = z.shape
    low_m, high_m = _radial_freq_masks(h, w, radius_frac, z.device, z.dtype)
    low_m = low_m.view(1, 1, h, w)
    high_m = high_m.view(1, 1, h, w)
    z_fft = torch.fft.fft2(z, dim=(-2, -1))
    z_low = torch.fft.ifft2(z_fft * low_m, dim=(-2, -1)).real
    z_high = torch.fft.ifft2(z_fft * high_m, dim=(-2, -1)).real
    return z_low.to(orig_dtype), z_high.to(orig_dtype)
