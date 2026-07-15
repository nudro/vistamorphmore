"""
Differentiable visible-domain FFT loss (amplitude + phase L1), following
``Base/vistamorph_with_fft.py`` (rfft2 on luminance, magnitude + phase vs numpy script).
"""

from __future__ import annotations

import torch


def fft_inner_weighted(
    loss_amp: torch.Tensor,
    loss_phase: torch.Tensor,
    *,
    phase_scale: float,
) -> torch.Tensor:
    """Combine amp and phase L1s before applying ``--lambda_fft`` (phase often dominates / jitters)."""
    return loss_amp + phase_scale * loss_phase


def visible_fft_amp_phase_l1(fake_bchw: torch.Tensor, real_bchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        fake_bchw, real_bchw: (B, 3, H, W), normalized ~[-1, 1] as in VMM trainers.
    Returns:
        loss_amp, loss_phase (scalars), each mean L1 in frequency domain.
    """
    t_f = ((fake_bchw.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    t_r = ((real_bchw.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    g_f = 0.299 * t_f[:, 0] + 0.587 * t_f[:, 1] + 0.114 * t_f[:, 2]
    g_r = 0.299 * t_r[:, 0] + 0.587 * t_r[:, 1] + 0.114 * t_r[:, 2]
    z_f = torch.fft.rfft2(g_f, norm="ortho")
    z_r = torch.fft.rfft2(g_r, norm="ortho")
    # Real ops only: complex .abs() / .angle() can NVRTC-JIT kernels whose -arch
    # mismatches very new GPUs (e.g. sm_121) under some PyTorch builds.
    mag_f = torch.hypot(z_f.real, z_f.imag)
    mag_r = torch.hypot(z_r.real, z_r.imag)
    ph_f = torch.atan2(z_f.imag, z_f.real)
    ph_r = torch.atan2(z_r.imag, z_r.real)
    loss_amp = (mag_f - mag_r).abs().mean()
    loss_phase = (ph_f - ph_r).abs().mean()
    return loss_amp, loss_phase
