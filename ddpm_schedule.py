"""Cosine β schedule and DDPM constants (Nichol & Dhariwal)."""

from __future__ import annotations

import math

import torch


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar_fn(j: float) -> float:
        return math.cos((j + 0.008) / (num_diffusion_timesteps + 0.008) * math.pi * 0.5) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def make_diffusion_constants(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]])
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1.0),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }
