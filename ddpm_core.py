"""Latent conditional DDPM: IR + EO ref + SLIC graph cond; generate visible EO from IR."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from ddpm_schedule import betas_for_alpha_bar, make_diffusion_constants
from graph_builder import GRAPH_COND_CH, graph_cond_spatial
from latent_space import EOEncoder, EODecoder, image_to_latent_spatial, latent_freq_split
from unet_eps import EpsUNet2D


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.to(device=t.device, dtype=torch.float32).gather(0, t)
    return out.view(b, *((1,) * (len(x_shape) - 1)))


class LatentConditionalDiffusion(nn.Module):
    """
    Train: ε-prediction L2 on ``z0 = E(real_A)`` with IR from ``real_B`` and EO/graph from ``real_A``.
    Sample: ``fake_A = decode(DDIM(z_T | src_B, ref_A))``.
    """

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        latent_ch: int = 8,
        T: int = 1000,
        unet_base: int = 64,
        slic_n_segments: int = 98,
        slic_compactness: float = 10.0,
        slic_sigma: float = 0.0,
        latent_hf_radius: float = 0.25,
        lambda_eps_lf: float = 1.0,
        lambda_eps_hf: float = 1.0,
        lambda_latent_hf: float = 0.0,
        latent_hf_t_max: int = -1,
    ):
        super().__init__()
        assert image_height % 4 == 0 and image_width % 4 == 0
        self.H = image_height
        self.W = image_width
        self.h_lat = image_height // 4
        self.w_lat = image_width // 4
        self.latent_ch = latent_ch
        self.T = T
        self.slic_n_segments = int(slic_n_segments)
        self.slic_compactness = float(slic_compactness)
        self.slic_sigma = float(slic_sigma)
        self.latent_hf_radius = float(latent_hf_radius)
        self.lambda_eps_lf = float(lambda_eps_lf)
        self.lambda_eps_hf = float(lambda_eps_hf)
        self.lambda_latent_hf = float(lambda_latent_hf)
        self.latent_hf_t_max = int(latent_hf_t_max)

        betas = betas_for_alpha_bar(T)
        for k, v in make_diffusion_constants(betas).items():
            self.register_buffer(k, v)

        self.encoder = EOEncoder(3, latent_ch, unet_base)
        self.decoder = EODecoder(latent_ch, 3, unet_base)
        self.eps_net = EpsUNet2D(latent_ch, graph_ch=GRAPH_COND_CH, base=unet_base)

    def _graph_lat(self, ref_a: torch.Tensor) -> torch.Tensor:
        return graph_cond_spatial(
            ref_a,
            self.h_lat,
            self.w_lat,
            n_segments=self.slic_n_segments,
            compactness=self.slic_compactness,
            slic_sigma=self.slic_sigma,
        )

    def cond_maps(self, src_b: torch.Tensor, ref_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ir_lat = image_to_latent_spatial(src_b, self.h_lat, self.w_lat)
        a_lat = image_to_latent_spatial(ref_a, self.h_lat, self.w_lat)
        graph_lat = self._graph_lat(ref_a).detach()
        return ir_lat, a_lat, graph_lat

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = _extract(self.sqrt_alphas_cumprod, t, z0.shape)
        s2 = _extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return s1 * z0 + s2 * noise

    def training_loss_l2(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z0 = self.encoder(real_a)
        b = z0.size(0)
        t = torch.randint(0, self.T, (b,), device=z0.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)
        ir_lat, a_lat, graph_lat = self.cond_maps(real_b, real_a)
        eps_pred = self.eps_net(zt, ir_lat, a_lat, graph_lat, t)

        rf = self.latent_hf_radius
        eps_low, eps_high = latent_freq_split(eps_pred, rf)
        noise_low, noise_high = latent_freq_split(noise, rf)
        loss_eps_lf = torch.mean((eps_low - noise_low) ** 2)
        loss_eps_hf = torch.mean((eps_high - noise_high) ** 2)
        loss_eps = self.lambda_eps_lf * loss_eps_lf + self.lambda_eps_hf * loss_eps_hf

        loss_z0_hf = eps_pred.new_zeros(())
        if self.lambda_latent_hf > 0.0:
            z0_pred = self.predict_x0_from_eps(zt, t, eps_pred)
            _, z0_hf = latent_freq_split(z0, rf)
            _, z0_pred_hf = latent_freq_split(z0_pred, rf)
            per = torch.mean((z0_pred_hf - z0_hf) ** 2, dim=(1, 2, 3))
            t_max = self.latent_hf_t_max if self.latent_hf_t_max >= 0 else self.T - 1
            valid = (t <= t_max).float()
            denom = valid.sum().clamp(min=1.0)
            loss_z0_hf = (per * valid).sum() / denom

        loss_total = loss_eps + self.lambda_latent_hf * loss_z0_hf
        if not return_parts:
            return loss_total
        return loss_total, {
            "eps_lf": loss_eps_lf.detach(),
            "eps_hf": loss_eps_hf.detach(),
            "z0_hf": loss_z0_hf.detach(),
        }

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        s1 = _extract(self.sqrt_recip_alphas_cumprod, t, xt.shape)
        s2 = _extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape)
        return s1 * xt - s2 * eps

    @staticmethod
    def _ddim_step(
        xt: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        ab = _extract(alphas_cumprod, t, xt.shape)
        abn = _extract(alphas_cumprod, t_next, xt.shape)
        pred_x0 = (xt - torch.sqrt(torch.clamp(1.0 - ab, min=0.0)) * eps) / torch.sqrt(ab + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
        return torch.sqrt(abn) * pred_x0 + torch.sqrt(torch.clamp(1.0 - abn, min=0.0)) * eps

    def sample_eo_ddim(
        self,
        src_b: torch.Tensor,
        ref_a: torch.Tensor,
        steps: int,
        *,
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """Differentiable DDIM: IR from ``src_b``, EO ref + graph from ``ref_a``."""
        b, _, h, w = src_b.shape
        assert h == self.H and w == self.W
        ctx = nullcontext() if enable_grad else torch.no_grad()
        with ctx:
            ir_lat, a_lat, graph_lat = self.cond_maps(src_b, ref_a)
            z = torch.randn(
                b,
                self.latent_ch,
                self.h_lat,
                self.w_lat,
                device=src_b.device,
                dtype=torch.float32,
            )
            steps = max(2, min(int(steps), self.T))
            ts = torch.linspace(self.T - 1, 0, steps=steps, device=src_b.device).long().clamp(0, self.T - 1)
            for i in range(steps - 1):
                t_cur = ts[i].expand(b)
                t_next = ts[i + 1].expand(b)
                eps = self.eps_net(z, ir_lat, a_lat, graph_lat, t_cur)
                z = self._ddim_step(z, eps, t_cur, t_next, self.alphas_cumprod)
            t0 = torch.zeros(b, dtype=torch.long, device=src_b.device)
            eps0 = self.eps_net(z, ir_lat, a_lat, graph_lat, t0)
            z0 = self.predict_x0_from_eps(z, t0, eps0)
            return self.decoder(z0, ir_lat, a_lat)

    @torch.no_grad()
    def sample(self, src_b: torch.Tensor, ref_a: torch.Tensor, steps: int) -> torch.Tensor:
        return self.sample_eo_ddim(src_b, ref_a, steps, enable_grad=False)
