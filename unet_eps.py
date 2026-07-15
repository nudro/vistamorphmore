"""Small conv UNet for ε-prediction on latent ``z_t``, conditioned on pooled IR + timestep."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def _film_on_gn(h: torch.Tensor, gb: torch.Tensor | None) -> torch.Tensor:
    if gb is None:
        return h
    gb = gb.to(dtype=h.dtype)
    g, b = gb.chunk(2, dim=1)
    return h * (1.0 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)


class IrGlobalFilmConditioner(nn.Module):
    """Global FiLM from spatially pooled IR (``ir_lat``)."""

    def __init__(self, base: int, hidden: int = 128):
        super().__init__()
        b2 = base * 2
        self.shared = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {
                "down1": nn.Linear(hidden, 2 * b2),
                "mid1_n1": nn.Linear(hidden, 2 * b2),
                "mid1_n2": nn.Linear(hidden, 2 * b2),
                "mid2_n1": nn.Linear(hidden, 2 * b2),
                "mid2_n2": nn.Linear(hidden, 2 * b2),
                "up1": nn.Linear(hidden, 2 * base),
                "out1_n1": nn.Linear(hidden, 2 * base),
                "out1_n2": nn.Linear(hidden, 2 * base),
                "out_norm": nn.Linear(hidden, 2 * base),
            }
        )
        for lin in self.heads.values():
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, ir_lat: torch.Tensor) -> dict[str, torch.Tensor]:
        v = F.adaptive_avg_pool2d(ir_lat, 1).flatten(1).float()
        s = self.shared(v)
        return {k: lin(s) for k, lin in self.heads.items()}


class ResBlock(nn.Module):
    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        gn = max(1, min(8, ch))
        self.norm1 = nn.GroupNorm(gn, ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(gn, ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.time_proj = nn.Linear(time_dim, ch)

    def forward(
        self,
        x: torch.Tensor,
        te: torch.Tensor,
        film_n1: torch.Tensor | None = None,
        film_n2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = _film_on_gn(self.norm1(x), film_n1)
        h = self.conv1(self.act1(h))
        h = h + self.time_proj(te)[:, :, None, None]
        h = _film_on_gn(self.norm2(h), film_n2)
        h = self.conv2(self.act2(h))
        return x + h


class DownStemFiLM(nn.Module):
    def __init__(self, base: int):
        super().__init__()
        self.conv = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.norm = nn.GroupNorm(8, base * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, film_gb: torch.Tensor | None) -> torch.Tensor:
        x = self.conv(x)
        h = _film_on_gn(self.norm(x), film_gb)
        return self.act(h)


class UpStemFiLM(nn.Module):
    def __init__(self, base: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.norm = nn.GroupNorm(8, base)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, film_gb: torch.Tensor | None) -> torch.Tensor:
        x = self.deconv(x)
        h = _film_on_gn(self.norm(x), film_gb)
        return self.act(h)


class EpsUNet2D(nn.Module):
    """Predict ε on ``z_t`` with ``concat(z_t, ir_lat, a_lat, graph_lat)`` and global IR FiLM."""

    def __init__(
        self,
        latent_ch: int,
        time_dim: int = 128,
        base: int = 64,
        *,
        graph_ch: int = 4,
        ir_film_hidden: int = 128,
    ):
        super().__init__()
        self.graph_ch = int(graph_ch)
        in_ch = latent_ch + 3 + 3 + self.graph_ch
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.ir_film = IrGlobalFilmConditioner(base, hidden=ir_film_hidden)
        self.in_conv = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.down1 = DownStemFiLM(base)
        self.mid1 = ResBlock(base * 2, time_dim)
        self.mid2 = ResBlock(base * 2, time_dim)
        self.up1 = UpStemFiLM(base)
        self.out1 = ResBlock(base, time_dim)
        gn = max(1, min(8, base))
        self.out_norm = nn.GroupNorm(gn, base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base, latent_ch, 3, 1, 1)

    def forward(
        self,
        z_t: torch.Tensor,
        ir_lat: torch.Tensor,
        a_lat: torch.Tensor,
        graph_lat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        te = self.time_mlp(t)
        film = self.ir_film(ir_lat)
        x = torch.cat([z_t, ir_lat, a_lat, graph_lat], dim=1)
        x = self.in_conv(x)
        x = self.down1(x, film["down1"])
        x = self.mid2(
            self.mid1(
                x,
                te,
                film_n1=film["mid1_n1"],
                film_n2=film["mid1_n2"],
            ),
            te,
            film_n1=film["mid2_n1"],
            film_n2=film["mid2_n2"],
        )
        x = self.up1(x, film["up1"])
        x = self.out1(
            x,
            te,
            film_n1=film["out1_n1"],
            film_n2=film["out1_n2"],
        )
        h = _film_on_gn(self.out_norm(x), film["out_norm"])
        return self.out_conv(self.out_act(h))
