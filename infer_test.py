#!/usr/bin/env python3
"""
Diffusion_B inference on the test split only.

Saves per stem:
  {stem}_strip.png — [real_A | warped_B | diff_after | real_A | real_B | diff_before]
  metrics.csv      — NCC and MI/NMI before (A vs B) and after (A vs warped_B)

Example (from repo root):

  python infer_test.py --epoch 100 --max_batches 4

Results default to Test_Results/{experiment}/e{epoch}/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from repo_paths import normalize_data_root_to_tier, package_root, workspace_root

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import colormaps
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from paired_dataset import TestImageDataset
from ddpm_core import LatentConditionalDiffusion
from registration_gnn import DeformableRegistrationNetWithGNNAffineOnly
from train import load_ld_checkpoint, load_reg_checkpoint


def _default_dataset() -> str:
    return os.path.join(workspace_root(__file__), "Data", "all_pairs")


def _default_output_dir() -> str:
    return os.path.join(package_root(__file__), "outputs")


def _default_test_results_dir() -> str:
    return os.path.join(package_root(__file__), "Test_Results")


def parse_args():
    p = argparse.ArgumentParser(description="Diffusion_B inference on test split")
    p.add_argument(
        "--dataset",
        type=str,
        default=_default_dataset(),
        help="Tier root with train/ and test/ (default: Data/all_pairs)",
    )
    p.add_argument("--output_dir", type=str, default=_default_output_dir())
    p.add_argument("--experiment", type=str, default="diffusion_b_all_pairs_p1_50_reg_100")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--reg_start_epoch", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_cpu", type=int, default=4)
    p.add_argument("--gpu_num", type=int, default=0)
    p.add_argument("--img_height", type=int, default=256)
    p.add_argument("--img_width", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--phase", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--ddpm_T", type=int, default=500)
    p.add_argument("--ddim_steps", type=int, default=32)
    p.add_argument("--latent_ch", type=int, default=8)
    p.add_argument("--viz_grid_step", type=int, default=20, help="0 = no red grid on A/B panels")
    p.add_argument("--diff_cmap", type=str, default="RdBu_r")
    p.add_argument("--mi_bins", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test set")
    p.add_argument("--no_superpixel_graph", action="store_true")
    p.add_argument("--slic_segments", type=int, default=98)
    p.add_argument("--slic_compactness", type=float, default=10.0)
    p.add_argument("--slic_sigma", type=float, default=0.0)
    p.add_argument("--slic_backend", type=str, default="diff", choices=("skimage", "diff"))
    p.add_argument("--gnn_hidden", type=int, default=64)
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--gnn_dropout", type=float, default=0.0)
    p.add_argument("--graph_pool", type=str, default="mean", choices=("mean", "max"))
    p.add_argument("--diffslic_n_iter", type=int, default=5)
    p.add_argument("--diffslic_tau", type=float, default=0.01)
    p.add_argument("--diffslic_candidate_radius", type=int, default=1)
    p.add_argument("--no_diffslic_stable", action="store_true")
    p.add_argument("--diffslic_normalize", action="store_true")
    p.add_argument("--vit_patch_size", type=int, default=32)
    return p.parse_args()


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _luminance_01(rgb_01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb_01[:, 0:1], rgb_01[:, 1:2], rgb_01[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)


def _ncc_batch(ya: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """ya, yb: B1HW in [0,1]. Per-batch NCC."""
    b = ya.size(0)
    out = ya.new_zeros(b)
    for bi in range(b):
        a = ya[bi].reshape(-1)
        c = yb[bi].reshape(-1)
        a = a - a.mean()
        c = c - c.mean()
        den = a.std(unbiased=False) * c.std(unbiased=False)
        if den < 1e-8:
            out[bi] = 0.0
        else:
            out[bi] = (a * c).mean() / den
    return out


def _mi_nmi_from_joint(
    ya: torch.Tensor,
    yb: torch.Tensor,
    n_bins: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = ya.shape
    nb = int(n_bins)
    ya_i = (ya.view(b, -1) * (nb - 1)).long().clamp(0, nb - 1)
    yb_i = (yb.view(b, -1) * (nb - 1)).long().clamp(0, nb - 1)
    lin = ya_i * nb + yb_i
    mi_out = ya.new_zeros(b)
    nmi_out = ya.new_zeros(b)
    for bi in range(b):
        hflat = torch.zeros(nb * nb, device=ya.device, dtype=torch.float32)
        hflat.index_add_(0, lin[bi], torch.ones(lin.size(1), device=ya.device, dtype=torch.float32))
        p = (hflat / (hflat.sum() + eps)).view(nb, nb)
        p_a = p.sum(dim=1)
        p_b = p.sum(dim=0)
        h_a = -(p_a * torch.log(p_a + eps)).sum()
        h_b = -(p_b * torch.log(p_b + eps)).sum()
        h_ab = -(p * torch.log(p + eps)).sum()
        mi = h_a + h_b - h_ab
        mi_out[bi] = mi
        nmi_out[bi] = mi / (torch.sqrt(torch.clamp(h_a * h_b, min=eps)) + eps)
    return mi_out, nmi_out


def _draw_red_grid(img01_bchw: torch.Tensor, step: int) -> torch.Tensor:
    if step <= 0:
        return img01_bchw
    x = img01_bchw.clone()
    _, _, h, w = x.shape
    st = max(1, int(step))
    red = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    for xx in range(0, w, st):
        x[:, :, :, xx : xx + 1] = red
    for yy in range(0, h, st):
        x[:, :, yy : yy + 1, :] = red
    return x.clamp(0.0, 1.0)


def _luma_diff_rgb(
    real_a_neg1: torch.Tensor,
    other_neg1: torch.Tensor,
    cmap_name: str,
) -> torch.Tensor:
    """BCHW [-1,1] -> BCHW [0,1] RdBu_r (or other diverging cmap) of luma difference."""
    la = _luminance_01(_to_01(real_a_neg1))
    lb = _luminance_01(_to_01(other_neg1))
    d = la - lb
    b = d.size(0)
    out = torch.zeros(b, 3, d.size(2), d.size(3), device=d.device, dtype=torch.float32)
    cmap = colormaps.get_cmap(cmap_name)
    for bi in range(b):
        di = d[bi, 0].detach().cpu().numpy()
        v = float(np.quantile(np.abs(di), 0.99))
        if v < 1e-6:
            v = 1.0
        t = np.clip(di / v, -1.0, 1.0)
        rgba = cmap((t + 1.0) * 0.5)
        out[bi] = torch.from_numpy(rgba[..., :3].astype(np.float32)).permute(2, 0, 1)
    return out.to(device=real_a_neg1.device)


def _build_strip(
    real_a: torch.Tensor,
    warped_b: torch.Tensor,
    real_b: torch.Tensor,
    grid_step: int,
    cmap_name: str,
) -> torch.Tensor:
    """6-panel row: A|warped_B|diff_after | A|real_B|diff_before."""
    diff_after = _luma_diff_rgb(real_a, warped_b, cmap_name)
    diff_before = _luma_diff_rgb(real_a, real_b, cmap_name)
    a01 = _to_01(real_a)
    wb01 = _to_01(warped_b)
    b01 = _to_01(real_b)
    a_vis = _draw_red_grid(a01, grid_step)
    wb_vis = _draw_red_grid(wb01, grid_step)
    b_vis = _draw_red_grid(b01, grid_step)
    triple_after = torch.cat((a_vis, wb_vis, diff_after), dim=3)
    triple_before = torch.cat((a_vis, b_vis, diff_before), dim=3)
    return torch.cat((triple_after, triple_before), dim=3)


def main():
    opt = parse_args()
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "Diffusion_B inference requires torch-geometric when the registration graph is enabled.\n"
            "  pip install torch-geometric"
        ) from None

    tier_root = normalize_data_root_to_tier(os.path.abspath(os.path.expanduser(opt.dataset)))
    test_split_dir = os.path.join(tier_root, "test")
    if not os.path.isdir(test_split_dir):
        raise SystemExit(f"Expected test split at {test_split_dir!r}")

    opt.output_dir = os.path.abspath(os.path.expanduser(opt.output_dir))
    exp_dir = os.path.join(opt.output_dir, opt.experiment)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    if opt.save_dir:
        save_dir = os.path.abspath(os.path.expanduser(opt.save_dir))
    else:
        save_dir = os.path.join(
            _default_test_results_dir(), opt.experiment, f"e{opt.epoch}"
        )
    os.makedirs(save_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{opt.gpu_num}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(opt.gpu_num)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    test_ds = TestImageDataset(
        root=tier_root,
        transforms_=transforms_,
        mode="test",
        img_size=(opt.img_height, opt.img_width),
    )
    loader = DataLoader(
        test_ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        drop_last=False,
    )

    ld = LatentConditionalDiffusion(
        image_height=opt.img_height,
        image_width=opt.img_width,
        latent_ch=opt.latent_ch,
        T=opt.ddpm_T,
        slic_n_segments=opt.slic_segments,
        slic_compactness=opt.slic_compactness,
        slic_sigma=opt.slic_sigma,
    ).to(device)

    use_graph = not opt.no_superpixel_graph
    reg = DeformableRegistrationNetWithGNNAffineOnly(
        opt.channels,
        opt.img_height,
        opt.img_width,
        patch_size=opt.vit_patch_size,
        use_superpixel_graph=use_graph,
        slic_backend=opt.slic_backend,
        slic_n_segments=opt.slic_segments,
        slic_compactness=opt.slic_compactness,
        slic_sigma=opt.slic_sigma,
        diffslic_n_iter=opt.diffslic_n_iter,
        diffslic_tau=opt.diffslic_tau,
        diffslic_candidate_radius=opt.diffslic_candidate_radius,
        diffslic_stable=not opt.no_diffslic_stable,
        diffslic_normalize=opt.diffslic_normalize,
        gnn_hidden=opt.gnn_hidden,
        gnn_layers=opt.gnn_layers,
        gnn_dropout=opt.gnn_dropout,
        graph_pool=opt.graph_pool,
    ).to(device)

    ld_path = load_ld_checkpoint(ld, ckpt_dir, opt.epoch, opt.reg_start_epoch, device)
    reg_path = load_reg_checkpoint(reg, ckpt_dir, opt.epoch, device)
    if not ld_path:
        raise SystemExit(f"Missing ld checkpoint for epoch={opt.epoch} under {ckpt_dir!r}")
    if not reg_path:
        raise SystemExit(f"Missing reg checkpoint for epoch={opt.epoch} under {ckpt_dir!r}")
    print(f"==> loaded ld: {ld_path}")
    print(f"==> loaded reg: {reg_path}")

    ld.eval()
    reg.eval()

    metrics_path = os.path.join(save_dir, "metrics.csv")
    fieldnames = [
        "stem",
        "ncc_before",
        "ncc_after",
        "mi_before",
        "mi_after",
        "nmi_before",
        "nmi_after",
    ]
    rows: list[dict[str, float | str]] = []

    sum_ncc_b = sum_ncc_a = 0.0
    sum_mi_b = sum_mi_a = sum_nmi_b = sum_nmi_a = 0.0
    n_metrics = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if opt.max_batches > 0 and bi >= opt.max_batches:
                break
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            bsz = real_A.size(0)

            with autocast("cuda", enabled=cuda):
                fake_A1 = ld.sample(real_B, real_A, opt.ddim_steps)
                warped_B, _, _ = reg(real_A, fake_A1, real_B, training_phase=opt.phase)

            ya = _luminance_01(_to_01(real_A))
            yb_before = _luminance_01(_to_01(real_B))
            yb_after = _luminance_01(_to_01(warped_B))

            ncc_before = _ncc_batch(ya, yb_before)
            ncc_after = _ncc_batch(ya, yb_after)
            mi_before, nmi_before = _mi_nmi_from_joint(ya, yb_before, opt.mi_bins)
            mi_after, nmi_after = _mi_nmi_from_joint(ya, yb_after, opt.mi_bins)

            strip = _build_strip(real_A, warped_B, real_B, opt.viz_grid_step, opt.diff_cmap)

            for j in range(bsz):
                gidx = bi * opt.batch_size + j
                src_path = test_ds.files[gidx]
                stem = os.path.splitext(os.path.basename(src_path))[0]
                one = strip[j : j + 1]
                save_image(one, os.path.join(save_dir, f"{stem}_strip.png"), nrow=1, normalize=False)

                rows.append({
                    "stem": stem,
                    "ncc_before": float(ncc_before[j].item()),
                    "ncc_after": float(ncc_after[j].item()),
                    "mi_before": float(mi_before[j].item()),
                    "mi_after": float(mi_after[j].item()),
                    "nmi_before": float(nmi_before[j].item()),
                    "nmi_after": float(nmi_after[j].item()),
                })
                sum_ncc_b += float(ncc_before[j].item())
                sum_ncc_a += float(ncc_after[j].item())
                sum_mi_b += float(mi_before[j].item())
                sum_mi_a += float(mi_after[j].item())
                sum_nmi_b += float(nmi_before[j].item())
                sum_nmi_a += float(nmi_after[j].item())
                n_metrics += 1

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"dataset_tier={tier_root!r} split=test n={n_metrics} save_dir={save_dir!r}")
    if n_metrics > 0:
        print(
            f"mean_NCC before={sum_ncc_b / n_metrics:.6f} after={sum_ncc_a / n_metrics:.6f} "
            f"mean_MI before={sum_mi_b / n_metrics:.6f} after={sum_mi_a / n_metrics:.6f} "
            f"mean_NMI before={sum_nmi_b / n_metrics:.6f} after={sum_nmi_a / n_metrics:.6f}"
        )
    print(f"metrics: {metrics_path!r}")


if __name__ == "__main__":
    main()
