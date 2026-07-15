#!/usr/bin/env python3
"""
VistaMorphMore Diffusion_B: TRES + latent DDPM (phase-1 struct-B + weighted LPIPS-A).

  python train.py --data_root /path/to/tier_with_train_and_test
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from repo_paths import (
    data_root_example_flir_vmm_mild,
    format_training_run_example,
    normalize_data_root_to_tier,
    package_root,
)


def _default_output_dir() -> str:
    return os.path.join(package_root(__file__), "outputs")


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lpips_pytorch import LPIPS
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from trainer_tensorboard import (
    create_summary_writer,
    print_tensorboard_cli_help,
    resolve_tb_log_dir,
    write_startup_scalars,
)
from paired_dataset import ImageDataset, TestImageDataset
from weight_init import weights_init_normal
from ddpm_core import LatentConditionalDiffusion
from fft_loss import fft_inner_weighted, visible_fft_amp_phase_l1
from registration_gnn import DeformableRegistrationNetWithGNNAffineOnly
from slic_features import (
    linear_rgb_to_01,
    luma_bt601_bchw,
    sobel_mag_gray_bchw,
    sobel_mag_norm_bchw,
)


def _use_reg(epoch: int, reg_start_epoch: int) -> bool:
    return epoch >= int(reg_start_epoch)


def _ckpt_phase1_dir(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "phase1")


def _ckpt_phase2_dir(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "phase2")


def _ld_ckpt_path(ckpt_dir: str, epoch: int, reg_start_epoch: int) -> str:
    if epoch >= reg_start_epoch:
        return os.path.join(_ckpt_phase2_dir(ckpt_dir), f"latent_ddpm_{epoch}.pth")
    return os.path.join(_ckpt_phase1_dir(ckpt_dir), f"latent_ddpm_{epoch}.pth")


def _reg_ckpt_path(ckpt_dir: str, epoch: int) -> str:
    return os.path.join(_ckpt_phase2_dir(ckpt_dir), f"registration_{epoch}.pth")


def _legacy_ld_ckpt_path(ckpt_dir: str, epoch: int) -> str:
    return os.path.join(ckpt_dir, f"latent_ddpm_{epoch}.pth")


def _legacy_reg_ckpt_path(ckpt_dir: str, epoch: int) -> str:
    return os.path.join(ckpt_dir, f"registration_{epoch}.pth")


def load_ld_checkpoint(
    ld: LatentConditionalDiffusion,
    ckpt_dir: str,
    epoch: int,
    reg_start_epoch: int,
    device: torch.device,
) -> str | None:
    """Load ``ld`` from phase1 or phase2 checkpoint; returns path loaded or None."""
    candidates: list[str] = []
    if epoch >= reg_start_epoch:
        candidates.append(_ld_ckpt_path(ckpt_dir, epoch, reg_start_epoch))
        candidates.append(_ld_ckpt_path(ckpt_dir, reg_start_epoch, reg_start_epoch))
    else:
        candidates.append(_ld_ckpt_path(ckpt_dir, epoch, reg_start_epoch))
    candidates.append(_legacy_ld_ckpt_path(ckpt_dir, epoch))
    for path in candidates:
        if os.path.isfile(path):
            ld.load_state_dict(torch.load(path, map_location=device), strict=False)
            return path
    return None


def load_reg_checkpoint(
    reg: DeformableRegistrationNetWithGNNAffineOnly,
    ckpt_dir: str,
    epoch: int,
    device: torch.device,
) -> str | None:
    for path in (_reg_ckpt_path(ckpt_dir, epoch), _legacy_reg_ckpt_path(ckpt_dir, epoch)):
        if os.path.isfile(path):
            reg.load_state_dict(torch.load(path, map_location=device), strict=False)
            return path
    return None


def _checkpoint_epochs(n_epochs: int, interval: int, reg_start_epoch: int) -> list[int]:
    if interval <= 0:
        return []
    out: set[int] = set()
    for ep in range(1, n_epochs + 1):
        if ep % interval == 0 or ep == n_epochs or ep == reg_start_epoch:
            out.add(ep)
    return sorted(out)


def _should_save_checkpoint(ep_done: int, n_epochs: int, interval: int, reg_start_epoch: int) -> bool:
    if interval <= 0:
        return False
    return (
        ep_done % interval == 0
        or ep_done == n_epochs
        or ep_done == reg_start_epoch
    )


def _struct_b_loss(fake_a1: torch.Tensor, real_b: torch.Tensor) -> torch.Tensor:
    """L1 on p99-norm Sobel luma maps; B target + p99 scale stopgrad, grad via fake_a1 only."""
    with torch.no_grad():
        s_b = sobel_mag_norm_bchw(real_b.float())
    rgb = linear_rgb_to_01(fake_a1.float())
    lum = luma_bt601_bchw(rgb)
    mag = sobel_mag_gray_bchw(lum)
    with torch.no_grad():
        flat = mag.flatten(1)
        p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    s_fake = (mag / p99.to(mag.dtype).clamp_min(1e-6)).clamp(0.0, 1.0)
    return F.l1_loss(s_fake, s_b)


def parse_args():
    _epilog = format_training_run_example("train.py", "diffusion_b_run", __file__)
    p = argparse.ArgumentParser(
        description="TRES + latent DDPM: real_B->fake_A1, reg, warped_B->fake_A2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog,
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One Phase-1 + one Phase-2 step; assert finite losses and exit",
    )
    p.add_argument("--output_dir", type=str, default=_default_output_dir())
    p.add_argument("--experiment", type=str, default="diffusion_run")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--n_epochs", type=int, default=210)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_reg", type=float, default=None, help="Reg LR; default 0.1 * --lr")
    p.add_argument("--b1", type=float, default=0.5)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--n_cpu", type=int, default=8)
    p.add_argument("--img_height", type=int, default=256)
    p.add_argument("--img_width", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--sample_interval", type=int, default=5)
    p.add_argument("--checkpoint_interval", type=int, default=5)
    p.add_argument(
        "--reg_start_epoch",
        type=int,
        default=100,
        help="Epoch to enable STN registration + fake_A2 TRES cycle (0..reg_start_epoch-1: DDPM+struct-B+lpips-A only)",
    )
    p.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm for ld (and reg when active); 0=off")
    p.add_argument("--gpu_num", type=int, default=0)
    p.add_argument("--ddpm_T", type=int, default=500)
    p.add_argument("--ddim_steps", type=int, default=32, help="DDIM steps for unrolled training + preview")
    p.add_argument("--latent_ch", type=int, default=8)
    p.add_argument("--lambda_ddpm", type=float, default=1.0)
    p.add_argument("--lambda_lpips_a", type=float, default=0.2, help="Phase-1 LPIPS(fake_A1, real_A)")
    p.add_argument("--lambda_struct_b", type=float, default=1.0, help="Phase-1 L1 Sobel luma(fake_A1, real_B)")
    p.add_argument("--lambda_lpips", type=float, default=1.0, help="Phase-2 LPIPS(fake_A2, real_A)")
    p.add_argument(
        "--latent_hf_radius",
        type=float,
        default=0.25,
        help="Radial cutoff (Nyquist fraction) for latent LF/HF split in DDPM target",
    )
    p.add_argument(
        "--lambda_eps_lf",
        type=float,
        default=1.0,
        help="Weight on low-freq epsilon branch in dual-branch DDPM loss",
    )
    p.add_argument(
        "--lambda_eps_hf",
        type=float,
        default=1.0,
        help="Weight on high-freq epsilon branch (>1 emphasizes structure in eps target)",
    )
    p.add_argument(
        "--lambda_latent_hf",
        type=float,
        default=0.0,
        help="Auxiliary MSE on high-freq z0_pred vs z0 (via predict_x0_from_eps)",
    )
    p.add_argument(
        "--latent_hf_t_max",
        type=int,
        default=-1,
        help="Apply z0 HF aux only for t<=this; default -1 uses all timesteps",
    )
    p.add_argument(
        "--phase",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="1=GAN-free LPIPS only; 2+=FFT; 3=ViT+EO GNN spatial channels in reg",
    )
    p.add_argument("--lambda_fft", type=float, default=0.25, help="FFT fake_A2 vs real_A when phase>=2")
    p.add_argument("--lambda_fft_phase_scale", type=float, default=0.35)
    p.add_argument("--lambda_eo_graph", type=float, default=0.0)
    p.add_argument("--eo_graph_loss", type=str, default="l2", choices=("l2", "huber", "cosine"))
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensorboard_dir", type=str, default="")
    p.add_argument("--no_tensorboard", action="store_true")
    p.add_argument("--tb_log_interval", type=int, default=1)
    return p.parse_args()


def _run_smoke(opt) -> None:
    """One Phase-1 + one Phase-2-style step; assert finite losses."""
    torch.manual_seed(opt.seed)
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{opt.gpu_num}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(opt.gpu_num)

    ld = LatentConditionalDiffusion(
        image_height=opt.img_height,
        image_width=opt.img_width,
        latent_ch=opt.latent_ch,
        T=opt.ddpm_T,
        slic_n_segments=opt.slic_segments,
        slic_compactness=opt.slic_compactness,
        slic_sigma=opt.slic_sigma,
        latent_hf_radius=opt.latent_hf_radius,
        lambda_eps_lf=opt.lambda_eps_lf,
        lambda_eps_hf=opt.lambda_eps_hf,
        lambda_latent_hf=opt.lambda_latent_hf,
        latent_hf_t_max=opt.latent_hf_t_max,
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
    ld.encoder.apply(weights_init_normal)
    ld.decoder.apply(weights_init_normal)
    ld.eps_net.apply(weights_init_normal)
    reg.localization.apply(weights_init_normal)
    reg.fc_loc.apply(weights_init_normal)

    bs = min(2, max(1, int(opt.batch_size)))
    real_A = real_B = None
    try:
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dl = DataLoader(
            ImageDataset(
                root=opt.data_root,
                transforms_=transforms_,
                mode="train",
                img_size=(opt.img_height, opt.img_width),
            ),
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        batch = next(iter(dl))
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
    except Exception as e:
        print(f"[smoke] data load failed ({e}); using random tensors")
        real_A = torch.randn(bs, opt.channels, opt.img_height, opt.img_width, device=device)
        real_B = torch.randn(bs, opt.channels, opt.img_height, opt.img_width, device=device)

    criterion_lpips = LPIPS(net_type="vgg", version="0.1").to(device)
    ddim_steps = min(4, max(2, int(opt.ddim_steps)))
    scaler = GradScaler("cuda", enabled=cuda)

    # Phase-1
    opt_p1 = torch.optim.Adam(ld.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    opt_p1.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=cuda):
        loss_ddpm = ld.training_loss_l2(real_A, real_B)
    with autocast("cuda", enabled=False):
        fake_A1 = ld.sample_eo_ddim(real_B, real_A, ddim_steps, enable_grad=True)
        loss_lpips_a = criterion_lpips(fake_A1.float(), real_A.float()).mean()
        loss_struct_b = _struct_b_loss(fake_A1, real_B)
        loss_p1 = (
            float(opt.lambda_ddpm) * loss_ddpm
            + float(opt.lambda_lpips_a) * loss_lpips_a
            + float(opt.lambda_struct_b) * loss_struct_b
        )
    if not torch.isfinite(loss_p1):
        raise SystemExit(f"smoke FAIL: non-finite phase1 loss={loss_p1}")
    scaler.scale(loss_p1).backward()
    scaler.unscale_(opt_p1)
    torch.nn.utils.clip_grad_norm_(ld.parameters(), max_norm=1.0)
    scaler.step(opt_p1)
    scaler.update()

    # Phase-2-style (ld frozen inference + reg train)
    for p in ld.parameters():
        p.requires_grad_(False)
    ld.eval()
    for p in reg.parameters():
        p.requires_grad_(True)
    opt_p2 = torch.optim.Adam(reg.parameters(), lr=opt.lr * 0.1, betas=(opt.b1, opt.b2))
    opt_p2.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=False):
        fake_A1 = ld.sample_eo_ddim(real_B, real_A, ddim_steps, enable_grad=True)
        warped_B, _, _ = reg(real_A, fake_A1, real_B, training_phase=opt.phase)
        fake_A2 = ld.sample_eo_ddim(warped_B, real_A, ddim_steps, enable_grad=True)
        loss_lpips = criterion_lpips(fake_A2.float(), real_A.float()).mean()
        loss_p2 = float(opt.lambda_lpips) * loss_lpips
        if opt.phase >= 2 and opt.lambda_fft > 0:
            loss_fft_amp, loss_fft_phase = visible_fft_amp_phase_l1(fake_A2, real_A)
            loss_fft = fft_inner_weighted(
                loss_fft_amp, loss_fft_phase, phase_scale=opt.lambda_fft_phase_scale
            )
            loss_p2 = loss_p2 + float(opt.lambda_fft) * loss_fft
    if not torch.isfinite(loss_p2):
        raise SystemExit(f"smoke FAIL: non-finite phase2 loss={loss_p2}")
    scaler.scale(loss_p2).backward()
    scaler.unscale_(opt_p2)
    torch.nn.utils.clip_grad_norm_(reg.parameters(), max_norm=1.0)
    scaler.step(opt_p2)
    scaler.update()
    print(f"smoke OK phase1={float(loss_p1.detach()):.6f} phase2={float(loss_p2.detach()):.6f}")


def main():
    opt = parse_args()
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "Diffusion_B_VMM_Github requires torch-geometric.\n  pip install torch-geometric"
        ) from None

    _raw_data_root = opt.data_root
    opt.data_root = normalize_data_root_to_tier(opt.data_root)
    opt.output_dir = os.path.abspath(os.path.expanduser(opt.output_dir))
    if not os.path.isdir(opt.data_root):
        hint = ""
        if "/path/to" in _raw_data_root or "/path/to" in opt.data_root:
            hint = f"\n  Use an actual tier root (e.g. {data_root_example_flir_vmm_mild(__file__)!r})."
        raise SystemExit(f"--data_root is not a directory: {opt.data_root!r}{hint}")

    if opt.smoke:
        _run_smoke(opt)
        return

    lr_reg = float(opt.lr_reg if opt.lr_reg is not None else opt.lr * 0.1)
    torch.manual_seed(opt.seed)

    exp_dir = os.path.join(opt.output_dir, opt.experiment)
    img_dir = os.path.join(exp_dir, "images")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpt_p1_dir = _ckpt_phase1_dir(ckpt_dir)
    ckpt_p2_dir = _ckpt_phase2_dir(ckpt_dir)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ckpt_p1_dir, exist_ok=True)
    os.makedirs(ckpt_p2_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{opt.gpu_num}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(opt.gpu_num)

    ld = LatentConditionalDiffusion(
        image_height=opt.img_height,
        image_width=opt.img_width,
        latent_ch=opt.latent_ch,
        T=opt.ddpm_T,
        slic_n_segments=opt.slic_segments,
        slic_compactness=opt.slic_compactness,
        slic_sigma=opt.slic_sigma,
        latent_hf_radius=opt.latent_hf_radius,
        lambda_eps_lf=opt.lambda_eps_lf,
        lambda_eps_hf=opt.lambda_eps_hf,
        lambda_latent_hf=opt.lambda_latent_hf,
        latent_hf_t_max=opt.latent_hf_t_max,
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

    if opt.epoch > 0:
        ld_loaded = load_ld_checkpoint(ld, ckpt_dir, opt.epoch, opt.reg_start_epoch, device)
        reg_loaded = None
        if opt.epoch >= opt.reg_start_epoch:
            reg_loaded = load_reg_checkpoint(reg, ckpt_dir, opt.epoch, device)
        if ld_loaded:
            print(f"==> loaded ld: {ld_loaded}")
        if reg_loaded:
            print(f"==> loaded reg: {reg_loaded}")
        if not ld_loaded:
            print(f"[WARN] no ld checkpoint for epoch={opt.epoch}; initializing weights")
            ld.encoder.apply(weights_init_normal)
            ld.decoder.apply(weights_init_normal)
            ld.eps_net.apply(weights_init_normal)
        if opt.epoch >= opt.reg_start_epoch and not reg_loaded:
            reg.localization.apply(weights_init_normal)
            reg.fc_loc.apply(weights_init_normal)
    else:
        ld.encoder.apply(weights_init_normal)
        ld.decoder.apply(weights_init_normal)
        ld.eps_net.apply(weights_init_normal)
        reg.localization.apply(weights_init_normal)
        reg.fc_loc.apply(weights_init_normal)

    criterion_lpips = LPIPS(net_type="vgg", version="0.1").to(device)
    z_fft = torch.tensor(0.0, device=device)

    reg_in_optimizer = opt.epoch >= opt.reg_start_epoch
    for p in reg.parameters():
        p.requires_grad_(reg_in_optimizer)

    if reg_in_optimizer:
        # Phase 2: ld is inference-only (frozen); train reg only.
        for p in ld.parameters():
            p.requires_grad_(False)
        ld.eval()
        optimizer = torch.optim.Adam(reg.parameters(), lr=lr_reg, betas=(opt.b1, opt.b2))
    else:
        optimizer = torch.optim.Adam(ld.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scaler = GradScaler("cuda", enabled=cuda)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    img_size = (opt.img_height, opt.img_width)
    dataloader = DataLoader(
        ImageDataset(root=opt.data_root, transforms_=transforms_, mode="train", img_size=img_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        TestImageDataset(root=opt.data_root, transforms_=transforms_, mode="test", img_size=img_size),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    f = open(os.path.join(log_dir, f"{opt.experiment}.txt"), "a+", encoding="utf-8")
    tb_run_dir, tb_dir_overridden = resolve_tb_log_dir(opt.output_dir, opt.experiment, opt.tensorboard_dir)
    tb_writer: SummaryWriter | None = None
    if not opt.no_tensorboard:
        tb_writer = create_summary_writer(tb_run_dir)
        cfg = (
            f"TRES+LDM: phased DDPM pretrain then ViT+GNN reg cycle\n"
            f"ddpm_T={opt.ddpm_T} ddim_steps={opt.ddim_steps} latent_ch={opt.latent_ch} "
            f"lambda_ddpm={opt.lambda_ddpm} lambda_lpips_a={opt.lambda_lpips_a} "
            f"lambda_struct_b={opt.lambda_struct_b} lambda_lpips={opt.lambda_lpips} "
            f"reg_start_epoch={opt.reg_start_epoch} checkpoint_interval={opt.checkpoint_interval} "
            f"ckpt_phase1={ckpt_p1_dir} ckpt_phase2={ckpt_p2_dir} "
            f"phase={opt.phase}\n"
            f"data_root={opt.data_root}\nexperiment={opt.experiment}\n"
        )
        tb_writer.add_text("config/run", cfg, 0)
        write_startup_scalars(tb_writer, learning_rate=opt.lr)
        print_tensorboard_cli_help(tb_run_dir, opt.output_dir, tb_dir_overridden)

    ckpt_epochs = _checkpoint_epochs(opt.n_epochs, opt.checkpoint_interval, opt.reg_start_epoch)
    if ckpt_epochs:
        print(f"==> checkpoint saves at epochs: {ckpt_epochs}")

    prev_time = time.time()

    def sample_images(batches_done: int, epoch: int):
        ld.eval()
        reg.eval()
        use_reg_now = _use_reg(epoch, opt.reg_start_epoch)
        try:
            batch = next(iter(test_dataloader))
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            with torch.no_grad(), autocast("cuda", enabled=cuda):
                fake_A1 = ld.sample(real_B, real_A, opt.ddim_steps)
                ab_pair = torch.cat((real_A, real_B), dim=-1)
                if use_reg_now:
                    warped_B, _, _ = reg(real_A, fake_A1, real_B, training_phase=opt.phase)
                    fake_A2 = ld.sample(warped_B, real_A, opt.ddim_steps)
                    aw_pair = torch.cat((real_A, warped_B), dim=-1)
                    row = torch.cat((fake_A2, ab_pair, aw_pair), dim=-1)
                else:
                    row = torch.cat((fake_A1, ab_pair), dim=-1)
            save_image(row, os.path.join(img_dir, f"{batches_done}.png"), nrow=1, normalize=True)
            if tb_writer is not None:
                vis = row.detach().float().cpu().squeeze(0).clamp(-1.0, 1.0)
                tb_writer.add_image("preview/strip", (vis + 1.0) * 0.5, global_step=batches_done)
        finally:
            if _use_reg(epoch, opt.reg_start_epoch):
                reg.train()
            else:
                ld.train()

    try:
        for epoch in range(opt.epoch, opt.n_epochs):
            use_reg_now = _use_reg(epoch, opt.reg_start_epoch)
            if use_reg_now and not reg_in_optimizer:
                # Phase 2: freeze ld (inference-only) and train reg only.
                for p in ld.parameters():
                    p.requires_grad_(False)
                ld.eval()
                for p in reg.parameters():
                    p.requires_grad_(True)
                optimizer = torch.optim.Adam(reg.parameters(), lr=lr_reg, betas=(opt.b1, opt.b2))
                reg_in_optimizer = True
                p1_final = os.path.join(ckpt_p1_dir, f"latent_ddpm_{opt.reg_start_epoch}.pth")
                p2_seed = os.path.join(ckpt_p2_dir, f"latent_ddpm_{opt.reg_start_epoch}.pth")
                if os.path.isfile(p1_final) and not os.path.isfile(p2_seed):
                    shutil.copy2(p1_final, p2_seed)
                    print(f"\n==> reg_start_epoch={opt.reg_start_epoch}: seeded phase2 ld from {p1_final}")
                elif not os.path.isfile(p2_seed):
                    torch.save(ld.state_dict(), p2_seed)
                    print(f"\n==> reg_start_epoch={opt.reg_start_epoch}: saved phase2 ld seed {p2_seed}")
                print(f"\n==> reg_start_epoch={opt.reg_start_epoch}: enabling STN + TRES cycle")

            curriculum = 2.0 if use_reg_now else 1.0

            for i, batch in enumerate(dataloader):
                batches_done = epoch * len(dataloader) + i
                real_A = batch["A"].to(device, non_blocking=True)
                real_B = batch["B"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=cuda):
                    loss_ddpm, ddpm_parts = ld.training_loss_l2(real_A, real_B, return_parts=True)

                with autocast("cuda", enabled=False):
                    fake_A1 = ld.sample_eo_ddim(real_B, real_A, opt.ddim_steps, enable_grad=True)

                    if use_reg_now:
                        warped_B, _, _ = reg(real_A, fake_A1, real_B, training_phase=opt.phase)
                        fake_A2 = ld.sample_eo_ddim(warped_B, real_A, opt.ddim_steps, enable_grad=True)
                        loss_lpips_a = z_fft
                        loss_lpips = criterion_lpips(fake_A2.float(), real_A.float()).mean()
                        loss_struct_b = z_fft

                        use_fft = opt.phase >= 2 and opt.lambda_fft > 0
                        if use_fft:
                            loss_fft_amp, loss_fft_phase = visible_fft_amp_phase_l1(fake_A2, real_A)
                            loss_fft = fft_inner_weighted(
                                loss_fft_amp,
                                loss_fft_phase,
                                phase_scale=opt.lambda_fft_phase_scale,
                            )
                        else:
                            loss_fft = z_fft

                        if opt.lambda_eo_graph > 0:
                            loss_eo_graph = reg.eo_readout_alignment_loss(
                                real_A, fake_A2, loss_type=opt.eo_graph_loss
                            )
                        else:
                            loss_eo_graph = z_fft

                        loss_G = (
                            float(opt.lambda_ddpm) * loss_ddpm
                            + float(opt.lambda_lpips) * loss_lpips
                            + float(opt.lambda_fft) * loss_fft
                            + float(opt.lambda_eo_graph) * loss_eo_graph
                        )
                    else:
                        loss_lpips_a = criterion_lpips(fake_A1.float(), real_A.float()).mean()
                        loss_lpips = loss_lpips_a
                        loss_struct_b = _struct_b_loss(fake_A1, real_B)
                        loss_fft = z_fft
                        loss_eo_graph = z_fft
                        loss_G = (
                            float(opt.lambda_ddpm) * loss_ddpm
                            + float(opt.lambda_lpips_a) * loss_lpips_a
                            + float(opt.lambda_struct_b) * loss_struct_b
                        )

                if not torch.isfinite(loss_G):
                    sys.stdout.write(
                        f"\n[WARN] non-finite loss at epoch={epoch} batch={i}; skipping step\n"
                    )
                    sys.stdout.flush()
                    f.write(f"epoch={epoch} batch={i} nan_skip=1\n")
                    if tb_writer is not None:
                        tb_writer.add_scalar("meta/nan_skip", 1.0, batches_done)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss_G).backward()
                ld_gn = None
                clip_params = reg.parameters() if use_reg_now else ld.parameters()
                if opt.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    ld_gn = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(opt.grad_clip))
                else:
                    scaler.unscale_(optimizer)
                    ld_gn = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float("inf"))
                scaler.step(optimizer)
                scaler.update()

                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()
                if use_reg_now:
                    status = (
                        f"\r[E {epoch}/{opt.n_epochs - 1}] [C {int(curriculum)}] [B {i + 1}/{len(dataloader)}] "
                        f"G:{loss_G.item():.4f} DDPM:{loss_ddpm.item():.4f} "
                        f"LPIPS:{loss_lpips.item():.4f} FFT:{loss_fft.item():.4f} ETA:{time_left}"
                    )
                else:
                    status = (
                        f"\r[E {epoch}/{opt.n_epochs - 1}] [C {int(curriculum)}] [B {i + 1}/{len(dataloader)}] "
                        f"G:{loss_G.item():.4f} DDPM:{loss_ddpm.item():.4f} "
                        f"LPIPS_A:{loss_lpips_a.item():.4f} StructB:{loss_struct_b.item():.4f} ETA:{time_left}"
                    )
                sys.stdout.write(status)
                sys.stdout.flush()
                if use_reg_now:
                    f.write(
                        f"epoch={epoch} batch={i} curriculum={int(curriculum)} "
                        f"G={loss_G.item():.6f} ddpm={loss_ddpm.item():.6f} "
                        f"lpips={loss_lpips.item():.6f} fft={loss_fft.item():.6f}\n"
                    )
                else:
                    f.write(
                        f"epoch={epoch} batch={i} curriculum={int(curriculum)} "
                        f"G={loss_G.item():.6f} ddpm={loss_ddpm.item():.6f} "
                        f"lpips_a={loss_lpips_a.item():.6f} struct_b={loss_struct_b.item():.6f}\n"
                    )

                if tb_writer is not None and opt.tb_log_interval > 0 and batches_done % opt.tb_log_interval == 0:
                    tb_writer.add_scalar("loss/G_total", loss_G.item(), batches_done)
                    tb_writer.add_scalar("loss/ddpm_l2", loss_ddpm.item(), batches_done)
                    tb_writer.add_scalar("loss/ddpm_eps_lf", ddpm_parts["eps_lf"].item(), batches_done)
                    tb_writer.add_scalar("loss/ddpm_eps_hf", ddpm_parts["eps_hf"].item(), batches_done)
                    tb_writer.add_scalar("loss/ddpm_z0_hf", ddpm_parts["z0_hf"].item(), batches_done)
                    if use_reg_now:
                        tb_writer.add_scalar("loss/lpips", loss_lpips.item(), batches_done)
                        tb_writer.add_scalar("loss/fft", loss_fft.item(), batches_done)
                    else:
                        tb_writer.add_scalar("loss/lpips_a", loss_lpips_a.item(), batches_done)
                        tb_writer.add_scalar("loss/struct_b", loss_struct_b.item(), batches_done)
                    tb_writer.add_scalar("loss/eo_graph", loss_eo_graph.item(), batches_done)
                    tb_writer.add_scalar("meta/training_phase", float(opt.phase), batches_done)
                    tb_writer.add_scalar("meta/curriculum_phase", curriculum, batches_done)
                    if ld_gn is not None:
                        tb_writer.add_scalar("meta/grad_norm_ld", float(ld_gn), batches_done)
                    tb_writer.add_scalar("meta/amp_scale", float(scaler.get_scale()), batches_done)

                if batches_done % opt.sample_interval == 0:
                    sample_images(batches_done, epoch)

            ep_done = epoch + 1
            if _should_save_checkpoint(
                ep_done, opt.n_epochs, opt.checkpoint_interval, opt.reg_start_epoch
            ):
                if ep_done <= opt.reg_start_epoch:
                    p1_path = os.path.join(ckpt_p1_dir, f"latent_ddpm_{ep_done}.pth")
                    torch.save(ld.state_dict(), p1_path)
                    print(f"\n==> saved phase1 ld: {p1_path}")
                if ep_done >= opt.reg_start_epoch:
                    p2_ld = os.path.join(ckpt_p2_dir, f"latent_ddpm_{ep_done}.pth")
                    torch.save(ld.state_dict(), p2_ld)
                    p2_reg = os.path.join(ckpt_p2_dir, f"registration_{ep_done}.pth")
                    torch.save(reg.state_dict(), p2_reg)
                    print(f"\n==> saved phase2 ld+reg: {p2_ld} {p2_reg}")

    finally:
        f.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
