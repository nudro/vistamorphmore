#!/usr/bin/env python3
"""
Compose per-stem comparison PNGs: one file per test image, each row = one model checkpoint.

Default rows (top → bottom):
  Diffusion_B e80, e90, e100; Diffusion e60; REG_WFCG e210

Example (from repo root):

  python VMM/Diffusion_B_VMM_Github/compose_strip_montage.py

  python VMM/Diffusion_B_VMM_Github/compose_strip_montage.py \\
    --stems FLIR_08863 FLIR_08864 FLIR_08909 FLIR_08902
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from repo_paths import repo_root_containing_vmm

DEFAULT_GOOD = ["FLIR_08863", "FLIR_08864", "FLIR_08909", "FLIR_08902"]
DEFAULT_BAD = ["FLIR_08866", "FLIR_09209", "FLIR_09292", "FLIR_09076"]


def _default_rows():
    root = repo_root_containing_vmm(__file__)
    b_base = os.path.join(
        root, "VMM", "Diffusion_B", "Test_Results", "diffusion_b_all_pairs_p1_50_reg_100"
    )
    d_base = os.path.join(
        root, "VMM", "Diffusion", "Test_Results", "diffusion_all_pairs_phased"
    )
    w_base = os.path.join(
        root, "VMM", "REG_WFCG", "Test_Results", "reg_wfcg_all_warps"
    )
    return [
        ("Diffusion_B", "e80", os.path.join(b_base, "e80")),
        ("Diffusion_B", "e90", os.path.join(b_base, "e90")),
        ("Diffusion_B", "e100", os.path.join(b_base, "e100")),
        ("Diffusion", "e60", os.path.join(d_base, "e60")),
        ("REG_WFCG", "e210", os.path.join(w_base, "e210")),
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Per-stem strip montage (one row per model)")
    p.add_argument("--stems", nargs="*", default=[], help="Stems to render (default: good+bad set)")
    p.add_argument("--good_only", action="store_true")
    p.add_argument("--bad_only", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (default: .../compare_per_stem/)",
    )
    p.add_argument("--label_w", type=int, default=300)
    p.add_argument("--row_gap", type=int, default=4)
    return p.parse_args()


def _fonts():
    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13),
        )
    except Exception:
        f = ImageFont.load_default()
        return f, f


def compose_stem(stem: str, rows, out_path: str, label_w: int, row_gap: int) -> None:
    strips: list[Image.Image] = []
    labels: list[str] = []
    for model, epoch, strip_dir in rows:
        path = os.path.join(strip_dir, f"{stem}_strip.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        strips.append(Image.open(path).convert("RGB"))
        labels.append(f"{model}  {epoch}")

    sw, sh = strips[0].size
    n = len(strips)
    H = n * sh + (n - 1) * row_gap
    W = label_w + sw
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font_lg, _font_sm = _fonts()

    y = 0
    for img, label in zip(strips, labels):
        draw.text((10, y + sh // 2 - 10), label, fill=(230, 230, 230), font=font_lg)
        canvas.paste(img, (label_w, y))
        y += sh + row_gap

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def main():
    opt = parse_args()
    rows = _default_rows()

    if opt.stems:
        stems = opt.stems
    elif opt.good_only:
        stems = DEFAULT_GOOD
    elif opt.bad_only:
        stems = DEFAULT_BAD
    else:
        stems = DEFAULT_GOOD + DEFAULT_BAD

    root = repo_root_containing_vmm(__file__)
    if opt.out_dir:
        out_dir = os.path.abspath(os.path.expanduser(opt.out_dir))
    else:
        out_dir = os.path.join(
            root,
            "VMM",
            "Diffusion_B",
            "Test_Results",
            "diffusion_b_all_pairs_p1_50_reg_100",
            "compare_per_stem",
        )
    os.makedirs(out_dir, exist_ok=True)

    for stem in stems:
        out_path = os.path.join(out_dir, f"{stem}_compare.png")
        compose_stem(stem, rows, out_path, opt.label_w, opt.row_gap)
        print(f"Wrote {out_path!r}")

    print(f"Done: {len(stems)} PNGs -> {out_dir!r}")


if __name__ == "__main__":
    main()
