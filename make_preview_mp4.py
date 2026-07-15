#!/usr/bin/env python3
"""
Encode training preview PNGs (batches_done.png strips) to MP4 with epoch / phase labels.

Example (from repo root):

  python VMM/Diffusion_B_VMM_Github/make_preview_mp4.py \\
    --images_dir VMM/Diffusion_B_VMM_Github/outputs/diffusion_b_all_pairs_p1_50_reg_100/images
"""

from __future__ import annotations

import argparse
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(description="Training preview strip -> MP4 with epoch labels")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out_mp4", type=str, default="")
    p.add_argument("--batches_per_epoch", type=int, default=253)
    p.add_argument("--reg_start_epoch", type=int, default=50)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--phase2_label", type=str, default="Phase 2 begins")
    return p.parse_args()


def _annotate_frame(img_rgb: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    h = pil.height
    bar_h = 22
    draw.rectangle([0, h - bar_h, pil.width, h], fill=(0, 0, 0))
    draw.text((6, h - 18), text, fill=(255, 255, 255))
    return np.asarray(pil)


def main():
    opt = parse_args()
    images_dir = os.path.abspath(os.path.expanduser(opt.images_dir))
    if not os.path.isdir(images_dir):
        raise SystemExit(f"--images_dir is not a directory: {images_dir!r}")

    stems = []
    for name in os.listdir(images_dir):
        m = re.match(r"^(\d+)\.png$", name)
        if m:
            stems.append(int(m.group(1)))
    stems.sort()
    if not stems:
        raise SystemExit(f"No preview PNGs in {images_dir!r}")

    if opt.out_mp4:
        out_mp4 = os.path.abspath(os.path.expanduser(opt.out_mp4))
    else:
        out_mp4 = os.path.join(os.path.dirname(images_dir), "training_preview.mp4")

    first = np.asarray(Image.open(os.path.join(images_dir, f"{stems[0]}.png")).convert("RGB"))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, float(opt.fps), (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {out_mp4!r}")

    phase2_announced = False
    bpe = max(1, int(opt.batches_per_epoch))
    reg_start = int(opt.reg_start_epoch)

    for batches_done in stems:
        path = os.path.join(images_dir, f"{batches_done}.png")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = np.asarray(
                Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
            )

        epoch = batches_done // bpe
        label = f"Epoch {epoch}"
        if epoch >= reg_start:
            label += f"  |  {opt.phase2_label}"
            phase2_announced = True
        elif not phase2_announced and batches_done == reg_start * bpe:
            label += f"  |  {opt.phase2_label}"

        annotated = _annotate_frame(rgb, label)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Wrote {len(stems)} frames -> {out_mp4!r} ({w}x{h} @ {opt.fps} fps)")


if __name__ == "__main__":
    main()
