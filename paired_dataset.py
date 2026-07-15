"""Paired EO–IR tiles: left half = A (visible), right half = B (thermal).

Extracted from the former ``CUT_model.datasets`` module — only the loaders
Diffusion_B needs (no CUT/GAN trainer code).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def _normalize_tier_root(root: str) -> str:
    """Tier root contains ``train/`` and ``test/``; accept ``.../Mild/train`` and strip to ``.../Mild``."""
    p = os.path.abspath(os.path.expanduser(root))
    if os.path.basename(p.rstrip(os.sep)) in ("train", "test"):
        return os.path.dirname(p.rstrip(os.sep))
    return p


def _list_pair_images(root: str, mode: str) -> List[str]:
    root = _normalize_tier_root(root)
    root = os.path.abspath(os.path.expanduser(root))
    split_dir = os.path.join(root, mode)
    if not os.path.isdir(split_dir):
        return []
    exts = ("*.bmp", "*.BMP", "*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.tif", "*.tiff", "*.webp")
    found: List[str] = []
    for pat in exts:
        found.extend(glob.glob(os.path.join(split_dir, pat)))
    if not found:
        found = sorted(glob.glob(os.path.join(split_dir, "*.*")))
    return sorted(set(found))


def _require_non_empty(files: List[str], root: str, mode: str) -> None:
    if files:
        return
    root = _normalize_tier_root(root)
    root = os.path.abspath(os.path.expanduser(root))
    split_dir = os.path.join(root, mode)
    raise ValueError(
        f"No images found for mode={mode!r} under data root {root!r}.\n"
        f"  Expected an existing directory {split_dir!r} with image files "
        f"(e.g. .bmp, .png, .jpg).\n"
        f"  Pass --data_root as an absolute path to a tier folder with train/ and test/.\n"
        f"  Current working directory: {os.getcwd()!r}"
    )


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms_: Optional[List] = None,
        mode: str = "train",
        img_size: Tuple[int, int] = (256, 256),
    ):
        self.transform = transforms.Compose(transforms_) if transforms_ else transforms.Compose([transforms.ToTensor()])
        self.img_size = img_size
        root_n = _normalize_tier_root(root)
        self.files = _list_pair_images(root_n, mode)
        _require_non_empty(self.files, root_n, mode)

    def __getitem__(self, index: int):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        w, h = img.size
        img_a = img.crop((0, 0, w // 2, h))
        img_b = img.crop((w // 2, 0, w, h))
        img_a = img_a.resize(self.img_size, Image.Resampling.BICUBIC)
        img_b = img_b.resize(self.img_size, Image.Resampling.BICUBIC)
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        return {"A": img_a, "B": img_b}

    def __len__(self) -> int:
        return len(self.files)


class TestImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms_: Optional[List] = None,
        mode: str = "test",
        img_size: Tuple[int, int] = (256, 256),
    ):
        self.transform = transforms.Compose(transforms_) if transforms_ else transforms.Compose([transforms.ToTensor()])
        self.img_size = img_size
        root_n = _normalize_tier_root(root)
        self.files = _list_pair_images(root_n, mode)
        _require_non_empty(self.files, root_n, mode)

    def __getitem__(self, index: int):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        w, h = img.size
        img_a = img.crop((0, 0, w // 2, h))
        img_b = img.crop((w // 2, 0, w, h))
        img_a = img_a.resize(self.img_size, Image.Resampling.BICUBIC)
        img_b = img_b.resize(self.img_size, Image.Resampling.BICUBIC)
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        return {"A": img_a, "B": img_b}

    def __len__(self) -> int:
        return len(self.files)
