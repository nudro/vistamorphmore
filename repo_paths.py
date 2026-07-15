"""Package / workspace path helpers for standalone GitHub root or nested under VistaMorphMore."""

from __future__ import annotations

import os


def package_root(here: str) -> str:
    """Directory containing this package's ``train.py``."""
    d = os.path.realpath(here)
    if os.path.isfile(d):
        d = os.path.dirname(d)
    return d


def workspace_root(here: str) -> str:
    """
    Prefer a parent that contains ``Data/`` (VistaMorphMore-style).
    Otherwise return ``package_root`` (standalone clone).
    """
    pkg = package_root(here)
    cur = pkg
    for _ in range(32):
        if os.path.isdir(os.path.join(cur, "Data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return pkg


# Back-compat alias used by montage / older call sites
def repo_root_containing_vmm(here: str) -> str:
    return workspace_root(here)


def data_root_example_flir_vmm_mild(here: str) -> str:
    """Example tier path when a ``Data/`` tree exists adjacent; else a placeholder."""
    root = workspace_root(here)
    candidate = os.path.join(root, "Data", "FLIR_VMM", "Mild", "train")
    if os.path.isdir(os.path.dirname(candidate)) or os.path.isdir(candidate):
        return candidate
    return "/path/to/tier_with_train_and_test"


def normalize_data_root_to_tier(data_root: str) -> str:
    """
    ``ImageDataset`` expects the tier directory (parent of ``train/`` and ``test/``).
    If ``data_root`` ends with ``train`` or ``test``, return its parent.
    """
    p = os.path.abspath(os.path.expanduser(data_root))
    bn = os.path.basename(p.rstrip(os.sep))
    if bn in ("train", "test"):
        return os.path.dirname(p.rstrip(os.sep))
    return p


def format_training_run_example(relpath_from_repo: str, experiment: str, here: str) -> str:
    root = package_root(here)
    data = data_root_example_flir_vmm_mild(here)
    return (
        "Typical invocation (package root):\n"
        f"  cd {root}\n"
        f"  python {relpath_from_repo} --data_root {data} --experiment {experiment}"
    )
