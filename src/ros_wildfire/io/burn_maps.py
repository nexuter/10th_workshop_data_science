from __future__ import annotations

import os
from PIL import Image
import numpy as np
import torch


def find_mask_root(case_root: str) -> str | None:
    """Support common misspelling and naming variations in datasets."""
    candidates = [
        "Satellite_Images_Mask",
        "Satellite_Image_Mask",
        "Satellite_Image_Masks",
        "Satellite_Images_Masks",
        "Satelllite_Images_Mask",
        "Satelllite_Images_Masks",
    ]
    for name in candidates:
        p = os.path.join(case_root, name)
        if os.path.isdir(p):
            return p

    cand_lower = {c.lower() for c in candidates}
    try:
        for entry in os.listdir(case_root):
            p = os.path.join(case_root, entry)
            if os.path.isdir(p) and entry.lower() in cand_lower:
                return p
    except FileNotFoundError:
        return None
    return None


def load_burn_maps(case_root: str, idx: int, T: int, H: int, W: int) -> torch.Tensor:
    """
    Load burn masks out1..outT from Satellite_Images_Mask/<pre>_<idx:05d>/.
    Returns (T,H,W) float32 with values {0,1}.
    """
    pre = os.path.basename(case_root)
    mask_root = find_mask_root(case_root)
    if mask_root is None:
        raise FileNotFoundError(f"Mask root not found under: {case_root}")

    run_dir = os.path.join(mask_root, f"{pre}_{idx:05d}")

    burn_layers = []
    exts = (".jpg", ".jpeg", ".png")
    for t in range(1, T + 1):
        path = None
        for ext in exts:
            cand = os.path.join(run_dir, f"out{t}{ext}")
            if os.path.exists(cand):
                path = cand
                break

        if path is None:
            burn_layers.append(torch.zeros((H, W), dtype=torch.float32))
            continue

        with Image.open(path) as img:
            if img.size != (W, H):
                img = img.resize((W, H))
            arr = (np.array(img).astype(np.float32) / 255.0) >= 0.5
            burn_layers.append(torch.from_numpy(arr.astype(np.float32)))

    return torch.stack(burn_layers, dim=0)
