from __future__ import annotations

import os
import numpy as np
import rasterio
import torch


def read_tif_1band(path: str) -> torch.Tensor:
    """Read a single-band GeoTIFF into a float32 torch tensor (H,W)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return torch.from_numpy(arr)


def create_spatial_stack(case_root: str) -> torch.Tensor:
    """Create spatial stack (5,H,W): [aspect, elevation, slope, fuel_model, canopy_cover]."""
    topo_dir = os.path.join(case_root, "Topography_Map")
    fuel_dir = os.path.join(case_root, "Fuel_Map")

    aspect = read_tif_1band(os.path.join(topo_dir, "Aspect.tif"))
    elev = read_tif_1band(os.path.join(topo_dir, "Elevation.tif"))
    slope = read_tif_1band(os.path.join(topo_dir, "Slope.tif"))
    fuel = read_tif_1band(os.path.join(fuel_dir, "FBFM13.tif"))
    cc = read_tif_1band(os.path.join(fuel_dir, "Canopy_Cover.tif"))

    H, W = aspect.shape
    for name, x in [("Elevation", elev), ("Slope", slope), ("Fuel", fuel), ("Canopy_Cover", cc)]:
        if x.shape != (H, W):
            raise ValueError(f"Shape mismatch: Aspect {(H,W)} vs {name} {tuple(x.shape)}")

    return torch.stack([aspect, elev, slope, fuel, cc], dim=0)
