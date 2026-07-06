from __future__ import annotations

import torch
import torch.nn.functional as F


def get_all_frontiers(burn_THW: torch.Tensor) -> torch.Tensor:
    """
    burn_THW: (T,H,W) float/bool cumulative burn
    returns:  (T,H,W) bool - burned pixels that have at least one unburned neighbor
    """
    burn = burn_THW > 0.5
    T, H, W = burn.shape

    kernel = torch.tensor(
        [[1,1,1],[1,0,1],[1,1,1]], dtype=torch.float32, device=burn.device
    ).view(1, 1, 3, 3)

    neighbor_burn_sum = F.conv2d(burn.float().unsqueeze(1), kernel, padding=1).squeeze(1)
    has_unburned_neighbor = neighbor_burn_sum < 8.0
    return burn & has_unburned_neighbor


@torch.no_grad()
def compute_frontier_batched(burn_BHW: torch.Tensor, burnable_BHW: torch.Tensor) -> torch.Tensor:
    """
    burning-side frontier:
      burning pixels with at least one unburned burnable neighbor.
    burn_BHW, burnable_BHW: (B,H,W) bool
    returns: (B,H,W) bool
    """
    unburned = (~burn_BHW) & burnable_BHW
    k = torch.ones((1, 1, 3, 3), device=burn_BHW.device, dtype=torch.float32)
    nb_unburned = F.conv2d(unburned.float().unsqueeze(1), k, padding=1)[:, 0] > 0
    return (burn_BHW & burnable_BHW) & nb_unburned


def elevation_grad_at_coords(elev_THW: torch.Tensor, t_idx, y_idx, x_idx, cell_size_m: float):
    """Central-difference local DEM gradient at given coords (m/m)."""
    T, H, W = elev_THW.shape
    xm1 = (x_idx - 1).clamp(0, W - 1)
    xp1 = (x_idx + 1).clamp(0, W - 1)
    ym1 = (y_idx - 1).clamp(0, H - 1)
    yp1 = (y_idx + 1).clamp(0, H - 1)

    z_xp = elev_THW[t_idx, y_idx, xp1]
    z_xm = elev_THW[t_idx, y_idx, xm1]
    z_yp = elev_THW[t_idx, yp1, x_idx]
    z_ym = elev_THW[t_idx, ym1, x_idx]

    zx = (z_xp - z_xm) / (2.0 * cell_size_m)
    zy = (z_yp - z_ym) / (2.0 * cell_size_m)
    return zx, zy


def create_frontier_dynamics_stack(ros, upsilon, lb, coords, T: int, H: int, W: int, device):
    """Create (3,T,H,W) stack with (ros,upsilon,lb) values at frontier coords."""
    out = torch.zeros((3, T, H, W), device=device, dtype=torch.float32)
    if coords.numel() == 0:
        return out
    t, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    out[0, t, y, x] = ros
    out[1, t, y, x] = upsilon
    out[2, t, y, x] = lb
    return out
