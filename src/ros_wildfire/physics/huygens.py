from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ros_wildfire.physics.frontier import compute_frontier_batched
from ros_wildfire.physics.rothermel import get_frontier_dynamics_parametric


def rasterize_richards_wavelet_kernel_from_ros(
    R_m_per_min: float,
    dt_minutes: float,
    LB: float,
    theta: float,
    cell_size_m: float,
    device=None,
    min_radius_px: float = 0.5,
) -> torch.Tensor:
    """
    Richards/FARSITE elliptical wavelet (Alexander 1985). Ignition at rear focus.
    Returns mask (k,k) float tensor.
    """
    LB = max(float(LB), 1.0)
    R_m_per_min = float(R_m_per_min)
    dt_minutes = float(dt_minutes)
    cell_size_m = float(cell_size_m)

    if R_m_per_min <= 0.0 or dt_minutes <= 0.0 or cell_size_m <= 0.0:
        return torch.ones((1, 1), device=device, dtype=torch.float32)

    R = R_m_per_min * dt_minutes

    root = math.sqrt(max(LB * LB - 1.0, 0.0))
    denom = max(LB - root, 1e-6)
    HB = (LB + root) / denom

    a_m = 0.5 * (R + R / HB) / LB
    b_m = 0.5 * (R + R / HB)
    c_m = b_m - (R / HB)

    a_px = max(a_m / cell_size_m, float(min_radius_px))
    b_px = max(b_m / cell_size_m, float(min_radius_px))
    c_px = c_m / cell_size_m

    r = int(math.ceil(max(a_px, c_px + b_px)))
    k = 2 * r + 1

    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )

    ct = math.cos(theta)
    st = math.sin(theta)

    xprime = ct * xs + st * ys
    yprime = -st * xs + ct * ys

    mask = ((xprime - c_px) / b_px) ** 2 + (yprime / a_px) ** 2 <= 1.0
    return mask.float()


def _mean_accel_multiplier(t0_min: torch.Tensor, dt_min: float, a_a: float) -> torch.Tensor:
    """Mean of (1 - exp(-a t)) over [t0, t0+dt]."""
    aa = float(a_a)
    dt = float(dt_min)
    t1 = t0_min + dt
    exp0 = torch.exp(-aa * t0_min)
    exp1 = torch.exp(-aa * t1)
    m = 1.0 - (exp0 - exp1) / max(aa * dt, 1e-6)
    return torch.clamp(m, 0.0, 1.0)


@torch.no_grad()
def huygens_substeps_parallel(
    full_stack: torch.Tensor,  # (C,T,H,W)
    lookups: dict,
    *,
    params: dict | None,
    cell_size_m: float,
    dt_seconds: float,
    n_substeps: int,
    a_a: float,
    n_theta: int,
    head_bins_px: Sequence[float],
    lb_bins: Sequence[float],
    max_head_px: float | None = None,
    substep_viz_time: Optional[int] = None,
):
    """
    Teacher forcing:
      For each hour t, start from burn[t] and predict burn[t+1].
    Parallel:
      Treat B=T-1 as batch. Loop only over substeps.

    Returns:
      burn_next_BHW, new_BHW, age_BHW
    """
    device = full_stack.device
    C, T, H, W = full_stack.shape
    B = T - 1

    burn_T = full_stack[9].bool()
    fuel_T = full_stack[3]
    if fuel_T.ndim == 2:
        fuel_T = fuel_T.unsqueeze(0).expand(T, -1, -1)

    burnable_T = ~((fuel_T == 0) | (fuel_T >= 14))
    burnable_B = burnable_T[:-1].contiguous()

    burn0_B = (burn_T[:-1] & burnable_B).contiguous()
    burn_state = burn0_B.clone()

    dt_min_total = dt_seconds / 60.0
    dt_min = dt_min_total / float(n_substeps)

    age_state = torch.zeros((B, H, W), device=device, dtype=torch.float32)
    age_state[burn_state] = 0.0

    head_bins_px_t = torch.tensor(head_bins_px, device=device, dtype=torch.float32)
    lb_bins_t = torch.tensor(lb_bins, device=device, dtype=torch.float32)
    theta_centers = [(i + 0.5) * (2 * math.pi / n_theta) for i in range(n_theta)]

    # Precompute kernels per (theta_bin, lb_bin, head_bin) to avoid rerasterization.
    # (This is cheap relative to conv2d, but helps.)
    kernel_cache = {}
    for i_th in range(n_theta):
        theta0 = theta_centers[i_th]
        for i_lb, lb0 in enumerate(lb_bins):
            for i_h, head_px_bin in enumerate(head_bins_px):
                head_m_bin = float(head_px_bin) * float(cell_size_m)
                R_m_per_min_bin = head_m_bin / max(dt_min, 1e-6)
                K = rasterize_richards_wavelet_kernel_from_ros(
                    R_m_per_min=R_m_per_min_bin,
                    dt_minutes=dt_min,
                    LB=float(lb0),
                    theta=float(theta0),
                    cell_size_m=float(cell_size_m),
                    device=device,
                )
                kernel_cache[(i_th, i_lb, i_h)] = K

    # Reused source tensor to reduce allocations
    S = torch.zeros((B, H, W), device=device, dtype=torch.float32)

    for _s in range(n_substeps):
        src_B = compute_frontier_batched(burn_state, burnable_B)
        if not src_B.any():
            age_state[burn_state] += dt_min
            continue

        stack_B = full_stack[:, :B].contiguous()  # treat time t as batch element b
        ros_ftmin, ups, lb, coords, _, _ = get_frontier_dynamics_parametric(
            stack_B, src_B, lookups, cell_size_m=cell_size_m, params=params
        )
        if coords.numel() == 0:
            age_state[burn_state] += dt_min
            continue

        b_idx = coords[:, 0].long()
        y_idx = coords[:, 1].long()
        x_idx = coords[:, 2].long()

        t0 = age_state[b_idx, y_idx, x_idx]
        accel = _mean_accel_multiplier(t0, dt_min=dt_min, a_a=a_a)

        ros_m_per_min = ros_ftmin * 0.3048
        head_m = ros_m_per_min * dt_min * accel
        head_px = head_m / float(cell_size_m)

        if max_head_px is not None:
            head_px = head_px.clamp_max(float(max_head_px))

        theta = torch.remainder(ups, 2 * math.pi)
        k_theta = torch.clamp((theta / (2 * math.pi) * n_theta).long(), 0, n_theta - 1)
        k_h = torch.argmin((head_px[:, None] - head_bins_px_t[None, :]).abs(), dim=1)
        k_lb = torch.argmin((lb[:, None] - lb_bins_t[None, :]).abs(), dim=1)

        reached_B = torch.zeros((B, H, W), device=device, dtype=torch.bool)

        for i_th in range(n_theta):
            sel_th = (k_theta == i_th)
            if not sel_th.any():
                continue
            for i_lb in range(len(lb_bins)):
                sel_lb = sel_th & (k_lb == i_lb)
                if not sel_lb.any():
                    continue
                for i_h in range(len(head_bins_px)):
                    sel = sel_lb & (k_h == i_h)
                    if not sel.any():
                        continue

                    # reuse allocation
                    S.zero_()
                    S[b_idx[sel], y_idx[sel], x_idx[sel]] = 1.0

                    K = kernel_cache[(i_th, i_lb, i_h)]
                    pad_y, pad_x = K.shape[0] // 2, K.shape[1] // 2

                    out = F.conv2d(
                        S.unsqueeze(1),
                        K.unsqueeze(0).unsqueeze(0),
                        padding=(pad_y, pad_x),
                    )[:, 0] > 0

                    reached_B |= out

        reached_B &= burnable_B
        old_burn = burn_state
        new_cells = reached_B & (~old_burn)

        age_state[old_burn] += dt_min
        age_state[new_cells] = 0.0
        burn_state = old_burn | reached_B

    burn_next_BHW = burn_state
    new_BHW = burn_next_BHW & (~burn0_B)
    return burn_next_BHW, new_BHW, age_state
