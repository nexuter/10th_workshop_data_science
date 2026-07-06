from __future__ import annotations

import math
import torch

from ros_wildfire.physics.fuel_lookup import lookup_dead_fuel_moisture
from ros_wildfire.physics.frontier import elevation_grad_at_coords


def get_frontier_dynamics_parametric(
    full_stack: torch.Tensor,
    all_frontiers: torch.Tensor,
    lookups: dict,
    *,
    cell_size_m: float,
    params: dict | None = None,
):
    """
    Compute ROS, head direction, and length-to-breadth ratio at frontier coords.

    full_stack: (C,T,H,W) where channels:
      0 aspect(deg), 1 elev(m), 2 slope(deg), 3 fuel, 4 canopy,
      5 temp(F), 6 RH(%), 7 wind_speed(mph), 8 wind_dir(deg), 9 burn

    all_frontiers: (T,H,W) bool

    Returns:
      ros_max (N,), upsilon (N,), lb (N,), coords(N,3), R0 (N,), phi_eff (N,)
    """
    if params is None:
        params = {}

    device = full_stack.device

    def k(name: str, default: float = 1.0) -> torch.Tensor:
        v = params.get(name, default)
        if not torch.is_tensor(v):
            v = torch.tensor(float(v), device=device, dtype=torch.float32)
        return v

    coords = torch.nonzero(all_frontiers)
    if coords.numel() == 0:
        empty = torch.tensor([], device=device)
        return empty, empty, empty, coords, empty, empty

    t_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]

    slope_tan = torch.tan(full_stack[2, t_idx, y_idx, x_idx] * (torch.pi / 180.0))
    aspect = full_stack[0, t_idx, y_idx, x_idx] * (torch.pi / 180.0)
    temp = full_stack[5, t_idx, y_idx, x_idx]
    rh = full_stack[6, t_idx, y_idx, x_idx]
    u = full_stack[7, t_idx, y_idx, x_idx] * 88.0  # mph -> ft/min (keep fixed)
    u_dir = full_stack[8, t_idx, y_idx, x_idx] * (torch.pi / 180.0)
    cc = full_stack[4, t_idx, y_idx, x_idx]

    fuel_idx = full_stack[3, t_idx, y_idx, x_idx].long()
    is_non_burnable = (fuel_idx == 0) | (fuel_idx >= 14)
    max_valid_idx = lookups["delta"].size(0) - 1
    safe_fuel_idx = torch.where(
        (fuel_idx >= 14) | (fuel_idx < 0) | (fuel_idx > max_valid_idx),
        torch.zeros_like(fuel_idx),
        fuel_idx,
    )
    safe_fuel_idx = torch.clamp(safe_fuel_idx, 0, max_valid_idx)

    # Wind adjustment factor (WAF) tables
    waf_unsheltered_table = torch.tensor(
        [0.0, 0.4, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5],
        device=device,
        dtype=torch.float32,
    )
    waf_unsheltered = waf_unsheltered_table[torch.clamp(safe_fuel_idx, 0, 13)]

    waf_sheltered = torch.full_like(cc, 0.30)
    waf_sheltered = torch.where((cc > 10.0) & (cc <= 15.0), torch.tensor(0.25, device=device), waf_sheltered)
    waf_sheltered = torch.where((cc > 15.0) & (cc <= 30.0), torch.tensor(0.20, device=device), waf_sheltered)
    waf_sheltered = torch.where((cc > 30.0) & (cc <= 50.0), torch.tensor(0.15, device=device), waf_sheltered)
    waf_sheltered = torch.where(cc > 50.0, torch.tensor(0.10, device=device), waf_sheltered)

    waf = torch.where(cc > 5.0, waf_sheltered, waf_unsheltered)
    u_mid_eff = waf * u

    w_o = lookups["w_o"][safe_fuel_idx]
    delta = lookups["delta"][safe_fuel_idx]
    sigma = lookups["sigma"][safe_fuel_idx]
    rho_p = lookups["rho_p"][safe_fuel_idx]
    Mx = lookups["Mx"][safe_fuel_idx]
    h = lookups["h"][safe_fuel_idx]
    Se = lookups["Se"][safe_fuel_idx]
    S_T = lookups["S_T"][safe_fuel_idx]
    rho_b = lookups["rho_b"][safe_fuel_idx]
    rat = lookups["rat"][safe_fuel_idx]

    beta = rho_b / rho_p
    beta_op = 3.348 * (sigma ** -0.8189)

    A = (k("k_A", 1.0) * 133.0) * (sigma ** -0.7913)
    Gamma_prime_max = (sigma**1.5) / (495.0 + 0.0594 * (sigma**1.5))
    Gamma_prime = Gamma_prime_max * (rat**A) * torch.exp(A * (1.0 - rat))

    # Dead fuel moisture
    Mf_pct = lookup_dead_fuel_moisture(temp, rh)
    is_shaded = cc > 50.0
    Mf_pct = Mf_pct + 1.0 + 2.0 * is_shaded
    Mf = torch.clamp(Mf_pct / 100.0, min=0.01, max=0.40)

    r_M = torch.clamp(Mf / Mx, max=1.0)
    eta_M = 1.0 - 2.59*r_M + 5.11*(r_M**2) - 3.52*(r_M**3)
    eta_M = torch.clamp(eta_M, min=0.0)

    eta_s = 0.174 * (Se ** -0.19)
    xi = torch.exp((0.792 + 0.681 * (sigma**0.5)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    Qig = 250.0 + 1116.0 * Mf
    epsilon = torch.exp(-138.0 / sigma)

    W_n = w_o * (1.0 - S_T)
    Ir = Gamma_prime * W_n * h * eta_M * eta_s
    R0 = (Ir * xi) / (rho_b * epsilon * Qig)
    R0 = k("k_R0", 1.0) * R0

    phi_s = 5.275 * beta ** (-0.3) * (slope_tan ** 2)
    phi_s = k("k_phi_s", 1.0) * phi_s

    C = k("k_C", 1.0) * 7.47 * torch.exp(-0.133 * (sigma**0.55))
    B = k("k_B", 1.0) * 0.02526 * (sigma**0.54)
    E = k("k_E", 1.0) * 0.715 * torch.exp(-3.59e-4 * sigma)

    u_mid_eff_ltd = torch.clamp(u_mid_eff, max=0.9 * Ir)
    phi_w = C * (u_mid_eff_ltd**B) * (rat**-E)

    # Direction conventions (kept as in your code)
    theta_w = u_dir + torch.pi
    theta_s = aspect

    res_x = phi_w * torch.cos(theta_w) + phi_s * torch.cos(theta_s)
    res_y = phi_w * torch.sin(theta_w) + phi_s * torch.sin(theta_s)

    phi_eff = torch.sqrt(res_x**2 + res_y**2)
    ros_max = R0 * (1.0 + phi_eff)
    upsilon = torch.atan2(res_y, res_x)

    # Project along terrain gradient into horizontal ROS
    zx, zy = elevation_grad_at_coords(full_stack[1], t_idx, y_idx, x_idx, cell_size_m)
    ux = torch.cos(upsilon)
    uy = torch.sin(upsilon)
    dz_ds = zx * ux + zy * uy
    proj = 1.0 / torch.sqrt(1.0 + dz_ds * dz_ds + 1e-6)
    ros_max = ros_max * proj

    ros_max = torch.nan_to_num(ros_max, nan=0.0)
    upsilon = torch.nan_to_num(upsilon, nan=0.0)

    ros_max = torch.where(is_non_burnable, torch.tensor(0.0, device=device), ros_max)
    upsilon = torch.where(is_non_burnable, torch.tensor(0.0, device=device), upsilon)
    phi_eff = torch.where(is_non_burnable, torch.tensor(0.0, device=device), phi_eff)
    R0 = torch.where(is_non_burnable, torch.tensor(0.0, device=device), R0)

    # Length-to-breadth
    u_mid_eff_mph = u_mid_eff / 88.0
    lb = 0.936 * torch.exp(0.2566 * u_mid_eff_mph) + 0.461 * torch.exp(-0.1548 * u_mid_eff_mph) - 0.397
    lb = torch.clamp(lb, min=1.0, max=8.0)
    lb = torch.where(is_non_burnable, torch.tensor(1.0, device=device), lb)
    lb = torch.nan_to_num(lb, nan=1.0)

    return ros_max, upsilon, lb, coords, R0, phi_eff
