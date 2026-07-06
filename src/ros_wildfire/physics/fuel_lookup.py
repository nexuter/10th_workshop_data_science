from __future__ import annotations

import torch
from ros_wildfire.constants import TEMP_BINS_F, RH_BINS, FUEL_MOISTURE_TABLE


def create_param_lookups(device: torch.device) -> dict:
    """
    Create parameter tensors indexed by fuel model id (0..99).
    Fuel models 1..13 are Anderson 13; others map to safe non-burnable defaults.
    """
    ANDERSON_PARAMS = {
        1: {"Mx":0.12,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":3500,"delta":1.0,"w_o":0.03745,"rho_b":0.03,"rat": 0.25},
        2: {"Mx":0.15,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":2784,"delta":1.0,"w_o":0.12552,"rho_b":0.18,"rat": 1.14},
        3: {"Mx":0.25,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1500,"delta":2.5,"w_o":0.15183,"rho_b":0.06,"rat": 0.21},
        4: {"Mx":0.20,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1739,"delta":6.0,"w_o":0.50257,"rho_b":0.12,"rat": 0.52},
        5: {"Mx":0.20,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1683,"delta":2.0,"w_o":0.15107,"rho_b":0.08,"rat": 0.33},
        6: {"Mx":0.25,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1564,"delta":2.5,"w_o":0.08098,"rho_b":0.11,"rat": 0.43},
        7: {"Mx":0.40,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1552,"delta":2.5,"w_o":0.07885,"rho_b":0.09,"rat": 0.34},
        8: {"Mx":0.30,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1889,"delta":0.2,"w_o":0.07541,"rho_b":1.15,"rat": 5.17},
        9: {"Mx":0.25,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":2484,"delta":0.2,"w_o":0.14551,"rho_b":0.80,"rat": 4.50},
        10:{"Mx":0.25,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1764,"delta":1.0,"w_o":0.25204,"rho_b":0.55,"rat": 2.35},
        11:{"Mx":0.15,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1182,"delta":1.0,"w_o":0.11388,"rho_b":0.53,"rat": 1.62},
        12:{"Mx":0.20,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1145,"delta":2.3,"w_o":0.33656,"rho_b":0.69,"rat": 2.06},
        13:{"Mx":0.25,"Se":0.01,"S_T":0.055,"rho_p":32,"h":8000,"sigma":1159,"delta":3.0,"w_o":0.56381,"rho_b":0.89,"rat": 2.68},
    }

    NON_BURNABLE_DEFAULT = {"Mx":0.01, "Se":0.01, "S_T":0.055, "rho_p":32,
                            "h":0.1, "sigma":1000, "delta":1.0, "w_o":0.0001,
                            "rho_b":0.0001, "rat":0.1}

    lookups = {k: torch.full((100,), float(v), device=device, dtype=torch.float32)
               for k, v in NON_BURNABLE_DEFAULT.items()}

    for model_id, params in ANDERSON_PARAMS.items():
        for k, v in params.items():
            lookups[k][model_id] = float(v)

    return lookups


def lookup_dead_fuel_moisture(temp_F: torch.Tensor, rh: torch.Tensor) -> torch.Tensor:
    """Vectorized table lookup: returns moisture (%) for each (temp,rh)."""
    device = temp_F.device
    t_idx = torch.bucketize(temp_F, TEMP_BINS_F.to(device))
    rh_idx = torch.bucketize(rh, RH_BINS.to(device))

    t_idx = torch.clamp(t_idx, 0, FUEL_MOISTURE_TABLE.shape[0] - 1)
    rh_idx = torch.clamp(rh_idx, 0, FUEL_MOISTURE_TABLE.shape[1] - 1)

    return FUEL_MOISTURE_TABLE.to(device)[t_idx, rh_idx]
