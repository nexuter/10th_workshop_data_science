from __future__ import annotations

import os
import numpy as np
import torch


def load_full_weather(case_root: str, idx: int) -> torch.Tensor:
    """
    Load weather time series for a run idx from Weather_Data.
    Returns (4,T): [temp(F), rh(%), wind_speed(mph), wind_dir(deg)].
    """
    pre = os.path.basename(case_root)
    fname = f"{pre}_{idx:05d}.txt"
    weather_path = os.path.join(case_root, "Weather_Data", fname)
    data = np.loadtxt(weather_path)

    temp = torch.from_numpy(data[:, 4].astype(np.float32))
    rh = torch.from_numpy(data[:, 5].astype(np.float32))
    ws = torch.from_numpy(data[:, 7].astype(np.float32))
    wd = torch.from_numpy(data[:, 8].astype(np.float32))

    return torch.stack([temp, rh, ws, wd], dim=0)
