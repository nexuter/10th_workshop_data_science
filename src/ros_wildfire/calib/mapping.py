from __future__ import annotations

import torch
from ros_wildfire.config import ExperimentConfig


def pso_params_to_dict(cfg: ExperimentConfig, pso_params) -> dict:
    """
    Map unconstrained z -> bounded multipliers k.
    pso_params: nn.ParameterDict or dict-like mapping name->tensor
    """
    out = {}
    lo, hi = cfg.ros.k_lo, cfg.ros.k_hi
    span = hi - lo

    for name, z in pso_params.items():
        if name == "k_N_substeps":
            k = cfg.ros.n_substeps_min + (cfg.ros.n_substeps_max - cfg.ros.n_substeps_min) * torch.sigmoid(z)
        else:
            k = lo + span * torch.sigmoid(z)
        out[name] = k
    return out
