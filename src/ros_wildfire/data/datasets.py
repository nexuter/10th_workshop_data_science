from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.data.discovery import discover_idx_samples, resolve_case_root
from ros_wildfire.io.raster import create_spatial_stack
from ros_wildfire.io.weather import load_full_weather
from ros_wildfire.io.burn_maps import load_burn_maps


class MultiIdxStackDataset(Dataset):
    """Each item is one run idx inside one scenario_dir. Returns (C,T,H,W) + meta."""
    def __init__(
        self,
        cfg: ExperimentConfig,
        scenario_dir: str,
        idx_list: Optional[Sequence[int]] = None,
        max_runs: Optional[int] = None,
        strict: bool = False,
    ):
        self.cfg = cfg
        self.scenario_dir = scenario_dir
        self.samples = discover_idx_samples(
            scenario_dir,
            idx_list=idx_list,
            max_runs=max_runs,
            random_per_scenario=cfg.data.random_per_scenario,
            seed=cfg.runtime.seed,
            strict=strict,
            t_cap=cfg.data.t_cap,
            drop_if_weather_short=cfg.data.drop_if_weather_short,
        )
        self._spatial_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        scenario_dir, idx = self.samples[i]
        case_root = resolve_case_root(scenario_dir)

        if case_root not in self._spatial_cache:
            self._spatial_cache[case_root] = create_spatial_stack(case_root)  # CPU
        spatial = self._spatial_cache[case_root]
        C_s, H, W = spatial.shape

        weather = load_full_weather(case_root, idx=idx)  # (4,Tw)
        _, Tw = weather.shape
        if self.cfg.data.t_cap is not None:
            if self.cfg.data.drop_if_weather_short and Tw < self.cfg.data.t_cap:
                raise IndexError(f"Weather too short idx={idx}: Tw={Tw} < t_cap={self.cfg.data.t_cap}")
            T = min(Tw, int(self.cfg.data.t_cap))
            weather = weather[:, :T]
        else:
            T = Tw

        burn = load_burn_maps(case_root, idx=idx, T=T, H=H, W=W)  # (T,H,W)

        spatial_4d = spatial.unsqueeze(1).expand(C_s, T, H, W)
        weather_4d = weather.view(4, T, 1, 1).expand(4, T, H, W)
        physical_4d = torch.cat([spatial_4d, weather_4d], dim=0)  # (9,T,H,W)

        full_4d = torch.cat([physical_4d, burn.unsqueeze(0)], dim=0)  # (10,T,H,W)

        meta = {"scenario_dir": scenario_dir, "case_root": case_root, "idx": idx}
        return full_4d, meta


class MultiScenarioIdxDataset(Dataset):
    """Each item is one (scenario_dir, idx) across many scenarios."""
    def __init__(
        self,
        cfg: ExperimentConfig,
        scenario_dirs: Sequence[str],
        idx_list: Optional[Sequence[int]] = None,
        max_runs: Optional[int] = None,
        strict: bool = False,
    ):
        self.cfg = cfg
        self.scenario_dirs = list(scenario_dirs)
        self._spatial_cache: Dict[str, torch.Tensor] = {}
        self.samples: List[Tuple[str, int]] = []
        self.per_scenario_counts: Dict[str, int] = {}

        for scen in self.scenario_dirs:
            salt = int(hashlib.md5(str(scen).encode("utf-8")).hexdigest()[:8], 16)
            seed = cfg.runtime.seed + salt if cfg.data.random_per_scenario else cfg.runtime.seed

            smp = discover_idx_samples(
                scen,
                idx_list=idx_list,
                max_runs=cfg.data.max_runs_per_scenario,
                random_per_scenario=cfg.data.random_per_scenario,
                seed=seed,
                strict=strict,
                t_cap=cfg.data.t_cap,
                drop_if_weather_short=cfg.data.drop_if_weather_short,
            )
            self.per_scenario_counts[scen] = len(smp)
            self.samples.extend(smp)

        if max_runs is not None:
            self.samples = self.samples[: int(max_runs)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        scenario_dir, idx = self.samples[i]
        case_root = resolve_case_root(scenario_dir)

        if case_root not in self._spatial_cache:
            self._spatial_cache[case_root] = create_spatial_stack(case_root)
        spatial = self._spatial_cache[case_root]
        C_s, H, W = spatial.shape

        weather = load_full_weather(case_root, idx=idx)
        _, Tw = weather.shape

        if self.cfg.data.t_cap is not None:
            if self.cfg.data.drop_if_weather_short and Tw < self.cfg.data.t_cap:
                raise IndexError(f"Weather too short idx={idx}: Tw={Tw} < t_cap={self.cfg.data.t_cap}")
            T = min(Tw, int(self.cfg.data.t_cap))
            weather = weather[:, :T]
        else:
            T = Tw

        burn = load_burn_maps(case_root, idx=idx, T=T, H=H, W=W)

        spatial_4d = spatial.unsqueeze(1).expand(C_s, T, H, W)
        weather_4d = weather.view(4, T, 1, 1).expand(4, T, H, W)
        physical_4d = torch.cat([spatial_4d, weather_4d], dim=0)
        full_4d = torch.cat([physical_4d, burn.unsqueeze(0)], dim=0)

        meta = {"scenario_dir": scenario_dir, "case_root": case_root, "idx": idx}
        return full_4d, meta
