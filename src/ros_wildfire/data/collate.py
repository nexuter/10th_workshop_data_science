from __future__ import annotations

from typing import List, Tuple, Dict, Any, Sequence

from torch.utils.data import DataLoader

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.data.datasets import MultiIdxStackDataset, MultiScenarioIdxDataset


def collate_as_lists(batch):
    stacks = [x[0] for x in batch]
    metas = [x[1] for x in batch]
    return stacks, metas


def build_loader(cfg: ExperimentConfig, *, scenario_dirs: Sequence[str] | None = None):
    """
    If scenario_dirs is provided and non-empty, uses MultiScenarioIdxDataset.
    Otherwise uses MultiIdxStackDataset on cfg.data.scenario_dir.
    """
    if scenario_dirs:
        ds = MultiScenarioIdxDataset(
            cfg,
            scenario_dirs=scenario_dirs,
            idx_list=cfg.data.idx_list,
            max_runs=cfg.data.max_runs,
            strict=False,
        )
    else:
        ds = MultiIdxStackDataset(
            cfg,
            scenario_dir=cfg.data.scenario_dir,
            idx_list=cfg.data.idx_list,
            max_runs=cfg.data.max_runs,
            strict=False,
        )

    dl = DataLoader(
        ds,
        batch_size=cfg.pso.batch_size,
        shuffle=cfg.data.shuffle_samples,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=collate_as_lists,
    )
    return ds, dl
