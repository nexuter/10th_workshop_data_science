from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import random
import torch


@dataclass
class RuntimeConfig:
    seed: int = 123
    device_preference: str = "cuda"  # "cuda" or "cpu"
    force_cpu: bool = False

    def torch_device(self) -> torch.device:
        if self.force_cpu:
            return torch.device("cpu")
        if self.device_preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def seed_everything(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


@dataclass
class DataConfig:
    scenario_dir: str = "./output/pkl_materialized/0001_000_pkl"
    idx: int = 1

    # Multi-run evaluation/training
    use_multi_idx: bool = True
    idx_list: Optional[Tuple[int, ...]] = None
    max_runs: Optional[int] = None

    # Multi-scenario
    max_runs_per_scenario: Optional[int] = 100
    random_per_scenario: bool = True

    shuffle_samples: bool = True
    num_workers: int = 0
    t_cap: Optional[int] = 48
    drop_if_weather_short: bool = True

    burn_channel: int = 9
    fuel_channel: int = 3


@dataclass
class ROSConfig:
    # PSO-tuned multipliers
    param_names: Tuple[str, ...] = ("k_A", "k_R0", "k_phi_s", "k_C", "k_B", "k_E")

    # Generic multiplier bounds for k_* mapped via sigmoid
    k_lo: float = 0.8
    k_hi: float = 1.2

    # Special mappings
    n_substeps_min: int = 4
    n_substeps_max: int = 8
    a_a_exp_base: float = 4.0

    init_z0: float = 0.0


@dataclass
class PhysicalModelConfig:
    cell_size_m: float = 30.0
    dt_seconds: float = 3600.0

    base_n_substeps: int = 8
    a_a: float = 0.115
    n_theta: int = 8
    head_bins_px: Tuple[float, ...] = (0.5, 1, 1.5, 2, 4, 6, 10, 12, 16, 20, 30)
    lb_bins: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.5, 5.0)
    max_head_px: Optional[float] = 30.0


@dataclass
class EvalConfig:
    dice_eps: float = 1e-6
    use_new_ignition: bool = True


@dataclass
class VizConfig:
    plot_t: int = 40
    save_figs: bool = False
    out_dir: str = "./figs"


@dataclass
class PSOConfig:
    num_iters: int = 30
    num_particles: int = 12  # Sweet spot: good accuracy, not unnecessarily slow
    inertial_weight: float = 0.9
    cognitive_coefficient: float = 1.0
    social_coefficient: float = 1.0
    max_param_value: float = 3.0
    min_param_value: float = -3.0

    # Batched objective
    batch_size: int = 10
    batches_per_iter: int = 1
    
    # Early stopping: MAIN SPEEDUP (30-40% faster via plateau detection)
    early_stop_patience: int = 8
    early_stop_threshold: float = 1e-4
    
    # Grid downsampling for faster evaluation (Phase 2 optimization)
    # 1 = no downsampling, 2 = 2x2 downsampling (4x faster), 4 = 4x4 (16x faster)
    grid_downsample: int = 2


@dataclass
class ExperimentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ros: ROSConfig = field(default_factory=ROSConfig)
    phys: PhysicalModelConfig = field(default_factory=PhysicalModelConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    viz: VizConfig = field(default_factory=VizConfig)
