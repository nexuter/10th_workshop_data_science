from __future__ import annotations

import json
import hashlib
import site
import sys
from pathlib import Path

import numpy as np


DEFAULT_PSO_PARAMS = Path(__file__).resolve().parents[1] / "pso" / "pso_params_a100.json"


class PsoWarmStartGenerator:
    """Adapter around the bundled ros_wildfire PSO/Huygens prediction code (src/ros_wildfire)."""

    def __init__(
        self,
        params_file: str | Path = DEFAULT_PSO_PARAMS,
        device: str = "auto",
        t_cap: int = 48,
        ros_wildfire_src: str | Path | None = None,
        include_user_site: bool = True,
        cache_dir: str | Path | None = "outputs/pso_cache",
    ) -> None:
        self.params_file = Path(params_file)
        self.device = device
        self.t_cap = int(t_cap)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if include_user_site:
            self._prepend_user_site()
        if ros_wildfire_src is not None:
            self._prepend_path(Path(ros_wildfire_src))
        self.params = self._load_params(self.params_file)
        self.params_hash = self._hash_file(self.params_file)
        self._prediction_cache: dict[tuple[str, int, int, str], np.ndarray] = {}

    def predict_frame(self, scenario_root: str | Path, event_id: str, frame_index: int) -> np.ndarray:
        idx = self.event_id_to_idx(event_id)
        sequence = self.predict_sequence(scenario_root, idx)
        if frame_index < 0 or frame_index >= sequence.shape[0]:
            raise IndexError(
                f"frame_index={frame_index} is outside PSO prediction sequence length {sequence.shape[0]} "
                f"for event_id={event_id}"
            )
        return sequence[frame_index].astype(float)

    def predict_sequence(self, scenario_root: str | Path, idx: int) -> np.ndarray:
        effective_t_cap = self._effective_t_cap(scenario_root, idx)
        key = (str(Path(scenario_root).resolve()), int(idx), effective_t_cap, self.device)
        if key not in self._prediction_cache:
            cached = self._load_disk_cache(scenario_root, idx, effective_t_cap)
            if cached is None:
                cached = self._run_huygens(scenario_root, idx, effective_t_cap)
                self._write_disk_cache(scenario_root, idx, effective_t_cap, cached)
            self._prediction_cache[key] = cached
        return self._prediction_cache[key]

    @staticmethod
    def event_id_to_idx(event_id: str) -> int:
        return int(str(event_id).split("_")[-1])

    def _run_huygens(self, scenario_root: str | Path, idx: int, t_cap: int) -> np.ndarray:
        try:
            import torch
            from ros_wildfire.config import ExperimentConfig
            from ros_wildfire.data.datasets import MultiIdxStackDataset
            from ros_wildfire.physics.fuel_lookup import create_param_lookups
            from ros_wildfire.physics.huygens import huygens_substeps_parallel
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(
                "PSO warm start requires torch and the bundled ros_wildfire package "
                f"(src/ros_wildfire). Import failed for: {missing}. Install the "
                "'pso' extra (pip install -e .[pso]) in the active environment, or "
                "pass --ros-wildfire-src <path> to use an alternate ros_wildfire copy."
            ) from exc

        cfg = ExperimentConfig()
        cfg.data.scenario_dir = str(scenario_root)
        cfg.data.use_multi_idx = True
        cfg.data.idx_list = (int(idx),)
        cfg.data.t_cap = int(t_cap)
        cfg.data.drop_if_weather_short = True
        cfg.data.random_per_scenario = False
        cfg.data.shuffle_samples = False
        if self.device == "cpu":
            cfg.runtime.force_cpu = True
        elif self.device == "cuda":
            cfg.runtime.device_preference = "cuda"

        cfg.runtime.seed_everything()
        device = cfg.runtime.torch_device()
        dataset = MultiIdxStackDataset(cfg, scenario_dir=str(scenario_root), idx_list=(int(idx),), strict=True)
        if len(dataset) == 0:
            raise RuntimeError(f"no PSO-compatible sample for scenario={scenario_root}, idx={idx}")

        full_stack, _meta = dataset[0]
        full_stack = full_stack.to(device)
        lookups = create_param_lookups(device)
        with torch.no_grad():
            pred_burn, _, _ = huygens_substeps_parallel(
                full_stack=full_stack,
                lookups=lookups,
                params=self.params,
                cell_size_m=cfg.phys.cell_size_m,
                dt_seconds=cfg.phys.dt_seconds,
                n_substeps=cfg.phys.base_n_substeps,
                a_a=cfg.phys.a_a,
                n_theta=cfg.phys.n_theta,
                head_bins_px=cfg.phys.head_bins_px,
                lb_bins=cfg.phys.lb_bins,
                max_head_px=cfg.phys.max_head_px,
            )
        return pred_burn.detach().float().cpu().numpy()

    @staticmethod
    def _load_params(params_file: Path) -> dict[str, float]:
        params = json.loads(params_file.read_text(encoding="utf-8"))
        required = {"k_A", "k_R0", "k_phi_s", "k_C", "k_B", "k_E"}
        missing = sorted(required - set(params))
        if missing:
            raise ValueError(f"PSO params file missing keys: {missing}")
        return {key: float(params[key]) for key in required}

    def _cache_path(self, scenario_root: str | Path, idx: int, t_cap: int) -> Path | None:
        if self.cache_dir is None:
            return None
        scenario = str(Path(scenario_root).resolve())
        stable = json.dumps(
            {
                "scenario": scenario,
                "idx": int(idx),
                "t_cap": int(t_cap),
                "params_hash": self.params_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def _load_disk_cache(self, scenario_root: str | Path, idx: int, t_cap: int) -> np.ndarray | None:
        cache_path = self._cache_path(scenario_root, idx, t_cap)
        if cache_path is None or not cache_path.exists():
            return None
        with np.load(cache_path) as data:
            sequence = data["prediction"].astype(float)
            if int(data["t_cap"]) != int(t_cap):
                return None
            if str(data["params_hash"]) != self.params_hash:
                return None
            return sequence

    def _write_disk_cache(self, scenario_root: str | Path, idx: int, t_cap: int, sequence: np.ndarray) -> None:
        cache_path = self._cache_path(scenario_root, idx, t_cap)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            prediction=np.asarray(sequence, dtype=np.float32),
            t_cap=np.array(int(t_cap)),
            params_hash=np.array(self.params_hash),
            idx=np.array(int(idx)),
            scenario=np.array(str(Path(scenario_root).resolve())),
        )

    def _effective_t_cap(self, scenario_root: str | Path, idx: int) -> int:
        available = self._contiguous_mask_frame_count(scenario_root, idx)
        if available is None:
            return self.t_cap
        return max(2, min(self.t_cap, available))

    @staticmethod
    def _contiguous_mask_frame_count(scenario_root: str | Path, idx: int) -> int | None:
        root = Path(scenario_root)
        case_id = root.name
        candidates = [
            root / "Satellite_Images_Mask" / f"{case_id}_{int(idx):05d}",
            root / "Satellite_Image_Mask" / f"{case_id}_{int(idx):05d}",
        ]
        run_dir = next((path for path in candidates if path.is_dir()), None)
        if run_dir is None:
            return None
        count = 0
        for index in range(1, 10000):
            if any((run_dir / f"out{index}{ext}").exists() for ext in (".jpg", ".jpeg", ".png")):
                count = index
                continue
            break
        return count

    @staticmethod
    def _hash_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _prepend_path(path: Path) -> None:
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)

    @classmethod
    def _prepend_user_site(cls) -> None:
        try:
            user_site = Path(site.getusersitepackages())
        except Exception:
            return
        cls._prepend_path(user_site)
