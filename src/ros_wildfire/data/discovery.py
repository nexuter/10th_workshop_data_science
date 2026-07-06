from __future__ import annotations

import glob
import os
import random
import re
from typing import List, Optional, Sequence, Tuple

from ros_wildfire.io.weather import load_full_weather
from ros_wildfire.io.burn_maps import find_mask_root

SEQ_RE = re.compile(r"_(\d{5})\.txt$")


def resolve_case_root(scenario_dir: str) -> str:
    """
    Resolve the directory that actually contains Weather_Data / Topography_Map.
    Handles common layouts:
      - scenario_dir/0001/Weather_Data
      - scenario_dir/Weather_Data
      - scenario_dir/extr/Weather_Data
      - scenario_dir/<any>/Weather_Data
    """
    scenario_dir = scenario_dir.rstrip("/")
    base = os.path.basename(scenario_dir)
    prefix = base[:4]

    cand = os.path.join(scenario_dir, prefix)
    if os.path.isdir(os.path.join(cand, "Weather_Data")):
        return cand

    if os.path.isdir(os.path.join(scenario_dir, "Weather_Data")):
        return scenario_dir

    for name in ["extr", "case", "data"]:
        cand = os.path.join(scenario_dir, name)
        if os.path.isdir(os.path.join(cand, "Weather_Data")):
            return cand

    for d in os.listdir(scenario_dir):
        p = os.path.join(scenario_dir, d)
        if os.path.isdir(os.path.join(p, "Weather_Data")):
            return p

    for root, dirs, _ in os.walk(scenario_dir):
        if "Weather_Data" in dirs:
            return root

    raise FileNotFoundError(f"Could not find Weather_Data under: {scenario_dir}")


def discover_available_idxs(case_root: str) -> List[int]:
    wdir = os.path.join(case_root, "Weather_Data")
    txts = sorted(glob.glob(os.path.join(wdir, "*.txt")))
    idxs: List[int] = []
    for f in txts:
        m = SEQ_RE.search(os.path.basename(f))
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))


def discover_idx_samples(
    scenario_dir: str,
    *,
    idx_list: Optional[Sequence[int]] = None,
    max_runs: Optional[int] = None,
    random_per_scenario: bool = False,
    seed: Optional[int] = None,
    strict: bool = False,
    t_cap: Optional[int] = None,
    drop_if_weather_short: bool = True,
) -> List[Tuple[str, int]]:
    """
    Return list of (scenario_dir, idx) where:
      - Weather_Data exists for idx
      - Mask folder exists
      - If t_cap is set: weather length >= t_cap (if drop_if_weather_short), and out1..outT_need exist
    """
    case_root = resolve_case_root(scenario_dir)
    pre = os.path.basename(case_root)

    available = discover_available_idxs(case_root)
    if not available:
        raise RuntimeError(f"No Weather_Data txt files under: {os.path.join(case_root, 'Weather_Data')}")

    if idx_list is not None:
        wanted = [int(x) for x in idx_list]
        missing = [i for i in wanted if i not in available]
        kept = [i for i in wanted if i in available]
        if strict and missing:
            raise RuntimeError(f"Requested idx missing weather files: {missing}")
        idxs = kept
    else:
        idxs = available

    if random_per_scenario:
        rng = random.Random(int(seed or 0))
        rng.shuffle(idxs)

    if max_runs is not None:
        idxs = idxs[: int(max_runs)]

    mask_root = find_mask_root(case_root)
    if mask_root is None:
        if strict:
            raise RuntimeError(f"No mask root found under: {case_root}")
        return []

    def frames_complete(run_dir: str, need: int) -> bool:
        exts = (".jpg", ".jpeg", ".png")
        for t in range(1, need + 1):
            ok = any(os.path.exists(os.path.join(run_dir, f"out{t}{ext}")) for ext in exts)
            if not ok:
                return False
        return True

    samples: List[Tuple[str, int]] = []
    for idx in idxs:
        run_dir = os.path.join(mask_root, f"{pre}_{idx:05d}")
        if not os.path.isdir(run_dir):
            if strict:
                raise RuntimeError(f"Missing mask folder: {run_dir}")
            continue

        weather = load_full_weather(case_root, idx=idx)  # (4,Tw)
        T_weather = int(weather.shape[1])
        if t_cap is None:
            T_need = T_weather
        else:
            if drop_if_weather_short and T_weather < t_cap:
                if strict:
                    raise RuntimeError(f"Weather too short: idx={idx}, T={T_weather} < t_cap={t_cap}")
                continue
            T_need = min(T_weather, int(t_cap))

        if not frames_complete(run_dir, need=T_need):
            if strict:
                raise RuntimeError(f"Incomplete frames: {run_dir} need out1..out{T_need}")
            continue

        samples.append((scenario_dir, idx))

    if not samples:
        raise RuntimeError(f"No valid samples found in scenario_dir={scenario_dir}")
    return samples
