import argparse
from pathlib import Path
from datetime import datetime
import json

import torch

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.data.collate import build_loader
from ros_wildfire.physics.fuel_lookup import create_param_lookups
from ros_wildfire.calib.pso_runner import run_pso


def run_pso_scenario(scenario_dir: str, t_cap: int, iters: int, particles: int, device: str, params_out: str | None = None):
    """Run PSO calibration for a single scenario."""
    start_time = datetime.now()
    scenario_name = Path(scenario_dir).name
    
    cfg = ExperimentConfig()
    cfg.data.scenario_dir = scenario_dir
    cfg.data.t_cap = t_cap
    cfg.pso.num_iters = iters
    cfg.pso.num_particles = particles

    cfg.runtime.seed_everything()
    print(f"[*] Running PSO on {scenario_name} on {str(device).upper()} - Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    ds, dl = build_loader(cfg, scenario_dirs=None)
    lookups = create_param_lookups(device)

    mapped_final, logs = run_pso(cfg, loader=dl, lookups=lookups, device=device)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"[+] {scenario_name} - Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed:.1f}s)")
    print(f"[+] {scenario_name} - Final params: {mapped_final}")
    if params_out:
        out_path = Path(params_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapped_final, f, indent=2)
        print(f"[+] {scenario_name} - Saved params: {out_path}")
    return mapped_final, logs


def run_pso_multi_scenario(scenario_dirs: list[str], t_cap: int, iters: int, particles: int, device: str, params_out: str | None = None):
    """Run ONE PSO across multiple scenarios to learn global coefficients."""
    start_time = datetime.now()
    cfg = ExperimentConfig()
    cfg.data.t_cap = t_cap
    cfg.pso.num_iters = iters
    cfg.pso.num_particles = particles

    cfg.runtime.seed_everything()
    print(
        f"[*] Running GLOBAL PSO on {len(scenario_dirs)} scenarios on {str(device).upper()} - "
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    ds, dl = build_loader(cfg, scenario_dirs=scenario_dirs)
    lookups = create_param_lookups(device)

    mapped_final, logs = run_pso(cfg, loader=dl, lookups=lookups, device=device)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"[+] GLOBAL PSO - Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed:.1f}s)")
    print(f"[+] GLOBAL PSO - Final params: {mapped_final}")
    if params_out:
        out_path = Path(params_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapped_final, f, indent=2)
        print(f"[+] GLOBAL PSO - Saved params: {out_path}")
    return mapped_final, logs


def _filter_scenarios(scenarios: list[Path], case_filter: list[str] | None) -> list[Path]:
    if not case_filter:
        return scenarios
    raw = [str(x).strip() for x in case_filter if str(x).strip()]
    if not raw:
        return scenarios

    def _case_id(p: Path) -> str:
        name = p.name
        return name.split("_")[0]

    filtered = [s for s in scenarios if any(_case_id(s).startswith(f) for f in raw)]
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", required=True, help="Path to single scenario or directory with multiple scenarios")
    ap.add_argument("--t-cap", type=int, default=48)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--global", dest="global_run", action="store_true", help="Run one PSO across ALL scenarios (global coefficients)")
    ap.add_argument("--case-filter", nargs="*", default=None, help="Scenario prefixes to include (e.g., 0001 0002)")
    ap.add_argument("--params-out", default=None, help="Path to save final PSO coefficients as JSON")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.runtime.seed_everything()
    device = cfg.runtime.torch_device()
    print("device:", device)

    scenario_dir = Path(args.scenario_dir)

    # Determine if single scenario or multi-scenario
    scenarios = []
    if scenario_dir.is_dir():
        subdirs = {d.name for d in scenario_dir.iterdir() if d.is_dir()}
        if {"Weather_Data", "Vegetation_Map", "Topography_Map", "Fuel_Map"}.issubset(subdirs) or \
           any((scenario_dir / subdir / "Weather_Data").exists() for subdir in scenario_dir.iterdir() if subdir.is_dir()):
            # Single scenario
            scenarios = [scenario_dir]
        else:
            # Multiple scenarios
            scenarios = sorted([d for d in scenario_dir.iterdir() if d.is_dir()])

    # Optional scenario filtering
    scenarios = _filter_scenarios(scenarios, args.case_filter)
    if not scenarios:
        raise RuntimeError("No scenarios matched --case-filter")

    # Run PSO
    if len(scenarios) == 1 and not args.global_run:
        run_pso_scenario(str(scenarios[0]), args.t_cap, args.iters, args.particles, device, params_out=args.params_out)
        return

    if args.global_run:
        run_pso_multi_scenario([str(s) for s in scenarios], args.t_cap, args.iters, args.particles, device, params_out=args.params_out)
        return

    # Multi-scenario sequential (per-scenario coefficients)
    print(f"Processing {len(scenarios)} scenarios sequentially (PSO is GPU-intensive)...")
    for scenario in scenarios:
        try:
            run_pso_scenario(str(scenario), args.t_cap, args.iters, args.particles, device, params_out=args.params_out)
        except Exception as e:
            print(f"[!] Error processing {scenario.name}: {e}")


if __name__ == "__main__":
    main()
