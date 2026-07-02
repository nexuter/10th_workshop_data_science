import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import gc
import re

import torch

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.data.collate import build_loader
from ros_wildfire.physics.fuel_lookup import create_param_lookups
from ros_wildfire.physics.huygens import huygens_substeps_parallel
from ros_wildfire.eval.metrics import binary_metrics_over_time, average_precision
from ros_wildfire.eval.reporting import format_metrics


def load_params(args) -> dict:
    if not args.params_file:
        raise ValueError("--params-file is required (no default coefficients allowed)")

    with open(args.params_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    required = {"k_A", "k_R0", "k_phi_s", "k_C", "k_B", "k_E"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError(f"Coefficients file missing keys: {missing}")

    return params


def infer_scenarios(path: Path) -> list[Path]:
    subdirs = {d.name for d in path.iterdir() if d.is_dir()}
    if {"Weather_Data", "Vegetation_Map", "Topography_Map", "Fuel_Map"}.issubset(subdirs) or \
       any((path / subdir / "Weather_Data").exists() for subdir in path.iterdir() if subdir.is_dir()):
        return [path]
    return sorted([d for d in path.iterdir() if d.is_dir()])


def get_next_eval_file(scenario_dir: Path, base_name: str) -> tuple[Path, int]:
    """Find next evaluation file number for a scenario.
    
    Returns (csv_path, file_number)
    """
    existing = sorted(scenario_dir.glob(f"{base_name}_*.csv"))
    if not existing:
        return scenario_dir / f"{base_name}_1.csv", 1
    
    # Extract numbers from filenames like eval_0001_1.csv
    pattern = re.compile(rf"{base_name}_(\d+)\.csv")
    numbers = [int(pattern.match(f.name).group(1)) for f in existing if pattern.match(f.name)]
    next_num = max(numbers) + 1 if numbers else 1
    return scenario_dir / f"{base_name}_{next_num}.csv", next_num


def load_evaluated_idxs(csv_path: Path) -> set[int]:
    """Load already-evaluated indices from CSV file."""
    if not csv_path.exists():
        return set()
    
    try:
        evaluated = set()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                evaluated.add(int(row["idx"]))
        return evaluated
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", required=True)
    ap.add_argument("--t-cap", type=int, default=48)
    ap.add_argument("--params-file", required=True, help="Path to JSON file of PSO params")
    ap.add_argument("--out-dir", default="./eval_reports")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available indices per scenario (no max_runs_per_scenario cap)",
    )
    ap.add_argument(
        "--max-runs-per-scenario",
        type=int,
        default=None,
        help="Override per-scenario sample cap (default: config value)",
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Run GC and CUDA cache cleanup after each sample",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing evaluations and create new numbered file",
    )
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.data.t_cap = args.t_cap
    cfg.data.shuffle_samples = False
    cfg.data.random_per_scenario = False  # Always use sequential order, not random sampling
    cfg.pso.batch_size = 1  # Set to 1 for evaluation (prevent OOM)
    cfg.runtime.seed_everything()
    device = cfg.runtime.torch_device()

    if args.all:
        cfg.data.max_runs_per_scenario = None
        cfg.data.idx_list = None
    elif args.max_runs_per_scenario is not None:
        cfg.data.max_runs_per_scenario = args.max_runs_per_scenario

    params = load_params(args)
    lookups = create_param_lookups(device)

    scenario_dir = Path(args.scenario_dir)
    scenarios = infer_scenarios(scenario_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_path in scenarios:
        # Load one scenario at a time to minimize memory footprint
        cfg.data.scenario_dir = str(scenario_path)
        ds, dl = build_loader(cfg, scenario_dirs=None)

        # Use parent folder name if scenario_path is a direct child of a *_pkl folder
        # (e.g., "0114_00004_pkl/0114" should be named "0114_00004_pkl")
        if scenario_path.parent.name.endswith("_pkl"):
            scenario_name = scenario_path.parent.name
        else:
            scenario_name = scenario_path.name
        scenario_out_dir = out_dir / scenario_name
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        # Determine eval file path
        if args.refresh:
            eval_csv_path, file_num = get_next_eval_file(scenario_out_dir, "eval")
            evaluated_idxs = set()
            is_new_file = True
        else:
            # Check if eval_1.csv or eval_2.csv etc. exists
            existing = sorted(scenario_out_dir.glob("eval_*.csv"))
            if existing:
                eval_csv_path = existing[-1]  # Use most recent
                evaluated_idxs = load_evaluated_idxs(eval_csv_path)
                is_new_file = False
                file_num = int(re.search(r"eval_(\d+)\.csv", eval_csv_path.name).group(1))
            else:
                eval_csv_path, file_num = get_next_eval_file(scenario_out_dir, "eval")
                evaluated_idxs = set()
                is_new_file = True

        # Write header if new file
        write_header = is_new_file or not eval_csv_path.exists()
        sample_count = 0

        with open(eval_csv_path, "a", encoding="utf-8", newline="") as csvf:
            writer = None

            for batch_stacks, batch_meta in dl:
                for full_stack, meta in zip(batch_stacks, batch_meta):
                    idx = meta.get("idx")
                    
                    # Skip if already evaluated
                    if idx in evaluated_idxs and not args.refresh:
                        continue

                    full_stack = full_stack.to(device)

                    pred_burn, _, _ = huygens_substeps_parallel(
                        full_stack=full_stack,
                        lookups=lookups,
                        params=params,
                        cell_size_m=cfg.phys.cell_size_m,
                        dt_seconds=cfg.phys.dt_seconds,
                        n_substeps=cfg.phys.base_n_substeps,
                        a_a=cfg.phys.a_a,
                        n_theta=cfg.phys.n_theta,
                        head_bins_px=cfg.phys.head_bins_px,
                        lb_bins=cfg.phys.lb_bins,
                        max_head_px=cfg.phys.max_head_px,
                    )

                    pred = pred_burn.bool()
                    gt = full_stack[cfg.data.burn_channel][1:].bool()

                    if cfg.eval.use_new_ignition:
                        burn_t = full_stack[cfg.data.burn_channel][:-1].bool()
                        pred = pred & (~burn_t)
                        gt = gt & (~burn_t)

                    m = binary_metrics_over_time(pred, gt)
                    m["auprc"] = average_precision(pred.float(), gt).item()

                    row = {
                        "scenario_dir": meta.get("scenario_dir"),
                        "case_root": meta.get("case_root"),
                        "idx": meta.get("idx"),
                        **{k: float(v) if torch.is_tensor(v) else v for k, v in m.items() if not k.endswith("_t")},
                    }

                    if writer is None:
                        writer = csv.DictWriter(csvf, fieldnames=list(row.keys()))
                        if write_header:
                            writer.writeheader()

                    writer.writerow(row)
                    sample_count += 1

                    print(f"[eval] {scenario_name} idx={idx} :: {format_metrics({**m, 'auprc': m['auprc']})}")

                    if args.cleanup:
                        del full_stack, pred_burn, pred, gt
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # Clear dataset/loader for this scenario to free memory
        del ds, dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if sample_count > 0:
            print(f"[eval] {scenario_name}: Evaluated {sample_count} samples -> {eval_csv_path}")
        else:
            if evaluated_idxs:
                print(f"[eval] {scenario_name}: All {len(evaluated_idxs)} samples already evaluated (skip)")
            else:
                print(f"[eval] {scenario_name}: No samples to evaluate")


if __name__ == "__main__":
    main()
