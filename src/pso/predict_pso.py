import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.data.datasets import MultiIdxStackDataset
from ros_wildfire.data.discovery import discover_available_idxs, discover_idx_samples, resolve_case_root
from ros_wildfire.physics.frontier import get_all_frontiers
from ros_wildfire.physics.huygens import huygens_substeps_parallel
from ros_wildfire.physics.fuel_lookup import create_param_lookups
from ros_wildfire.viz.plots import visualize_fire_step


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(log_file_arg: str | None) -> tuple[Path, object, object]:
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file_arg:
        log_path = Path(log_file_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"predict_pso_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_fh = open(log_path, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = TeeStream(sys.stdout, log_fh)
    sys.stderr = TeeStream(sys.stderr, log_fh)
    print(f"[*] Logging to {log_path}")
    return log_path, old_stdout, old_stderr


def append_crop_performance_row(
    crop_log_file: Path,
    *,
    scenario_name: str,
    num_samples: int,
    avg_orig_h: float,
    avg_orig_w: float,
    avg_orig_area: float,
    avg_new_h: float,
    avg_new_w: float,
    avg_new_area: float,
    reduction_pct: float,
) -> None:
    crop_log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not crop_log_file.exists()
    with open(crop_log_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp",
                "scenario",
                "num_samples",
                "avg_non_crop_h",
                "avg_non_crop_w",
                "avg_non_crop_area",
                "avg_crop_h",
                "avg_crop_w",
                "avg_crop_area",
                "avg_area_reduction_pct",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            scenario_name,
            num_samples,
            f"{avg_orig_h:.4f}",
            f"{avg_orig_w:.4f}",
            f"{avg_orig_area:.4f}",
            f"{avg_new_h:.4f}",
            f"{avg_new_w:.4f}",
            f"{avg_new_area:.4f}",
            f"{reduction_pct:.4f}",
        ])


def load_params(params_file: str | None) -> dict | None:
    if not params_file:
        return None
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    required = {"k_A", "k_R0", "k_phi_s", "k_C", "k_B", "k_E"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError(f"Coefficients file missing keys: {missing}")
    return params


def print_image_shape(scenario_dir: str, idx: int, t_cap: int) -> None:
    cfg = ExperimentConfig()
    cfg.data.scenario_dir = scenario_dir
    cfg.data.use_multi_idx = True
    cfg.data.idx_list = (idx,)
    cfg.data.t_cap = t_cap
    cfg.data.drop_if_weather_short = True

    try:
        ds = MultiIdxStackDataset(cfg, scenario_dir=scenario_dir, idx_list=(idx,))
        sample = ds[0][0]
        shape = tuple(sample.shape)
        if len(shape) >= 4:
            print(f"[*] Image size: {shape[-2]}x{shape[-1]}")
        else:
            print(f"[*] Image shape: {shape}")
    except Exception as e:
        print(f"[!] Unable to read image shape: {e}")


def process_scenario(
    scenario_dir: str,
    t: int,
    t_cap: int,
    plot_dir: Path,
    device: str = "cpu",
    *,
    params: dict | None = None,
    crop: bool = False,
    all_t: bool = False,
    crop_log_file: Path | None = None,
    retry_on_oom: bool = True,
    batch_size_override: int | None = None,
):
    """Process all indices in a scenario and save visualizations.
    
    Args:
        scenario_dir: Path to scenario directory
        t: Time step to visualize (0-indexed, so t=30 means timestep 30 out of T total timesteps)
        t_cap: Maximum time cap for data loading
        plot_dir: Directory to save visualizations
        device: 'cuda' or 'cpu'
        params: Optional PSO coefficients dict for prediction visualization
        retry_on_oom: If True, retry on CPU if GPU runs out of memory
        batch_size_override: Override default batch size (default: 2 for physics, 12 for GT only)
    
    Output:
        - GT PNGs: *_gt.png
        - Pred PNGs: *_pred.png (when params are provided and prediction succeeds)
    """
    scenario_path = Path(scenario_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        case_root = resolve_case_root(scenario_dir)
        available = discover_available_idxs(case_root)
        if not available:
            print(f"[!] No indices found in {scenario_path.name}")
            return

        # Pre-filter indices using the same logic as the dataset
        # This avoids checking for plots for indices that will be filtered out anyway
        try:
            valid_samples = discover_idx_samples(
                scenario_dir,
                idx_list=available,
                t_cap=t_cap,
                drop_if_weather_short=True,
                strict=False
            )
            valid_indices = [idx for (_, idx) in valid_samples]
        except RuntimeError as e:
            print(f"[!] Error discovering valid indices: {e}")
            return
        
        if not valid_indices:
            print(f"[!] No valid indices in {scenario_path.name} (all filtered out)")
            return
        
        num_filtered = len(available) - len(valid_indices)
        if num_filtered > 0:
            print(f"[*] Pre-filtered: {num_filtered} indices excluded (incomplete data or short weather)")

        print(f"[*] Processing {scenario_path.name} with {len(valid_indices)} valid indices on {device.upper()}")
        files_per_idx = 2 if params else 1
        if all_t:
            print(f"    will generate multiple PNGs per index across all timesteps (up to t_cap={t_cap})")
        else:
            print(f"    will generate up to {len(valid_indices) * files_per_idx} PNGs ({files_per_idx} per valid index, timestep t={t})")

        # Get scenario name for output directory
        scenario_name = scenario_path.name
        if not scenario_name.endswith('_pkl'):
            # If passed inner directory (e.g., "0114"), use parent directory name
            scenario_name = scenario_path.parent.name
        
        scenario_id = scenario_name.split('_')[0]  # Extract scenario ID
        scenario_subdir = plot_dir / scenario_name
        scenario_subdir.mkdir(parents=True, exist_ok=True)
        
        # Check which valid indices already have plots and skip
        indices_to_process = []
        require_pred = params is not None
        for idx in valid_indices:
            if all_t or require_pred:
                indices_to_process.append(idx)
            else:
                out_no = min(max(t, 0) + 1, t_cap)
                gt_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt.png"
                gt_frontier_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt_frontier.png"
                pred_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_pred.png"
                gt_exists = gt_save_path.exists()
                gt_frontier_exists = gt_frontier_save_path.exists()
                pred_exists = pred_save_path.exists()
                done = gt_exists and gt_frontier_exists and (pred_exists if require_pred else True)
                if not done:
                    indices_to_process.append(idx)
        
        if not indices_to_process:
            print_image_shape(scenario_dir, valid_indices[0], t_cap)
            print(f"[*] All {len(valid_indices)} valid indices already have plots, skipping...")
            return
        
        if len(indices_to_process) < len(valid_indices):
            print(f"[*] Resuming: {len(indices_to_process)} indices remaining ({len(valid_indices) - len(indices_to_process)} already done)")

        lookups = create_param_lookups(torch.device(device)) if params else None

        cfg = ExperimentConfig()
        cfg.data.scenario_dir = scenario_dir
        cfg.data.use_multi_idx = True
        cfg.data.idx_list = tuple(indices_to_process)
        cfg.data.t_cap = t_cap
        cfg.data.drop_if_weather_short = True

        # Load dataset with only remaining indices
        ds = MultiIdxStackDataset(cfg, scenario_dir=scenario_dir, idx_list=tuple(indices_to_process))
        try:
            sample = ds[0][0]
            shape = tuple(sample.shape)
            if len(shape) >= 4:
                print(f"[*] Image size: {shape[-2]}x{shape[-1]}")
            else:
                print(f"[*] Image shape: {shape}")
        except Exception as e:
            print(f"[!] Unable to read image shape: {e}")
        
        if len(ds) == 0:
            print(f"[!] No valid samples in {scenario_path.name}")
            return

        # This shouldn't happen now since we pre-filtered, but check just in case
        if len(ds) < len(indices_to_process):
            print(f"[!] Warning: {len(indices_to_process) - len(ds)} indices unexpectedly filtered during loading")

        # Clamp t to valid range [0, T-1] where T=t_cap
        # NOTE: Some sequences may have T < t_cap, which will be caught during processing
        t_clamped = min(t, t_cap - 1)
        if t_clamped != t:
            print(f"[!] Warning: requested t={t} clamped to t={t_clamped} (max allowed: 0-{t_cap-1})")

        # Adaptive DataLoader params optimized for RTX 5080 (24 cores, 32 threads)
        if batch_size_override is not None:
            batch_size = batch_size_override
        
        if device == "cuda":
            if batch_size_override is None:
                # Set default batch_size if not overridden
                if os.name == "nt":
                    batch_size = 2 if params else 12
                else:
                    batch_size = 2 if params else 8
            num_workers = 6  # RTX 5080: 6-8 workers (24 cores available)
        else:
            if batch_size_override is None:
                batch_size = 1  # CPU processes one-by-one
            num_workers = 4  # CPU async loading (plenty of cores)

        # Windows shared-memory can fail with DataLoader workers (error 1455)
        if os.name == "nt":
            num_workers = 0

        dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        print(f"[*] DataLoader: batch_size={batch_size}, num_workers={num_workers}")

        # Process all indices
        total = len(ds)
        processed = 0
        crop_orig_areas: list[int] = []
        crop_new_areas: list[int] = []
        crop_orig_h: list[int] = []
        crop_orig_w: list[int] = []
        crop_new_h: list[int] = []
        crop_new_w: list[int] = []
        
        for batch_data, batch_meta in dataloader:
            batch_size_actual = batch_data.shape[0]
            
            # Move batch to device
            if device == "cuda":
                batch_data = batch_data.cuda()
            
            # Process each item in batch
            for j in range(batch_size_actual):
                full_4d = batch_data[j]
                idx = None
                if isinstance(batch_meta, dict) and "idx" in batch_meta:
                    idx_field = batch_meta["idx"]
                    if torch.is_tensor(idx_field):
                        idx = int(idx_field[j].item())
                    elif isinstance(idx_field, (list, tuple)):
                        idx = int(idx_field[j])
                    else:
                        idx = int(idx_field)
                elif isinstance(batch_meta, (list, tuple)) and j < len(batch_meta):
                    item_meta = batch_meta[j]
                    if isinstance(item_meta, dict) and "idx" in item_meta:
                        idx = int(item_meta["idx"])

                if idx is None:
                    idx = indices_to_process[processed] if processed < len(indices_to_process) else processed
                
                # Check actual sequence length first
                seq_len = full_4d.shape[1]  # Dimension 1 is T
                if t_clamped >= seq_len:
                    print(f"  [{processed+1:3d}/{total}] Skipping index {idx} (sequence too short: {seq_len} timesteps, requested t={t_clamped})")
                    processed += 1
                    continue
                
                try:
                    # Get frontiers; now safe because sequence is long enough
                    frontiers = get_all_frontiers(full_4d[9])

                    pred_burn = None
                    pred_frontiers = None
                    files_written = 0
                    frames_written = 0
                    frames_skipped = 0
                    if params:
                        try:
                            pred_burn_raw, _, _ = huygens_substeps_parallel(
                                full_stack=full_4d,
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
                            pred_burn = pred_burn_raw.bool()
                            del pred_burn_raw  # Free the raw tensor immediately
                            pred_frontiers = get_all_frontiers(pred_burn)
                        except (RuntimeError, IndexError) as e:
                            if "out of memory" in str(e).lower() and device == "cuda" and retry_on_oom:
                                print(f"  [{processed+1:3d}/{total}] OOM on CUDA, retrying index {idx} on CPU...")
                                full_4d_cpu = full_4d.cpu()
                                lookups_cpu = create_param_lookups(torch.device("cpu"))
                                try:
                                    pred_burn_raw, _, _ = huygens_substeps_parallel(
                                        full_stack=full_4d_cpu,
                                        lookups=lookups_cpu,
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
                                    pred_burn = pred_burn_raw.bool()
                                    del pred_burn_raw  # Free the raw tensor immediately
                                    pred_frontiers = get_all_frontiers(pred_burn)
                                    torch.cuda.empty_cache()
                                except Exception:
                                    # Skip prediction for this sample
                                    print(f"  [{processed+1:3d}/{total}] Skipping prediction for index {idx} (sequence length issue)")
                                    pred_burn = None
                                    pred_frontiers = None
                            else:
                                # Skip prediction but continue with visualization
                                print(f"  [{processed+1:3d}/{total}] Skipping prediction for index {idx} (huygens error)")
                                pred_burn = None
                                pred_frontiers = None

                    if params and pred_burn is not None:
                        pred_seq_len = int(pred_burn.shape[0])
                        pred_t_values = list(range(pred_seq_len)) if all_t else [min(t_clamped, pred_seq_len - 1)]
                        for pred_t in pred_t_values:
                            gt_t = pred_t + 1
                            out_no = gt_t + 1
                            gt_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt.png"
                            gt_frontier_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt_frontier.png"
                            pred_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_pred.png"
                            pred_frontier_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_pred_frontier.png"
                            if all_t and gt_save_path.exists() and gt_frontier_save_path.exists() and pred_save_path.exists() and pred_frontier_save_path.exists():
                                frames_skipped += 1
                                continue

                            vis_stats = visualize_fire_step(
                                full_4d,
                                frontiers,
                                t=gt_t,
                                pred_burn=pred_burn,
                                pred_frontiers=pred_frontiers,
                                pred_t=pred_t,
                                frame_label=f"out{out_no}",
                                gt_save_path=str(gt_save_path),
                                pred_save_path=str(pred_save_path),
                                gt_frontier_save_path=str(gt_frontier_save_path),
                                pred_frontier_save_path=str(pred_frontier_save_path),
                                crop=crop,
                            )

                            if crop and isinstance(vis_stats, dict):
                                orig_h, orig_w = vis_stats.get("orig_hw", (0, 0))
                                new_h, new_w = vis_stats.get("crop_hw", (0, 0))
                                if orig_h > 0 and orig_w > 0 and new_h > 0 and new_w > 0:
                                    crop_orig_h.append(int(orig_h))
                                    crop_orig_w.append(int(orig_w))
                                    crop_new_h.append(int(new_h))
                                    crop_new_w.append(int(new_w))
                                    crop_orig_areas.append(int(orig_h) * int(orig_w))
                                    crop_new_areas.append(int(new_h) * int(new_w))

                            files_written += 4
                            frames_written += 1
                    else:
                        gt_t_values = list(range(seq_len)) if all_t else [t_clamped]
                        for gt_t in gt_t_values:
                            out_no = gt_t + 1
                            gt_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt.png"
                            gt_frontier_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_gt_frontier.png"
                            pred_save_path = scenario_subdir / f"{scenario_id}_{idx:05d}_out{out_no}_pred.png"
                            if all_t and gt_save_path.exists() and gt_frontier_save_path.exists():
                                frames_skipped += 1
                                continue

                            vis_stats = visualize_fire_step(
                                full_4d,
                                frontiers,
                                t=gt_t,
                                pred_burn=None,
                                pred_frontiers=None,
                                frame_label=f"out{out_no}",
                                gt_save_path=str(gt_save_path),
                                pred_save_path=None,
                                gt_frontier_save_path=str(gt_frontier_save_path),
                                pred_frontier_save_path=None,
                                crop=crop,
                            )

                            if crop and isinstance(vis_stats, dict):
                                orig_h, orig_w = vis_stats.get("orig_hw", (0, 0))
                                new_h, new_w = vis_stats.get("crop_hw", (0, 0))
                                if orig_h > 0 and orig_w > 0 and new_h > 0 and new_w > 0:
                                    crop_orig_h.append(int(orig_h))
                                    crop_orig_w.append(int(orig_w))
                                    crop_new_h.append(int(new_h))
                                    crop_new_w.append(int(new_w))
                                    crop_orig_areas.append(int(orig_h) * int(orig_w))
                                    crop_new_areas.append(int(new_h) * int(new_w))

                            files_written += 2
                            frames_written += 1

                    if params and pred_burn is not None:
                        print(
                            f"  [{processed+1:3d}/{total}] idx={idx:05d} "
                            f"frames_written={frames_written} files_written={files_written} "
                            f"frames_skipped={frames_skipped} (gt,gt_frontier,pred,pred_frontier)"
                        )
                    else:
                        print(
                            f"  [{processed+1:3d}/{total}] idx={idx:05d} "
                            f"frames_written={frames_written} files_written={files_written} "
                            f"frames_skipped={frames_skipped} (gt,gt_frontier)"
                        )
                    
                    # Clean up tensors to free memory
                    del full_4d, frontiers
                    if pred_burn is not None:
                        del pred_burn
                    if pred_frontiers is not None:
                        del pred_frontiers
                    
                    # Close matplotlib figures to prevent memory accumulation
                    import matplotlib.pyplot as plt
                    plt.close('all')
                        
                except Exception as e:
                    print(f"  [{processed+1:3d}/{total}] Error with index {idx}: {e}")
                
                processed += 1
            
            # Clean up batch and free GPU memory after each batch
            del batch_data
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection
            
            # Extra GPU cleanup to prevent memory fragmentation
            if device == "cuda":
                torch.cuda.empty_cache()

        if crop:
            if crop_orig_areas:
                n = len(crop_orig_areas)
                avg_orig_h = sum(crop_orig_h) / n
                avg_orig_w = sum(crop_orig_w) / n
                avg_new_h = sum(crop_new_h) / n
                avg_new_w = sum(crop_new_w) / n
                avg_orig_area = sum(crop_orig_areas) / n
                avg_new_area = sum(crop_new_areas) / n
                reduction_pct = (1.0 - (avg_new_area / avg_orig_area)) * 100.0 if avg_orig_area > 0 else 0.0
                print(f"[*] Crop performance ({scenario_name}):")
                print(f"    avg non-crop size: {avg_orig_h:.1f}x{avg_orig_w:.1f} px (area {avg_orig_area:.1f})")
                print(f"    avg crop size:     {avg_new_h:.1f}x{avg_new_w:.1f} px (area {avg_new_area:.1f})")
                print(f"    avg area reduction: {reduction_pct:.2f}%")
                if crop_log_file is not None:
                    append_crop_performance_row(
                        crop_log_file,
                        scenario_name=scenario_name,
                        num_samples=n,
                        avg_orig_h=avg_orig_h,
                        avg_orig_w=avg_orig_w,
                        avg_orig_area=avg_orig_area,
                        avg_new_h=avg_new_h,
                        avg_new_w=avg_new_w,
                        avg_new_area=avg_new_area,
                        reduction_pct=reduction_pct,
                    )
                    print(f"    crop metrics saved: {crop_log_file}")
            else:
                print(f"[*] Crop performance ({scenario_name}): no new images processed, nothing to compare")

    except Exception as e:
        if "out of memory" in str(e).lower() and device == "cuda" and retry_on_oom:
            print(f"[!] GPU OOM on {scenario_path.name}, retrying entire scenario on CPU...")
            torch.cuda.empty_cache()
            try:
                process_scenario(
                    scenario_dir,
                    t,
                    t_cap,
                    plot_dir,
                    device="cpu",
                    params=params,
                    crop=crop,
                    all_t=all_t,
                    crop_log_file=crop_log_file,
                    retry_on_oom=False,
                )
                return
            except Exception as e2:
                print(f"[!] Error on CPU retry for {scenario_path.name}: {e2}")
        else:
            print(f"[!] Error processing {scenario_path.name}: {e}")


def build_python_executable() -> str:
    venv_python = Path("C:/Users/jk2347/Workspace/ros-based-wildfire-prediction/.venv/Scripts/python.exe")
    return str(venv_python) if venv_python.exists() else sys.executable


def run_scenario_subprocess(args: argparse.Namespace, scenario_path: Path, plot_dir: Path) -> bool:
    python_exe = build_python_executable()
    cmd = [
        python_exe,
        "scripts/pso/predict_pso.py",
        "--scenario-dir", str(scenario_path),
        "--t", str(args.t),
        "--t-cap", str(args.t_cap),
        "--plot-dir", str(plot_dir),
    ]
    if args.params_file:
        cmd.extend(["--params-file", args.params_file])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.crop:
        cmd.append("--crop")
    if args.all_t:
        cmd.append("--all-t")
    if args.log_file:
        cmd.extend(["--log-file", args.log_file])
    if args.crop_log_file:
        cmd.extend(["--crop-log-file", args.crop_log_file])

    env = os.environ.copy()
    env["WF_CHILD"] = "1"
    try:
        subprocess.run(cmd, env=env, check=False, timeout=900)
        return True
    except subprocess.TimeoutExpired:
        print("time out")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", required=True, help="Path to single scenario or directory with multiple scenarios")
    ap.add_argument("--t", type=int, default=48, help="Time step to visualize (default: final timestep)")
    ap.add_argument("--t-cap", type=int, default=48, help="Maximum time cap for data loading")
    ap.add_argument("--plot-dir", type=str, default=None, help="Directory to save visualizations")
    ap.add_argument("--list-indices", action="store_true", help="List available indices and exit")
    ap.add_argument("--params-file", default=None, help="JSON file with PSO coefficients to visualize predictions")
    ap.add_argument("--batch-size", type=int, default=None, help="Override default batch size (default: 2 for physics, 12 for GT only)")
    ap.add_argument("--crop", action="store_true", help="Crop output to burn-area bounding box (+5 px margin), shared for GT and Pred")
    ap.add_argument("--all-t", action="store_true", help="Generate images for all timesteps in each sequence")
    ap.add_argument("--log-file", default=None, help="Path to save run logs (disabled by default)")
    ap.add_argument("--crop-log-file", default=None, help="Path to save crop performance CSV (default: ./logs/crop_performance_YYYYmmdd_HHMMSS.csv)")
    args = ap.parse_args()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    if args.log_file:
        log_path, old_stdout, old_stderr = setup_run_logging(args.log_file)
        args.log_file = str(log_path)
    if args.crop_log_file:
        crop_log_file = Path(args.crop_log_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_log_file = Path("./logs") / f"crop_performance_{stamp}.csv"
    args.crop_log_file = str(crop_log_file)

    try:
        scenario_dir = Path(args.scenario_dir)
        if args.plot_dir:
            plot_dir = Path(args.plot_dir)
        else:
            plot_dir = Path("./output/fire_plots_crop" if args.crop else "./output/fire_plots")

        # Handle --list-indices mode
        if args.list_indices:
            try:
                case_root = resolve_case_root(str(scenario_dir))
                available = discover_available_idxs(case_root)
                print(f"Available indices for {scenario_dir.name}:")
                print(f"  {available}")
                return
            except Exception as e:
                print(f"Error: {e}")
                return

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

        # Parent process: run scenarios with timeout enforcement
        if os.environ.get("WF_CHILD") != "1":
            run_start = time.perf_counter()
            scenario_times: list[tuple[str, float]] = []
            for i, scenario in enumerate(scenarios, 1):
                print(f"\n[{i}/{len(scenarios)}] Processing {scenario.name}...")
                scenario_start = time.perf_counter()
                ok = run_scenario_subprocess(args, scenario, plot_dir)
                scenario_elapsed = time.perf_counter() - scenario_start
                scenario_times.append((scenario.name, scenario_elapsed))
                if not ok and len(scenarios) == 1:
                    return
            total_elapsed = time.perf_counter() - run_start
            lines = []
            lines.append("=== PSO Performance Report ===")
            lines.append("\n--- Per-Scenario Timing ---")
            for name, elapsed in scenario_times:
                lines.append(f"  {name}: elapsed={elapsed:.2f}s")
            lines.append(f"\ntotal_running_time_sec: {total_elapsed:.2f}")
            report_text = "\n".join(lines)
            print("\n" + report_text)
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = log_dir / f"pso_performance_{stamp}.txt"
            report_path.write_text(report_text, encoding="utf-8")
            print(f"[*] Performance report saved: {report_path}")
            return

        # Child process: do actual work
        params = load_params(args.params_file) if args.params_file else None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Processing {len(scenarios)} scenarios serially on {device.upper()}...")

        child_run_start = time.perf_counter()
        child_scenario_times: list[tuple[str, float]] = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] Processing {scenario.name}...")
            scenario_start = time.perf_counter()
            process_scenario(
                str(scenario),
                args.t,
                args.t_cap,
                plot_dir,
                device,
                params=params,
                crop=args.crop,
                all_t=args.all_t,
                crop_log_file=crop_log_file,
                batch_size_override=args.batch_size,
            )
            scenario_elapsed = time.perf_counter() - scenario_start
            child_scenario_times.append((scenario.name, scenario_elapsed))
            print(f"[*] {scenario.name}: scenario_elapsed={scenario_elapsed:.2f}s")
        child_total_elapsed = time.perf_counter() - child_run_start
        lines = []
        lines.append("=== PSO Performance Report (child) ===")
        lines.append("\n--- Per-Scenario Timing ---")
        for name, elapsed in child_scenario_times:
            lines.append(f"  {name}: elapsed={elapsed:.2f}s")
        lines.append(f"\ntotal_running_time_sec: {child_total_elapsed:.2f}")
        report_text = "\n".join(lines)
        print("\n" + report_text)
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = log_dir / f"pso_performance_child_{stamp}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"[*] Performance report saved: {report_path}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == "__main__":
    main()
