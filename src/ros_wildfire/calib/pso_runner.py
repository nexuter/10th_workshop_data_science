from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch_pso import ParticleSwarmOptimizer

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.utils import ensure_dir, to_py_floats
from ros_wildfire.calib.objective import make_batched_dice_closure
from ros_wildfire.calib.mapping import pso_params_to_dict


def run_pso(
    cfg: ExperimentConfig,
    *,
    loader,
    lookups: dict,
    device: torch.device,
    log_dir: str = "./pso_logs",
):
    ensure_dir(log_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = f"{log_dir}/pso_trace_{run_id}.jsonl"
    csv_path = f"{log_dir}/pso_trace_{run_id}.csv"

    param_names = list(cfg.ros.param_names)
    pso_params = nn.ParameterDict({name: nn.Parameter(torch.tensor(cfg.ros.init_z0, device=device))
                                  for name in param_names})

    optimizer = ParticleSwarmOptimizer(
        params=pso_params.values(),
        inertial_weight=cfg.pso.inertial_weight,
        cognitive_coefficient=cfg.pso.cognitive_coefficient,
        social_coefficient=cfg.pso.social_coefficient,
        num_particles=cfg.pso.num_particles,
        max_param_value=cfg.pso.max_param_value,
        min_param_value=cfg.pso.min_param_value,
    )

    closure, set_batch = make_batched_dice_closure(
        cfg,
        optimizer=optimizer,
        lookups=lookups,
        pso_params=pso_params,
        device=device,
        cell_size_m=cfg.phys.cell_size_m,
        dt_seconds=cfg.phys.dt_seconds,
        burn_channel=cfg.data.burn_channel,
        use_new_ignition=cfg.eval.use_new_ignition,
        eps=cfg.eval.dice_eps,
    )

    history: List[Dict[str, Any]] = []
    baseline = to_py_floats(pso_params_to_dict(cfg, pso_params))

    data_iter = iter(loader)
    best_loss = None
    best_loss_global = None
    no_improve_count = 0
    iter_start_time = datetime.now()

    for it in range(cfg.pso.num_iters):
        batch_meta = []
        batch_stacks = []

        for _ in range(max(1, cfg.pso.batches_per_iter)):
            try:
                stacks_cpu, metas = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                stacks_cpu, metas = next(data_iter)
            batch_stacks.extend([s.to(device, non_blocking=True) for s in stacks_cpu])
            batch_meta.extend(metas)

        set_batch(batch_stacks)
        loss = optimizer.step(closure)
        loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)

        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val

        # Early stopping logic
        if best_loss_global is None:
            best_loss_global = loss_val
            no_improve_count = 0
        else:
            relative_improvement = (best_loss_global - loss_val) / (abs(best_loss_global) + 1e-8)
            if relative_improvement > cfg.pso.early_stop_threshold:
                best_loss_global = loss_val
                no_improve_count = 0
            else:
                no_improve_count += 1

        # Calculate iteration elapsed time
        iter_end_time = datetime.now()
        iter_elapsed = (iter_end_time - iter_start_time).total_seconds()

        mapped_now = to_py_floats(pso_params_to_dict(cfg, pso_params))
        delta_now = {k: mapped_now[k] - baseline.get(k, mapped_now[k]) for k in mapped_now}
        ex = batch_meta[0] if batch_meta else {}

        row = {
            "iter": it + 1,
            "loss": loss_val,
            "best": best_loss,
            "timestamp": iter_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "iter_elapsed_s": round(iter_elapsed, 2),
            "ex_scenario_dir": ex.get("scenario_dir"),
            "ex_case_root": ex.get("case_root"),
            "ex_idx": ex.get("idx"),
        }
        for k in param_names:
            row[f"{k}"] = mapped_now.get(k)
            row[f"d_{k}"] = delta_now.get(k)

        history.append(row)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # free CUDA tensors
        batch_stacks = None
        batch_meta = None
        if torch.cuda.is_available() and ((it + 1) % 5 == 0):
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print(f"[PSO] iter {it+1}/{cfg.pso.num_iters} loss={loss_val:.6f} best={best_loss:.6f} elapsed={iter_elapsed:.1f}s")

        # Check early stopping condition
        if no_improve_count >= cfg.pso.early_stop_patience:
            print(f"[+] Early stopping at iteration {it+1}/{cfg.pso.num_iters} "
                  f"(no improvement for {no_improve_count} iterations, best_loss={best_loss_global:.6f})")
            break
        
        # Reset timer for next iteration
        iter_start_time = datetime.now()

    # Write CSV
    if history:
        keys = sorted({k for r in history for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in history:
                w.writerow(r)

    mapped_final = to_py_floats(pso_params_to_dict(cfg, pso_params))
    return mapped_final, {"jsonl": jsonl_path, "csv": csv_path}
