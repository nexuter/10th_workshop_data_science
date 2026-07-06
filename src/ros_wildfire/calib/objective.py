from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_pso import ParticleSwarmOptimizer

from ros_wildfire.config import ExperimentConfig
from ros_wildfire.calib.mapping import pso_params_to_dict
from ros_wildfire.physics.huygens import huygens_substeps_parallel


def make_batched_dice_closure(
    cfg: ExperimentConfig,
    *,
    optimizer: ParticleSwarmOptimizer,
    lookups: dict,
    pso_params,
    device: torch.device,
    cell_size_m: float,
    dt_seconds: float,
    burn_channel: int,
    use_new_ignition: bool,
    eps: float,
):
    current_batch = []

    def set_batch(batch_stacks_on_device):
        nonlocal current_batch
        current_batch = batch_stacks_on_device

    def downsample_stack(stack: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample spatial dimensions by factor using bilinear interpolation."""
        if factor == 1:
            return stack
        C, T, H, W = stack.shape
        h_new, w_new = H // factor, W // factor
        # Reshape to (C*T, 1, H, W) for interpolation
        stack_flat = stack.view(C * T, 1, H, W)
        downsampled = F.interpolate(stack_flat, size=(h_new, w_new), mode='bilinear', align_corners=False)
        return downsampled.view(C, T, h_new, w_new)

    def upsample_tensor(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Upsample tensor back to target resolution."""
        if tensor.shape[-2:] == (target_h, target_w):
            return tensor
        # Handle both 3D (B, H, W) and 4D (B, T, H, W) tensors
        if tensor.ndim == 3:
            B, H, W = tensor.shape
            tensor_unsqueezed = tensor.unsqueeze(1)  # (B, 1, H, W)
            upsampled = F.interpolate(tensor_unsqueezed, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return upsampled.squeeze(1)  # (B, H, W)
        else:  # 4D
            B, T, H, W = tensor.shape
            tensor_flat = tensor.view(B * T, 1, H, W)
            upsampled = F.interpolate(tensor_flat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return upsampled.view(B, T, target_h, target_w)

    def closure():
        optimizer.zero_grad()
        params = pso_params_to_dict(cfg, pso_params)

        losses = []
        for full_stack in current_batch:
            # Store original resolution for later upsampling
            orig_h, orig_w = full_stack.shape[-2:]
            
            # Downsample for faster evaluation if configured
            if cfg.pso.grid_downsample > 1:
                stack_eval = downsample_stack(full_stack, cfg.pso.grid_downsample)
            else:
                stack_eval = full_stack

            pred_burn, _, _ = huygens_substeps_parallel(
                full_stack=stack_eval,
                lookups=lookups,
                params=params,
                cell_size_m=cell_size_m,
                dt_seconds=dt_seconds,
                n_substeps=cfg.phys.base_n_substeps,
                a_a=cfg.phys.a_a,
                n_theta=cfg.phys.n_theta,
                head_bins_px=cfg.phys.head_bins_px,
                lb_bins=cfg.phys.lb_bins,
                max_head_px=cfg.phys.max_head_px,
            )

            # Upsample predictions back to original resolution for loss computation
            if cfg.pso.grid_downsample > 1:
                pred_burn_float = pred_burn.float()  # Convert to float for interpolation
                pred_burn_upsampled = upsample_tensor(pred_burn_float, orig_h, orig_w)
                pred = pred_burn_upsampled.bool()  # Convert back to bool
            else:
                pred = pred_burn.bool()

            # Ground truth uses original full_stack
            gt = full_stack[burn_channel][1:].bool()

            if use_new_ignition:
                burn_t = full_stack[burn_channel][:-1].bool()
                pred = pred & (~burn_t)
                gt = gt & (~burn_t)

            tp = (pred & gt).sum(dim=(1, 2)).float()
            p = pred.sum(dim=(1, 2)).float()
            g = gt.sum(dim=(1, 2)).float()
            dice_t = (2.0 * tp + eps) / (p + g + eps)
            losses.append(1.0 - dice_t.mean())

        if not losses:
            return torch.tensor(1.0, device=device)
        return torch.stack(losses).mean()

    return closure, set_batch

    def downsample_stack(stack: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample spatial dimensions by factor using bilinear interpolation."""
        if factor == 1:
            return stack
        C, T, H, W = stack.shape
        h_new, w_new = H // factor, W // factor
        # Reshape for interpolation: (C*T, 1, H, W)
        stack_flat = stack.view(C * T, 1, H, W)
        downsampled = F.interpolate(stack_flat, size=(h_new, w_new), mode='bilinear', align_corners=False)
        return downsampled.view(C, T, h_new, w_new)

    def upsample_predictions(pred: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Upsample predictions to target resolution using bilinear interpolation."""
        if pred.shape[-2:] == (target_h, target_w):
            return pred
        # pred shape: (B, T, H, W) or (B, H, W)
        if pred.ndim == 4:
            B, T, H, W = pred.shape
            pred_flat = pred.view(B * T, 1, H, W)
            upsampled = F.interpolate(pred_flat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return upsampled.view(B, T, target_h, target_w)
        else:
            pred_unsqueezed = pred.unsqueeze(1)
            upsampled = F.interpolate(pred_unsqueezed, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return upsampled.squeeze(1)

    def closure():
        optimizer.zero_grad()
        params = pso_params_to_dict(cfg, pso_params)

        losses = []
        for full_stack in current_batch:
            # Store original resolution
            orig_h, orig_w = full_stack.shape[-2:]
            
            # Downsample for faster evaluation
            if grid_downsample > 1:
                stack_eval = downsample_stack(full_stack, grid_downsample)
            else:
                stack_eval = full_stack

            pred_burn, _, _ = huygens_substeps_parallel(
                full_stack=stack_eval,
                lookups=lookups,
                params=params,
                cell_size_m=cell_size_m,
                dt_seconds=dt_seconds,
                n_substeps=cfg.phys.base_n_substeps,
                a_a=cfg.phys.a_a * float(params.get("k_a_a", 1.0).detach().cpu().item()) if "k_a_a" in params else cfg.phys.a_a,
                n_theta=cfg.phys.n_theta,
                head_bins_px=cfg.phys.head_bins_px,
                lb_bins=cfg.phys.lb_bins,
                max_head_px=cfg.phys.max_head_px,
            )

            # Upsample predictions back to original resolution for loss computation (do this on float before bool conversion)
            if grid_downsample > 1:
                pred_burn = upsample_predictions(pred_burn.float(), orig_h, orig_w)

            pred = pred_burn.bool()
            gt = full_stack[burn_channel][1:].bool()

            if use_new_ignition:
                burn_t = full_stack[burn_channel][:-1].bool()
                pred = pred & (~burn_t)
                gt = gt & (~burn_t)

            tp = (pred & gt).sum(dim=(1, 2)).float()
            p = pred.sum(dim=(1, 2)).float()
            g = gt.sum(dim=(1, 2)).float()
            dice_t = (2.0 * tp + eps) / (p + g + eps)
            losses.append(1.0 - dice_t.mean())

        if not losses:
            return torch.tensor(1.0, device=device)
        return torch.stack(losses).mean()

    return closure, set_batch
