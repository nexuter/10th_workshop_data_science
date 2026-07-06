from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


def visualize_fire_step(
    full_stack: torch.Tensor,
    frontiers: torch.Tensor,
    t: int,
    *,
    pred_burn: torch.Tensor | None = None,
    pred_frontiers: torch.Tensor | None = None,
    pred_t: int | None = None,
    frame_label: str | None = None,
    gt_save_path: str | None = None,
    pred_save_path: str | None = None,
    gt_frontier_save_path: str | None = None,
    pred_frontier_save_path: str | None = None,
    crop: bool = False,
    crop_margin: int = 5,
    save_path: str | None = None,
):
    plt.close('all')

    seq_len_frontiers = frontiers.shape[0]
    seq_len_burn = full_stack.shape[1]
    seq_len_gt = min(seq_len_frontiers, seq_len_burn)
    t_actual = min(t, seq_len_gt - 1)

    pred_t_actual = None
    if pred_burn is not None:
        pred_seq = pred_burn.shape[0]
        pred_t_actual = min(pred_t if pred_t is not None else t, pred_seq - 1)
    elif pred_frontiers is not None:
        pred_seq = pred_frontiers.shape[0]
        pred_t_actual = min(pred_t if pred_t is not None else t, pred_seq - 1)

    burn_map_t = full_stack[9, t_actual].detach().cpu()
    orig_h, orig_w = int(burn_map_t.shape[0]), int(burn_map_t.shape[1])
    frontier_t = frontiers[t_actual].detach().cpu()
    pred_map_t = None
    pred_frontier_t = None
    if pred_burn is not None:
        pred_map_t = pred_burn[pred_t_actual].detach().cpu()
    if pred_frontiers is not None:
        pred_frontier_t = pred_frontiers[pred_t_actual].detach().cpu()

    if crop:
        nonzero = [burn_map_t > 0, frontier_t > 0]
        if pred_map_t is not None:
            nonzero.append(pred_map_t > 0)
        if pred_frontier_t is not None:
            nonzero.append(pred_frontier_t > 0)

        all_nonzero = torch.zeros_like(burn_map_t, dtype=torch.bool)
        for mask in nonzero:
            all_nonzero |= mask

        if torch.any(all_nonzero):
            ys, xs = torch.where(all_nonzero)
            y_min = max(int(ys.min().item()) - crop_margin, 0)
            y_max = min(int(ys.max().item()) + crop_margin + 1, burn_map_t.shape[0])
            x_min = max(int(xs.min().item()) - crop_margin, 0)
            x_max = min(int(xs.max().item()) + crop_margin + 1, burn_map_t.shape[1])

            burn_map_t = burn_map_t[y_min:y_max, x_min:x_max]
            frontier_t = frontier_t[y_min:y_max, x_min:x_max]
            if pred_map_t is not None:
                pred_map_t = pred_map_t[y_min:y_max, x_min:x_max]
            if pred_frontier_t is not None:
                pred_frontier_t = pred_frontier_t[y_min:y_max, x_min:x_max]

    crop_h, crop_w = int(burn_map_t.shape[0]), int(burn_map_t.shape[1])

    burn_map = burn_map_t.numpy()
    frontier = frontier_t.numpy()
    pred_map = None
    pred_frontier = None
    if pred_map_t is not None:
        pred_map = pred_map_t.numpy()
    if pred_frontier_t is not None:
        pred_frontier = pred_frontier_t.numpy()

    has_pred = pred_map is not None
    t_label = frame_label if frame_label is not None else (f"t={t_actual}" if t_actual == t else f"t={t_actual} (requested {t})")

    def _is_binary(arr: np.ndarray) -> bool:
        if arr.dtype == np.bool_:
            return True
        vals = np.unique(arr)
        if vals.size == 0 or vals.size > 2:
            return False
        return np.all(np.isclose(vals, 0.0) | np.isclose(vals, 1.0))

    gt_frontier_is_binary = _is_binary(frontier)
    gt_frontier_cmap = "gray" if gt_frontier_is_binary else "YlOrRd"

    def _save_binary_mask(path: str | None, arr: np.ndarray):
        if path is None:
            return
        mask_u8 = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(mask_u8, mode="L").save(path)

    if gt_save_path is not None or pred_save_path is not None or gt_frontier_save_path is not None or pred_frontier_save_path is not None:
        _save_binary_mask(gt_save_path, burn_map)
        _save_binary_mask(gt_frontier_save_path, frontier)
        if has_pred:
            if pred_frontier is None:
                pred_frontier = pred_map > 0.5
            _save_binary_mask(pred_save_path, pred_map)
            _save_binary_mask(pred_frontier_save_path, pred_frontier)
        plt.close('all')
        return {
            "orig_hw": (orig_h, orig_w),
            "crop_hw": (crop_h, crop_w),
            "t_actual": int(t_actual),
        }

    ncols = 4 if has_pred else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    axes[0].imshow(burn_map, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"GT Burn ({t_label})")
    axes[0].axis("off")

    axes[1].imshow(frontier, cmap=gt_frontier_cmap, interpolation="nearest")
    axes[1].set_title(f"GT Frontier ({t_label})")
    axes[1].axis("off")

    if has_pred:
        axes[2].imshow(pred_map, cmap="gray", interpolation="nearest")
        axes[2].set_title(f"Pred Burn ({t_label})")
        axes[2].axis("off")

        if pred_frontier is None:
            pred_frontier = pred_map > 0.5
        pred_frontier_is_binary = _is_binary(pred_frontier)
        pred_frontier_cmap = "gray" if pred_frontier_is_binary else "YlOrRd"
        axes[3].imshow(pred_frontier, cmap=pred_frontier_cmap, interpolation="nearest")
        axes[3].set_title(f"Pred Frontier ({t_label})")
        axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')
    else:
        plt.show()

    return {
        "orig_hw": (orig_h, orig_w),
        "crop_hw": (crop_h, crop_w),
        "t_actual": int(t_actual),
        "pred_t_actual": int(pred_t_actual) if pred_t_actual is not None else None,
    }
