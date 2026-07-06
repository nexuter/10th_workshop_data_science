from __future__ import annotations

import torch


@torch.no_grad()
def binary_metrics_over_time(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    pred = pred.bool()
    gt = gt.bool()

    tp = (pred & gt).sum(dim=(1, 2)).float()
    fp = (pred & (~gt)).sum(dim=(1, 2)).float()
    fn = ((~pred) & gt).sum(dim=(1, 2)).float()
    tn = ((~pred) & (~gt)).sum(dim=(1, 2)).float()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice_t": dice,
        "iou_t": iou,
        "precision_t": precision,
        "recall_t": recall,
        "f1_t": f1,
        "acc_t": acc,
        "dice_mean": dice.mean(),
        "iou_mean": iou.mean(),
        "f1_mean": f1.mean(),
        "acc_mean": acc.mean(),
        "dice_last": dice[-1],
        "iou_last": iou[-1],
        "f1_last": f1[-1],
        "acc_last": acc[-1],
    }


@torch.no_grad()
def average_precision(scores: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Average Precision (AUPRC) for binary labels.
    scores: arbitrary float scores (higher = more positive)
    gt: boolean ground truth
    """
    scores = scores.flatten().float()
    gt = gt.flatten().bool()

    if gt.numel() == 0:
        return torch.tensor(0.0, device=scores.device)

    pos = gt.sum().float()
    if pos.item() == 0:
        return torch.tensor(0.0, device=scores.device)

    # Sort by descending score
    order = torch.argsort(scores, descending=True)
    gt_sorted = gt[order].float()

    tp = torch.cumsum(gt_sorted, dim=0)
    fp = torch.cumsum(1.0 - gt_sorted, dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (pos + eps)

    # AP = sum over recall steps of precision
    recall_diff = torch.cat([recall[:1], recall[1:] - recall[:-1]])
    ap = torch.sum(precision * recall_diff)
    return ap
