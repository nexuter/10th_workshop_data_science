from __future__ import annotations

from typing import Dict, Any


def format_metrics(m: Dict[str, Any]) -> str:
    return (
        f"dice_mean={float(m['dice_mean']):.4f} "
        f"iou_mean={float(m['iou_mean']):.4f} "
        f"f1_mean={float(m['f1_mean']):.4f} "
        f"acc_mean={float(m['acc_mean']):.4f} "
        f"auprc={float(m.get('auprc', 0.0)):.4f}"
    )
