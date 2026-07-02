from __future__ import annotations

import numpy as np

from wildfire_llm_agent.array_ops import as_binary, boundary, shift, wind_to_offset


def classification_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray | None = None,
    *,
    empty_iou: float = 1.0,
) -> dict[str, float]:
    pred = as_binary(prediction).astype(bool)
    truth = as_binary(target).astype(bool)
    if current_burn is not None:
        current = as_binary(current_burn).astype(bool)
        pred = pred & ~current
        truth = truth & ~current

    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": tp / union if union else empty_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def average_precision_score(prediction: np.ndarray, target: np.ndarray, current_burn: np.ndarray | None = None) -> float:
    scores = np.asarray(prediction, dtype=float).ravel()
    truth = as_binary(target).astype(bool).ravel()
    if current_burn is not None:
        current = as_binary(current_burn).astype(bool).ravel()
        keep = ~current
        scores = scores[keep]
        truth = truth[keep]
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[order]
    tp = np.cumsum(sorted_truth, dtype=float)
    fp = np.cumsum(~sorted_truth, dtype=float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(positives)
    recall_step = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * recall_step))


def recall_at_new_burn_area(
    prediction: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray,
    *,
    budget_multiplier: float = 1.0,
) -> float:
    scores = np.asarray(prediction, dtype=float).ravel()
    truth = as_binary(target).astype(bool).ravel()
    current = as_binary(current_burn).astype(bool).ravel()
    eligible = ~current
    truth_new = truth & eligible
    positive_count = int(truth_new.sum())
    if positive_count == 0:
        return 0.0
    budget = max(int(round(positive_count * budget_multiplier)), 1)
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) == 0:
        return 0.0
    budget = min(budget, len(eligible_indices))
    eligible_scores = scores[eligible_indices]
    top_order = np.argsort(-eligible_scores, kind="mergesort")[:budget]
    selected = eligible_indices[top_order]
    return float(truth_new[selected].sum()) / float(positive_count)


def burned_area_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_area = float(as_binary(prediction).sum())
    truth_area = float(as_binary(target).sum())
    if truth_area == 0.0:
        return pred_area
    return abs(pred_area - truth_area) / truth_area


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    pred_b = boundary(prediction).astype(bool)
    truth_b = boundary(target).astype(bool)
    if tolerance > 0:
        from wildfire_llm_agent.array_ops import dilate

        pred_match = dilate(truth_b, tolerance).astype(bool)
        truth_match = dilate(pred_b, tolerance).astype(bool)
    else:
        pred_match = truth_b
        truth_match = pred_b
    tp_pred = float(np.logical_and(pred_b, pred_match).sum())
    tp_truth = float(np.logical_and(truth_b, truth_match).sum())
    precision = tp_pred / pred_b.sum() if pred_b.sum() else 0.0
    recall = tp_truth / truth_b.sum() if truth_b.sum() else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def new_boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray,
    tolerance: int = 1,
) -> float:
    current = as_binary(current_burn).astype(bool)
    pred_new = as_binary(prediction).astype(bool) & ~current
    truth_new = as_binary(target).astype(bool) & ~current
    return boundary_f1(pred_new, truth_new, tolerance=tolerance)


def wind_aligned_spread_error(
    prediction: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray,
    wind_direction_deg: float,
) -> float:
    row_offset, col_offset = wind_to_offset(wind_direction_deg)
    current = as_binary(current_burn)
    downwind_zone = shift(current, row_offset, col_offset).astype(bool)
    pred_new = (as_binary(prediction) > 0) & (current == 0)
    target_new = (as_binary(target) > 0) & (current == 0)
    pred_fraction = float(np.logical_and(pred_new, downwind_zone).sum()) / max(float(pred_new.sum()), 1.0)
    truth_fraction = float(np.logical_and(target_new, downwind_zone).sum()) / max(float(target_new.sum()), 1.0)
    return abs(pred_fraction - truth_fraction)


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray,
    wind_direction_deg: float,
) -> dict[str, float]:
    metrics = classification_metrics(prediction, target)
    new_metrics = classification_metrics(prediction, target, current_burn=current_burn, empty_iou=0.0)
    return {
        **metrics,
        "auprc": average_precision_score(prediction, target),
        "new_iou": new_metrics["iou"],
        "new_precision": new_metrics["precision"],
        "new_recall": new_metrics["recall"],
        "new_f1": new_metrics["f1"],
        "new_auprc": average_precision_score(prediction, target, current_burn=current_burn),
        "new_recall_at_true_area": recall_at_new_burn_area(
            prediction,
            target,
            current_burn,
            budget_multiplier=1.0,
        ),
        "new_recall_at_2x_true_area": recall_at_new_burn_area(
            prediction,
            target,
            current_burn,
            budget_multiplier=2.0,
        ),
        "burned_area_error": burned_area_error(prediction, target),
        "boundary_f1": boundary_f1(prediction, target),
        "new_boundary_f1": new_boundary_f1(prediction, target, current_burn),
        "wind_aligned_spread_error": wind_aligned_spread_error(prediction, target, current_burn, wind_direction_deg),
    }
