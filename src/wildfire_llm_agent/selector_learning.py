from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from wildfire_llm_agent.data import MaterializedScenarioReader


METHODS = ("warm_start", "llm_only", "agent")


def build_selector_dataset(evaluation_dir: Path, dataset_root: Path, output_path: Path) -> list[dict[str, object]]:
    metrics_path = evaluation_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, str]]] = {}
    for row in metric_rows:
        key = (row["scenario"], row["fire_id"], row["event_id"], int(row["frame_index"]))
        grouped.setdefault(key, {})[row["method"]] = row

    rows: list[dict[str, object]] = []
    static_cache = {}
    frame_cache = {}
    for (scenario, fire_id, event_id, frame_index), methods in sorted(grouped.items()):
        if not all(method in methods for method in METHODS):
            continue
        scenario_root = dataset_root / scenario / fire_id
        reader = MaterializedScenarioReader(scenario_root)
        if scenario_root not in static_cache:
            static_cache[scenario_root] = reader.load_static_layers()
        static_layers = static_cache[scenario_root]

        frame_key = (scenario_root, event_id)
        if frame_key not in frame_cache:
            event = next(item for item in reader.list_events() if item.event_id == event_id)
            frame_cache[frame_key] = (reader, event, reader.load_mask_sequence(event, max_frames=frame_index + 1))
        cached_reader, event, frames = frame_cache[frame_key]
        if len(frames) <= frame_index:
            frames = cached_reader.load_mask_sequence(event, max_frames=frame_index + 1)
            frame_cache[frame_key] = (cached_reader, event, frames)
        burn = (frames[frame_index] >= 0.5).astype(np.uint8)
        weather = reader.load_weather(event, index=frame_index)

        agent_selected = methods.get("agent_selected", {})
        burn_cells = int(burn.sum())
        warm_new_cells = _to_int(agent_selected.get("selection_warm_new_cells"), default=0)
        agent_new_cells = _to_int(agent_selected.get("selection_agent_new_cells"), default=0)
        added_over_warm = _to_int(agent_selected.get("selection_added_over_warm_cells"), default=0)
        selector_threshold = _to_float(agent_selected.get("selection_max_allowed_added_cells"), default=0.0)

        row = {
            "scenario": scenario,
            "fire_id": fire_id,
            "event_id": event_id,
            "frame_index": frame_index,
            "burned_cells_t": burn_cells,
            "warm_new_cells": warm_new_cells,
            "agent_new_cells": agent_new_cells,
            "added_over_warm_cells": added_over_warm,
            "max_allowed_added_cells": selector_threshold,
            "agent_new_to_current_ratio": agent_new_cells / max(burn_cells, 1),
            "warm_new_to_current_ratio": warm_new_cells / max(burn_cells, 1),
            "agent_to_warm_new_ratio": agent_new_cells / max(warm_new_cells, 1),
            "added_over_warm_to_current_ratio": added_over_warm / max(burn_cells, 1),
            "temperature_f": weather.temperature_f,
            "relative_humidity": weather.relative_humidity,
            "wind_speed_mph": weather.wind_speed_mph,
            "wind_direction_deg": weather.wind_direction_deg,
            "dryness_index": weather.dryness_index(),
            "slope_mean": float(np.nanmean(static_layers.slope)),
            "slope_p90": float(np.nanpercentile(static_layers.slope, 90)),
            "canopy_cover_mean": float(np.nanmean(static_layers.canopy_cover)),
            "dominant_fbfm13": int(np.bincount(static_layers.fbfm13.astype(int).ravel()).argmax()),
        }
        for method in METHODS:
            for metric in (
                "new_f1",
                "new_auprc",
                "new_iou",
                "burned_area_error",
                "boundary_f1",
                "wind_aligned_spread_error",
                "f1",
                "iou",
                "auprc",
            ):
                row[f"{method}_{metric}"] = _to_float(methods[method].get(metric), default=0.0)
        rows.append(row)

    _write_csv(output_path, rows)
    return rows


def train_threshold_selector(
    selector_dataset_path: Path,
    output_dir: Path,
    *,
    fallback_method: str = "llm_only",
    auprc_weight: float = 0.25,
    area_error_weight: float = 0.05,
    wind_error_weight: float = 0.05,
) -> dict[str, object]:
    with selector_dataset_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {selector_dataset_path}")
    if fallback_method not in {"llm_only", "warm_start"}:
        raise ValueError("fallback_method must be llm_only or warm_start")

    thresholds = sorted(
        {
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            *(_to_float(row["agent_new_to_current_ratio"], default=0.0) for row in rows),
        }
    )
    candidates = []
    for threshold in thresholds:
        selected_methods = [
            fallback_method if _to_float(row["agent_new_to_current_ratio"], default=0.0) > threshold else "agent"
            for row in rows
        ]
        candidates.append(
            {
                "threshold": threshold,
                "fallback_method": fallback_method,
                **_summarize_policy(rows, selected_methods, auprc_weight, area_error_weight, wind_error_weight),
            }
        )
    best = max(candidates, key=lambda item: item["mean_utility"])
    oracle_methods = [_oracle_method(row, auprc_weight, area_error_weight, wind_error_weight) for row in rows]
    always_agent = ["agent" for _ in rows]
    always_warm = ["warm_start" for _ in rows]
    always_llm = ["llm_only" for _ in rows]
    report = {
        "selector_dataset": str(selector_dataset_path),
        "rows": len(rows),
        "objective": {
            "utility": "new_f1 + auprc_weight*new_auprc + boundary_f1 - area_error_weight*burned_area_error - wind_error_weight*wind_aligned_spread_error",
            "auprc_weight": auprc_weight,
            "area_error_weight": area_error_weight,
            "wind_error_weight": wind_error_weight,
        },
        "best_threshold_policy": best,
        "baselines": {
            "always_agent": _summarize_policy(rows, always_agent, auprc_weight, area_error_weight, wind_error_weight),
            "always_warm_start": _summarize_policy(rows, always_warm, auprc_weight, area_error_weight, wind_error_weight),
            "always_llm_only": _summarize_policy(rows, always_llm, auprc_weight, area_error_weight, wind_error_weight),
            "oracle": _summarize_policy(rows, oracle_methods, auprc_weight, area_error_weight, wind_error_weight),
        },
        "threshold_candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "threshold_selector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_threshold_markdown(output_dir / "threshold_selector_report.md", report)
    return report


def _oracle_method(row: dict[str, str], auprc_weight: float, area_error_weight: float, wind_error_weight: float) -> str:
    return max(METHODS, key=lambda method: _utility(row, method, auprc_weight, area_error_weight, wind_error_weight))


def _summarize_policy(
    rows: list[dict[str, str]],
    selected_methods: Iterable[str],
    auprc_weight: float,
    area_error_weight: float,
    wind_error_weight: float,
) -> dict[str, object]:
    selected_methods = list(selected_methods)
    summary = {
        "mean_utility": float(np.mean([_utility(row, method, auprc_weight, area_error_weight, wind_error_weight) for row, method in zip(rows, selected_methods)])),
        "mean_new_f1": _mean_metric(rows, selected_methods, "new_f1"),
        "mean_new_auprc": _mean_metric(rows, selected_methods, "new_auprc"),
        "mean_burned_area_error": _mean_metric(rows, selected_methods, "burned_area_error"),
        "mean_boundary_f1": _mean_metric(rows, selected_methods, "boundary_f1"),
        "mean_wind_aligned_spread_error": _mean_metric(rows, selected_methods, "wind_aligned_spread_error"),
        "selection_counts": {method: selected_methods.count(method) for method in METHODS},
    }
    return summary


def _utility(row: dict[str, str], method: str, auprc_weight: float, area_error_weight: float, wind_error_weight: float) -> float:
    return (
        _to_float(row[f"{method}_new_f1"], default=0.0)
        + auprc_weight * _to_float(row[f"{method}_new_auprc"], default=0.0)
        + _to_float(row[f"{method}_boundary_f1"], default=0.0)
        - area_error_weight * _to_float(row[f"{method}_burned_area_error"], default=0.0)
        - wind_error_weight * _to_float(row[f"{method}_wind_aligned_spread_error"], default=0.0)
    )


def _mean_metric(rows: list[dict[str, str]], methods: list[str], metric: str) -> float:
    return float(np.mean([_to_float(row[f"{method}_{metric}"], default=0.0) for row, method in zip(rows, methods)]))


def _write_threshold_markdown(path: Path, report: dict[str, object]) -> None:
    best = report["best_threshold_policy"]
    baselines = report["baselines"]
    lines = [
        "# Learned Threshold Selector Report",
        "",
        f"- Rows: {report['rows']}",
        f"- Best threshold: `{best['threshold']:.4f}`",
        f"- Fallback method: `{best['fallback_method']}`",
        "",
        "| Policy | Utility | New F1 | New AUPRC | Area Error | Boundary F1 | Wind Error | Counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    policies = {"best_threshold": best, **baselines}
    for name, values in policies.items():
        counts = values["selection_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{float(values['mean_utility']):.4f}",
                    f"{float(values['mean_new_f1']):.4f}",
                    f"{float(values['mean_new_auprc']):.4f}",
                    f"{float(values['mean_burned_area_error']):.4f}",
                    f"{float(values['mean_boundary_f1']):.4f}",
                    f"{float(values['mean_wind_aligned_spread_error']):.4f}",
                    ", ".join(f"{method}={counts.get(method, 0)}" for method in METHODS),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _to_int(value, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))
