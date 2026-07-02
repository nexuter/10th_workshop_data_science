from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from wildfire_llm_agent.array_ops import as_binary, bbox, burn_front, normalize01, wind_to_offset
from wildfire_llm_agent.schemas import PredictionInput


@dataclass(frozen=True)
class AgentContext:
    summary: dict[str, Any]
    panel_path: str | None = None


class ContextBuilder:
    def build(self, inputs: PredictionInput, panel_path: str | Path | None = None) -> AgentContext:
        inputs.validate()
        burn = as_binary(inputs.burn_map_t)
        metadata = dict(inputs.metadata)
        uses_physical_warm_start = bool(metadata.get("uses_physical_warm_start", True))
        prior = np.asarray(inputs.pso_forecast_t_plus_h, dtype=float)
        new_prior = np.maximum((prior >= 0.5).astype(np.uint8) - burn, 0)
        front = burn_front(burn)
        row_offset, col_offset = wind_to_offset(inputs.weather_t.wind_direction_deg)
        front_box = bbox(np.maximum(front, new_prior), pad=4)

        layers = inputs.static_layers
        burnable = (layers.fbfm13 > 0).astype(np.uint8)
        aspect_rad = np.deg2rad(np.asarray(layers.aspect, dtype=float))
        aspect_sin = float(np.nanmean(np.sin(aspect_rad)))
        aspect_cos = float(np.nanmean(np.cos(aspect_rad)))
        dominant_aspect_deg = float((np.rad2deg(np.arctan2(aspect_sin, aspect_cos)) + 360.0) % 360.0)
        fuel_values, fuel_counts = np.unique(layers.fbfm13.astype(int), return_counts=True)
        top_fuels = sorted(zip(fuel_values.tolist(), fuel_counts.tolist()), key=lambda item: item[1], reverse=True)[:5]
        summary = {
            "shape": list(burn.shape),
            "burned_cells_t": int(burn.sum()),
            "uses_physical_warm_start": uses_physical_warm_start,
            "warm_start_label": "pso_physics_forecast" if uses_physical_warm_start else "current_burn_only_no_physics",
            "prior_new_burn_cells": int(new_prior.sum()),
            "pso_new_burn_cells": int(new_prior.sum()) if uses_physical_warm_start else 0,
            "front_cells": int(front.sum()),
            "front_bbox": list(front_box) if front_box else None,
            "wind_downstream_offset": [row_offset, col_offset],
            "weather": {
                "temperature_f": inputs.weather_t.temperature_f,
                "relative_humidity": inputs.weather_t.relative_humidity,
                "wind_speed_mph": inputs.weather_t.wind_speed_mph,
                "wind_direction_deg": inputs.weather_t.wind_direction_deg,
                "precipitation_in": inputs.weather_t.precipitation_in,
                "dryness_index": inputs.weather_t.dryness_index(),
            },
            "static_layer_stats": {
                "elevation_mean": float(np.nanmean(layers.elevation)),
                "elevation_p10": float(np.nanpercentile(layers.elevation, 10)),
                "elevation_p90": float(np.nanpercentile(layers.elevation, 90)),
                "slope_mean": float(np.nanmean(layers.slope)),
                "slope_p50": float(np.nanpercentile(layers.slope, 50)),
                "slope_p90": float(np.nanpercentile(layers.slope, 90)),
                "canopy_cover_mean": float(np.nanmean(layers.canopy_cover)),
                "canopy_cover_p90": float(np.nanpercentile(layers.canopy_cover, 90)),
                "burnable_fraction": float(burnable.mean()),
                "nonburnable_fraction": float(1.0 - burnable.mean()),
                "dominant_fbfm13": int(np.bincount(layers.fbfm13.astype(int).ravel()).argmax()),
                "top_fbfm13_counts": [{"fuel": int(fuel), "cells": int(count)} for fuel, count in top_fuels],
                "dominant_aspect_deg": dominant_aspect_deg,
            },
            "metadata": metadata,
        }

        rendered_path = None
        if panel_path is not None:
            rendered_path = str(self.render_panel(inputs, panel_path))
        return AgentContext(summary=summary, panel_path=rendered_path)

    def render_panel(self, inputs: PredictionInput, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        panels = [
            self._gray(inputs.burn_map_t, "burn_t"),
            self._gray(
                inputs.pso_forecast_t_plus_h,
                "pso_t+h" if bool(inputs.metadata.get("uses_physical_warm_start", True)) else "base_t",
            ),
            self._gray(inputs.static_layers.aspect, "aspect"),
            self._gray(inputs.static_layers.elevation, "elev"),
            self._gray(inputs.static_layers.slope, "slope"),
            self._gray(inputs.static_layers.fbfm13, "fbfm13"),
            self._gray(inputs.static_layers.canopy_cover, "canopy"),
        ]
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        canvas.save(output)
        return output

    def _gray(self, values: np.ndarray, label: str) -> Image.Image:
        arr = (normalize01(values) * 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="L").convert("RGB")
        image = image.resize((max(96, image.width), max(96, image.height)))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(90, image.width - 1), 14), fill=(255, 255, 255))
        draw.text((3, 2), label, fill=(0, 0, 0))
        return image
