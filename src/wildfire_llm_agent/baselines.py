from __future__ import annotations

import numpy as np

from wildfire_llm_agent.array_ops import as_binary, burn_front, dilate, shift, wind_to_offset
from wildfire_llm_agent.schemas import PredictionInput


def persistence(inputs: PredictionInput) -> np.ndarray:
    return as_binary(inputs.burn_map_t)


def pso_only(inputs: PredictionInput, threshold: float = 0.5) -> np.ndarray:
    return np.maximum(as_binary(inputs.burn_map_t), as_binary(inputs.pso_forecast_t_plus_h, threshold))


def simple_physics_warm_start(inputs: PredictionInput) -> np.ndarray:
    burn = as_binary(inputs.burn_map_t)
    dryness = inputs.weather_t.dryness_index()
    radius = 2 if inputs.weather_t.wind_speed_mph >= 12 or dryness > 0.65 else 1
    row_offset, col_offset = wind_to_offset(inputs.weather_t.wind_direction_deg, radius)
    growth = shift(dilate(burn_front(burn), radius), row_offset, col_offset)
    burnable = (inputs.static_layers.fbfm13 > 0) & (inputs.static_layers.fbfm13 <= 13)
    probability = np.maximum(inputs.pso_forecast_t_plus_h, growth * (0.35 + 0.45 * dryness))
    probability = np.where(burnable | (burn > 0), probability, 0.0)
    return np.maximum(probability, burn)
