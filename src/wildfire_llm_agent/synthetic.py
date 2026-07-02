from __future__ import annotations

import numpy as np

from wildfire_llm_agent.array_ops import dilate, shift, wind_to_offset
from wildfire_llm_agent.schemas import PredictionInput, StaticLayers, WeatherObservation


def make_synthetic_case(size: tuple[int, int] = (96, 96), seed: int = 7) -> tuple[PredictionInput, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, cols = size
    rr, cc = np.indices(size)
    center = np.array([rows // 2, cols // 3])
    distance = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    burn_t = (distance <= 7).astype(np.uint8)

    slope = np.clip((cc / cols) * 35 + rng.normal(0, 2, size), 0, 50)
    aspect = np.full(size, 90.0)
    elevation = 500 + rr * 3 + cc * 2
    fbfm13 = np.where((rr > 8) & (cc > 8), 4, 0).astype(np.int16)
    canopy = np.clip(40 + rng.normal(0, 8, size), 0, 100)
    static = StaticLayers(aspect=aspect, elevation=elevation, slope=slope, fbfm13=fbfm13, canopy_cover=canopy)
    weather = WeatherObservation(
        temperature_f=91,
        relative_humidity=18,
        wind_speed_mph=14,
        wind_direction_deg=270,
        precipitation_in=0.0,
    )

    row_offset, col_offset = wind_to_offset(weather.wind_direction_deg, 2)
    target = np.maximum(burn_t, shift(dilate(burn_t, 3), row_offset, col_offset))
    pso = np.maximum(burn_t, shift(dilate(burn_t, 2), row_offset, col_offset - 1)) * 0.55
    pso = np.maximum(pso, burn_t)
    inputs = PredictionInput(
        burn_map_t=burn_t,
        pso_forecast_t_plus_h=pso,
        static_layers=static,
        weather_t=weather,
        metadata={"case_id": "synthetic_wind_east", "horizon_hours": 1, "resolution": "synthetic"},
    )
    return inputs, target.astype(np.uint8)
