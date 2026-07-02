from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class WeatherObservation:
    temperature_f: float
    relative_humidity: float
    wind_speed_mph: float
    wind_direction_deg: float
    precipitation_in: float = 0.0
    cloud_cover: float | None = None

    def dryness_index(self) -> float:
        humidity_term = 1.0 - np.clip(self.relative_humidity, 0.0, 100.0) / 100.0
        temp_term = np.clip((self.temperature_f - 50.0) / 55.0, 0.0, 1.0)
        wind_term = np.clip(self.wind_speed_mph / 30.0, 0.0, 1.0)
        rain_penalty = np.clip(self.precipitation_in / 0.25, 0.0, 1.0)
        return float(np.clip(0.45 * humidity_term + 0.30 * temp_term + 0.25 * wind_term - 0.5 * rain_penalty, 0.0, 1.0))


@dataclass(frozen=True)
class StaticLayers:
    aspect: Array
    elevation: Array
    slope: Array
    fbfm13: Array
    canopy_cover: Array

    def validate(self, shape: tuple[int, int]) -> None:
        for name, layer in self.as_dict().items():
            if layer.shape != shape:
                raise ValueError(f"static layer {name} has shape {layer.shape}, expected {shape}")

    def as_dict(self) -> dict[str, Array]:
        return {
            "aspect": self.aspect,
            "elevation": self.elevation,
            "slope": self.slope,
            "fbfm13": self.fbfm13,
            "canopy_cover": self.canopy_cover,
        }


@dataclass(frozen=True)
class PredictionInput:
    burn_map_t: Array
    pso_forecast_t_plus_h: Array
    static_layers: StaticLayers
    weather_t: WeatherObservation
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.burn_map_t.ndim != 2:
            raise ValueError("burn_map_t must be a 2-D raster")
        if self.pso_forecast_t_plus_h.shape != self.burn_map_t.shape:
            raise ValueError("pso_forecast_t_plus_h must match burn_map_t shape")
        self.static_layers.validate(self.burn_map_t.shape)


@dataclass(frozen=True)
class CorrectionOperation:
    name: str
    strength: float = 1.0
    radius: int = 1
    direction_deg: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectionPlan:
    operations: list[CorrectionOperation]
    confidence: float
    rationale: str
    retrieved_snippet_ids: list[str] = field(default_factory=list)
    source: str = "unknown"
    evidence_checklist: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionOutput:
    predicted_burn_probability_map_t_plus_h: Array
    predicted_binary_burn_map_t_plus_h: Array
    correction_plan: CorrectionPlan
    uncertainty_map: Array
    physical_rationale: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
