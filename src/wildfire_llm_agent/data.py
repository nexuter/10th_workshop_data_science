from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from wildfire_llm_agent.schemas import StaticLayers, WeatherObservation


def load_raster(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".jpg", ".jpeg", ".png"}:
        return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("GeoTIFF loading requires the optional rasterio dependency") from exc
        with rasterio.open(path) as dataset:
            return dataset.read(1)
    raise ValueError(f"unsupported raster format: {path}")


def parse_weather_file(path: str | Path) -> list[WeatherObservation]:
    observations: list[WeatherObservation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        observations.append(
            WeatherObservation(
                temperature_f=float(parts[4]),
                relative_humidity=float(parts[5]),
                precipitation_in=float(parts[6]),
                wind_speed_mph=float(parts[7]),
                wind_direction_deg=float(parts[8]),
                cloud_cover=float(parts[9]),
            )
        )
    if not observations:
        raise ValueError(f"no weather observations found in {path}")
    return observations


@dataclass(frozen=True)
class MaterializedEvent:
    event_id: str
    mask_dir: Path
    weather_file: Path


class MaterializedScenarioReader:
    def __init__(self, scenario_root: str | Path) -> None:
        self.scenario_root = Path(scenario_root)

    def list_events(self) -> list[MaterializedEvent]:
        mask_root = self.scenario_root / "Satellite_Images_Mask"
        weather_root = self.scenario_root / "Weather_Data"
        events: list[MaterializedEvent] = []
        for mask_dir in sorted(mask_root.iterdir()):
            if not mask_dir.is_dir():
                continue
            weather_file = weather_root / f"{mask_dir.name}.txt"
            if weather_file.exists():
                events.append(MaterializedEvent(mask_dir.name, mask_dir, weather_file))
        return events

    def load_static_layers(self) -> StaticLayers:
        topography = self.scenario_root / "Topography_Map"
        fuel = self.scenario_root / "Fuel_Map"
        return StaticLayers(
            aspect=load_raster(topography / "Aspect.tif"),
            elevation=load_raster(topography / "Elevation.tif"),
            slope=load_raster(topography / "Slope.tif"),
            fbfm13=load_raster(fuel / "FBFM13.tif"),
            canopy_cover=load_raster(fuel / "Canopy_Cover.tif"),
        )

    def load_mask_sequence(self, event: MaterializedEvent, max_frames: int | None = None) -> list[np.ndarray]:
        frames = sorted(event.mask_dir.glob("out*.jpg"), key=lambda p: int(p.stem.replace("out", "")))
        if max_frames is not None:
            frames = frames[:max_frames]
        if not frames:
            raise ValueError(f"no mask frames found in {event.mask_dir}")
        return [load_raster(frame) for frame in frames]

    def load_weather(self, event: MaterializedEvent, index: int = 0) -> WeatherObservation:
        observations = parse_weather_file(event.weather_file)
        return observations[min(index, len(observations) - 1)]


def discover_materialized_scenarios(dataset_root: str | Path) -> list[Path]:
    """Find scenario folders that contain the expected materialized data layout."""
    root = Path(dataset_root)
    scenarios: list[Path] = []
    for candidate in sorted(root.glob("*/*")):
        if not candidate.is_dir():
            continue
        required = [
            candidate / "Fuel_Map",
            candidate / "Topography_Map",
            candidate / "Weather_Data",
            candidate / "Satellite_Images_Mask",
        ]
        if all(path.exists() for path in required):
            scenarios.append(candidate)
    return scenarios
