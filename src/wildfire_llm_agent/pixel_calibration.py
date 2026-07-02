from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildfire_llm_agent.array_ops import burn_front, dilate, normalize01, shift, wind_to_offset
from wildfire_llm_agent.schemas import PredictionInput


FEATURE_NAMES = [
    "warm_probability",
    "agent_probability",
    "llm_only_probability",
    "simple_probability",
    "front_neighborhood",
    "downwind_front",
    "slope",
    "canopy_cover",
    "elevation",
    "burnable_fuel",
    "aspect_sin",
    "aspect_cos",
    "dryness_index",
    "wind_speed_norm",
    "relative_humidity_norm",
    "temperature_norm",
    "wind_dir_sin",
    "wind_dir_cos",
]


@dataclass(frozen=True)
class PixelCalibrator:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_new_probability(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape((-1, features.shape[-1]))
        standardized = (flat - self.mean) / self.scale
        logits = standardized @ self.weights + self.bias
        return _sigmoid(logits).reshape(features.shape[:2])


def build_pixel_features(
    inputs: PredictionInput,
    *,
    warm_probability: np.ndarray,
    agent_probability: np.ndarray,
    llm_only_probability: np.ndarray,
    simple_probability: np.ndarray,
) -> np.ndarray:
    burn = (np.asarray(inputs.burn_map_t) >= 0.5).astype(np.uint8)
    front = dilate(burn_front(burn), 2)
    row_offset, col_offset = wind_to_offset(inputs.weather_t.wind_direction_deg, 2)
    downwind = shift(front, row_offset, col_offset)
    layers = inputs.static_layers
    aspect_rad = np.deg2rad(np.asarray(layers.aspect, dtype=float))
    burnable = ((layers.fbfm13 >= 1) & (layers.fbfm13 <= 13)).astype(float)
    weather_planes = _weather_planes(inputs, burn.shape)
    features = [
        np.asarray(warm_probability, dtype=float),
        np.asarray(agent_probability, dtype=float),
        np.asarray(llm_only_probability, dtype=float),
        np.asarray(simple_probability, dtype=float),
        np.asarray(front, dtype=float),
        np.asarray(downwind, dtype=float),
        normalize01(layers.slope),
        normalize01(layers.canopy_cover),
        normalize01(layers.elevation),
        burnable,
        np.sin(aspect_rad),
        np.cos(aspect_rad),
        *weather_planes,
    ]
    return np.stack(features, axis=-1).astype(float)


def mask_feature_channels(
    features: np.ndarray,
    feature_names: list[str],
    disabled_feature_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    masked = np.array(features, dtype=float, copy=True)
    for feature_name in disabled_feature_names:
        try:
            feature_index = feature_names.index(feature_name)
        except ValueError as exc:
            raise ValueError(f"unknown feature for masking: {feature_name}") from exc
        masked[..., feature_index] = 0.0
    return masked


def sample_new_burn_training_pixels(
    features: np.ndarray,
    target: np.ndarray,
    current_burn: np.ndarray,
    *,
    negative_ratio: int = 4,
    max_negative_pixels: int = 20000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    truth_new = (np.asarray(target) >= 0.5) & ~(np.asarray(current_burn) >= 0.5)
    eligible = ~(np.asarray(current_burn) >= 0.5)
    positive_indices = np.flatnonzero(truth_new.ravel())
    negative_indices = np.flatnonzero((eligible & ~truth_new).ravel())
    rng = np.random.default_rng(seed)
    if len(positive_indices):
        negative_count = min(len(negative_indices), max(len(positive_indices) * negative_ratio, 1), max_negative_pixels)
    else:
        negative_count = min(len(negative_indices), max_negative_pixels)
    if negative_count < len(negative_indices):
        negative_indices = rng.choice(negative_indices, size=negative_count, replace=False)
    selected = np.concatenate([positive_indices, negative_indices])
    if len(selected) == 0:
        return np.empty((0, features.shape[-1]), dtype=float), np.empty((0,), dtype=float)
    labels = np.concatenate([np.ones(len(positive_indices), dtype=float), np.zeros(len(negative_indices), dtype=float)])
    flat_features = features.reshape((-1, features.shape[-1]))[selected]
    order = rng.permutation(len(selected))
    return flat_features[order], labels[order]


def fit_pixel_calibrator(
    feature_batches: list[np.ndarray],
    label_batches: list[np.ndarray],
    *,
    iterations: int = 600,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
    positive_weight: float | None = None,
) -> PixelCalibrator:
    x = np.vstack([batch for batch in feature_batches if len(batch)])
    y = np.concatenate([batch for batch in label_batches if len(batch)])
    if x.size == 0 or y.size == 0:
        raise ValueError("no training pixels available for calibration")
    if y.max() == y.min():
        raise ValueError("calibration labels must include both positive and negative pixels")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    if positive_weight is None:
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        positive_weight = min(negatives / positives, 25.0)
    sample_weight = np.where(y > 0.5, positive_weight, 1.0)
    sample_weight = sample_weight / sample_weight.mean()

    for _ in range(iterations):
        logits = z @ weights + bias
        pred = _sigmoid(logits)
        error = (pred - y) * sample_weight
        grad_w = z.T @ error / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return PixelCalibrator(
        feature_names=list(FEATURE_NAMES),
        mean=mean,
        scale=scale,
        weights=weights,
        bias=float(bias),
    )


def calibrated_burn_probability(inputs: PredictionInput, features: np.ndarray, calibrator: PixelCalibrator) -> np.ndarray:
    new_probability = calibrator.predict_new_probability(features)
    burn = (np.asarray(inputs.burn_map_t) >= 0.5).astype(float)
    return np.maximum(burn, new_probability)


def _weather_planes(inputs: PredictionInput, shape: tuple[int, int]) -> list[np.ndarray]:
    weather = inputs.weather_t
    wind_rad = np.deg2rad(weather.wind_direction_deg)
    values = [
        weather.dryness_index(),
        np.clip(weather.wind_speed_mph / 30.0, 0.0, 2.0),
        np.clip(weather.relative_humidity / 100.0, 0.0, 1.0),
        np.clip((weather.temperature_f - 30.0) / 90.0, 0.0, 1.5),
        float(np.sin(wind_rad)),
        float(np.cos(wind_rad)),
    ]
    return [np.full(shape, value, dtype=float) for value in values]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
