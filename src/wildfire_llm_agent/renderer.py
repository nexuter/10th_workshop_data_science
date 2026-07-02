from __future__ import annotations

import numpy as np

from wildfire_llm_agent.array_ops import as_binary, burn_front, dilate, normalize01, shift, wind_to_offset
from wildfire_llm_agent.guardrails import Guardrail
from wildfire_llm_agent.schemas import CorrectionPlan, PredictionInput


class ToolRenderer:
    def __init__(self, threshold: float = 0.5, guardrail: Guardrail | None = None) -> None:
        self.threshold = threshold
        self.guardrail = guardrail or Guardrail()

    def render(self, inputs: PredictionInput, plan: CorrectionPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.guardrail.validate_plan(plan)
        burn = as_binary(inputs.burn_map_t)
        base_probability = np.asarray(inputs.pso_forecast_t_plus_h, dtype=float).copy()
        if base_probability.max() > 1.0 or base_probability.min() < 0.0:
            base_probability = normalize01(base_probability)
        base_probability = np.maximum(base_probability, burn)
        correction_probability = np.zeros_like(base_probability, dtype=float)
        correction_allowed = np.ones_like(burn, dtype=bool)

        for op in plan.operations:
            if op.name == "expand_downwind":
                direction = inputs.weather_t.wind_direction_deg if op.direction_deg is None else op.direction_deg
                row_offset, col_offset = wind_to_offset(direction, max(1, op.radius))
                candidate = shift(dilate(burn_front(burn), op.radius), row_offset, col_offset).astype(bool)
                candidate &= correction_allowed
                candidate = self._cap_candidate(candidate, inputs, op.params)
                correction_probability = np.maximum(correction_probability, candidate * op.strength)
            elif op.name == "fuel_mask":
                min_fuel = int(op.params.get("min_fbfm13", 1))
                max_fuel = int(op.params.get("max_fbfm13", 13))
                fuel = inputs.static_layers.fbfm13
                nonburnable_codes = set(op.params.get("nonburnable_codes", [0, 91, 92, 93, 98, 99]))
                burnable = (fuel >= min_fuel) & (fuel <= max_fuel)
                burnable |= ~np.isin(fuel.astype(int), list(nonburnable_codes))
                correction_allowed &= burnable | (burn > 0)
            elif op.name == "monotonic_burn":
                base_probability = np.maximum(base_probability, burn)
            elif op.name == "suppress_growth":
                only_new = bool(op.params.get("only_new_growth", True))
                factor = 1.0 - min(op.strength, 0.95)
                if only_new:
                    correction_probability = np.where(burn > 0, correction_probability, correction_probability * factor)
                else:
                    correction_probability *= factor
            elif op.name == "slope_boost":
                slope = normalize01(inputs.static_layers.slope)
                front = dilate(burn_front(burn), max(1, op.radius)).astype(bool)
                front &= correction_allowed
                front = self._cap_candidate(front, inputs, op.params)
                correction_probability = np.maximum(correction_probability, front * slope * op.strength)
            elif op.name == "clip_probability":
                base_probability = np.clip(base_probability, 0.0, 1.0)
                correction_probability = np.clip(correction_probability, 0.0, 1.0)

        probability = np.maximum(base_probability, correction_probability)
        probability = np.clip(probability, 0.0, 1.0)
        binary = (probability >= self.threshold).astype(np.uint8)
        binary = np.maximum(binary, burn).astype(np.uint8)
        uncertainty = np.clip(1.0 - np.abs(probability - self.threshold) * 2.0, 0.0, 1.0)
        self.guardrail.validate_output(inputs, probability, binary)
        return probability, binary, uncertainty

    def _cap_candidate(self, candidate: np.ndarray, inputs: PredictionInput, params: dict) -> np.ndarray:
        max_factor = params.get("max_added_cells_factor")
        if max_factor is None:
            return candidate
        candidate = np.asarray(candidate, dtype=bool)
        count = int(candidate.sum())
        if count == 0:
            return candidate
        burn = as_binary(inputs.burn_map_t)
        pso_new = np.maximum(as_binary(inputs.pso_forecast_t_plus_h) - burn, 0)
        budget = max(int(params.get("min_added_cells", 1)), int(max(1, pso_new.sum()) * float(max_factor)))
        if count <= budget:
            return candidate
        slope_score = normalize01(inputs.static_layers.slope)
        rows, cols = np.where(candidate)
        scores = slope_score[rows, cols]
        keep_idx = np.argsort(scores)[-budget:]
        capped = np.zeros_like(candidate, dtype=bool)
        capped[rows[keep_idx], cols[keep_idx]] = True
        return capped
