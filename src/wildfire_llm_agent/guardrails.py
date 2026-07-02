from __future__ import annotations

import numpy as np

from wildfire_llm_agent.schemas import CorrectionPlan, PredictionInput


SUPPORTED_OPERATIONS = {
    "expand_downwind",
    "fuel_mask",
    "monotonic_burn",
    "suppress_growth",
    "slope_boost",
    "clip_probability",
}


class GuardrailError(ValueError):
    pass


class Guardrail:
    def validate_plan(self, plan: CorrectionPlan) -> None:
        if not (0.0 <= plan.confidence <= 1.0):
            raise GuardrailError("correction plan confidence must be in [0, 1]")
        for op in plan.operations:
            if op.name not in SUPPORTED_OPERATIONS:
                raise GuardrailError(f"unsupported correction operation: {op.name}")
            if op.radius < 0:
                raise GuardrailError(f"operation {op.name} has negative radius")
            if not (0.0 <= op.strength <= 1.5):
                raise GuardrailError(f"operation {op.name} strength must be in [0, 1.5]")

    def validate_output(self, inputs: PredictionInput, probability: np.ndarray, binary: np.ndarray) -> None:
        if probability.shape != inputs.burn_map_t.shape:
            raise GuardrailError("probability output shape does not match input")
        if binary.shape != inputs.burn_map_t.shape:
            raise GuardrailError("binary output shape does not match input")
        if np.nanmin(probability) < -1e-9 or np.nanmax(probability) > 1.0 + 1e-9:
            raise GuardrailError("probability output must stay in [0, 1]")
        current = inputs.burn_map_t >= 0.5
        if np.any(binary[current] == 0):
            raise GuardrailError("prediction violates monotonic burn constraint")
