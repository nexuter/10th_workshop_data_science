from __future__ import annotations

import numpy as np

from wildfire_llm_agent.schemas import PredictionInput


def without_physical_warm_start(inputs: PredictionInput) -> PredictionInput:
    """Build an LLM-only ablation input that hides the physical forecast."""
    metadata = dict(inputs.metadata)
    metadata.update(
        {
            "warm_start": "none_current_burn_only",
            "uses_physical_warm_start": False,
            "ablation": "llm_only_no_physical_prior",
        }
    )
    return PredictionInput(
        burn_map_t=inputs.burn_map_t,
        pso_forecast_t_plus_h=np.asarray(inputs.burn_map_t, dtype=float),
        static_layers=inputs.static_layers,
        weather_t=inputs.weather_t,
        metadata=metadata,
    )
