from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildfire_llm_agent.array_ops import as_binary
from wildfire_llm_agent.schemas import PredictionInput, PredictionOutput


@dataclass(frozen=True)
class SelectionDecision:
    selected: str
    reason: str
    confidence: float
    warm_new_cells: int
    agent_new_cells: int
    added_over_warm_cells: int
    max_allowed_added_cells: int
    agent_to_warm_area_ratio: float


class AreaBudgetSelector:
    """Choose LLM correction only when it stays within a conservative area budget."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.45,
        max_added_cells_factor: float = 3.0,
        min_allowed_added_cells: int = 20,
        max_agent_to_warm_area_ratio: float = 4.0,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.max_added_cells_factor = float(max_added_cells_factor)
        self.min_allowed_added_cells = int(min_allowed_added_cells)
        self.max_agent_to_warm_area_ratio = float(max_agent_to_warm_area_ratio)

    def select(self, inputs: PredictionInput, output: PredictionOutput) -> tuple[np.ndarray, SelectionDecision]:
        burn = as_binary(inputs.burn_map_t).astype(bool)
        warm = np.maximum(as_binary(inputs.pso_forecast_t_plus_h).astype(bool), burn)
        agent = as_binary(output.predicted_binary_burn_map_t_plus_h).astype(bool)

        warm_new = warm & ~burn
        agent_new = agent & ~burn
        added_over_warm = agent & ~warm

        warm_new_cells = int(warm_new.sum())
        agent_new_cells = int(agent_new.sum())
        added_over_warm_cells = int(added_over_warm.sum())
        warm_area = max(int(warm.sum()), 1)
        agent_area = int(agent.sum())
        ratio = float(agent_area / warm_area)
        budget = max(self.min_allowed_added_cells, int(max(1, warm_new_cells) * self.max_added_cells_factor))
        confidence = float(output.correction_plan.confidence)

        if confidence < self.min_confidence:
            return warm.astype(np.uint8), SelectionDecision(
                selected="warm_start",
                reason="confidence_below_threshold",
                confidence=confidence,
                warm_new_cells=warm_new_cells,
                agent_new_cells=agent_new_cells,
                added_over_warm_cells=added_over_warm_cells,
                max_allowed_added_cells=budget,
                agent_to_warm_area_ratio=ratio,
            )

        if added_over_warm_cells > budget:
            return warm.astype(np.uint8), SelectionDecision(
                selected="warm_start",
                reason="added_area_budget_exceeded",
                confidence=confidence,
                warm_new_cells=warm_new_cells,
                agent_new_cells=agent_new_cells,
                added_over_warm_cells=added_over_warm_cells,
                max_allowed_added_cells=budget,
                agent_to_warm_area_ratio=ratio,
            )

        if ratio > self.max_agent_to_warm_area_ratio:
            return warm.astype(np.uint8), SelectionDecision(
                selected="warm_start",
                reason="agent_to_warm_area_ratio_exceeded",
                confidence=confidence,
                warm_new_cells=warm_new_cells,
                agent_new_cells=agent_new_cells,
                added_over_warm_cells=added_over_warm_cells,
                max_allowed_added_cells=budget,
                agent_to_warm_area_ratio=ratio,
            )

        return agent.astype(np.uint8), SelectionDecision(
            selected="agent",
            reason="within_budget",
            confidence=confidence,
            warm_new_cells=warm_new_cells,
            agent_new_cells=agent_new_cells,
            added_over_warm_cells=added_over_warm_cells,
            max_allowed_added_cells=budget,
            agent_to_warm_area_ratio=ratio,
        )
