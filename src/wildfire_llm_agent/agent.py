from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from wildfire_llm_agent.context import ContextBuilder
from wildfire_llm_agent.rag import RuleBasedRetriever
from wildfire_llm_agent.reasoners import HeuristicReasoner, Reasoner
from wildfire_llm_agent.renderer import ToolRenderer
from wildfire_llm_agent.schemas import PredictionInput, PredictionOutput


class WildfireAgent:
    def __init__(
        self,
        context_builder: ContextBuilder | None = None,
        retriever: RuleBasedRetriever | None = None,
        reasoner: Reasoner | None = None,
        renderer: ToolRenderer | None = None,
    ) -> None:
        self.context_builder = context_builder or ContextBuilder()
        self.retriever = retriever or RuleBasedRetriever()
        self.reasoner = reasoner or HeuristicReasoner()
        self.renderer = renderer or ToolRenderer()

    def predict(self, inputs: PredictionInput, panel_path: str | Path | None = None) -> PredictionOutput:
        start = perf_counter()
        context = self.context_builder.build(inputs, panel_path=panel_path)
        query = self._query_from_context(context.summary)
        snippets = self.retriever.retrieve(query)
        plan = self.reasoner.propose_correction(inputs, context, snippets)
        probability, binary, uncertainty = self.renderer.render(inputs, plan)
        elapsed = perf_counter() - start
        return PredictionOutput(
            predicted_burn_probability_map_t_plus_h=probability,
            predicted_binary_burn_map_t_plus_h=binary,
            correction_plan=plan,
            uncertainty_map=uncertainty,
            physical_rationale=plan.rationale,
            diagnostics={
                "latency_seconds": elapsed,
                "context_summary": context.summary,
                "panel_path": context.panel_path,
                "retrieved_snippets": [snippet.title for snippet in snippets],
                "reasoner_source": plan.source,
            },
        )

    def _query_from_context(self, summary: dict[str, Any]) -> str:
        weather = summary["weather"]
        stats = summary["static_layer_stats"]
        return (
            f"wildfire spread correction fbfm13 wind {weather['wind_speed_mph']} "
            f"direction {weather['wind_direction_deg']} humidity {weather['relative_humidity']} "
            f"slope {stats['slope_p90']} tool-use guardrail"
        )
