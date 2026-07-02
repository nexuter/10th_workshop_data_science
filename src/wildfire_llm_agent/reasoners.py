from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from wildfire_llm_agent.context import AgentContext
from wildfire_llm_agent.rag import KnowledgeSnippet
from wildfire_llm_agent.schemas import CorrectionOperation, CorrectionPlan, PredictionInput


class Reasoner(Protocol):
    def propose_correction(
        self,
        inputs: PredictionInput,
        context: AgentContext,
        snippets: list[KnowledgeSnippet],
    ) -> CorrectionPlan:
        ...


class HeuristicReasoner:
    """Deterministic stand-in for an off-the-shelf multimodal LLM reasoner."""

    def propose_correction(
        self,
        inputs: PredictionInput,
        context: AgentContext,
        snippets: list[KnowledgeSnippet],
    ) -> CorrectionPlan:
        weather = inputs.weather_t
        dryness = weather.dryness_index()
        wind = weather.wind_speed_mph
        humidity = weather.relative_humidity
        ops: list[CorrectionOperation] = [
            CorrectionOperation(
                "fuel_mask",
                strength=1.0,
                params={"min_fbfm13": 1, "max_fbfm13": 13, "nonburnable_codes": [0, 91, 92, 93, 98, 99]},
            ),
            CorrectionOperation("monotonic_burn", strength=1.0),
        ]

        if dryness > 0.35 or wind >= 8:
            radius = 2 if wind >= 12 or dryness > 0.65 else 1
            strength = 0.20 + 0.45 * dryness + min(wind / 60.0, 0.25)
            ops.append(
                CorrectionOperation(
                    "expand_downwind",
                    strength=float(min(strength, 0.85)),
                    radius=radius,
                    direction_deg=weather.wind_direction_deg,
                    params={"max_added_cells_factor": 1.5, "min_added_cells": 2},
                )
            )

        if humidity >= 55 or weather.precipitation_in > 0.01:
            ops.append(
                CorrectionOperation(
                    "suppress_growth",
                    strength=float(min(0.65, 0.25 + humidity / 160.0 + weather.precipitation_in)),
                    params={"only_new_growth": True},
                )
            )

        if context.summary["static_layer_stats"]["slope_p90"] >= 20:
            ops.append(
                CorrectionOperation(
                    "slope_boost",
                    strength=0.15,
                    radius=1,
                    params={"max_added_cells_factor": 1.0, "min_added_cells": 2},
                )
            )

        confidence = 0.55 + 0.25 * dryness
        if context.summary["pso_new_burn_cells"] == 0 and dryness > 0.45:
            confidence -= 0.1
        snippet_ids = [snippet.snippet_id for snippet in snippets]
        if context.summary.get("uses_physical_warm_start", True):
            rationale = (
                "PSO forecast is treated as the base map. Corrections emphasize downwind spread under dry or windy conditions, "
                "suppress aggressive growth under humidity or precipitation, and enforce burnable-fuel and monotonic-burn constraints."
            )
        else:
            rationale = (
                "No PSO/physics warm start is provided. The current burn map is treated as the base map, and corrections rely on "
                "weather, terrain, fuel, and retrieved rules only."
            )
        return CorrectionPlan(
            operations=ops,
            confidence=float(max(0.05, min(confidence, 0.95))),
            rationale=rationale,
            retrieved_snippet_ids=snippet_ids,
            source="heuristic",
            evidence_checklist={
                "burn_front_and_pso_prior": "uses PSO/base map and current burn front counts",
                "wind_speed_and_direction": f"wind {wind:.2f} mph from {weather.wind_direction_deg:.1f} deg",
                "temperature_humidity_dryness": f"dryness {dryness:.3f}, RH {humidity:.1f}",
                "slope_elevation_aspect": "uses slope summary; elevation/aspect are not explicit heuristic triggers",
                "fbfm13_fuel_and_canopy": "applies FBFM13 burnable-fuel mask",
                "retrieved_rules": ", ".join(snippet_ids),
            },
        )


class OllamaReasoner:
    """Ollama-backed structured correction planner."""

    def __init__(
        self,
        model: str = "llama4:latest",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_seconds: int = 180,
        cache_dir: str | Path | None = "outputs/ollama_cache",
        use_cache: bool = True,
        fallback_reasoner: Reasoner | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_seconds = int(timeout_seconds)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = bool(use_cache)
        self.fallback_reasoner = fallback_reasoner

    def propose_correction(
        self,
        inputs: PredictionInput,
        context: AgentContext,
        snippets: list[KnowledgeSnippet],
    ) -> CorrectionPlan:
        prompt = self._build_prompt(context, snippets)
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        if context.panel_path:
            payload["messages"][0]["images"] = [self._image_b64(context.panel_path)]

        try:
            content = self._cached_chat(payload)
            plan = self._parse_plan(content, snippets)
            return plan
        except Exception:
            if self.fallback_reasoner is None:
                raise
            fallback = self.fallback_reasoner.propose_correction(inputs, context, snippets)
            return CorrectionPlan(
                operations=fallback.operations,
                confidence=fallback.confidence,
                rationale=f"ollama_failed_fallback: {fallback.rationale}",
                retrieved_snippet_ids=fallback.retrieved_snippet_ids,
                source=f"ollama_failed_fallback:{self.model}",
                evidence_checklist=fallback.evidence_checklist,
            )

    def _chat(self, payload: dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for model {self.model}: {exc}") from exc
        return str(raw.get("message", {}).get("content", ""))

    def _cached_chat(self, payload: dict) -> str:
        if not self.use_cache or self.cache_dir is None:
            return self._chat(payload)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{self._cache_key(payload)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(cached["content"])
        content = self._chat(payload)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "host": self.host,
                    "temperature": self.temperature,
                    "content": content,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return content

    def _cache_key(self, payload: dict) -> str:
        stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def _build_prompt(self, context: AgentContext, snippets: list[KnowledgeSnippet]) -> str:
        snippet_text = [
            {"id": snippet.snippet_id, "title": snippet.title, "text": snippet.text}
            for snippet in snippets
        ]
        uses_physical_warm_start = bool(context.summary.get("uses_physical_warm_start", True))
        if uses_physical_warm_start:
            task_framing = (
                "Given a PSO/physics warm-start forecast, weather, static layer statistics, and retrieved rules, "
                "return only a JSON correction plan. Preserve the PSO warm start; propose only conservative structured corrections."
            )
        else:
            task_framing = (
                "No PSO/physics warm-start forecast is provided. Use the current burn map as the base state and reason from "
                "weather, static layer statistics, the image panel, and retrieved rules only. Return only a JSON correction plan."
            )
        schema = {
            "confidence": "float between 0 and 1",
            "rationale": (
                "short physical explanation that explicitly mentions evidence from "
                "burn front/PSO prior, wind, temperature/RH/dryness, slope/elevation/aspect, "
                "FBFM13 fuel/canopy, and retrieved rules"
            ),
            "evidence_checklist": {
                "burn_front_and_pso_prior": "one short phrase",
                "wind_speed_and_direction": "one short phrase",
                "temperature_humidity_dryness": "one short phrase",
                "slope_elevation_aspect": "one short phrase",
                "fbfm13_fuel_and_canopy": "one short phrase",
                "retrieved_rules": "one short phrase",
            },
            "operations": [
                {
                    "name": "fuel_mask | monotonic_burn | expand_downwind | suppress_growth | slope_boost | clip_probability",
                    "strength": "float between 0 and 1.5",
                    "radius": "integer 0 to 3",
                    "direction_deg": "number or null",
                    "params": "object; for expansion include max_added_cells_factor and min_added_cells",
                }
            ],
        }
        return (
            "You are a wildfire spread correction planner. Do not generate pixels. "
            f"{task_framing} "
            "Prefer no expansion when evidence is weak. Use small radii and area caps. "
            "You must inspect every provided modality before choosing operations: "
            "the burn-front/current burn map, PSO/base forecast panel, aspect panel, elevation panel, slope panel, "
            "FBFM13 fuel panel, canopy panel, weather values, metadata, and retrieved rules. "
            "If a modality is weak, noisy, or not decisive, say so in the evidence_checklist instead of ignoring it. "
            "Do not let visual intuition override the structured weather and static-layer JSON. "
            "Allowed operations are fuel_mask, monotonic_burn, expand_downwind, suppress_growth, slope_boost, clip_probability.\n\n"
            f"Required JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Context summary:\n{json.dumps(self._json_ready(context.summary), indent=2)}\n\n"
            f"Retrieved knowledge:\n{json.dumps(snippet_text, indent=2)}\n\n"
            "Return JSON only, with no markdown."
        )

    def _parse_plan(self, content: str, snippets: list[KnowledgeSnippet]) -> CorrectionPlan:
        parsed = self._loads_json(content)
        operations = []
        for item in parsed.get("operations", []):
            operations.append(
                CorrectionOperation(
                    name=str(item.get("name", "")),
                    strength=float(item.get("strength", 1.0)),
                    radius=int(item.get("radius", 1)),
                    direction_deg=None if item.get("direction_deg") is None else float(item.get("direction_deg")),
                    params=dict(item.get("params", {})),
                )
            )
        if not operations:
            operations = [
                CorrectionOperation(
                    "fuel_mask",
                    strength=1.0,
                    params={"min_fbfm13": 1, "max_fbfm13": 13, "nonburnable_codes": [0, 91, 92, 93, 98, 99]},
                ),
                CorrectionOperation("monotonic_burn", strength=1.0),
            ]
        return CorrectionPlan(
            operations=operations,
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=str(parsed.get("rationale", "Ollama generated structured correction plan.")),
            retrieved_snippet_ids=[snippet.snippet_id for snippet in snippets],
            source=f"ollama:{self.model}",
            evidence_checklist={
                str(key): str(value)
                for key, value in dict(parsed.get("evidence_checklist", {})).items()
            },
        )

    def _loads_json(self, content: str) -> dict:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _image_b64(self, path: str) -> str:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    def _json_ready(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_ready(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._json_ready(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
