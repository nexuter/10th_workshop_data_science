from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeSnippet:
    snippet_id: str
    title: str
    text: str
    tags: tuple[str, ...]


DEFAULT_SNIPPETS = [
    KnowledgeSnippet(
        "fbfm13_rothermel",
        "FBFM13 as a fire-behavior prior",
        "The Anderson 13 fire behavior fuel models represent surface fuel distributions and are used as inputs to Rothermel-style surface fire spread models.",
        ("fuel", "fbfm13", "rothermel"),
    ),
    KnowledgeSnippet(
        "wind_alignment",
        "Wind-aligned spread",
        "Higher wind speed increases directional spread in the downwind direction; forecasts that expand mainly upwind should be treated as low-confidence unless terrain dominates.",
        ("weather", "wind", "direction"),
    ),
    KnowledgeSnippet(
        "slope_effect",
        "Slope and aspect effects",
        "Fire generally spreads faster upslope because flames and convective heat preheat fuels above the active front.",
        ("topography", "slope", "aspect"),
    ),
    KnowledgeSnippet(
        "humidity_suppression",
        "Humidity and precipitation suppression",
        "High relative humidity and recent precipitation reduce ignition likelihood and should suppress aggressive perimeter growth.",
        ("weather", "humidity", "precipitation"),
    ),
    KnowledgeSnippet(
        "llm_guardrail",
        "LLM correction guardrail",
        "A language model should not directly emit pixels for high-stakes geospatial prediction; it should emit structured corrections that deterministic tools can validate and render.",
        ("llm", "tool-use", "guardrail"),
    ),
]


class RuleBasedRetriever:
    def __init__(self, snippets: list[KnowledgeSnippet] | None = None) -> None:
        self.snippets = snippets or list(DEFAULT_SNIPPETS)

    def retrieve(self, query: str, top_k: int = 4) -> list[KnowledgeSnippet]:
        terms = {term.strip(".,:;()[]").lower() for term in query.split()}

        def score(snippet: KnowledgeSnippet) -> int:
            haystack = " ".join((snippet.title, snippet.text, " ".join(snippet.tags))).lower()
            return sum(1 for term in terms if term and term in haystack)

        ranked = sorted(self.snippets, key=score, reverse=True)
        nonzero = [snippet for snippet in ranked if score(snippet) > 0]
        return (nonzero or ranked)[:top_k]
