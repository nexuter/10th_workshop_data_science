"""Repository-local import shim for the src-layout package.

This lets `python -m wildfire_llm_agent.cli ...` work from the repository root
without setting PYTHONPATH or installing the package in editable mode.
"""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "wildfire_llm_agent"
if _SRC_PACKAGE.exists():
    __path__.insert(0, str(_SRC_PACKAGE))

from .agent import WildfireAgent
from .schemas import (
    CorrectionOperation,
    CorrectionPlan,
    PredictionInput,
    PredictionOutput,
    StaticLayers,
    WeatherObservation,
)

__all__ = [
    "CorrectionOperation",
    "CorrectionPlan",
    "PredictionInput",
    "PredictionOutput",
    "StaticLayers",
    "WeatherObservation",
    "WildfireAgent",
]
