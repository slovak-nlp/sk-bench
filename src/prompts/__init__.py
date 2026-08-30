# Import the modules to trigger their prompt registrations
from src.prompts import (
    analogy,
    chiby,
    completion,
    copa,
    declension,
    gradation,
    hate_speech,
    identity_form,
    ifeval,
    inflection,
    mqa,
    nli,
    oneruler,
    qa,
    sentiment_analysis,
    skj1,
    summarization,
    toxicity,
)
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


__all__ = [
    "PROMPT_REGISTRY",
    "PromptCategory",
    "analogy",
    "chiby",
    "completion",
    "copa",
    "declension",
    "gradation",
    "hate_speech",
    "identity_form",
    "ifeval",
    "inflection",
    "mqa",
    "nli",
    "oneruler",
    "qa",
    "sentiment_analysis",
    "skj1",
    "summarization",
    "toxicity",
]
