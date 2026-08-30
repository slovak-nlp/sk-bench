"""Category definitions for prompts.

This module defines an Enum for prompt category names used in prompt
registrations. Using an Enum ensures categories are well-defined in one place
and provides type safety.

Prompt categories use the `p_{category}` naming convention (e.g., `p_sentiment_binary`)
to distinguish them from task categories.
"""

from enum import StrEnum


class PromptCategory(StrEnum):
    """Enum for prompt categories.

    Prompt categories use the `p_` prefix (e.g., p_sentiment_binary) to distinguish
    them from task categories (which use `c_`).
    """

    p_analogy = "p_analogy"
    p_chiby = "p_chiby"
    p_completion = "p_completion"
    p_copa = "p_copa"
    p_declension = "p_declension"
    p_gradation = "p_gradation"
    p_hate_speech = "p_hate_speech"
    p_identity_form = "p_identity_form"
    p_ifeval = "p_ifeval"
    p_inflection = "p_inflection"
    p_multi_choice = "p_multi_choice"
    p_multi_select = "p_multi_select"
    p_nli_binary = "p_nli_binary"
    p_nli_3class = "p_nli_3class"
    p_question_answering = "p_question_answering"
    p_sentiment_binary = "p_sentiment_binary"
    p_sentiment_3class = "p_sentiment_3class"
    p_skj1 = "p_skj1"
    p_summarization = "p_summarization"
    p_toxicity = "p_toxicity"
    p_oneruler = "p_oneruler"


__all__ = [
    "PromptCategory",
]
