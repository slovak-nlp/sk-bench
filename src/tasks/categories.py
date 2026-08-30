"""Category definitions for tasks.

This module defines an Enum for task category names used in task registrations.
Using an Enum ensures categories are well-defined in one place and provides
type safety.

Task categories use the `c_{category}` naming convention (e.g., `c_sentiment`)
to easily distinguish between task_name and category.
"""

from enum import StrEnum


class TaskCategory(StrEnum):
    """Enum for task categories.

    Categories are named with the `c_` prefix (e.g., c_sentiment) to easily
    distinguish between task_name and category. The string values also use the
    c_ prefix.
    """

    c_analogy = "c_analogy"
    c_completion = "c_completion"
    c_copa = "c_copa"
    c_grammar = "c_grammar"
    c_hate_speech = "c_hate_speech"
    c_ifeval = "c_ifeval"
    c_multi_choice = "c_multi_choice"
    c_nli = "c_nli"
    c_question_answering = "c_question_answering"
    c_sentiment = "c_sentiment"
    c_summarization = "c_summarization"
    c_toxicity = "c_toxicity"
    c_oneruler = "c_oneruler"


__all__ = [
    "TaskCategory",
]
