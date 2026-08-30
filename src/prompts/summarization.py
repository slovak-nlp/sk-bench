from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_summarization_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating summarization prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for text
        query_template: Template for the query with placeholder for text

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        text = transformed["text"]
        summary = transformed["summary"]

        query = query_template.format(text=text)

        return Doc(
            task_name=task_name,
            query=query,
            choices=[summary],
            gold_index=0,
            instruction=instruction_template,
        )

    return prompt_fn


SUMMARIZATION_PROMPT_CONFIGS = {
    "prompt1_summarization_en": {
        "query_template": "{text}\n\nWrite a summary for the text above with a maximum of 50 words. The output has to be in Slovak.\n\nSummary:",
        "instruction_template": None,
    },
    "prompt1_summarization_sk": {
        "query_template": "{text}\n\nNapíš zhrnutie vyššie uvedeného textu s využitím maximálne 50 slov. Výstup musí byť v slovenčine.\n\nZhrnutie:",
        "instruction_template": None,
    },
    "prompt2_summarization_en": {
        "query_template": "Text: {text}\n\nSummary:",
        "instruction_template": "Summarize the following text in Slovak into a maximum of 50 words.",
    },
    "prompt2_summarization_sk": {
        "query_template": "Text: {text}\n\nSúhrn:",
        "instruction_template": "Sumarizuj nasledujúci text v slovenčine do maximálne 50 slov.",
    },
    "prompt3_summarization_en": {
        "query_template": "Text: {text}\n\nSummary:",
        "instruction_template": "Read the following text and summarize it briefly in Slovak, using no more than 50 words.",
    },
    "prompt3_summarization_sk": {
        "query_template": "Text: {text}\n\nZhrnutie:",
        "instruction_template": "Prečítaj si nasledujúci text a stručne ho sumarizuj v slovenčine, s využitím maximálne 50 slov.",
    },
}


for name, config in SUMMARIZATION_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_summarization_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_summarization)(prompt_fn)
