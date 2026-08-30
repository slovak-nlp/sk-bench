from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_toxicity_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating toxicity prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for text
        query_template: Template for the query with placeholder for text
        choices: List of choice texts

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        text = transformed["text"]
        gold_idx = transformed["gold_idx"]

        query = query_template.format(text=text)

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction_template,
        )

    return prompt_fn


TOXICITY_PROMPT_CONFIGS = {
    "prompt1_toxicity_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Determine whether the given text includes toxic speech. Answer only "yes" or "no".',
        "choices": ["No", "Yes"],
    },
    "prompt1_toxicity_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Urči, či daný text obsahuje toxické prejavy. Odpovedz len „áno“ alebo „nie“.",
        "choices": ["Nie", "Áno"],
    },
    "prompt2_toxicity_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Is the following text toxic? Answer only "yes" or "no".',
        "choices": ["No", "Yes"],
    },
    "prompt2_toxicity_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Je nasledujúci text toxický? Odpovedz len „áno“ alebo „nie“.",
        "choices": ["Nie", "Áno"],
    },
    "prompt3_toxicity_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Check the text below to see if it contains any form of toxic content. Write "yes" if the text contains toxic content, otherwise "no".',
        "choices": ["No", "Yes"],
    },
    "prompt3_toxicity_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Zisti, či v texte nižšie nájdeš akúkoľvek formu toxického obsahu. Napíš „áno“, ak sa v texte vyskytuje toxický obsah, inak „nie“.",
        "choices": ["Nie", "Áno"],
    },
}


for name, config in TOXICITY_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_toxicity_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_toxicity)(prompt_fn)
