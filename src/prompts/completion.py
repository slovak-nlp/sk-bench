from functools import partial
from string import ascii_uppercase
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_completion_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating completion prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question and choices

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        question = transformed["question"]
        choices = transformed["choices"]
        gold_idx = transformed["gold_idx"]

        letter_prefixes = [f"{letter}) " for letter in ascii_uppercase[: len(choices)]]
        choice_text = "\n".join(f"{prefix}{choice}" for prefix, choice in zip(letter_prefixes, choices))
        letter_string = ", ".join(ascii_uppercase[: len(choices)])

        query = query_template.format(question=question, choices=choice_text)

        output_choices = [f"{letter}" for letter in ascii_uppercase[: len(choices)]]
        instruction = instruction_template.format(letter_string=letter_string) if instruction_template else None

        return Doc(
            task_name=task_name,
            query=query,
            choices=output_choices,
            gold_index=gold_idx,
            instruction=instruction,
        )

    return prompt_fn


COMPLETION_PROMPT_CONFIGS = {
    "prompt1_completion_en": {
        "query_template": "Text: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Complete the sentence by selecting one of the following options. Answer only by selecting the letter corresponding to the correct answer ({letter_string}).",
    },
    "prompt1_completion_sk": {
        "query_template": "Text: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Doplň pokračovanie textu výberom jednej z nasledujúcich možností. Odpovedz iba použitím písmena správnej odpovede ({letter_string}).",
    },
    "prompt2_completion_en": {
        "query_template": "Text: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Task: Select the correct continuation of the following text from options: {letter_string}. Answer only with the letter of the correct answer.",
    },
    "prompt2_completion_sk": {
        "query_template": "Text: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Úloha: Vyber správne pokračovanie nasledujúceho textu z možností: {letter_string}. Odpovedz len písmenom správnej odpovede.",
    },
    "prompt3_completion_en": {
        "query_template": "Text: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Choose the best option to complete the following text. Write only the letter of the correct answer.",
    },
    "prompt3_completion_sk": {
        "query_template": "Text: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Zvoľ najvhodnejšiu možnosť na dokončenie nasledujúceho textu. Napíš len písmeno správnej odpovede.",
    },
}

for name, config in COMPLETION_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_completion_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_completion)(prompt_fn)
