from functools import partial
from string import ascii_uppercase
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_skj1_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating SKJ1 prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by mqa_prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question
        choices: List of possible answer choices
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        sentence = transformed["sentence"]
        choices = transformed["choices"]
        gold_idx = transformed["gold_idx"]

        letter_prefixes = [f"{letter}) " for letter in ascii_uppercase[: len(choices)]]
        choice_text = "\n".join(f"{prefix}{choice}" for prefix, choice in zip(letter_prefixes, choices))
        letter_string = ", ".join(ascii_uppercase[: len(choices)])

        output_choices = [f"{letter}" for letter in ascii_uppercase[: len(choices)]]

        query = query_template.format(sentence=sentence, choices=choice_text)
        instruction = instruction_template.format(letter_string=letter_string) if instruction_template else None

        return Doc(
            task_name=task_name,
            query=query,
            choices=output_choices,
            gold_index=gold_idx,
            instruction=instruction,
        )

    return prompt_fn


SKJ1_PROMPT_CONFIGS = {
    "prompt1_skj1_en": {
        "query_template": "Sentence: {sentence}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Check the correctness of the following sentence and select the option with the correct form ({letter_string}). Answer only with the letter of the correct option.",
    },
    "prompt1_skj1_sk": {
        "query_template": "Veta: {sentence}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Skontroluj správnosť nasledujúcej vety a vyber možnosť, v ktorej sa nachádza správny tvar ({letter_string}). Odpovedz len písmenom správnej možnosti.",
    },
    "prompt2_skj1_en": {
        "query_template": "Sentence: {sentence}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Select the grammatically correct version of the following sentence from the options: {letter_string}. Answer only with the letter of the correct option.",
    },
    "prompt2_skj1_sk": {
        "query_template": "Veta: {sentence}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Vyber gramaticky správnu verziu nasledujúcej vety z možností: {letter_string}. Odpovedz len písmenom správnej možnosti.",
    },
    "prompt3_skj1_en": {
        "query_template": "Sentence: {sentence}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Your task is to choose the grammatically correct version of the following sentence. Answer only with the letter of the correct option: {letter_string}.",
    },
    "prompt3_skj1_sk": {
        "query_template": "Veta: {sentence}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Tvojou úlohou je vybrať gramaticky správny tvar nasledujúcej vety. Odpovedz len písmenom správnej možnosti: {letter_string}.",
    },
}


for name, config in SKJ1_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_skj1_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_skj1)(prompt_fn)
