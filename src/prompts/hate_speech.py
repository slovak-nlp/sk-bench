from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_hate_speech_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating hate speech prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for context
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


HATE_SPEECH_PROMPT_CONFIGS = {
    "prompt1_hate_speech_en": {
        "query_template": "Claim: {text}\nAnswer:",
        "instruction_template": 'Determine whether the given claim includes hate speech. Answer only "yes" or "no".',
        "choices": ["No", "Yes"],
    },
    "prompt1_hate_speech_sk": {
        "query_template": "Výrok: {text}\nOdpoveď:",
        "instruction_template": "Urči, či zadaný výrok obsahuje nenávistné prejavy. Odpovedz len slovom „áno“ alebo „nie“.",
        "choices": ["Nie", "Áno"],
    },
    "prompt2_hate_speech_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": "Does the following text contain hate speech? Answer yes/no.",
        "choices": ["No", "Yes"],
    },
    "prompt2_hate_speech_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Obsahuje nasledujúci text nenávistné prejavy? Odpovedz áno/nie.",
        "choices": ["Nie", "Áno"],
    },
    "prompt3_hate_speech_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Determine whether the following text contains signs of hate speech. Respond only with the word "yes" or "no".',
        "choices": ["No", "Yes"],
    },
    "prompt3_hate_speech_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Rozhodni, či nasledujúci text obsahuje známky nenávistného prejavu (hate speech). Odpovedz iba slovom „áno“ alebo „nie“.",
        "choices": ["Nie", "Áno"],
    },
}

for name, config in HATE_SPEECH_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_hate_speech_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_hate_speech)(prompt_fn)
