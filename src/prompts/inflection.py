from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_inflection_prompt_base(
    transform: Callable,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Inflection prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        query_template: Template for the query with placeholder for question
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        lemma = transformed["lemma"]
        target_description = transformed["target_description"]
        answer = transformed["answer"]

        query = query_template.format(lemma=lemma, target_description=target_description)

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=None,
        )

    return prompt_fn


INFLECTION_PROMPT_CONFIGS = {
    "prompt1_inflection_sk": {
        "query_template": "Uveď tvar slovesa „{lemma}“ {target_description}. Odpovedz iba samotným tvarom slovesa.",
    },
    "prompt2_inflection_sk": {
        "query_template": "Vyčasuj sloveso „{lemma}“ {target_description}. Odpovedz iba jedným slovom.",
    },
    "prompt3_inflection_sk": {
        "query_template": "Úloha: pre sloveso „{lemma}“ napíš tvar {target_description}. V odpovedi uveď iba jedno slovo.",
    },
}

for name, config in INFLECTION_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_inflection_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_inflection)(prompt_fn)
